/*
 * run_gptoss.c — GPT-OSS 20B C inference engine
 *
 * Built from scratch for OpenAI GPT-OSS 20B (Mixture of Experts).
 * 20.9B total params, 3.6B active per token.
 *
 * Architecture:
 *   - 24 layers, hidden_dim=2880, head_dim=64
 *   - GQA 8:1 (64 query heads, 8 KV heads)
 *   - MoE: 32 experts, top-4 active (MXFP4 weights + E8M0 scales)
 *   - float16 attention weights, embeddings, biases
 *   - Learnable RMSNorm, attention sinks, SwiGLU activation
 *   - Alternating sliding window (even=128, odd=full)
 *   - Half-split RoPE with theta=150000
 *   - tiktoken BPE tokenizer (o200k, 201088 vocab)
 */

#include "run_gptoss.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Platform-specific includes
#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#include <io.h>
#include <fcntl.h>
#else
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#endif

// SIMD support (compile with -mavx2 -mfma -mf16c to enable)
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
#include <immintrin.h>
#define USE_SIMD 1
#else
#define USE_SIMD 0
#endif

// ============================================================================
// Section 1: Float16 conversion
// ============================================================================

static inline float f16_to_f32(uint16_t h) {
    // IEEE 754 float16 → float32
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0) {
        if (mant == 0) {
            // ±zero
            union { uint32_t u; float f; } u = { sign };
            return u.f;
        }
        // Subnormal: normalize
        while (!(mant & 0x400)) { mant <<= 1; exp--; }
        exp++; mant &= 0x3FF;
    } else if (exp == 31) {
        // Inf or NaN
        union { uint32_t u; float f; } u = { sign | 0x7F800000 | ((uint32_t)mant << 13) };
        return u.f;
    }

    union { uint32_t u; float f; } u = { sign | ((uint32_t)(exp + 112) << 23) | ((uint32_t)mant << 13) };
    return u.f;
}

// ============================================================================
// Section 2: Math helpers
// ============================================================================

static void rmsnorm_scaled(float* out, const float* x, const uint16_t* scale_f16, int size) {
    // RMSNorm with learnable scale: out = (x / sqrt(mean(x²) + eps)) * scale
    float ss = 0.0f;
#if USE_SIMD
    __m256 acc = _mm256_setzero_ps();
    int j;
    for (j = 0; j + 7 < size; j += 8) {
        __m256 xv = _mm256_loadu_ps(x + j);
        acc = _mm256_fmadd_ps(xv, xv, acc);
    }
    // Horizontal sum
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 s4 = _mm_add_ps(lo, hi);
    s4 = _mm_hadd_ps(s4, s4);
    s4 = _mm_hadd_ps(s4, s4);
    ss = _mm_cvtss_f32(s4);
    for (; j < size; j++) ss += x[j] * x[j];
#else
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
#endif
    ss /= size;
    ss += 1e-5f;  // GPT-OSS uses eps=1e-5
    ss = 1.0f / sqrtf(ss);
#if USE_SIMD
    __m256 ss_v = _mm256_set1_ps(ss);
    for (j = 0; j + 7 < size; j += 8) {
        __m128i h8 = _mm_loadu_si128((__m128i*)(scale_f16 + j));
        __m256 sc = _mm256_cvtph_ps(h8);
        __m256 xv = _mm256_loadu_ps(x + j);
        __m256 res = _mm256_mul_ps(_mm256_mul_ps(xv, ss_v), sc);
        _mm256_storeu_ps(out + j, res);
    }
    for (; j < size; j++) {
        out[j] = x[j] * ss * f16_to_f32(scale_f16[j]);
    }
#else
    for (int j = 0; j < size; j++) {
        out[j] = x[j] * ss * f16_to_f32(scale_f16[j]);
    }
#endif
}

static void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// Float16 matrix-vector multiply: out = W @ x  (W is float16, x is float32)
// W is (d, n) row-major, x is (n,), out is (d,)
static void f16_matmul(float* out, const float* x, const uint16_t* w, int n, int d) {
    #pragma omp parallel for
    for (int i = 0; i < d; i++) {
        const uint16_t* wi = w + (size_t)i * n;
#if USE_SIMD
        __m256 acc = _mm256_setzero_ps();
        int j;
        for (j = 0; j + 7 < n; j += 8) {
            __m128i h8 = _mm_loadu_si128((__m128i*)(wi + j));
            __m256 w8 = _mm256_cvtph_ps(h8);
            __m256 x8 = _mm256_loadu_ps(x + j);
            acc = _mm256_fmadd_ps(w8, x8, acc);
        }
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 s4 = _mm_add_ps(lo, hi);
        s4 = _mm_hadd_ps(s4, s4);
        s4 = _mm_hadd_ps(s4, s4);
        float val = _mm_cvtss_f32(s4);
        for (; j < n; j++) val += f16_to_f32(wi[j]) * x[j];
        out[i] = val;
#else
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += f16_to_f32(wi[j]) * x[j];
        }
        out[i] = val;
#endif
    }
}

// Float16 matrix-vector multiply with float16 bias
static void f16_matmul_bias(float* out, const float* x, const uint16_t* w,
                             const uint16_t* bias, int n, int d) {
    #pragma omp parallel for
    for (int i = 0; i < d; i++) {
        const uint16_t* wi = w + (size_t)i * n;
#if USE_SIMD
        __m256 acc = _mm256_setzero_ps();
        int j;
        for (j = 0; j + 7 < n; j += 8) {
            __m128i h8 = _mm_loadu_si128((__m128i*)(wi + j));
            __m256 w8 = _mm256_cvtph_ps(h8);
            __m256 x8 = _mm256_loadu_ps(x + j);
            acc = _mm256_fmadd_ps(w8, x8, acc);
        }
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 s4 = _mm_add_ps(lo, hi);
        s4 = _mm_hadd_ps(s4, s4);
        s4 = _mm_hadd_ps(s4, s4);
        float val = _mm_cvtss_f32(s4);
        for (; j < n; j++) val += f16_to_f32(wi[j]) * x[j];
        out[i] = val + f16_to_f32(bias[i]);
#else
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += f16_to_f32(wi[j]) * x[j];
        }
        out[i] = val + f16_to_f32(bias[i]);
#endif
    }
}

// ============================================================================
// Section 3: MXFP4 dequantization + dot product
// ============================================================================

// Dot product of one MXFP4 row with a float32 input vector.
// Fused dequant: never materializes the full float32 weight row.
//   blocks: packed uint8 bytes (2 FP4 values per byte)
//   scales: E8M0 scale bytes (one per group of 32 elements)
//   x: float32 input vector
//   n_groups: number of scale groups (= input_dim / 32)
static inline float mxfp4_dot_row(const uint8_t* blocks, const uint8_t* scales,
                                   const float* x, int n_groups) {
#if USE_SIMD
    __m256 total_acc = _mm256_setzero_ps();
    for (int g = 0; g < n_groups; g++) {
        float scale = ldexpf(1.0f, (int)scales[g] - 127);
        __m256 scale_v = _mm256_set1_ps(scale);
        __m256 group_acc = _mm256_setzero_ps();
        const uint8_t* gb = blocks + g * 16;
        const float* gx = x + g * 32;

        // Process 16 bytes (32 FP4 values) in 4 iterations of 8 values
        for (int j = 0; j < 16; j += 4) {
            __m256 wv = _mm256_set_ps(
                FP4_LUT[gb[j+3] >> 4],   FP4_LUT[gb[j+3] & 0x0F],
                FP4_LUT[gb[j+2] >> 4],   FP4_LUT[gb[j+2] & 0x0F],
                FP4_LUT[gb[j+1] >> 4],   FP4_LUT[gb[j+1] & 0x0F],
                FP4_LUT[gb[j]   >> 4],   FP4_LUT[gb[j]   & 0x0F]
            );
            __m256 xv = _mm256_loadu_ps(gx + 2 * j);
            group_acc = _mm256_fmadd_ps(wv, xv, group_acc);
        }
        total_acc = _mm256_fmadd_ps(group_acc, scale_v, total_acc);
    }
    // Horizontal sum
    __m128 hi = _mm256_extractf128_ps(total_acc, 1);
    __m128 lo = _mm256_castps256_ps128(total_acc);
    __m128 s4 = _mm_add_ps(lo, hi);
    s4 = _mm_hadd_ps(s4, s4);
    s4 = _mm_hadd_ps(s4, s4);
    return _mm_cvtss_f32(s4);
#else
    float total = 0.0f;
    for (int g = 0; g < n_groups; g++) {
        float scale = ldexpf(1.0f, (int)scales[g] - 127);
        float group_dot = 0.0f;
        const uint8_t* gb = blocks + g * 16;
        const float* gx = x + g * 32;

        for (int j = 0; j < 16; j++) {
            uint8_t byte = gb[j];
            group_dot += FP4_LUT[byte & 0x0F] * gx[2 * j];
            group_dot += FP4_LUT[byte >> 4]   * gx[2 * j + 1];
        }
        total += group_dot * scale;
    }
    return total;
#endif
}

// Full MXFP4 matrix-vector multiply with float16 bias:
//   out[i] = mxfp4_dot(row_i, x) + bias[i]
static void mxfp4_matmul_bias(float* out, const float* x,
                                const uint8_t* blocks, const uint8_t* scales,
                                const uint16_t* bias,
                                int in_dim, int out_dim,
                                int packed_cols, int n_groups) {
    #pragma omp parallel for
    for (int i = 0; i < out_dim; i++) {
        const uint8_t* row_blocks = blocks + (size_t)i * packed_cols;
        const uint8_t* row_scales = scales + (size_t)i * n_groups;
        out[i] = mxfp4_dot_row(row_blocks, row_scales, x, n_groups)
                 + f16_to_f32(bias[i]);
    }
}

// ============================================================================
// Section 4: SwiGLU activation (GPT-OSS variant)
// ============================================================================

// GPT-OSS SwiGLU: interleaved gate+linear in mlp1 output
//   h = [gate_0, linear_0, gate_1, linear_1, ...]  (size = 2 * intermediate_size)
//   out[i] = clamp(gate) * sigmoid(1.702 * clamp(gate)) * (clamp(linear) + 1.0)
static void swiglu(float* out, const float* h, int intermediate_size) {
    for (int i = 0; i < intermediate_size; i++) {
        float gate   = h[2 * i];       // even indices
        float linear = h[2 * i + 1];   // odd indices

        // Clamp: gate max=7, linear [-7, 7]
        if (gate > 7.0f) gate = 7.0f;
        if (linear < -7.0f) linear = -7.0f;
        if (linear >  7.0f) linear =  7.0f;

        // gate_act = gate * sigmoid(1.702 * gate)
        float gate_act = gate * (1.0f / (1.0f + expf(-1.702f * gate)));
        out[i] = gate_act * (linear + 1.0f);
    }
}

// ============================================================================
// Section 5: Top-k selection for MoE routing
// ============================================================================

static void topk(const float* logits, int n, int k, int* indices, float* values) {
    // Initialize with very negative values
    for (int i = 0; i < k; i++) {
        values[i] = -1e30f;
        indices[i] = -1;
    }
    // Simple O(n*k) selection — fine for n=32, k=4
    for (int j = 0; j < n; j++) {
        // Find the minimum in current top-k
        int min_pos = 0;
        for (int p = 1; p < k; p++) {
            if (values[p] < values[min_pos]) min_pos = p;
        }
        if (logits[j] > values[min_pos]) {
            values[min_pos] = logits[j];
            indices[min_pos] = j;
        }
    }
    // Sort top-k descending by value (bubble sort, k=4 so this is trivial)
    for (int i = 0; i < k - 1; i++) {
        for (int j = i + 1; j < k; j++) {
            if (values[j] > values[i]) {
                float tv = values[i]; values[i] = values[j]; values[j] = tv;
                int ti = indices[i]; indices[i] = indices[j]; indices[j] = ti;
            }
        }
    }
}

// ============================================================================
// Section 6: Memory allocation
// ============================================================================

// Aligned allocation for SIMD (32-byte alignment for AVX2)
static void* aligned_calloc(size_t count, size_t size) {
    size_t total = count * size;
#ifdef _WIN32
    void* p = _aligned_malloc(total, 32);
    if (p) memset(p, 0, total);
#else
    void* p = NULL;
    posix_memalign(&p, 32, total);
    if (p) memset(p, 0, total);
#endif
    return p;
}

static void aligned_free(void* p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

static void malloc_run_state(RunState* s, Config* p, int qkv_dim, int attn_out_dim,
                              int kv_dim, int mlp1_out_dim) {
    s->x          = (float*)aligned_calloc(p->hidden_size, sizeof(float));
    s->xb         = (float*)aligned_calloc(p->hidden_size, sizeof(float));
    s->xb2        = (float*)aligned_calloc(p->hidden_size, sizeof(float));
    s->qkv        = (float*)aligned_calloc(qkv_dim, sizeof(float));
    s->attn_out   = (float*)aligned_calloc(attn_out_dim, sizeof(float));
    s->att        = (float*)aligned_calloc(p->num_heads * (p->max_seq_len + 1), sizeof(float));
    s->gate_logits = (float*)aligned_calloc(p->num_experts, sizeof(float));
    for (int e = 0; e < 4; e++) {
        s->expert_bufs[e] = (float*)aligned_calloc(mlp1_out_dim, sizeof(float));
        s->expert_acts[e] = (float*)aligned_calloc(p->intermediate_size, sizeof(float));
        s->expert_outs[e] = (float*)aligned_calloc(p->hidden_size, sizeof(float));
    }
    s->moe_out    = (float*)aligned_calloc(p->hidden_size, sizeof(float));
    s->logits     = (float*)aligned_calloc(p->vocab_size, sizeof(float));
    s->key_cache  = (float*)aligned_calloc((size_t)p->num_layers * p->max_seq_len * kv_dim, sizeof(float));
    s->value_cache = (float*)aligned_calloc((size_t)p->num_layers * p->max_seq_len * kv_dim, sizeof(float));

    if (!s->x || !s->xb || !s->xb2 || !s->qkv || !s->attn_out || !s->att ||
        !s->gate_logits || !s->expert_bufs[0] || !s->expert_acts[0] || !s->expert_outs[0] ||
        !s->moe_out || !s->logits || !s->key_cache || !s->value_cache) {
        fprintf(stderr, "ERROR: malloc failed for RunState! Need ~%zu MB\n",
                ((size_t)p->num_layers * p->max_seq_len * kv_dim * 2 * 4) / (1024 * 1024));
        exit(EXIT_FAILURE);
    }
}

static void free_run_state(RunState* s) {
    aligned_free(s->x);       aligned_free(s->xb);        aligned_free(s->xb2);
    aligned_free(s->qkv);     aligned_free(s->attn_out);  aligned_free(s->att);
    aligned_free(s->gate_logits);
    for (int e = 0; e < 4; e++) {
        aligned_free(s->expert_bufs[e]);
        aligned_free(s->expert_acts[e]);
        aligned_free(s->expert_outs[e]);
    }
    aligned_free(s->moe_out);    aligned_free(s->logits);
    aligned_free(s->key_cache);  aligned_free(s->value_cache);
}

// ============================================================================
// Section 7: Model loading via memory mapping
// ============================================================================

static void memory_map_weights(TransformerWeights* w, Config* cfg, uint8_t* ptr,
                                int qkv_dim, int attn_out_dim, int mlp1_out_dim,
                                int packed_cols, int num_groups) {
    int H = cfg->hidden_size;
    int E = cfg->num_experts;

    // Section 1: Embedding (vocab_size × hidden_size × 2 bytes)
    w->embedding = (uint16_t*)ptr;
    ptr += (size_t)cfg->vocab_size * H * 2;

    // Section 2: Per-layer weights
    for (int l = 0; l < cfg->num_layers; l++) {
        LayerWeights* lw = &w->layers[l];

        // Attention
        lw->attn_norm_scale = (uint16_t*)ptr;  ptr += H * 2;
        lw->qkv_weight      = (uint16_t*)ptr;  ptr += (size_t)qkv_dim * H * 2;
        lw->qkv_bias        = (uint16_t*)ptr;  ptr += qkv_dim * 2;
        lw->attn_sinks       = (uint16_t*)ptr;  ptr += cfg->num_heads * 2;
        lw->out_weight       = (uint16_t*)ptr;  ptr += (size_t)H * attn_out_dim * 2;
        lw->out_bias         = (uint16_t*)ptr;  ptr += H * 2;

        // MoE
        lw->mlp_norm_scale   = (uint16_t*)ptr;  ptr += H * 2;
        lw->gate_weight      = (uint16_t*)ptr;  ptr += (size_t)E * H * 2;
        lw->gate_bias        = (uint16_t*)ptr;  ptr += E * 2;

        lw->mlp1_blocks      = (uint8_t*)ptr;   ptr += (size_t)E * mlp1_out_dim * packed_cols;
        lw->mlp1_scales      = (uint8_t*)ptr;   ptr += (size_t)E * mlp1_out_dim * num_groups;
        lw->mlp1_bias        = (uint16_t*)ptr;  ptr += (size_t)E * mlp1_out_dim * 2;

        lw->mlp2_blocks      = (uint8_t*)ptr;   ptr += (size_t)E * H * packed_cols;
        lw->mlp2_scales      = (uint8_t*)ptr;   ptr += (size_t)E * H * num_groups;
        lw->mlp2_bias        = (uint16_t*)ptr;  ptr += (size_t)E * H * 2;
    }

    // Section 3: Final norm + unembedding
    w->final_norm_scale = (uint16_t*)ptr;  ptr += H * 2;
    w->unembedding      = (uint16_t*)ptr;  ptr += (size_t)cfg->vocab_size * H * 2;
}

void build_transformer(Transformer* t, const char* checkpoint_path) {
    // Read config header (64 bytes = 16 × int32)
    FILE* file = fopen(checkpoint_path, "rb");
    if (!file) {
        fprintf(stderr, "ERROR: Could not open model file: %s\n", checkpoint_path);
        exit(EXIT_FAILURE);
    }
    if (fread(&t->config, sizeof(int), 16, file) != 16) {
        fprintf(stderr, "ERROR: Failed to read 64-byte header\n");
        exit(EXIT_FAILURE);
    }
    fclose(file);

    Config* p = &t->config;

    // Validate magic number
    if (p->magic != GOSS_MAGIC) {
        fprintf(stderr, "ERROR: Invalid magic number 0x%08X (expected 0x%08X)\n",
                p->magic, GOSS_MAGIC);
        exit(EXIT_FAILURE);
    }

    // Compute derived constants
    t->qkv_dim        = p->head_dim * (p->num_heads + 2 * p->num_kv_heads);
    t->attn_out_dim   = p->head_dim * p->num_heads;
    t->kv_dim         = p->head_dim * p->num_kv_heads;
    t->gqa_ratio      = p->num_heads / p->num_kv_heads;
    t->mlp1_out_dim   = p->intermediate_size * 2;
    t->mlp1_packed_cols = p->hidden_size / 2;
    t->num_groups     = p->hidden_size / 32;

    fprintf(stderr, "=== GPT-OSS 20B Configuration ===\n");
    fprintf(stderr, "  hidden_size       = %d\n", p->hidden_size);
    fprintf(stderr, "  intermediate_size = %d\n", p->intermediate_size);
    fprintf(stderr, "  num_layers        = %d\n", p->num_layers);
    fprintf(stderr, "  num_heads         = %d (query)\n", p->num_heads);
    fprintf(stderr, "  num_kv_heads      = %d (GQA %d:1)\n", p->num_kv_heads, t->gqa_ratio);
    fprintf(stderr, "  head_dim          = %d\n", p->head_dim);
    fprintf(stderr, "  vocab_size        = %d\n", p->vocab_size);
    fprintf(stderr, "  num_experts       = %d (top-%d active)\n", p->num_experts, p->experts_per_token);
    fprintf(stderr, "  sliding_window    = %d\n", p->sliding_window);
    fprintf(stderr, "  max_seq_len       = %d\n", p->max_seq_len);
    fprintf(stderr, "  rope_theta        = %d\n", p->rope_theta);
    fprintf(stderr, "  eos_token_id      = %d\n", p->eos_token_id);
    fprintf(stderr, "  qkv_dim           = %d\n", t->qkv_dim);
    fprintf(stderr, "  attn_out_dim      = %d\n", t->attn_out_dim);
    fprintf(stderr, "  kv_dim            = %d\n", t->kv_dim);
    fprintf(stderr, "  mlp1_out_dim      = %d (SwiGLU interleaved)\n", t->mlp1_out_dim);
    fprintf(stderr, "  MXFP4 groups      = %d (group_size=32)\n", t->num_groups);

    // Memory-map the file
#ifdef _WIN32
    HANDLE hFile = CreateFileA(checkpoint_path, GENERIC_READ, FILE_SHARE_READ,
                               NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "ERROR: CreateFile failed for %s\n", checkpoint_path);
        exit(EXIT_FAILURE);
    }
    LARGE_INTEGER fileSize;
    GetFileSizeEx(hFile, &fileSize);
    t->file_size = (size_t)fileSize.QuadPart;

    HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hMapping) {
        fprintf(stderr, "ERROR: CreateFileMapping failed\n");
        CloseHandle(hFile);
        exit(EXIT_FAILURE);
    }
    t->data = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
    if (!t->data) {
        fprintf(stderr, "ERROR: MapViewOfFile failed\n");
        CloseHandle(hMapping);
        CloseHandle(hFile);
        exit(EXIT_FAILURE);
    }
    CloseHandle(hMapping);
    CloseHandle(hFile);
    t->fd = -1;
#else
    t->fd = open(checkpoint_path, O_RDONLY);
    if (t->fd == -1) {
        fprintf(stderr, "ERROR: open failed for %s\n", checkpoint_path);
        exit(EXIT_FAILURE);
    }
    t->file_size = lseek(t->fd, 0, SEEK_END);
    t->data = mmap(NULL, t->file_size, PROT_READ, MAP_PRIVATE, t->fd, 0);
    if (t->data == MAP_FAILED) {
        fprintf(stderr, "ERROR: mmap failed\n");
        close(t->fd);
        exit(EXIT_FAILURE);
    }
#endif

    fprintf(stderr, "  Model file size   = %.2f GB\n", t->file_size / (1024.0 * 1024.0 * 1024.0));

    // Point weights into mmap'd data (skip 64-byte header)
    uint8_t* weights_ptr = (uint8_t*)t->data + 64;
    memory_map_weights(&t->weights, &t->config, weights_ptr,
                       t->qkv_dim, t->attn_out_dim, t->mlp1_out_dim,
                       t->mlp1_packed_cols, t->num_groups);

    // Allocate run state buffers
    malloc_run_state(&t->state, &t->config, t->qkv_dim, t->attn_out_dim,
                     t->kv_dim, t->mlp1_out_dim);

    // Precompute RoPE sin/cos tables
    {
        int half_dim = p->head_dim / 2;
        int max_seq = p->max_seq_len;
        float theta = (float)p->rope_theta;
        t->rope_cos = (float*)malloc((size_t)max_seq * half_dim * sizeof(float));
        t->rope_sin = (float*)malloc((size_t)max_seq * half_dim * sizeof(float));
        if (!t->rope_cos || !t->rope_sin) {
            fprintf(stderr, "ERROR: Failed to allocate RoPE tables\n");
            exit(EXIT_FAILURE);
        }
        for (int pos = 0; pos < max_seq; pos++) {
            for (int i = 0; i < half_dim; i++) {
                float freq = 1.0f / powf(theta, (float)(2 * i) / (float)p->head_dim);
                float angle = (float)pos * freq;
                t->rope_cos[pos * half_dim + i] = cosf(angle);
                t->rope_sin[pos * half_dim + i] = sinf(angle);
            }
        }
        fprintf(stderr, "  RoPE tables       = %.1f KB\n",
                (2.0 * max_seq * half_dim * sizeof(float)) / 1024.0);
    }

    // Print memory usage
    size_t kv_bytes = (size_t)p->num_layers * p->max_seq_len * t->kv_dim * 2 * sizeof(float);
    size_t act_bytes = (p->hidden_size * 3 + t->qkv_dim + t->attn_out_dim +
                        p->num_heads * (p->max_seq_len + 1) + p->num_experts +
                        t->mlp1_out_dim + p->intermediate_size + p->hidden_size * 2 +
                        p->vocab_size) * sizeof(float);

    fprintf(stderr, "\n=== Memory Usage ===\n");
    fprintf(stderr, "  Model (mmap'd)    = %.2f GB\n", t->file_size / (1024.0 * 1024.0 * 1024.0));
    fprintf(stderr, "  KV cache          = %.1f MB\n", kv_bytes / (1024.0 * 1024.0));
    fprintf(stderr, "  Activations       = %.1f MB\n", act_bytes / (1024.0 * 1024.0));
    fprintf(stderr, "  Total RAM alloc   = %.1f MB\n", (kv_bytes + act_bytes) / (1024.0 * 1024.0));
    fprintf(stderr, "================================\n\n");
}

void free_transformer(Transformer* t) {
    free_run_state(&t->state);
    free(t->rope_cos);
    free(t->rope_sin);
#ifdef _WIN32
    if (t->data) UnmapViewOfFile(t->data);
#else
    if (t->data != MAP_FAILED) munmap(t->data, t->file_size);
    if (t->fd != -1) close(t->fd);
#endif
}

void reset_kv_cache(Transformer* t) {
    size_t cache_size = (size_t)t->config.num_layers * t->config.max_seq_len * t->kv_dim;
    memset(t->state.key_cache, 0, cache_size * sizeof(float));
    memset(t->state.value_cache, 0, cache_size * sizeof(float));
}

// ============================================================================
// Section 8: Forward pass
// ============================================================================

float* forward(Transformer* transformer, int token, int pos, Metrics* metrics) {
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;

    int H = p->hidden_size;
    int head_dim = p->head_dim;
    int half_dim = head_dim / 2;
    int qkv_dim = transformer->qkv_dim;
    int attn_out_dim = transformer->attn_out_dim;
    int kv_dim = transformer->kv_dim;
    int gqa_ratio = transformer->gqa_ratio;
    int mlp1_out_dim = transformer->mlp1_out_dim;
    int packed_cols = transformer->mlp1_packed_cols;
    int num_groups = transformer->num_groups;

    // 1. Token embedding lookup (float16 → float32)
    uint16_t* emb_row = w->embedding + (size_t)token * H;
    for (int j = 0; j < H; j++) {
        s->x[j] = f16_to_f32(emb_row[j]);
    }

    // 2. Forward through all transformer layers
    for (int l = 0; l < p->num_layers; l++) {
        LayerWeights* lw = &w->layers[l];

        // ---- Attention Block ----

        // 2a. Pre-attention RMSNorm with learnable scale
        rmsnorm_scaled(s->xb, s->x, lw->attn_norm_scale, H);

        // 2b. Fused QKV projection + bias
        //     qkv = xb @ qkv_weight.T + qkv_bias → (qkv_dim,)
        f16_matmul_bias(s->qkv, s->xb, lw->qkv_weight, lw->qkv_bias, H, qkv_dim);

        // Split QKV: Q = [0:attn_out_dim], K = [attn_out_dim:+kv_dim], V = [+kv_dim:]
        float* q = s->qkv;                          // (4096,)
        float* k = s->qkv + attn_out_dim;           // (512,)
        float* v = s->qkv + attn_out_dim + kv_dim;  // (512,)

        // 2c. Apply RoPE (half-split, precomputed tables)
        float* cos_row = transformer->rope_cos + pos * half_dim;
        float* sin_row = transformer->rope_sin + pos * half_dim;
        // Query heads (64 heads)
        for (int h = 0; h < p->num_heads; h++) {
            float* qh = q + h * head_dim;
            for (int i = 0; i < half_dim; i++) {
                float x1 = qh[i], x2 = qh[i + half_dim];
                qh[i]            = x1 * cos_row[i] - x2 * sin_row[i];
                qh[i + half_dim] = x2 * cos_row[i] + x1 * sin_row[i];
            }
        }
        // Key heads (8 heads)
        for (int h = 0; h < p->num_kv_heads; h++) {
            float* kh = k + h * head_dim;
            for (int i = 0; i < half_dim; i++) {
                float x1 = kh[i], x2 = kh[i + half_dim];
                kh[i]            = x1 * cos_row[i] - x2 * sin_row[i];
                kh[i + half_dim] = x2 * cos_row[i] + x1 * sin_row[i];
            }
        }

        // 2d. Store K, V in cache
        int loff = l * p->max_seq_len * kv_dim;
        float* kcache_pos = s->key_cache + loff + pos * kv_dim;
        float* vcache_pos = s->value_cache + loff + pos * kv_dim;
        memcpy(kcache_pos, k, kv_dim * sizeof(float));
        memcpy(vcache_pos, v, kv_dim * sizeof(float));

        // 2e. Attention with sinks + sliding window
        // Even layers: sliding window (128), Odd layers: full attention
        int is_sliding = (l % 2 == 0);
        int window = is_sliding ? p->sliding_window : (pos + 1);
        int start = pos - window + 1;
        if (start < 0) start = 0;

        // Pre-convert sinks for this layer (64 values)
        float sinks_f32[64];
        for (int h = 0; h < p->num_heads; h++) {
            sinks_f32[h] = f16_to_f32(lw->attn_sinks[h]);
        }

        float scale = 1.0f / sqrtf((float)head_dim);

        #pragma omp parallel for
        for (int h = 0; h < p->num_heads; h++) {
            float* qh = q + h * head_dim;
            int kv_h = h / gqa_ratio;  // which KV head (0..7)
            float* att = s->att + h * (p->max_seq_len + 1);
            int num_keys = pos - start + 1;

            // Compute QK dot products for positions in window
            for (int t = start; t <= pos; t++) {
                float* kh = s->key_cache + loff + t * kv_dim + kv_h * head_dim;
                float score = 0.0f;
                for (int j = 0; j < head_dim; j++) {
                    score += qh[j] * kh[j];
                }
                att[t - start] = score * scale;
            }

            // Append sink score at position num_keys
            att[num_keys] = sinks_f32[h];

            // Softmax over [0..num_keys] inclusive (num_keys + 1 elements)
            softmax(att, num_keys + 1);

            // Weighted sum of values (exclude sink — its weight gets "absorbed")
            float* oh = s->attn_out + h * head_dim;
            memset(oh, 0, head_dim * sizeof(float));
            for (int t = start; t <= pos; t++) {
                float* vh = s->value_cache + loff + t * kv_dim + kv_h * head_dim;
                float a = att[t - start];
                for (int j = 0; j < head_dim; j++) {
                    oh[j] += a * vh[j];
                }
            }
        }

        // 2f. Output projection + residual
        //     xb = attn_out @ out_weight.T + out_bias → (H,)
        f16_matmul_bias(s->xb, s->attn_out, lw->out_weight, lw->out_bias, attn_out_dim, H);
        for (int j = 0; j < H; j++) {
            s->x[j] += s->xb[j];  // residual connection
        }

        // ---- MoE Block ----

        // 2g. Pre-MLP RMSNorm with learnable scale
        rmsnorm_scaled(s->xb, s->x, lw->mlp_norm_scale, H);

        // 2h. Router — expert selection
        //     gate_logits = xb @ gate_weight.T + gate_bias → (num_experts,)
        f16_matmul_bias(s->gate_logits, s->xb, lw->gate_weight, lw->gate_bias,
                        H, p->num_experts);

        // Top-k selection (k=4 from 32)
        int top_experts[4];
        float top_values[4];
        topk(s->gate_logits, p->num_experts, p->experts_per_token, top_experts, top_values);

        // Softmax over selected expert values
        softmax(top_values, p->experts_per_token);

        // Track expert usage in metrics
        if (metrics) {
            for (int e = 0; e < p->experts_per_token; e++) {
                if (top_experts[e] >= 0 && top_experts[e] < 32) {
                    metrics->expert_hits[top_experts[e]]++;
                }
            }
        }

        // 2i. Expert MLP computation (4 active experts)
        // Sequential over experts, parallel within each matmul (OpenMP inside mxfp4_matmul_bias)
        memset(s->moe_out, 0, H * sizeof(float));

        for (int e = 0; e < p->experts_per_token; e++) {
            int ei = top_experts[e];    // expert index (0..31)
            float ew = top_values[e];   // expert weight (softmax'd)

            // MLP1: MXFP4 matmul + bias → (mlp1_out_dim,)
            mxfp4_matmul_bias(
                s->expert_bufs[e], s->xb,
                lw->mlp1_blocks + (size_t)ei * mlp1_out_dim * packed_cols,
                lw->mlp1_scales + (size_t)ei * mlp1_out_dim * num_groups,
                lw->mlp1_bias   + (size_t)ei * mlp1_out_dim,
                H, mlp1_out_dim, packed_cols, num_groups
            );

            // SwiGLU activation → (intermediate_size,)
            swiglu(s->expert_acts[e], s->expert_bufs[e], p->intermediate_size);

            // MLP2: MXFP4 matmul + bias → (H,)
            mxfp4_matmul_bias(
                s->expert_outs[e], s->expert_acts[e],
                lw->mlp2_blocks + (size_t)ei * H * packed_cols,
                lw->mlp2_scales + (size_t)ei * H * num_groups,
                lw->mlp2_bias   + (size_t)ei * H,
                p->intermediate_size, H, packed_cols, num_groups
            );

            // Weighted accumulation
            for (int j = 0; j < H; j++) {
                s->moe_out[j] += ew * s->expert_outs[e][j];
            }
        }

        // MoE residual
        for (int j = 0; j < H; j++) {
            s->x[j] += s->moe_out[j];
        }
    }

    // 3. Final RMSNorm with learnable scale
    rmsnorm_scaled(s->x, s->x, w->final_norm_scale, H);

    // 4. Unembedding — logits (no tied embeddings, no softcap)
    f16_matmul(s->logits, s->x, w->unembedding, H, p->vocab_size);

    return s->logits;
}

// ============================================================================
// Section 9: Tokenizer — tiktoken BPE (no SentencePiece preprocessing)
// ============================================================================

void build_tokenizer(Tokenizer* t, const char* tokenizer_path) {
    // Read 16-byte header: [vocab_size, max_token_length, eos_token_id, pad_token_id]
    FILE* file = fopen(tokenizer_path, "rb");
    if (!file) {
        fprintf(stderr, "ERROR: Could not open tokenizer file: %s\n", tokenizer_path);
        exit(EXIT_FAILURE);
    }

    int header[4];
    if (fread(header, sizeof(int), 4, file) != 4) {
        fprintf(stderr, "ERROR: Failed to read tokenizer header\n");
        exit(EXIT_FAILURE);
    }

    t->vocab_size       = header[0];
    t->max_token_length = header[1];
    t->eos_token_id     = header[2];
    t->pad_token_id     = header[3];
    t->sorted_indices   = NULL;

    t->vocab        = (char**)malloc(t->vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(t->vocab_size * sizeof(float));
    t->vocab_lengths = (int*)malloc(t->vocab_size * sizeof(int));

    for (int i = 0; i < t->vocab_size; i++) {
        float score;
        int len;
        if (fread(&score, sizeof(float), 1, file) != 1 ||
            fread(&len, sizeof(int), 1, file) != 1) {
            fprintf(stderr, "ERROR: Failed to read token %d\n", i);
            exit(EXIT_FAILURE);
        }
        t->vocab_scores[i] = score;
        t->vocab_lengths[i] = len;
        t->vocab[i] = (char*)malloc(len + 1);
        if (fread(t->vocab[i], 1, len, file) != (size_t)len) {
            fprintf(stderr, "ERROR: Failed to read piece for token %d\n", i);
            exit(EXIT_FAILURE);
        }
        t->vocab[i][len] = '\0';
    }
    fclose(file);
    fprintf(stderr, "Tokenizer loaded: %d tokens, max_len=%d, eos=%d\n",
            t->vocab_size, t->max_token_length, t->eos_token_id);
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) {
        free(t->vocab[i]);
    }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->vocab_lengths);
    free(t->sorted_indices);
}

// Sorted index for binary search during encoding
static Tokenizer* _sort_tokenizer;
static int compare_tokens(const void* a, const void* b) {
    int ia = *(const int*)a;
    int ib = *(const int*)b;
    return strcmp(_sort_tokenizer->vocab[ia], _sort_tokenizer->vocab[ib]);
}

static void build_sorted_index(Tokenizer* t) {
    t->sorted_indices = (int*)malloc(t->vocab_size * sizeof(int));
    for (int i = 0; i < t->vocab_size; i++) {
        t->sorted_indices[i] = i;
    }
    _sort_tokenizer = t;
    qsort(t->sorted_indices, t->vocab_size, sizeof(int), compare_tokens);
}

static int str_lookup(const char* str, Tokenizer* t) {
    if (t->sorted_indices == NULL) {
        build_sorted_index(t);
    }
    int lo = 0, hi = t->vocab_size - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        int idx = t->sorted_indices[mid];
        int cmp = strcmp(str, t->vocab[idx]);
        if (cmp == 0) return idx;
        if (cmp < 0) hi = mid - 1;
        else lo = mid + 1;
    }
    return -1;
}

char* decode(Tokenizer* t, int prev_token, int token) {
    (void)prev_token;  // unused for tiktoken
    if (token < 0 || token >= t->vocab_size) return "";
    return t->vocab[token];
}

void encode(Tokenizer* t, const char* text, int bos, int eos,
            int* tokens, int* n_tokens) {
    // tiktoken BPE: no SentencePiece preprocessing (no ▁ space replacement)
    // Just encode raw text bytes, then merge via BPE
    if (text == NULL) { *n_tokens = 0; return; }
    if (t->sorted_indices == NULL) build_sorted_index(t);

    *n_tokens = 0;

    // BOS not typically used for GPT-OSS, but support it
    if (bos) {
        // Use <|startoftext|> if available (token 199998)
        tokens[(*n_tokens)++] = 199998;
    }

    if (*text == '\0') {
        if (eos) tokens[(*n_tokens)++] = t->eos_token_id;
        return;
    }

    char* str_buffer = (char*)malloc((t->max_token_length * 2 + 3) * sizeof(char));

    // First pass: encode each UTF-8 character as individual token
    const char* ptr = text;
    while (*ptr != '\0') {
        int char_len = 1;
        unsigned char c = (unsigned char)*ptr;
        if ((c & 0x80) == 0)      char_len = 1;
        else if ((c & 0xE0) == 0xC0) char_len = 2;
        else if ((c & 0xF0) == 0xE0) char_len = 3;
        else if ((c & 0xF8) == 0xF0) char_len = 4;

        memcpy(str_buffer, ptr, char_len);
        str_buffer[char_len] = '\0';

        int id = str_lookup(str_buffer, t);
        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            // Fallback: individual bytes
            for (int b = 0; b < char_len; b++) {
                unsigned char byte = (unsigned char)ptr[b];
                // tiktoken byte tokens are usually raw single bytes
                // Try the raw byte as a 1-char string
                str_buffer[0] = (char)byte;
                str_buffer[1] = '\0';
                id = str_lookup(str_buffer, t);
                if (id != -1) {
                    tokens[(*n_tokens)++] = id;
                }
            }
        }
        ptr += char_len;
    }

    // Second pass: BPE merge loop
    while (1) {
        float best_score = -1e10f;
        int best_idx = -1;
        int best_id = -1;

        for (int i = 0; i < (*n_tokens) - 1; i++) {
            snprintf(str_buffer, t->max_token_length * 2 + 2, "%s%s",
                     t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
            int id = str_lookup(str_buffer, t);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_idx = i;
                best_id = id;
            }
        }

        if (best_idx == -1) break;

        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < (*n_tokens) - 1; i++) {
            tokens[i] = tokens[i + 1];
        }
        (*n_tokens)--;
    }

    if (eos) {
        tokens[(*n_tokens)++] = t->eos_token_id;
    }

    free(str_buffer);
}

// ============================================================================
// Section 10: Sampler — top-k with temperature
// ============================================================================

static unsigned int random_u32(unsigned long long* state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (unsigned int)((*state * 0x2545F4914F6CDD1DULL) >> 32);
}

static float random_f32(unsigned long long* state) {
    return (random_u32(state) >> 8) / 16777216.0f;
}

static int compare_prob_index(const void* a, const void* b) {
    const ProbIndex* pa = (const ProbIndex*)a;
    const ProbIndex* pb = (const ProbIndex*)b;
    if (pa->prob > pb->prob) return -1;
    if (pa->prob < pb->prob) return 1;
    return 0;
}

void build_sampler(Sampler* s, float temperature, int top_k, unsigned long long rng_seed, int vocab_size) {
    s->temperature = temperature;
    s->top_k = top_k;
    s->rng_state = rng_seed;
    s->vocab_size = vocab_size;
    s->prob_index = (ProbIndex*)malloc((size_t)vocab_size * sizeof(ProbIndex));
    if (!s->prob_index) {
        fprintf(stderr, "ERROR: Failed to allocate sampler buffer\n");
        exit(EXIT_FAILURE);
    }
}

void free_sampler(Sampler* s) {
    free(s->prob_index);
}

int sample(Sampler* s, float* logits, int vocab_size) {
    if (s->temperature == 0.0f) {
        // Greedy
        int max_i = 0;
        float max_val = logits[0];
        for (int i = 1; i < vocab_size; i++) {
            if (logits[i] > max_val) {
                max_val = logits[i];
                max_i = i;
            }
        }
        return max_i;
    }

    // Temperature scaling
    for (int i = 0; i < vocab_size; i++) {
        logits[i] /= s->temperature;
    }
    softmax(logits, vocab_size);

    // Top-k filtering
    int n = vocab_size;
    int k = s->top_k;
    if (k > 0 && k < n) {
        ProbIndex* pi = s->prob_index;  // persistent buffer
        for (int i = 0; i < n; i++) {
            pi[i].prob = logits[i];
            pi[i].index = i;
        }
        qsort(pi, n, sizeof(ProbIndex), compare_prob_index);

        float cutoff = pi[k - 1].prob;
        for (int i = 0; i < n; i++) {
            if (logits[i] < cutoff) logits[i] = 0.0f;
        }

        float sum = 0.0f;
        for (int i = 0; i < n; i++) sum += logits[i];
        for (int i = 0; i < n; i++) logits[i] /= sum;
    }

    // Categorical sampling
    float coin = random_f32(&s->rng_state);
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += logits[i];
        if (coin < cdf) return i;
    }
    return n - 1;
}

// ============================================================================
// Section 11: Timing utility
// ============================================================================

static long time_in_ms(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (long)(counter.QuadPart * 1000 / freq.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
#endif
}

// ============================================================================
// Section 12: Metrics
// ============================================================================

void print_metrics(Metrics* m, long gen_end_ms) {
    fprintf(stderr, "\n=== Performance Metrics ===\n");

    // Model info
    fprintf(stderr, "  Model size (disk)     = %.2f GB\n",
            m->model_file_size / (1024.0 * 1024.0 * 1024.0));
    fprintf(stderr, "  KV cache              = %.1f MB\n",
            m->kv_cache_bytes / (1024.0 * 1024.0));
    fprintf(stderr, "  Activation buffers    = %.1f MB\n",
            m->activation_bytes / (1024.0 * 1024.0));
    fprintf(stderr, "  Active params/token   = ~3.6B\n");

    // Prompt processing
    if (m->prompt_tokens > 0 && m->prompt_end_ms > m->prompt_start_ms) {
        double prompt_time = (m->prompt_end_ms - m->prompt_start_ms) / 1000.0;
        fprintf(stderr, "  Prompt tokens         = %d\n", m->prompt_tokens);
        fprintf(stderr, "  Prompt time           = %.2f s\n", prompt_time);
        fprintf(stderr, "  Prompt speed          = %.1f tok/s\n",
                m->prompt_tokens / prompt_time);
    }

    // Generation
    if (m->gen_tokens > 0 && gen_end_ms > m->gen_start_ms) {
        double gen_time = (gen_end_ms - m->gen_start_ms) / 1000.0;
        fprintf(stderr, "  Generated tokens      = %d\n", m->gen_tokens);
        fprintf(stderr, "  Generation time       = %.2f s\n", gen_time);
        fprintf(stderr, "  Generation speed      = %.2f tok/s\n",
                m->gen_tokens / gen_time);
        fprintf(stderr, "  Time per token        = %.0f ms\n",
                gen_time * 1000.0 / m->gen_tokens);
    }

    // Total
    if (m->prompt_tokens + m->gen_tokens > 0) {
        double total_time = (gen_end_ms - m->prompt_start_ms) / 1000.0;
        fprintf(stderr, "  Total tokens          = %d\n",
                m->prompt_tokens + m->gen_tokens);
        fprintf(stderr, "  Total time            = %.2f s\n", total_time);
    }

    // Expert utilization — sort a copy and print top 5
    fprintf(stderr, "\n  Expert utilization (top 5):\n");
    int sorted_experts[32];
    for (int e = 0; e < 32; e++) sorted_experts[e] = e;
    // Simple selection sort for top 5
    for (int i = 0; i < 5 && i < 32; i++) {
        for (int j = i + 1; j < 32; j++) {
            if (m->expert_hits[sorted_experts[j]] > m->expert_hits[sorted_experts[i]]) {
                int tmp = sorted_experts[i];
                sorted_experts[i] = sorted_experts[j];
                sorted_experts[j] = tmp;
            }
        }
        if (m->expert_hits[sorted_experts[i]] > 0) {
            fprintf(stderr, "    Expert %2d: %d selections\n",
                    sorted_experts[i], m->expert_hits[sorted_experts[i]]);
        }
    }

    fprintf(stderr, "===========================\n");
}

// ============================================================================
// Section 13: Harmony chat template
// ============================================================================

void chat_state_init(ChatState* cs, int show_thinking, ReasoningLevel level) {
    memset(cs, 0, sizeof(ChatState));
    cs->show_thinking = show_thinking;
    cs->reasoning_level = level;
}

// Append n tokens from src to cs->history, guarding overflow
static void history_append(ChatState* cs, const int* src, int n) {
    for (int i = 0; i < n && cs->history_len < 4096; i++) {
        cs->history[cs->history_len++] = src[i];
    }
}

// Append a single token to cs->history
static void history_push(ChatState* cs, int token) {
    if (cs->history_len < 4096) {
        cs->history[cs->history_len++] = token;
    }
}

// Build system message tokens into history (called once on first turn)
static void chat_build_system_message(ChatState* cs, Tokenizer* t) {
    int tmp[64], n_tmp;

    history_push(cs, HARM_START);

    encode(t, "system", 0, 0, tmp, &n_tmp);
    history_append(cs, tmp, n_tmp);

    history_push(cs, HARM_MESSAGE);

    const char* body;
    switch (cs->reasoning_level) {
        case REASONING_HIGH:   body = "Reasoning: high\n"; break;
        case REASONING_LOW:    body = "Reasoning: low\n"; break;
        default:               body = "Reasoning: medium\n"; break;
    }
    encode(t, body, 0, 0, tmp, &n_tmp);
    history_append(cs, tmp, n_tmp);

    history_push(cs, HARM_END);
}

int chat_build_prompt(ChatState* cs, Tokenizer* t, const char* user_message,
                      int* out_tokens, int* n_tokens) {
    int tmp[2048], n_tmp;

    // First turn: prepend system message
    if (cs->history_len == 0) {
        chat_build_system_message(cs, t);
    }

    // User message: <|start|>user<|message|>{text}<|end|>
    history_push(cs, HARM_START);
    encode(t, "user", 0, 0, tmp, &n_tmp);
    history_append(cs, tmp, n_tmp);
    history_push(cs, HARM_MESSAGE);
    encode(t, user_message, 0, 0, tmp, &n_tmp);
    history_append(cs, tmp, n_tmp);
    history_push(cs, HARM_END);

    // Generation prefix: <|start|>assistant
    history_push(cs, HARM_START);
    encode(t, "assistant", 0, 0, tmp, &n_tmp);
    history_append(cs, tmp, n_tmp);

    // Context window management: if history too long, drop oldest turns
    if (cs->history_len > MAX_HISTORY_TOKENS) {
        // Find end of system message (first HARM_END)
        int sys_end = 0;
        for (int i = 0; i < cs->history_len; i++) {
            if (cs->history[i] == HARM_END) { sys_end = i + 1; break; }
        }
        // Drop enough old turns to fit
        int excess = cs->history_len - MAX_HISTORY_TOKENS;
        int cut = sys_end + excess;
        // Advance cut to next message boundary (HARM_START)
        while (cut < cs->history_len && cs->history[cut] != HARM_START) cut++;
        if (cut > sys_end && cut < cs->history_len) {
            int remaining = cs->history_len - cut;
            memmove(cs->history + sys_end, cs->history + cut, remaining * sizeof(int));
            cs->history_len = sys_end + remaining;
        }
    }

    memcpy(out_tokens, cs->history, cs->history_len * sizeof(int));
    *n_tokens = cs->history_len;
    return 0;
}

int chat_process_token(ChatState* cs, Tokenizer* t, int token, int* should_stop) {
    *should_stop = 0;

    // Stop on <|return|> (200002 = EOS)
    if (token == 200002) {
        *should_stop = 1;
        return 0;
    }

    switch (cs->parse_state) {
    case PARSE_CONTENT:
        if (token == HARM_START) {
            cs->parse_state = PARSE_SAW_START;
            cs->role_buf_len = 0;
            cs->channel_buf_len = 0;
            return 0;
        }
        if (token == HARM_END) {
            // End of a message segment (e.g. analysis → final transition)
            return 0;
        }
        // Regular content token — print based on channel
        if (cs->current_channel == CHANNEL_ANALYSIS) {
            cs->thinking_tokens++;
            if (cs->show_thinking) {
                if (!cs->analysis_printed_header) {
                    fprintf(stderr, "\n[Thinking...]\n");
                    cs->analysis_printed_header = 1;
                    cs->thinking_start_ms = time_in_ms();
                }
                return 1;  // Print thinking content
            }
            return 0;  // Suppress analysis
        }
        // FINAL or NONE — always print
        cs->response_tokens++;
        return 1;

    case PARSE_SAW_START:
        if (token == HARM_CHANNEL) {
            cs->parse_state = PARSE_SAW_CHANNEL;
            return 0;
        }
        if (token == HARM_MESSAGE) {
            // No channel → CHANNEL_NONE (fallback)
            cs->current_channel = CHANNEL_NONE;
            cs->parse_state = PARSE_CONTENT;
            return 0;
        }
        // Role text token (e.g. "assistant")
        {
            char* piece = decode(t, 0, token);
            int plen = (int)strlen(piece);
            if (cs->role_buf_len + plen < MAX_CHAT_BUF - 1) {
                memcpy(cs->role_buf + cs->role_buf_len, piece, plen);
                cs->role_buf_len += plen;
                cs->role_buf[cs->role_buf_len] = '\0';
            }
        }
        cs->parse_state = PARSE_IN_ROLE;
        return 0;

    case PARSE_IN_ROLE:
        if (token == HARM_CHANNEL) {
            cs->parse_state = PARSE_SAW_CHANNEL;
            return 0;
        }
        if (token == HARM_MESSAGE) {
            cs->current_channel = CHANNEL_NONE;
            cs->parse_state = PARSE_CONTENT;
            return 0;
        }
        // More role text
        {
            char* piece = decode(t, 0, token);
            int plen = (int)strlen(piece);
            if (cs->role_buf_len + plen < MAX_CHAT_BUF - 1) {
                memcpy(cs->role_buf + cs->role_buf_len, piece, plen);
                cs->role_buf_len += plen;
                cs->role_buf[cs->role_buf_len] = '\0';
            }
        }
        return 0;

    case PARSE_SAW_CHANNEL:
        // First token of channel name
        {
            char* piece = decode(t, 0, token);
            int plen = (int)strlen(piece);
            if (plen < MAX_CHAT_BUF - 1) {
                memcpy(cs->channel_buf, piece, plen);
                cs->channel_buf_len = plen;
                cs->channel_buf[plen] = '\0';
            }
        }
        cs->parse_state = PARSE_IN_CHANNEL_NAME;
        return 0;

    case PARSE_IN_CHANNEL_NAME:
        if (token == HARM_MESSAGE) {
            // Channel name complete — identify it
            if (strncmp(cs->channel_buf, "analysis", 8) == 0) {
                cs->current_channel = CHANNEL_ANALYSIS;
            } else if (strncmp(cs->channel_buf, "final", 5) == 0) {
                cs->current_channel = CHANNEL_FINAL;
                // Transition from thinking to response
                if (cs->analysis_printed_header && cs->show_thinking) {
                    long now = time_in_ms();
                    if (cs->thinking_start_ms > 0) {
                        double think_sec = (now - cs->thinking_start_ms) / 1000.0;
                        fprintf(stderr, "[Thought for %.1f seconds, %d tokens]\n\n",
                                think_sec, cs->thinking_tokens);
                    }
                }
            } else if (strncmp(cs->channel_buf, "commentary", 10) == 0) {
                cs->current_channel = CHANNEL_COMMENTARY;
            } else {
                cs->current_channel = CHANNEL_NONE;
            }
            cs->parse_state = PARSE_CONTENT;
            return 0;
        }
        // More channel name text
        {
            char* piece = decode(t, 0, token);
            int plen = (int)strlen(piece);
            if (cs->channel_buf_len + plen < MAX_CHAT_BUF - 1) {
                memcpy(cs->channel_buf + cs->channel_buf_len, piece, plen);
                cs->channel_buf_len += plen;
                cs->channel_buf[cs->channel_buf_len] = '\0';
            }
        }
        return 0;
    }
    return 0;
}

void chat_store_response(ChatState* cs, Tokenizer* t,
                         const int* gen_tokens, int n_gen) {
    // Walk generated tokens with a mini state machine to identify channels,
    // then store only final-channel (and structural) tokens in history.
    // Replace <|return|> (200002) with <|end|> (HARM_END).

    HarmonyParseState st = PARSE_CONTENT;
    int final_start = -1;  // Start of the final-channel segment in gen_tokens

    for (int i = 0; i < n_gen; i++) {
        int tok = gen_tokens[i];

        switch (st) {
        case PARSE_CONTENT:
            if (tok == HARM_START) { st = PARSE_SAW_START; }
            break;
        case PARSE_SAW_START:
            if (tok == HARM_CHANNEL) { st = PARSE_SAW_CHANNEL; }
            else if (tok == HARM_MESSAGE) { st = PARSE_CONTENT; }
            else { st = PARSE_IN_ROLE; }
            break;
        case PARSE_IN_ROLE:
            if (tok == HARM_CHANNEL) { st = PARSE_SAW_CHANNEL; }
            else if (tok == HARM_MESSAGE) { st = PARSE_CONTENT; }
            break;
        case PARSE_SAW_CHANNEL: {
            // Peek at the channel name token
            char* piece = decode(t, 0, tok);
            if (strncmp(piece, "final", 5) == 0) {
                // Walk back to find the <|start|> for this segment
                for (int j = i - 1; j >= 0; j--) {
                    if (gen_tokens[j] == HARM_START) { final_start = j; break; }
                }
            }
            st = PARSE_IN_CHANNEL_NAME;
            break;
        }
        case PARSE_IN_CHANNEL_NAME:
            if (tok == HARM_MESSAGE) { st = PARSE_CONTENT; }
            break;
        }
    }

    // If we found a final channel, store from final_start.
    // Otherwise, store everything (fallback for models that don't use channels).
    int copy_from = (final_start >= 0) ? final_start : 0;

    for (int i = copy_from; i < n_gen && cs->history_len < 4096; i++) {
        int tok = gen_tokens[i];
        // Skip analysis channel tokens when copying
        if (tok == 200002) {
            // Replace <|return|> with <|end|>
            history_push(cs, HARM_END);
        } else {
            history_push(cs, tok);
        }
    }

    // If no <|return|> was at the end, ensure we close with <|end|>
    if (n_gen > 0 && gen_tokens[n_gen - 1] != 200002) {
        history_push(cs, HARM_END);
    }
}

// ============================================================================
// Section 14: Main entry point
// ============================================================================

#ifndef __ANDROID__
#ifndef NO_MAIN

int main(int argc, char* argv[]) {
#ifdef _WIN32
    SetConsoleOutputCP(65001);
    SetConsoleCP(65001);

    int wargc;
    LPWSTR* wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
    if (wargv) {
        char** utf8_argv = (char**)malloc((wargc + 1) * sizeof(char*));
        for (int i = 0; i < wargc; i++) {
            int needed = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, NULL, 0, NULL, NULL);
            utf8_argv[i] = (char*)malloc(needed);
            WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, utf8_argv[i], needed, NULL, NULL);
        }
        utf8_argv[wargc] = NULL;
        LocalFree(wargv);
        argc = wargc;
        argv = utf8_argv;
    }
#endif

    // Default parameters
    char* model_path = NULL;
    char* tokenizer_path = NULL;
    float temperature = 0.7f;
    int top_k = 40;
    int max_tokens = 512;
    char* prompt = NULL;
    int chat_mode = 0;
    unsigned long long rng_seed = 0;
    ReasoningLevel reasoning_level = REASONING_MEDIUM;
    int show_thinking = 0;

    // Parse command line
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) { model_path = argv[++i]; }
        else if (strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) { tokenizer_path = argv[++i]; }
        else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) { temperature = atof(argv[++i]); }
        else if (strcmp(argv[i], "--top_k") == 0 && i + 1 < argc) { top_k = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--max_tokens") == 0 && i + 1 < argc) { max_tokens = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) { prompt = argv[++i]; }
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) { rng_seed = atoll(argv[++i]); }
        else if (strcmp(argv[i], "--chat") == 0) { chat_mode = 1; }
        else if (strcmp(argv[i], "--reasoning") == 0 && i + 1 < argc) {
            i++;
            if (strcmp(argv[i], "high") == 0) reasoning_level = REASONING_HIGH;
            else if (strcmp(argv[i], "low") == 0) reasoning_level = REASONING_LOW;
            else reasoning_level = REASONING_MEDIUM;
        }
        else if (strcmp(argv[i], "--show-thinking") == 0) { show_thinking = 1; }
        else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
        }
    }

    if (model_path == NULL) {
        fprintf(stderr, "GPT-OSS 20B C Inference Engine\n\n");
        fprintf(stderr, "Usage: %s --model <model.bin> --tokenizer <tokenizer.bin> [options]\n\n", argv[0]);
        fprintf(stderr, "Options:\n");
        fprintf(stderr, "  --prompt <text>     Input prompt (completion mode)\n");
        fprintf(stderr, "  --chat              Interactive chat mode (Harmony format)\n");
        fprintf(stderr, "  --reasoning <level> Reasoning effort: high|medium|low (default: medium)\n");
        fprintf(stderr, "  --show-thinking     Show chain-of-thought (analysis channel)\n");
        fprintf(stderr, "  --temp <float>      Temperature (default: 0.7)\n");
        fprintf(stderr, "  --top_k <int>       Top-k sampling (default: 40)\n");
        fprintf(stderr, "  --max_tokens <int>  Max tokens to generate (default: 512)\n");
        fprintf(stderr, "  --seed <int>        RNG seed (default: time-based)\n");
        return 1;
    }

    if (rng_seed == 0) rng_seed = (unsigned long long)time(NULL);

    // Load model
    Transformer transformer;
    build_transformer(&transformer, model_path);

    // Load tokenizer
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path);

    // Build sampler
    Sampler sampler;
    build_sampler(&sampler, temperature, top_k, rng_seed, transformer.config.vocab_size);

    // Initialize metrics
    Metrics metrics;
    memset(&metrics, 0, sizeof(Metrics));
    metrics.model_file_size = transformer.file_size;
    metrics.kv_cache_bytes = (size_t)transformer.config.num_layers *
                              transformer.config.max_seq_len *
                              transformer.kv_dim * 2 * sizeof(float);
    metrics.activation_bytes = (transformer.config.hidden_size * 3 +
                                transformer.qkv_dim + transformer.attn_out_dim +
                                transformer.config.num_heads * (transformer.config.max_seq_len + 1) +
                                transformer.config.num_experts + transformer.mlp1_out_dim +
                                transformer.config.intermediate_size +
                                transformer.config.hidden_size * 2 +
                                transformer.config.vocab_size) * sizeof(float);

    int* prompt_tokens = (int*)malloc(transformer.config.max_seq_len * sizeof(int));
    int vocab_size = transformer.config.vocab_size;
    int eos_id = transformer.config.eos_token_id;

    if (chat_mode) {
        // Harmony chat mode — multi-channel structured output
        ChatState chat;
        chat_state_init(&chat, show_thinking, reasoning_level);

        char input_buffer[2048];
        fprintf(stderr, "\n=== GPT-OSS 20B Chat (Harmony) ===\n");
        fprintf(stderr, "Reasoning: %s | Thinking: %s\n",
                reasoning_level == REASONING_HIGH ? "high" :
                reasoning_level == REASONING_LOW ? "low" : "medium",
                show_thinking ? "visible" : "hidden");
        fprintf(stderr, "Commands: 'quit' to exit, 'clear' to reset.\n\n");

        while (1) {
            fprintf(stdout, "You: ");
            fflush(stdout);
            if (fgets(input_buffer, sizeof(input_buffer), stdin) == NULL) break;

            // Strip newline
            int len = (int)strlen(input_buffer);
            while (len > 0 && (input_buffer[len - 1] == '\n' || input_buffer[len - 1] == '\r'))
                input_buffer[--len] = '\0';
            if (strcmp(input_buffer, "quit") == 0 || strcmp(input_buffer, "exit") == 0) break;
            if (strcmp(input_buffer, "clear") == 0) {
                chat_state_init(&chat, show_thinking, reasoning_level);
                reset_kv_cache(&transformer);
                fprintf(stderr, "[Conversation cleared]\n\n");
                continue;
            }
            if (len == 0) continue;

            // Re-process full history each turn (KV cache rebuilt)
            reset_kv_cache(&transformer);
            memset(metrics.expert_hits, 0, sizeof(metrics.expert_hits));

            // Reset parser state for this turn
            // Start in PARSE_IN_ROLE because prompt ends with <|start|>assistant
            chat.parse_state = PARSE_IN_ROLE;
            chat.current_channel = CHANNEL_NONE;
            chat.role_buf_len = 0;
            chat.channel_buf_len = 0;
            chat.thinking_tokens = 0;
            chat.response_tokens = 0;
            chat.analysis_printed_header = 0;
            chat.thinking_start_ms = 0;

            // Build Harmony-formatted prompt
            int n_prompt;
            chat_build_prompt(&chat, &tokenizer, input_buffer, prompt_tokens, &n_prompt);
            metrics.prompt_tokens = n_prompt;

            // Process prompt tokens (prefill)
            metrics.prompt_start_ms = time_in_ms();
            for (int pos = 0; pos < n_prompt; pos++) {
                forward(&transformer, prompt_tokens[pos], pos, &metrics);
            }
            metrics.prompt_end_ms = time_in_ms();

            fprintf(stdout, "GPT-OSS: ");
            fflush(stdout);

            // Generate with Harmony output parsing
            metrics.gen_start_ms = time_in_ms();
            metrics.gen_tokens = 0;
            float* logits = transformer.state.logits;
            int next_token = sample(&sampler, logits, vocab_size);
            int pos = n_prompt;

            // Track generated tokens for history storage
            int* gen_tokens = (int*)malloc(transformer.config.max_seq_len * sizeof(int));
            int n_gen = 0;

            while (pos < transformer.config.max_seq_len && metrics.gen_tokens < max_tokens) {
                int should_stop = 0;
                int should_print = chat_process_token(&chat, &tokenizer, next_token, &should_stop);

                if (n_gen < transformer.config.max_seq_len) {
                    gen_tokens[n_gen++] = next_token;
                }

                if (should_stop) break;

                if (should_print) {
                    char* piece = decode(&tokenizer, 0, next_token);
                    if (chat.current_channel == CHANNEL_ANALYSIS && chat.show_thinking) {
                        fprintf(stderr, "%s", piece);  // Thinking goes to stderr
                    } else {
                        fprintf(stdout, "%s", piece);   // Response goes to stdout
                        fflush(stdout);
                    }
                }

                logits = forward(&transformer, next_token, pos, &metrics);
                next_token = sample(&sampler, logits, vocab_size);
                pos++;
                metrics.gen_tokens++;
            }

            long gen_end = time_in_ms();
            fprintf(stdout, "\n");

            // Store assistant response in history (strip analysis, replace return→end)
            chat_store_response(&chat, &tokenizer, gen_tokens, n_gen);
            free(gen_tokens);

            // Print metrics
            if (chat.thinking_tokens > 0) {
                fprintf(stderr, "  [%d thinking + %d response tokens]\n",
                        chat.thinking_tokens, chat.response_tokens);
            }
            print_metrics(&metrics, gen_end);
            fprintf(stdout, "\n");
        }
    } else if (prompt != NULL) {
        // Single prompt mode
        int n_prompt;
        encode(&tokenizer, prompt, 0, 0, prompt_tokens, &n_prompt);
        metrics.prompt_tokens = n_prompt;

        fprintf(stderr, "Prompt: \"%s\" (%d tokens)\n\n", prompt, n_prompt);

        // Process prompt
        metrics.prompt_start_ms = time_in_ms();
        for (int pos = 0; pos < n_prompt; pos++) {
            forward(&transformer, prompt_tokens[pos], pos, &metrics);
        }
        metrics.prompt_end_ms = time_in_ms();

        // Generate
        metrics.gen_start_ms = time_in_ms();
        metrics.gen_tokens = 0;
        float* logits = transformer.state.logits;
        int next_token = sample(&sampler, logits, vocab_size);
        int pos = n_prompt;

        while (pos < transformer.config.max_seq_len && metrics.gen_tokens < max_tokens) {
            if (next_token == eos_id) break;

            char* piece = decode(&tokenizer, 0, next_token);
            printf("%s", piece);
            fflush(stdout);

            logits = forward(&transformer, next_token, pos, &metrics);
            next_token = sample(&sampler, logits, vocab_size);
            pos++;
            metrics.gen_tokens++;
        }

        long gen_end = time_in_ms();
        printf("\n");
        print_metrics(&metrics, gen_end);
    } else {
        fprintf(stderr, "Please specify --prompt or --chat\n");
    }

    // Cleanup
    free(prompt_tokens);
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}

#endif // NO_MAIN
#endif // __ANDROID__

# Devlog 1: SIMD & Compute Optimization

> **Goal**: Take GPT-OSS 20B inference from ~1.7 tok/s to 4-8 tok/s on the same
> hardware through targeted optimizations in the hot path — without changing the
> model format, binary layout, or public API.

> **Baseline**: 1.66 tok/s generation (604 ms/token), measured on a real chat query
> with `--reasoning high --show-thinking` (512 max tokens, 56 prompt tokens).

> **Hardware**:
> - **CPU**: Intel Core i7-14700HX (8P + 12E cores, 28 threads, 2.10 GHz base)
> - **RAM**: 16 GB DDR5 (15.7 GB usable)
> - **ISA**: AVX2, FMA, F16C confirmed
> - **OS**: Windows 11
>
> **Memory pressure note**: The model is 12.82 GB mmap'd + 386 MB runtime = ~13.2 GB.
> With 15.7 GB usable, that leaves ~2.5 GB for OS + other processes. The model won't
> fit entirely in RAM — the OS will page some of it. This means we are partially
> **memory-bandwidth-bound** even before optimization, and page faults on cold expert
> weights are a real factor. SIMD optimization helps most on the hot (cached) paths;
> for cold paths, the bottleneck is disk I/O, not compute.

---

## Table of Contents

1. [Why This Matters](#1-why-this-matters)
2. [Profiling the Hot Path](#2-profiling-the-hot-path)
3. [Optimization 1: SIMD Float16 Conversion](#3-optimization-1-simd-float16-conversion)
4. [Optimization 2: RoPE Precomputation](#4-optimization-2-rope-precomputation)
5. [Optimization 3: SIMD MXFP4 Dot Product](#5-optimization-3-simd-mxfp4-dot-product)
6. [Optimization 4: Parallel Expert Execution](#6-optimization-4-parallel-expert-execution)
7. [Optimization 5: Sampler Allocation Fix](#7-optimization-5-sampler-allocation-fix)
8. [Optimization 6: Minor Fixes](#8-optimization-6-minor-fixes)
9. [Implementation Order](#9-implementation-order)
10. [Expected Impact Summary](#10-expected-impact-summary)
11. [Testing & Validation Strategy](#11-testing--validation-strategy)
12. [Build Notes](#12-build-notes)
13. [Non-Goals](#13-non-goals)

---

## 1. Why This Matters

The current engine is **correct and complete** — every architectural detail (MoE routing,
MXFP4 fused dequant, attention sinks, Harmony chat, sliding window) works. But all compute
kernels are scalar C code. The compiler does what it can with `-O3`, but the critical inner
loops are structured in ways that prevent auto-vectorization:

- `f16_to_f32()` has data-dependent branching (subnormals, inf/nan)
- `mxfp4_dot_row()` does byte-level nibble extraction in a tight loop
- RoPE calls transcendental functions (`powf`, `cosf`, `sinf`) per element

These are the three places where the CPU spends the vast majority of its time, and all
three have well-known SIMD solutions.

### Current Performance Breakdown (estimated per token)

| Component | % of Time | Why |
|-----------|-----------|-----|
| MXFP4 matmuls (MoE experts) | ~55% | 4 experts × 2 matmuls each, 77% of all weights |
| Float16 matmuls (attention + unembedding) | ~30% | QKV proj, output proj, unembedding (201K×2880) |
| RoPE | ~5% | 72 heads × transcendentals per token |
| Attention (QK dot + softmax + V accumulate) | ~5% | Already has some OpenMP parallelism |
| Everything else (norm, routing, SwiGLU, sampling) | ~5% | Cheap operations |

The top three (MXFP4, f16, RoPE) account for ~90% of per-token time. All three are
SIMD-friendly.

---

## 2. Profiling the Hot Path

### Where the time goes in `forward()` (run_gptoss.c:452-661)

Per token, the forward pass does:

```
24 layers × {
    1× rmsnorm_scaled         (2880 elements — cheap)
    1× f16_matmul_bias        (5120 × 2880 = 14.7M f16→f32 conversions)     ← HOT
    1× RoPE                   (72 heads × 32 trig calls = 2304 trig ops)    ← HOT
    1× attention loop         (64 heads, windowed — moderate)
    1× f16_matmul_bias        (2880 × 4096 = 11.8M f16→f32 conversions)     ← HOT
    1× rmsnorm_scaled         (2880 elements — cheap)
    1× f16_matmul_bias        (32 × 2880 = 92K — routing, cheap)
    4× mxfp4_matmul_bias      (5760 × 1440 packed = 8.3M nibble ops each)   ← HOTTEST
    4× swiglu                 (2880 elements — cheap)
    4× mxfp4_matmul_bias      (2880 × 1440 packed = 4.1M nibble ops each)   ← HOTTEST
}
+ 1× f16_matmul              (201088 × 2880 = 579M f16→f32 conversions)     ← HOT
```

**Total per token**:
- ~636M float16 → float32 conversions (attention + unembedding)
- ~119M MXFP4 nibble lookups + dot products (MoE experts)
- ~55K transcendental function calls (RoPE)
- ~579M f16→f32 for unembedding alone

---

## 3. Optimization 1: SIMD Float16 Conversion

### Problem

`f16_to_f32()` (run_gptoss.c:42-65) is called **per-element** inside `f16_matmul` and
`f16_matmul_bias`. It does bit manipulation with branching for subnormals, inf, and NaN:

```c
static inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    if (exp == 0) { /* subnormal handling with while loop */ }
    else if (exp == 31) { /* inf/nan */ }
    // ...
}
```

In the unembedding matmul alone, this is called 579 million times per token.

### Solution: x86 F16C Intrinsics

Modern x86 CPUs (Haswell+, 2013) have the F16C extension which converts **8 float16
values to 8 float32 values in a single instruction**:

```c
#include <immintrin.h>

// Convert 8 float16 values to 8 float32 values
__m128i h8 = _mm_loadu_si128((__m128i*)&w[j]);    // load 8 × uint16
__m256 f8 = _mm256_cvtph_ps(h8);                   // convert 8 × f16 → 8 × f32
```

### New `f16_matmul_bias` with AVX2 + F16C

```c
static void f16_matmul_bias(float* out, const float* x, const uint16_t* w,
                             const uint16_t* bias, int n, int d) {
    #pragma omp parallel for
    for (int i = 0; i < d; i++) {
        const uint16_t* wi = w + (size_t)i * n;
        __m256 acc = _mm256_setzero_ps();
        int j;

        // Process 8 elements at a time
        for (j = 0; j + 7 < n; j += 8) {
            __m128i h8 = _mm_loadu_si128((__m128i*)(wi + j));
            __m256 w8 = _mm256_cvtph_ps(h8);
            __m256 x8 = _mm256_loadu_ps(x + j);
            acc = _mm256_fmadd_ps(w8, x8, acc);      // fused multiply-add
        }

        // Horizontal sum of 8 floats in acc
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 sum4 = _mm_add_ps(lo, hi);
        sum4 = _mm_hadd_ps(sum4, sum4);
        sum4 = _mm_hadd_ps(sum4, sum4);
        float val = _mm_cvtss_f32(sum4);

        // Scalar tail
        for (; j < n; j++) {
            val += f16_to_f32(wi[j]) * x[j];
        }

        out[i] = val + f16_to_f32(bias[i]);
    }
}
```

### Expected Impact

- **8x fewer conversion instructions** (8 values per instruction vs 1)
- **FMA instead of separate multiply + add** (2 ops → 1 op)
- **Better pipeline utilization** (no branches in hot loop)
- Estimated **3-5x speedup on f16 matmuls** (attention layers + unembedding)
- Since f16 matmuls are ~30% of total time → **~1.0-1.5x overall speedup**

### Compile Flags

```bash
gcc -O3 -fopenmp -mavx2 -mfma -mf16c -o run_gptoss.exe run_gptoss.c -lm -lshell32
```

### Fallback

Keep the scalar `f16_to_f32()` for platforms without F16C. Use a compile-time flag:

```c
#if defined(__F16C__) && defined(__AVX2__) && defined(__FMA__)
    // SIMD path
#else
    // Scalar fallback (current code)
#endif
```

---

## 4. Optimization 2: RoPE Precomputation

### Problem

RoPE frequencies are recomputed on every single `forward()` call (run_gptoss.c:495-521):

```c
for (int h = 0; h < p->num_heads; h++) {     // 64 heads
    for (int i = 0; i < half_dim; i++) {       // 32 dims
        float freq = 1.0f / powf(theta, (float)(2 * i) / (float)head_dim);
        float angle = (float)pos * freq;
        float cos_val = cosf(angle);           // transcendental
        float sin_val = sinf(angle);           // transcendental
        // ...
    }
}
```

Per token: 72 heads (64 Q + 8 K) × 32 dimensions = **2,304 `powf` + 2,304 `cosf` +
2,304 `sinf`** calls. These are identical for every head at the same position — only
`pos` and `i` matter.

### Solution: Precompute sin/cos table at model load time

```c
// In Transformer struct (or RunState):
float* rope_cos;    // (max_seq_len, half_dim)
float* rope_sin;    // (max_seq_len, half_dim)

// At load time (build_transformer):
void precompute_rope(Transformer* t) {
    int half_dim = t->config.head_dim / 2;
    int max_seq = t->config.max_seq_len;
    float theta = (float)t->config.rope_theta;

    t->rope_cos = (float*)malloc(max_seq * half_dim * sizeof(float));
    t->rope_sin = (float*)malloc(max_seq * half_dim * sizeof(float));

    for (int pos = 0; pos < max_seq; pos++) {
        for (int i = 0; i < half_dim; i++) {
            float freq = 1.0f / powf(theta, (float)(2 * i) / (float)(half_dim * 2));
            float angle = (float)pos * freq;
            t->rope_cos[pos * half_dim + i] = cosf(angle);
            t->rope_sin[pos * half_dim + i] = sinf(angle);
        }
    }
}
```

Then in `forward()`:

```c
// Replace the per-head trig computation with table lookup
float* cos_row = t->rope_cos + pos * half_dim;
float* sin_row = t->rope_sin + pos * half_dim;

for (int h = 0; h < p->num_heads; h++) {
    float* qh = q + h * head_dim;
    for (int i = 0; i < half_dim; i++) {
        float x1 = qh[i];
        float x2 = qh[i + half_dim];
        qh[i]            = x1 * cos_row[i] - x2 * sin_row[i];
        qh[i + half_dim] = x2 * cos_row[i] + x1 * sin_row[i];
    }
}
// Same for K heads — exact same cos_row/sin_row
```

### Memory Cost

`4096 positions × 32 dims × 2 (sin+cos) × 4 bytes = 1 MB`

Negligible compared to the 384 MB KV cache.

### Expected Impact

- **Eliminates all transcendental function calls** from the per-token hot path
- ~5% of per-token time → saves ~30 ms/token
- One-time cost of ~50 ms at model load (amortized over entire session)
- Also makes the RoPE loop auto-vectorizable (pure multiply-add, no function calls)

---

## 5. Optimization 3: SIMD MXFP4 Dot Product

### Problem

`mxfp4_dot_row()` (run_gptoss.c:138-155) is where **~55% of all compute time** lives.
It processes MXFP4 blocks one byte at a time:

```c
for (int g = 0; g < n_groups; g++) {           // 90 groups
    float scale = ldexpf(1.0f, (int)scales[g] - 127);
    float group_dot = 0.0f;
    const uint8_t* gb = blocks + g * 16;        // 16 bytes = 32 FP4 values
    for (int j = 0; j < 16; j++) {              // 16 bytes per group
        uint8_t byte = gb[j];
        group_dot += FP4_LUT[byte & 0x0F] * gx[2*j];
        group_dot += FP4_LUT[byte >> 4]   * gx[2*j + 1];
    }
    total += group_dot * scale;
}
```

Per expert MLP1: 5760 rows × 90 groups × 16 bytes = 8.3M byte operations.
4 experts × 2 matmuls × 24 layers = **192 matmul calls per token**.

### Solution: AVX2 Vectorized MXFP4 Dot

The key insight: we can use `_mm256_i32gather_ps` (or a register-based LUT) to look up
8 FP4 values simultaneously, then do vectorized FMA with the input vector.

**Strategy: Preload the 16-entry LUT into two AVX registers and use `vpshufb` for
parallel lookup.**

```c
static inline float mxfp4_dot_row_avx2(const uint8_t* blocks, const uint8_t* scales,
                                         const float* x, int n_groups) {
    // Preload FP4 LUT into two 128-bit registers (16 floats = 4 × __m128)
    // But actually: we use integer shuffle for nibble → index mapping,
    // then gather from a float LUT.

    // Alternative approach: use vpshufb on packed bytes to extract nibbles,
    // then use the 16-entry LUT via gather or sequential lookup.

    // Most practical AVX2 approach for MXFP4:
    __m256 total_acc = _mm256_setzero_ps();

    for (int g = 0; g < n_groups; g++) {
        float scale = ldexpf(1.0f, (int)scales[g] - 127);
        __m256 scale_v = _mm256_set1_ps(scale);
        __m256 group_acc = _mm256_setzero_ps();

        const uint8_t* gb = blocks + g * 16;
        const float* gx = x + g * 32;

        // Process 16 bytes = 32 FP4 values = 4 iterations of 8 values
        for (int j = 0; j < 16; j += 4) {
            // Unpack 4 bytes → 8 float values via LUT
            // Low nibbles → even positions, high nibbles → odd positions
            float w0 = FP4_LUT[gb[j]   & 0x0F]; float w1 = FP4_LUT[gb[j]   >> 4];
            float w2 = FP4_LUT[gb[j+1] & 0x0F]; float w3 = FP4_LUT[gb[j+1] >> 4];
            float w4 = FP4_LUT[gb[j+2] & 0x0F]; float w5 = FP4_LUT[gb[j+2] >> 4];
            float w6 = FP4_LUT[gb[j+3] & 0x0F]; float w7 = FP4_LUT[gb[j+3] >> 4];

            __m256 wv = _mm256_set_ps(w7, w6, w5, w4, w3, w2, w1, w0);
            __m256 xv = _mm256_loadu_ps(gx + 2 * j);
            group_acc = _mm256_fmadd_ps(wv, xv, group_acc);
        }

        total_acc = _mm256_fmadd_ps(group_acc, scale_v, total_acc);
    }

    // Horizontal sum
    __m128 hi = _mm256_extractf128_ps(total_acc, 1);
    __m128 lo = _mm256_castps256_ps128(total_acc);
    __m128 sum4 = _mm_add_ps(lo, hi);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    return _mm_cvtss_f32(sum4);
}
```

### Better Approach: Full SIMD Nibble Extraction

For maximum throughput, use integer SIMD to extract nibbles from 16 bytes in parallel:

```c
// Load 16 bytes (one full group)
__m128i raw = _mm_loadu_si128((__m128i*)gb);

// Extract low nibbles: raw & 0x0F
__m128i mask_lo = _mm_set1_epi8(0x0F);
__m128i lo_nibbles = _mm_and_si128(raw, mask_lo);    // 16 low nibble indices

// Extract high nibbles: raw >> 4
__m128i hi_nibbles = _mm_srli_epi16(raw, 4);
hi_nibbles = _mm_and_si128(hi_nibbles, mask_lo);     // 16 high nibble indices

// Use vpshufb to look up 16 FP4 values at once (as int8 scaled values)
// Then convert to float for the dot product...
```

This is more complex to implement but processes **32 FP4 values per group in ~10
instructions** instead of 32 scalar operations.

### Expected Impact

- **4-8x speedup on MXFP4 matmuls** depending on how well we vectorize
- Since MXFP4 matmuls are ~55% of total time → **~2-3x overall speedup**
- This is the single highest-impact optimization

### Implementation Note

The MXFP4 SIMD kernel is the most complex of all optimizations. The nibble extraction +
LUT lookup + interleaved accumulation has subtle ordering requirements. We should:

1. Implement the simpler "unpack 4 bytes → 8 floats" version first
2. Verify correctness against the scalar version
3. Then iterate toward the full `vpshufb` approach if needed

---

## 6. Optimization 4: Parallel Expert Execution

### Problem

The 4 active experts are processed sequentially (run_gptoss.c:617-646):

```c
for (int e = 0; e < p->experts_per_token; e++) {
    mxfp4_matmul_bias(..., s->expert_buf, ...);   // MLP1 → expert_buf
    swiglu(s->expert_act, s->expert_buf, ...);      // SwiGLU → expert_act
    mxfp4_matmul_bias(..., s->expert_out, ...);    // MLP2 → expert_out
    // accumulate into moe_out
}
```

The inner matmuls use OpenMP, but the outer loop is serial. This means threads fan out
and collapse 8 times per layer (4 experts × 2 matmuls), wasting synchronization overhead.

### Solution: Allocate per-expert buffers, run experts in parallel

```c
// In RunState — allocate 4 sets of expert buffers:
float* expert_bufs[4];    // 4 × (mlp1_out_dim,) = 4 × 5760 × 4 = 92 KB
float* expert_acts[4];    // 4 × (intermediate_size,) = 4 × 2880 × 4 = 46 KB
float* expert_outs[4];    // 4 × (hidden_size,) = 4 × 2880 × 4 = 46 KB
                          // Total: 184 KB — negligible
```

Then use OpenMP sections or tasks:

```c
#pragma omp parallel for
for (int e = 0; e < p->experts_per_token; e++) {
    int ei = top_experts[e];

    // Each expert uses its own buffer set — no conflicts
    mxfp4_matmul_bias_single_thread(
        expert_bufs[e], s->xb,
        lw->mlp1_blocks + ..., lw->mlp1_scales + ..., lw->mlp1_bias + ...,
        H, mlp1_out_dim, packed_cols, num_groups
    );

    swiglu(expert_acts[e], expert_bufs[e], p->intermediate_size);

    mxfp4_matmul_bias_single_thread(
        expert_outs[e], expert_acts[e],
        lw->mlp2_blocks + ..., lw->mlp2_scales + ..., lw->mlp2_bias + ...,
        p->intermediate_size, H, packed_cols, num_groups
    );
}

// Sequential accumulation (fast, just 4 × 2880 additions)
memset(s->moe_out, 0, H * sizeof(float));
for (int e = 0; e < 4; e++) {
    float ew = top_values[e];
    for (int j = 0; j < H; j++) {
        s->moe_out[j] += ew * expert_outs[e][j];
    }
}
```

### Trade-off: Nested vs Flat Parallelism

Two strategies:

| Strategy | How | Pros | Cons |
|----------|-----|------|------|
| **Current**: Serial experts, parallel matmul rows | Inner `#pragma omp parallel for` on matmul | Simple | 8 thread fan-out/collapse per layer |
| **Proposed**: Parallel experts, serial matmul rows | Outer `#pragma omp parallel for` on 4 experts | 1 fan-out per MoE block | Less parallelism per matmul (1 thread per expert) |
| **Hybrid**: Parallel experts with nested parallel matmul | `omp_set_nested(1)` | Maximum parallelism | Complex, overhead risk |

With 8 CPU threads and 4 experts, the flat "parallel experts" approach gives 2 threads
per expert — which is less parallelism per matmul but eliminates synchronization barriers.
The best approach depends on the hardware; we should benchmark both.

### Expected Impact

- Eliminates 8 thread synchronization barriers per layer (192 per token)
- Estimated **10-20% speedup** (synchronization overhead is real but not dominant)
- Memory cost: 184 KB (0.05% of current 385 MB allocation)

---

## 7. Optimization 5: Sampler Allocation Fix

### Problem

Every generated token, `sample()` (run_gptoss.c:903-914) allocates and frees a 1.6 MB
array:

```c
ProbIndex* pi = (ProbIndex*)malloc(n * sizeof(ProbIndex));  // n = 201088
// ... qsort ...
free(pi);
```

This is 201088 × 8 bytes = ~1.6 MB, allocated and freed on every single token. While
`malloc`/`free` are fast, this still causes:

- Memory allocator overhead (~1-5 us per call)
- Potential TLB misses on the new allocation
- Cache pollution (fresh memory, not warm)

### Solution: Persistent buffer in Sampler or RunState

```c
typedef struct {
    float temperature;
    int top_k;
    unsigned long long rng_state;
    ProbIndex* prob_index;    // persistent buffer, allocated once
} Sampler;

void build_sampler(Sampler* s, float temperature, int top_k,
                   unsigned long long rng_seed, int vocab_size) {
    s->temperature = temperature;
    s->top_k = top_k;
    s->rng_state = rng_seed;
    s->prob_index = (ProbIndex*)malloc(vocab_size * sizeof(ProbIndex));
}
```

### Expected Impact

- Eliminates ~1.6 MB malloc/free per token
- Small but free win: **~1-2% speedup**, eliminates unnecessary allocator pressure
- Zero memory cost (same total, just allocated once instead of per-token)

---

## 8. Optimization 6: Minor Fixes

### 8a. Thread-unsafe global tokenizer pointer

`_sort_tokenizer` (run_gptoss.c:724) is a static global used by `qsort`:

```c
static Tokenizer* _sort_tokenizer;
static int compare_tokens(const void* a, const void* b) {
    return strcmp(_sort_tokenizer->vocab[ia], _sort_tokenizer->vocab[ib]);
}
```

This is thread-unsafe and a landmine for the JNI/web use case.

**Fix**: Use `qsort_r` (POSIX) / `qsort_s` (Windows) which passes a context pointer:

```c
#ifdef _WIN32
static int compare_tokens_r(void* context, const void* a, const void* b) {
    Tokenizer* t = (Tokenizer*)context;
    // ...
}
// qsort_s(indices, n, sizeof(int), compare_tokens_r, t);
#else
static int compare_tokens_r(const void* a, const void* b, void* context) {
    Tokenizer* t = (Tokenizer*)context;
    // ...
}
// qsort_r(indices, n, sizeof(int), compare_tokens_r, t);
#endif
```

### 8b. BPE O(n^2) merge loop

The BPE encoder (run_gptoss.c:818-841) scans all adjacent pairs on every merge iteration.
For short prompts (<100 tokens) this is fine. For long prompts approaching 4096 tokens, it
becomes a bottleneck.

**Fix (deferred)**: Priority queue of merge candidates. Not critical for v1 since prompt
processing is already fast enough (~2 tok/s on 56-token prompts) and this only affects
the initial encode, not per-token generation.

### 8c. Missing bounds check in encode()

`encode()` writes into caller-provided `tokens` array with no size parameter. If input
text produces more tokens than `max_seq_len`, it writes past the buffer.

**Fix**: Add a `max_tokens` parameter to `encode()`:

```c
void encode(Tokenizer* t, const char* text, int bos, int eos,
            int* tokens, int* n_tokens, int max_tokens);
```

---

## 9. Implementation Order

The optimizations are ordered by impact and independence:

| Phase | What | Impact | Complexity | Dependencies |
|-------|------|--------|------------|--------------|
| **Phase 1** | RoPE precomputation | ~5% speedup | Low | None |
| **Phase 2** | SIMD f16 matmul (F16C + FMA) | ~20-30% speedup | Medium | Compile flags |
| **Phase 3** | SIMD MXFP4 dot product (AVX2) | ~40-60% speedup | High | Compile flags |
| **Phase 4** | Parallel expert execution | ~10-20% speedup | Medium | Phase 3 (buffer changes) |
| **Phase 5** | Sampler + minor fixes | ~2-5% speedup | Low | None |

### Why This Order

1. **RoPE first**: Simplest change, easy to verify, builds confidence
2. **F16 SIMD second**: Medium complexity, well-understood pattern (F16C is standard)
3. **MXFP4 SIMD third**: Hardest kernel, biggest payoff, benefits from learnings in Phase 2
4. **Parallel experts fourth**: Depends on understanding the OpenMP threading model
   established in earlier phases
5. **Minor fixes last**: Low impact, can be done alongside any phase

### Implementation Approach

For each phase:
1. Create a new SIMD version of the function alongside the scalar version
2. Gate with `#ifdef` based on compile-time CPU feature detection
3. Validate that SIMD output matches scalar output (bitwise comparison on test inputs)
4. Benchmark before and after (same prompt, same seed, 50+ tokens)
5. Commit working phase before starting next

---

## 10. Expected Impact Summary

| Optimization | Est. Speedup | Combined tok/s |
|--------------|-------------|----------------|
| Baseline | — | 1.66 tok/s |
| + RoPE precompute | +5% | ~1.74 |
| + SIMD f16 matmul | +25% | ~2.18 |
| + SIMD MXFP4 dot | +50% | ~3.27 |
| + Parallel experts | +15% | ~3.76 |
| + Minor fixes | +3% | ~3.87 |

**Conservative estimate: 3.0-4.0 tok/s** (1.8-2.4x speedup)

**Optimistic estimate: 5-7 tok/s** (3.0-4.2x speedup) if SIMD kernels achieve near-peak
throughput and most expert weights stay resident in RAM.

The gap between conservative and optimistic depends on:
- How well the MXFP4 kernel vectorizes (nibble extraction is tricky)
- **Page fault rate** — 16 GB RAM with a 12.82 GB model means cold experts get paged out
- OpenMP thread scheduling across 8P+12E heterogeneous cores (thread affinity matters)
- Whether the i7-14700HX E-cores help or hurt with OpenMP (different clock speeds)

**The 16 GB RAM constraint is the hard ceiling.** No amount of SIMD optimization can fix
a page fault stall. The biggest wins will come from compute-bound paths (attention matmuls,
hot experts, unembedding) where the weights are already cached in RAM.

### Memory Bandwidth Reality Check

At 2 tok/s, the model touches per token:
- 4 experts × ~(10 MB + 5 MB) MXFP4 per layer × 24 layers = ~1.4 GB/token
- Attention weights: ~100 MB/token
- Unembedding: ~1.1 GB/token
- Total: ~2.6 GB/token × 2 tok/s = ~5.2 GB/s

**Our hardware**: i7-14700HX with DDR5 (likely 4800-5600 MT/s):
- DDR5-5200 dual-channel: ~83 GB/s theoretical, ~40-50 GB/s practical
- **We have ~8-10x headroom on raw memory bandwidth**

**However**: with only 16 GB RAM and a 12.82 GB model, not all weights stay resident.
The OS will evict pages for cold experts. When a cold expert is selected, the page fault
goes to SSD (~3-5 GB/s sequential, but random 4K reads are much slower). This means:

- **Hot path** (frequently-used experts, attention weights): SIMD helps a lot (compute-bound)
- **Cold path** (rarely-used experts paged out): SIMD doesn't help (I/O-bound, waiting for SSD)
- The 4-expert MoE selection means ~12.5% of experts per token are active; over many tokens
  the working set is larger but still not all 32 experts per layer

**Realistic expectation**: SIMD will help more on shorter sessions (more weights cached)
and less on the very first tokens after a cold start. The 16 GB constraint is the ceiling
on maximum throughput — not SIMD.

---

## 11. Testing & Validation Strategy

### Correctness

For each SIMD kernel:

1. **Unit test**: Run scalar and SIMD versions on identical inputs, compare outputs
   - f16_matmul: random float16 weight matrix × float32 input vector
   - mxfp4_dot_row: random packed blocks + scales × float32 input
   - RoPE: compare table lookup vs computed values for all positions

2. **Integration test**: Run full `forward()` with scalar vs SIMD, compare logits
   - Use `--seed 42 --temp 0` (greedy) to get deterministic output
   - Same prompt should produce identical token sequence

3. **Chat test**: Verify Harmony output parsing still works correctly

### Performance

For each optimization:

1. **Microbenchmark**: Time the isolated kernel (1M iterations)
2. **Macro benchmark**: Full generation with fixed prompt and seed
   - Prompt: `"Explain the halting problem in one paragraph."`
   - Settings: `--reasoning medium --max_tokens 256 --seed 42 --temp 0.7`
   - Measure: total time, tok/s, ms/token

### Regression

After all optimizations:
- Re-run the original test: `"If you remove one ball from a bag..."` with
  `--reasoning high --show-thinking --max_tokens 1024`
- Compare output quality (should be identical or very close with same seed)
- Compare tok/s (should be measurably faster)

---

## 12. Build Notes

### Required Compile Flags

```bash
# Full SIMD build (recommended for modern x86):
gcc -O3 -fopenmp -mavx2 -mfma -mf16c -o run_gptoss.exe run_gptoss.c -lm -lshell32

# Safe fallback (any x86-64):
gcc -O3 -fopenmp -o run_gptoss.exe run_gptoss.c -lm -lshell32
```

### Runtime CPU Detection (optional, for distributable binaries)

If we want a single binary that works everywhere but uses SIMD where available:

```c
#include <cpuid.h>

static int has_avx2_fma_f16c(void) {
    unsigned int eax, ebx, ecx, edx;
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    int avx2 = (ebx >> 5) & 1;
    __cpuid(1, eax, ebx, ecx, edx);
    int fma  = (ecx >> 12) & 1;
    int f16c = (ecx >> 29) & 1;
    return avx2 && fma && f16c;
}
```

Then use function pointers to dispatch between scalar and SIMD paths. This adds ~20 lines
but makes the binary portable.

### Platform Notes

| Platform | SIMD Available | Notes |
|----------|---------------|-------|
| **i7-14700HX (our machine)** | **AVX2 + FMA + F16C** | **Full SIMD support, 8P+12E cores** |
| x86-64 (Haswell+, 2013) | AVX2 + FMA + F16C | Full SIMD support |
| x86-64 (older) | SSE4.2 only | Scalar fallback |
| ARM64 (Android) | NEON + FP16 | Different intrinsics (future work) |
| Apple Silicon | NEON + FP16 | Different intrinsics (future work) |

This devlog focuses on **x86 AVX2** only. ARM NEON support is a separate effort.

### OpenMP Thread Count Note

The i7-14700HX has 20 cores (8P + 12E) / 28 threads. By default, OpenMP will use all
available threads, but E-cores run at lower clocks (~1.5 GHz vs 5.5 GHz boost on P-cores).
This can cause **thread straggler effects** where fast P-core threads wait for slow E-core
threads to finish their chunk. We may want to experiment with:

```bash
# Limit to P-cores only (8 cores, 16 threads):
OMP_NUM_THREADS=16 ./run_gptoss.exe ...

# Or pin to P-cores:
OMP_PLACES=cores OMP_PROC_BIND=close OMP_NUM_THREADS=16 ./run_gptoss.exe ...
```

This is a tuning knob to test after SIMD work is done.

---

## 13. Non-Goals

Things we are explicitly **not** doing in this optimization pass:

1. **GPU support** — This is a CPU inference engine. GPU would require a complete rewrite.
2. **INT8/INT4 KV cache** — The 384 MB float32 KV cache fits in RAM fine. Quantizing it
   would add complexity for marginal benefit on PC.
3. **Speculative decoding** — Requires a draft model. Interesting but out of scope.
4. **Continuous batching** — Single-user inference only.
5. **YaRN RoPE for >4096 context** — The precomputed table supports up to `max_seq_len`;
   extending beyond that is a separate feature.
6. **Changing the binary format** — All optimizations are inference-side only. The
   `.bin` file format and export scripts are untouched.
7. **ARM NEON SIMD** — x86 first. ARM support is a future devlog.
8. **Restructuring the codebase** — We keep the single `.c` + `.h` architecture.
   No splitting into multiple files.

---

## Revision Log

| Date | Change |
|------|--------|
| 2026-03-06 | Initial devlog created. Baseline: 1.66 tok/s. |
| 2026-03-06 | Updated hardware specs: i7-14700HX, 16 GB RAM (not 32 GB). Added memory pressure analysis — 12.82 GB model in 15.7 GB usable RAM means page faults on cold experts. Adjusted expectations: conservative 3-4 tok/s, optimistic 5-7 tok/s. Added OpenMP E-core thread straggler note. |
| 2026-03-06 | **Implementation complete.** All phases implemented. Results below. |

## Benchmark Results

Test prompt: *"If you remove one ball from a bag containing 3 red and 2 blue balls..."*
Settings: `--reasoning high --show-thinking --temp 0.7 --max_tokens 512 --seed 42`

| Build | Flags | Prompt | Generation | ms/tok | Total | vs Baseline |
|-------|-------|--------|------------|--------|-------|-------------|
| **Baseline** (scalar) | `-O3 -fopenmp` | 1.9 tok/s | 1.66 tok/s | 604 ms | 337.9s | — |
| **SIMD v1** (parallel experts — reverted) | `-O3 -fopenmp -mavx2 -mfma -mf16c` | 0.7 tok/s | 1.50 tok/s | 665 ms | 424.9s | **-15% regression** |
| **SIMD v2** (sequential experts, inner OMP) | `-O3 -fopenmp -mavx2 -mfma -mf16c` | 2.5 tok/s | 1.99 tok/s | 504 ms | 280.2s | **+20% faster** |
| **Native** (CPU-tuned) | `-O3 -march=native -fopenmp` | **2.8 tok/s** | **2.36 tok/s** | **423 ms** | **236.9s** | **+42% faster** |

### Key Findings

1. **`-march=native` is the best build flag.** It implies `-mavx2 -mfma -mf16c` and adds CPU-specific scheduling for Raptor Lake. The explicit SIMD flags gave +20%, but native gave +42% — an extra +18% from microarchitecture tuning alone.

2. **Parallel expert execution was a regression.** 4 threads (1 per expert) with serial inner matmuls is far worse than sequential experts with all 20+ threads parallelizing each inner matmul. The `#pragma omp parallel for` over 4 experts gave each only 1 thread for matmuls with 5760+ rows of work. Reverted to sequential-experts + inner-parallel.

3. **F16C + FMA are the biggest wins.** Converting 8 half-floats per instruction (`_mm256_cvtph_ps`) and fused multiply-add (`_mm256_fmadd_ps`) directly speed up the 636M f16→f32 conversions per token.

4. **MXFP4 SIMD gains are modest.** The `_mm256_set_ps` with 8 scalar LUT lookups is still mostly scalar — the FMA accumulation helps but the nibble extraction bottleneck remains. A future `vpshufb`-based approach could improve this further.

5. **RoPE precomputation helps but is small (~5%).** Eliminates 2304 trig calls per token; 1 MB memory cost.

### Recommended Build Command

```bash
gcc -O3 -march=native -fopenmp -o run_gptoss.exe run_gptoss.c -lm -lshell32
```

### What's Left on the Table

- **Full SIMD nibble extraction** (`vpshufb` for MXFP4): could add another 10-20%
- **OpenMP thread affinity** (`OMP_NUM_THREADS=16` for P-cores only): untested
- **`-O3 -flto`** (link-time optimization): untested, may help inlining
- **Memory bandwidth ceiling**: 12.82 GB model in 16 GB RAM means page faults are still the hard limit

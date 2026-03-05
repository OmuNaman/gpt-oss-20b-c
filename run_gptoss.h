/*
 * run_gptoss.h — GPT-OSS 20B C inference engine header
 *
 * Built from scratch for OpenAI GPT-OSS 20B (Mixture of Experts).
 * Key architecture features:
 *   - MoE: 32 experts, top-4 active per token (3.6B active of 20.9B total)
 *   - MXFP4 expert weights with E8M0 scales (fused dequant + dot product)
 *   - float16 attention weights, embeddings, biases
 *   - Learnable RMSNorm (scale vector per norm)
 *   - Fused QKV projection with bias
 *   - Attention sinks (learned per-head scalar)
 *   - Alternating sliding window (even=128, odd=full)
 *   - SwiGLU activation (alpha=1.702, +1.0 linear bias, clamping)
 *   - GQA 8:1 (64 query heads, 8 KV heads)
 *   - Half-split RoPE with theta=150000
 *   - No tied embeddings (separate unembedding matrix)
 *   - No logit softcap, no QK norm, no per-layer mixing
 */

#ifndef RUN_GPTOSS_H
#define RUN_GPTOSS_H

#include <stdint.h>
#include <stddef.h>

// ----------------------------------------------------------------------------
// Constants

#define GOSS_MAGIC 0x474F5353  // "GOSS" in ASCII

// FP4 lookup table for MXFP4 dequantization (E2M1 format)
// Indices 0-7: positive values, 8-15: negative values
static const float FP4_LUT[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// ----------------------------------------------------------------------------
// Model configuration (read from 64-byte binary header)

typedef struct {
    int magic;              // 0x474F5353 ("GOSS")
    int version;            // 1
    int hidden_size;        // 2880
    int intermediate_size;  // 2880
    int num_layers;         // 24
    int num_heads;          // 64 (query heads)
    int num_kv_heads;       // 8 (KV heads)
    int head_dim;           // 64
    int vocab_size;         // 201088
    int num_experts;        // 32
    int experts_per_token;  // 4
    int sliding_window;     // 128
    int max_seq_len;        // 4096
    int rope_theta;         // 150000
    int eos_token_id;       // 200002
    int reserved;           // 0
} Config;

// Derived constants (computed from config):
//   qkv_dim           = head_dim * (num_heads + 2*num_kv_heads) = 5120
//   attn_out_dim      = head_dim * num_heads = 4096
//   kv_dim            = head_dim * num_kv_heads = 512
//   gqa_ratio         = num_heads / num_kv_heads = 8
//   mlp1_out_dim      = intermediate_size * 2 = 5760  (gate + linear interleaved)
//   mlp1_packed_cols  = hidden_size / 2 = 1440  (bytes per row in MXFP4)
//   num_groups        = hidden_size / 32 = 90  (MXFP4 groups per input dim)

// ----------------------------------------------------------------------------
// Per-layer weights (pointers into mmap'd binary file)
// Mixed types: uint16_t* for float16, uint8_t* for MXFP4/E8M0

typedef struct {
    // --- Attention (all float16) ---
    uint16_t* attn_norm_scale;   // (hidden_size,)
    uint16_t* qkv_weight;        // (qkv_dim, hidden_size)
    uint16_t* qkv_bias;          // (qkv_dim,)
    uint16_t* attn_sinks;        // (num_heads,)
    uint16_t* out_weight;        // (hidden_size, attn_out_dim)
    uint16_t* out_bias;          // (hidden_size,)

    // --- MoE ---
    uint16_t* mlp_norm_scale;    // (hidden_size,) float16
    uint16_t* gate_weight;       // (num_experts, hidden_size) float16
    uint16_t* gate_bias;         // (num_experts,) float16

    uint8_t*  mlp1_blocks;       // (num_experts, mlp1_out_dim, packed_cols) MXFP4
    uint8_t*  mlp1_scales;       // (num_experts, mlp1_out_dim, num_groups) E8M0
    uint16_t* mlp1_bias;         // (num_experts, mlp1_out_dim) float16

    uint8_t*  mlp2_blocks;       // (num_experts, hidden_size, packed_cols) MXFP4
    uint8_t*  mlp2_scales;       // (num_experts, hidden_size, num_groups) E8M0
    uint16_t* mlp2_bias;         // (num_experts, hidden_size) float16
} LayerWeights;

// Maximum number of layers supported
#define MAX_LAYERS 64

// ----------------------------------------------------------------------------
// Top-level weight container

typedef struct {
    uint16_t*    embedding;         // (vocab_size, hidden_size) float16
    LayerWeights layers[MAX_LAYERS];
    uint16_t*    final_norm_scale;  // (hidden_size,) float16
    uint16_t*    unembedding;       // (vocab_size, hidden_size) float16
} TransformerWeights;

// ----------------------------------------------------------------------------
// Runtime state (allocated float32 buffers for inference)

typedef struct {
    // Activation buffers
    float* x;           // (hidden_size,) main hidden state
    float* xb;          // (hidden_size,) buffer after norm
    float* xb2;         // (hidden_size,) second buffer

    // Attention buffers
    float* qkv;         // (qkv_dim,) fused QKV output
    float* attn_out;    // (attn_out_dim,) concatenated attention output
    float* att;         // (num_heads, max_seq_len+1) scores (+1 for sink)

    // MoE buffers (4 sets for parallel expert execution)
    float* gate_logits;    // (num_experts,) router output
    float* expert_bufs[4]; // 4 × (mlp1_out_dim,) mlp1 output per expert
    float* expert_acts[4]; // 4 × (intermediate_size,) after SwiGLU
    float* expert_outs[4]; // 4 × (hidden_size,) mlp2 output per expert
    float* moe_out;        // (hidden_size,) weighted sum of expert outputs

    // Output
    float* logits;      // (vocab_size,) final logits

    // KV cache
    float* key_cache;   // (num_layers, max_seq_len, kv_dim)
    float* value_cache; // (num_layers, max_seq_len, kv_dim)
} RunState;

// ----------------------------------------------------------------------------
// Top-level transformer handle

typedef struct {
    Config config;
    TransformerWeights weights;
    RunState state;

    // Derived constants (computed once at load time)
    int qkv_dim;
    int attn_out_dim;
    int kv_dim;
    int gqa_ratio;
    int mlp1_out_dim;
    int mlp1_packed_cols;
    int num_groups;

    // Memory-mapped file data
    int fd;
    void* data;         // void* since binary has mixed types
    size_t file_size;

    // Precomputed RoPE tables (allocated at load time)
    float* rope_cos;    // (max_seq_len, half_dim)
    float* rope_sin;    // (max_seq_len, half_dim)
} Transformer;

// ----------------------------------------------------------------------------
// Tokenizer (BPE, loaded from tokenizer_gptoss.bin)
// tiktoken-based: no SentencePiece preprocessing

typedef struct {
    char** vocab;           // token id -> piece string
    float* vocab_scores;    // token id -> BPE merge score
    int* vocab_lengths;     // token id -> piece byte length
    int vocab_size;         // 201088
    int max_token_length;
    int eos_token_id;       // 200002
    int pad_token_id;       // 199999
    int* sorted_indices;    // sorted for binary search during encode
} Tokenizer;

// ----------------------------------------------------------------------------
// Sampler (top-k sampling with temperature)

typedef struct {
    float prob;
    int index;
} ProbIndex;

typedef struct {
    float temperature;
    int top_k;
    unsigned long long rng_state;
    ProbIndex* prob_index;  // persistent buffer (vocab_size elements)
    int vocab_size;
} Sampler;

// ----------------------------------------------------------------------------
// Runtime metrics

typedef struct {
    size_t model_file_size;     // bytes (mmap'd)
    size_t kv_cache_bytes;      // bytes
    size_t activation_bytes;    // bytes
    size_t tokenizer_bytes;     // bytes (approx)
    long prompt_start_ms;       // timestamp
    long prompt_end_ms;         // timestamp
    long gen_start_ms;          // timestamp
    int prompt_tokens;          // count
    int gen_tokens;             // count
    int expert_hits[32];        // how many times each expert was selected
} Metrics;

// ----------------------------------------------------------------------------
// Harmony chat template (OpenAI's structured response format)

// Special token IDs from o200k_harmony tokenizer
#define HARM_CHANNEL     200005  // <|channel|>
#define HARM_START       200006  // <|start|>
#define HARM_END         200007  // <|end|>
#define HARM_MESSAGE     200008  // <|message|>
// Note: <|return|> = 200002 = eos_token_id (already in Config)

typedef enum {
    CHANNEL_NONE = 0,       // Before any channel identified / fallback
    CHANNEL_ANALYSIS,       // Chain-of-thought (hidden by default)
    CHANNEL_COMMENTARY,     // Tool calls (not used in basic chat)
    CHANNEL_FINAL           // User-facing response
} HarmonyChannel;

typedef enum {
    PARSE_CONTENT = 0,      // Inside message content
    PARSE_SAW_START,        // Saw <|start|>, expect role text
    PARSE_IN_ROLE,          // Accumulating role text
    PARSE_SAW_CHANNEL,      // Saw <|channel|>, expect channel name
    PARSE_IN_CHANNEL_NAME,  // Accumulating channel name
} HarmonyParseState;

typedef enum {
    REASONING_LOW = 0,
    REASONING_MEDIUM = 1,
    REASONING_HIGH = 2,
} ReasoningLevel;

#define MAX_HISTORY_TOKENS 3584   // Reserve ~512 for generation
#define MAX_CHAT_BUF 32           // Buffer for role/channel name

typedef struct {
    // Conversation history (token IDs)
    int history[4096];
    int history_len;

    // Output parser state machine
    HarmonyParseState parse_state;
    HarmonyChannel current_channel;
    char role_buf[MAX_CHAT_BUF];
    int role_buf_len;
    char channel_buf[MAX_CHAT_BUF];
    int channel_buf_len;

    // Settings
    int show_thinking;
    ReasoningLevel reasoning_level;

    // Per-turn counters
    int thinking_tokens;
    int response_tokens;
    int analysis_printed_header;
    long thinking_start_ms;
} ChatState;

// Chat functions
void chat_state_init(ChatState* cs, int show_thinking, ReasoningLevel level);
int  chat_build_prompt(ChatState* cs, Tokenizer* t, const char* user_message,
                       int* out_tokens, int* n_tokens);
int  chat_process_token(ChatState* cs, Tokenizer* t, int token, int* should_stop);
void chat_store_response(ChatState* cs, Tokenizer* t,
                         const int* gen_tokens, int n_gen);

// ----------------------------------------------------------------------------
// API functions

// Model loading and cleanup
void build_transformer(Transformer* t, const char* checkpoint_path);
void free_transformer(Transformer* t);

// Tokenizer
void build_tokenizer(Tokenizer* t, const char* tokenizer_path);
void free_tokenizer(Tokenizer* t);
void encode(Tokenizer* t, const char* text, int bos, int eos,
            int* tokens, int* n_tokens);
char* decode(Tokenizer* t, int prev_token, int token);

// Sampler
void build_sampler(Sampler* s, float temperature, int top_k, unsigned long long rng_seed, int vocab_size);
void free_sampler(Sampler* s);
int sample(Sampler* s, float* logits, int vocab_size);

// Inference — returns logits for next token
float* forward(Transformer* transformer, int token, int pos, Metrics* metrics);

// Reset KV cache (for new conversation)
void reset_kv_cache(Transformer* transformer);

// Metrics
void print_metrics(Metrics* m, long gen_end_ms);

#endif // RUN_GPTOSS_H

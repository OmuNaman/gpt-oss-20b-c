# GPT-OSS 20B C Inference Engine — Implementation Plan

## Context

We have a working C inference engine for nano_hindi (254M dense model) in `android/app/src/main/cpp/run.c`. Now we're building a **brand new** engine for OpenAI's GPT-OSS 20B — a 20.9B param MoE model with only 3.6B active per token. The model binary (`gpt_oss_20b.bin`, 12.82 GB) and tokenizer (`tokenizer_gptoss.bin`, 3 MB) are already exported and uploaded to HuggingFace at `omunaman/gpt-oss-20b-c`.

The goal: a single-file C engine that can run GPT-OSS 20B on PC (and eventually Android), similar to what we did with nano_hindi.

---

## Weight Formats — What's Actually In The Binary

The 12.82 GB binary has **three** different weight types:

| Type | What Uses It | How Stored | % of File |
|------|-------------|-----------|-----------|
| **float16** | Embeddings, attention weights, biases, norms, unembedding | 2 bytes per value, IEEE 754 half-precision | ~18% (~2.3 GB) |
| **MXFP4** (blocks) | MoE expert weights (mlp1, mlp2) | 4-bit values, 2 packed per byte, 16-entry LUT | ~77% (~9.9 GB) |
| **E8M0** (scales) | MoE expert scales | 1 byte per group of 32 elements, `2^(byte-127)` | ~5% (~0.6 GB) |

**MXFP4 dequantization** (the core new thing):
```
FP4_LUT = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6]

For each row of 2880 input values:
  → 90 groups of 32 elements each
  → Each group: 16 packed bytes (blocks) + 1 scale byte (E8M0)
  → value = FP4_LUT[nibble] * 2^(scale_byte - 127)
```

We do **fused dequant + dot product** — never materialize the full float32 matrix.

---

## File Structure

```
gpt_oss/
├── docs/
│   └── GPT_OSS_DEEP_DIVE.md        (existing — architecture reference)
├── raw_bin/                          (downloaded .bin files go here)
│   ├── gpt_oss_20b.bin              (12.82 GB model)
│   └── tokenizer_gptoss.bin         (3 MB tokenizer)
├── run_gptoss.h                      (NEW — all structs + constants + API)
├── run_gptoss.c                      (NEW — full engine + main())
├── Makefile                          (NEW — build for Win/Linux/Android)
├── export_model.py                   (existing — model exporter)
└── export_tokenizer.py               (existing — tokenizer exporter)
```

Single `.c` + `.h` — same proven pattern as nano_hindi. Expected ~1800-2200 lines.

---

## Data Structures

### Config (maps directly to 64-byte binary header)
```c
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
    int reserved;
} Config;
```

### LayerWeights (per-layer pointers into mmap'd file)
- **Attention** (all `uint16_t*` = float16): norm_scale(2880), qkv_weight(5120×2880), qkv_bias(5120), sinks(64), out_weight(2880×4096), out_bias(2880)
- **MoE** (mixed): norm_scale(2880 fp16), gate_weight(32×2880 fp16), gate_bias(32 fp16), mlp1_blocks(32×5760×1440 uint8), mlp1_scales(32×5760×90 uint8), mlp1_bias(32×5760 fp16), mlp2_blocks(32×2880×1440 uint8), mlp2_scales(32×2880×90 uint8), mlp2_bias(32×2880 fp16)

### RunState (~404 MB runtime)
- Activations: x, xb, xb2 (2880 each, float32)
- Attention: qkv(5120), q(4096), k(512), v(512), att(64×4097), attn_out(4096)
- MoE: gate_logits(32), expert_buf(5760), expert_act(2880), expert_out(2880), moe_out(2880)
- Output: logits(201088)
- **KV cache: ~384 MB** (24 layers × 4096 positions × 512 kv_dim × 4 bytes × 2)

---

## Forward Pass — Step by Step

### 1. Embedding lookup
float16 → float32 conversion for token row (2880 values)

### 2. Per layer (×24):

**2a. Pre-attention RMSNorm with learnable scale**
- `x_norm = (x / sqrt(mean(x²) + 1e-5)) * scale_vector`
- Scale is float16 (2880,) — convert on the fly
- Note: eps=1e-5 (not 1e-6 like nano_hindi)

**2b. Fused QKV projection + bias** (float16 matmul)
- Single matmul: (2880,) × (5120, 2880)^T + bias → (5120,)
- Split: Q = [0:4096], K = [4096:4608], V = [4608:5120]

**2c. RoPE** (half-split, theta=150000)
- Same half-split style as nano_hindi
- Different sign convention: `(x1*cos - x2*sin, x2*cos + x1*sin)`
- **No QK normalization** (unlike nano_hindi)

**2d. KV cache store** (same pattern as nano_hindi)

**2e. Attention with sinks + sliding window**
- Even layers (0,2,4...): 128-token sliding window
- Odd layers (1,3,5...): full attention
- **Sinks**: append learned per-head scalar to attention scores before softmax, drop after softmax
- GQA 8:1 (each of 64 query heads maps to one of 8 KV heads)

**2f. Output projection + residual** (float16 matmul + bias)

**2g. Pre-MLP RMSNorm with learnable scale**

**2h. Router** — expert selection
- gate_logits = xb × gate_weight^T + gate_bias → (32,)
- Top-4 selection, then softmax over those 4 values

**2i. Expert MLP (×4 active experts)**
- **MLP1**: MXFP4 matmul + bias → (5760,) — fused dequant, never materialize
- **SwiGLU**: interleaved gate/linear, alpha=1.702, +1.0 on linear branch, clamp [-7,7]
- **MLP2**: MXFP4 matmul + bias → (2880,) — fused dequant
- Weighted sum of 4 expert outputs + residual

### 3. Final RMSNorm with scale

### 4. Unembedding (float16 matmul, no softcap, no tied embeddings)

---

## Key New Functions (vs nano_hindi)

| Function | Purpose | Why New |
|----------|---------|---------|
| `f16_to_f32()` | Convert float16 → float32 | nano_hindi uses float32 weights |
| `f16_matmul_bias()` | Matmul with fp16 weights + fp16 bias | nano_hindi has no bias, float32 weights |
| `mxfp4_dot_row()` | Fused MXFP4 dequant + dot product | Entirely new — core of MoE |
| `mxfp4_matmul_bias()` | Full MXFP4 matmul using dot_row | Entirely new |
| `rmsnorm_scaled()` | RMSNorm with learnable fp16 scale | nano_hindi has no scale params |
| `swiglu()` | GPT-OSS SwiGLU variant | nano_hindi uses ReLU² |
| `topk()` | Top-k selection from 32 experts | New for MoE routing |

---

## Implementation Order

1. **`run_gptoss.h`** — All structs, constants, FP4_LUT, function declarations
2. **`run_gptoss.c`** — Implementation in this order:
   - `f16_to_f32()` — needed by everything
   - `rmsnorm_scaled()`, `softmax()` — basic math
   - `f16_matmul()`, `f16_matmul_bias()` — attention weights
   - `mxfp4_dot_row()`, `mxfp4_matmul_bias()` — MoE weights (the hard part)
   - `swiglu()` — activation function
   - `topk()` — expert selection
   - `build_transformer()` — header read + mmap + weight pointer setup
   - `forward()` — wire everything together
   - Tokenizer — adapted from nano_hindi (remove SentencePiece specifics, tiktoken BPE instead)
   - Sampler — copy from nano_hindi (identical)
   - `main()` — CLI parsing, generation loop, chat mode
3. **`Makefile`** — Build targets for Windows (MinGW), Linux, Android

---

## Key Architecture Differences from nano_hindi

| Feature | nano_hindi | GPT-OSS |
|---------|-----------|---------|
| Weights | float32, 968 MB | fp16 + MXFP4, 12.82 GB |
| MLP | Dense (ReLU²) | MoE 32 experts top-4 (SwiGLU) |
| Bias | None | Yes (QKV, output, MLP) |
| RMSNorm | No learnable params | Learnable scale vector |
| Tied embeddings | Yes | No (separate unembedding) |
| QK norm | Yes | No |
| Attention sinks | No | Yes |
| Logit softcap | Yes (15·tanh) | No |
| Per-layer mixing | resid_λ, x0_λ | No |
| Sliding window | SSSL pattern | Alternating even/odd |
| GQA ratio | 3:1 | 8:1 |
| Vocab | 68K (Sarvam) | 201K (o200k tiktoken) |

---

## Optimizations

1. **OpenMP** on all matmuls and attention heads (parallelizes across cores)
2. **Fused MXFP4 dequant** — never allocate full expert matrices in float32
3. **mmap** the 12.82 GB file — OS handles paging, no manual loading
4. **Pre-allocated KV cache** — zero dynamic allocation during generation
5. **Sliding window** — even layers only attend to 128 tokens (saves compute)
6. **MoE sparsity** — only 4/32 experts run per token (87.5% of MoE skipped)

Future SIMD optimizations (v2):
- ARM NEON `vcvt_f32_f16` for float16 conversion
- x86 F16C `_mm_cvtph_ps` for float16 conversion
- NEON/SSE for MXFP4 dot product (process 16 bytes at a time)

---

## Build & Test

**Build (Windows):**
```bash
gcc -O3 -fopenmp -o run_gptoss.exe run_gptoss.c -lm -lshell32
```

**Run:**
```bash
./run_gptoss.exe --model raw_bin/gpt_oss_20b.bin --tokenizer raw_bin/tokenizer_gptoss.bin --prompt "Hello"
./run_gptoss.exe --model raw_bin/gpt_oss_20b.bin --tokenizer raw_bin/tokenizer_gptoss.bin --chat
```

**Verification:**
- Compare C output against PyTorch reference (greedy sampling, same prompt)
- Test MXFP4 dot product in isolation first
- Verify attention sinks behavior
- Test sliding window at context boundary

---

## Runtime Metrics (printed automatically)

The engine will calculate and display:

1. **Memory usage** (printed at startup):
   - Model file size (mmap'd)
   - KV cache allocation
   - Activation buffers allocation
   - Total RAM footprint

2. **Per-token metrics** (printed during/after generation):
   - Tokens per second (tok/s)
   - Time per token (ms/tok)
   - Prompt processing speed (prompt tok/s)
   - Generation speed (gen tok/s)

3. **Summary stats** (printed at end):
   - Total tokens generated
   - Total time
   - Average tok/s
   - Peak memory (if available via OS API)
   - Active params per token (3.6B)
   - Expert utilization (which experts were selected, how often)

---

## Performance Estimate

| Platform | Speed (est.) | Notes |
|----------|-------------|-------|
| PC (multi-core CPU, 32GB RAM) | 1-3 tok/s | mmap fits in RAM, OpenMP helps |
| PC (16GB RAM) | 0.5-1.5 tok/s | Some page faults on 12.8 GB mmap |
| Android (8GB, Snapdragon 778G) | 0.1-0.5 tok/s | Heavy page faults, but functional |
| Android (12GB+) | 0.3-1.0 tok/s | Better with more RAM |

nano_hindi reference: 6.2 tok/s on PC, 12.0 tok/s on phone (254M, all float32, fits in RAM).
GPT-OSS is ~14x more active params but MXFP4 is 8x smaller per param, so memory bandwidth is ~2x worse.

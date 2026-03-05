# GPT-OSS 20B — Complete Technical Deep Dive

> This document is the single source of truth for porting OpenAI's GPT-OSS 20B
> to a custom C inference engine, following the same approach used for
> nano_hindi (254M). Everything discovered during research is here.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Official Links](#2-official-links)
3. [Architecture Summary](#3-architecture-summary)
4. [Config (exact values for 20B)](#4-config-exact-values-for-20b)
5. [Weight Format — MXFP4 + BF16](#5-weight-format--mxfp4--bf16)
6. [Tensor Names & Shapes (original/ format)](#6-tensor-names--shapes-original-format)
7. [Forward Pass — Step by Step](#7-forward-pass--step-by-step)
8. [MXFP4 Dequantization — The LUT Method](#8-mxfp4-dequantization--the-lut-method)
9. [Tokenizer — o200k_harmony (tiktoken)](#9-tokenizer--o200k_harmony-tiktoken)
10. [Our Binary Format (.bin)](#10-our-binary-format-bin)
11. [Comparison: nano_hindi vs GPT-OSS 20B](#11-comparison-nano_hindi-vs-gpt-oss-20b)
12. [Cloud Export Pipeline](#12-cloud-export-pipeline)
13. [C Engine Design Notes](#13-c-engine-design-notes)
14. [Performance Estimates](#14-performance-estimates)
15. [Open Questions & Risks](#15-open-questions--risks)

---

## 1. Overview

| Field | Value |
|-------|-------|
| **Full name** | gpt-oss-20b |
| **Released** | August 5, 2025 by OpenAI |
| **License** | Apache 2.0 (fully permissive, commercial OK) |
| **Total params** | 20.9B |
| **Active params/token** | 3.6B (MoE sparse activation) |
| **Architecture** | Transformer + Mixture-of-Experts (MoE) |
| **Checkpoint size** | 12.8 GiB (native MXFP4 + BF16) |
| **Min memory** | 16 GB |
| **Reasoning** | Chain-of-thought, comparable to o3-mini |
| **Context length** | 131,072 tokens (128K) |
| **Sibling model** | gpt-oss-120b (117B total, 5.1B active, ~o4-mini level) |

**Key claim**: gpt-oss-20b scored **98.7% on AIME 2025** (competition math),
matching/exceeding o3-mini on most benchmarks despite being a small open model.

The model was trained with MXFP4 quantization **during post-training** (not applied
after the fact), so the native 4-bit precision IS the intended precision.

---

## 2. Official Links

| Resource | URL |
|----------|-----|
| GitHub repo | https://github.com/openai/gpt-oss |
| HuggingFace model | https://huggingface.co/openai/gpt-oss-20b |
| OpenAI blog post | https://openai.com/index/introducing-gpt-oss/ |
| Model card (OpenAI) | https://openai.com/index/gpt-oss-model-card/ |
| arXiv paper | https://arxiv.org/abs/2508.10925 |
| Model card PDF | https://cdn.openai.com/pdf/419b6906-9da6-406c-a19d-1bb078ac7637/oai_gpt-oss_model_card.pdf |
| Ollama | https://ollama.com/library/gpt-oss:20b |
| LM Studio | https://lmstudio.ai/models/openai/gpt-oss-20b |
| Pre-converted GGUF (ggml-org) | https://huggingface.co/ggml-org/gpt-oss-20b-GGUF |
| Pre-converted GGUF (bartowski) | https://huggingface.co/bartowski/openai_gpt-oss-20b-GGUF |
| Architecture deep-dive (Modal) | https://modal.com/blog/gpt-oss-arch |
| Architecture deep-dive (Cameron Wolfe) | https://cameronrwolfe.substack.com/p/gpt-oss |
| Community reimplementation | https://github.com/HamzaElshafie/gpt-oss-20B |

---

## 3. Architecture Summary

```
Input tokens
    │
    ▼
┌─────────────────┐
│   Embedding      │  (201088, 2880) BF16
│   (no tied emb)  │
└────────┬────────┘
         │
    ┌────▼────┐  × 24 layers
    │         │
    │  RMSNorm │  (learnable scale)
    │    │     │
    │  Fused   │  QKV projection (2880 → 5120) BF16
    │  QKV     │  Q: 64 heads × 64 dim = 4096
    │    │     │  K: 8 heads × 64 dim = 512
    │    │     │  V: 8 heads × 64 dim = 512
    │  RoPE    │  YaRN scaling for 128K context
    │  (half-  │
    │  split)  │
    │    │     │
    │  GQA     │  64 query heads, 8 KV heads (8:1 ratio)
    │  Attn    │  + Attention sinks (learned per-head bias)
    │    │     │  + Sliding window (128 tokens on even layers)
    │    │     │
    │  O proj  │  (4096 → 2880) BF16 + bias
    │    │     │
    │  + Resid │
    │    │     │
    │  RMSNorm │  (learnable scale)
    │    │     │
    │  Router  │  Linear(2880 → 32) → sigmoid → top-4 → softmax
    │    │     │
    │  MoE     │  32 experts, top-4 active per token
    │  Expert  │  Each expert: SwiGLU MLP
    │  MLPs    │    mlp1: (2880 → 5760) MXFP4 + bias
    │    │     │    swiglu activation (clamped)
    │    │     │    mlp2: (2880 → 2880) MXFP4 + bias
    │    │     │
    │  Weighted│  sum of 4 expert outputs
    │  + Resid │
    │         │
    └────┬────┘
         │
    ┌────▼────┐
    │ RMSNorm  │  (learnable scale)
    │    │     │
    │ Unembed  │  Linear(2880 → 201088) BF16 (separate from embed)
    └────┬────┘
         │
         ▼
    Logits (201088)
```

### Parameter Breakdown

| Component | Parameters | % of Total |
|-----------|-----------|-----------|
| MLP / MoE experts | 19.12B | 91.4% |
| Attention layers | 0.64B | 3.1% |
| Embed + Unembed | 1.16B | 5.5% |
| **Total** | **20.91B** | 100% |
| **Active per token** | **3.61B** | 17.3% |

Active = attention (0.64B) + embed (1.16B) + 4/32 of MoE (19.12B × 4/32 ≈ 2.39B) ≈ 3.61B

---

## 4. Config (exact values for 20B)

From the HuggingFace `config.json`:

```json
{
    "architectures": ["GptOssForCausalLM"],
    "hidden_size": 2880,
    "intermediate_size": 2880,
    "num_hidden_layers": 24,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "head_dim": 64,
    "vocab_size": 201088,
    "num_local_experts": 32,
    "num_experts_per_tok": 4,
    "sliding_window": 128,
    "max_position_embeddings": 131072,
    "initial_context_length": 4096,
    "rope_theta": 150000,
    "rope_scaling": {
        "rope_type": "yarn",
        "factor": 32.0,
        "original_max_position_embeddings": 4096,
        "beta_fast": 32.0,
        "beta_slow": 1.0
    },
    "hidden_act": "silu",
    "swiglu_limit": 7.0,
    "attention_bias": true,
    "tie_word_embeddings": false,
    "rms_norm_eps": 1e-05,
    "eos_token_id": 200002,
    "pad_token_id": 199999
}
```

The `original/config.json` uses OpenAI's naming (from their `ModelConfig` dataclass):

```json
{
    "num_hidden_layers": 24,
    "num_experts": 32,
    "experts_per_token": 4,
    "vocab_size": 201088,
    "hidden_size": 2880,
    "intermediate_size": 2880,
    "swiglu_limit": 7.0,
    "head_dim": 64,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "sliding_window": 128,
    "initial_context_length": 4096,
    "rope_theta": 150000.0,
    "rope_scaling_factor": 32.0,
    "rope_ntk_alpha": 1.0,
    "rope_ntk_beta": 32.0
}
```

### Derived Constants

```
qkv_dim = head_dim * (num_attention_heads + 2 * num_key_value_heads)
        = 64 * (64 + 2*8) = 64 * 80 = 5120

attn_output_dim = head_dim * num_attention_heads = 64 * 64 = 4096

kv_dim = head_dim * num_key_value_heads = 64 * 8 = 512

gqa_ratio = num_attention_heads / num_key_value_heads = 64 / 8 = 8

mlp1_output_dim = intermediate_size * 2 = 5760  (interleaved gate+up for SwiGLU)

mlp1_packed_cols = hidden_size / 2 = 1440  (bytes, since 2 FP4 per byte)
```

---

## 5. Weight Format — MXFP4 + BF16

### Hybrid Precision Scheme

OpenAI uses a hybrid approach — different components have different dtypes:

| Component | Dtype | Bits/param | Notes |
|-----------|-------|-----------|-------|
| Embedding | BF16 | 16 | Standard bfloat16 |
| Attention weights | BF16 | 16 | Q, K, V, O projections |
| Attention biases | BF16 | 16 | Yes, this model HAS bias |
| Attention sinks | BF16 | 16 | Learned per-head scalar |
| RMSNorm scales | BF16 | 16 | Learnable (unlike nano_hindi) |
| Router weights | BF16 | 16 | Linear(2880, 32) |
| **MoE expert weights** | **MXFP4** | **~4.25** | **Blocks (FP4) + Scales (E8M0)** |
| MoE expert biases | BF16 | 16 | Standard bfloat16 |
| Unembedding | BF16 | 16 | Separate from embedding |

### Why MXFP4 Matters

The MoE experts contain **91.4% of all parameters** (19.12B out of 20.91B).
Quantizing them to ~4.25 bits/param reduces those 19.12B params from ~38 GB (BF16)
to ~10 GB (MXFP4), which is why the total checkpoint is only 12.8 GiB.

The model was **trained/fine-tuned with MXFP4** — this is not a post-hoc
quantization. The quality at MXFP4 is the intended quality.

### File Layout on HuggingFace

**original/ directory** (preferred for export):
- `config.json` — 376 bytes
- `dtypes.json` — 13.1 KB (dtype of every tensor)
- `model.safetensors` — 13.8 GB (single file, all weights)

**Root directory** (HuggingFace converted format):
- `config.json` — HuggingFace format config
- `model-00000-of-00002.safetensors` — shard 1
- `model-00001-of-00002.safetensors` — shard 2
- `model-00002-of-00002.safetensors` — shard 3
- `model.safetensors.index.json` — tensor→shard mapping

We use the **original/** format because it's a single file with straightforward naming.

---

## 6. Tensor Names & Shapes (original/ format)

### Global Tensors

| Tensor Name | Dtype | Shape | Size (bytes) |
|-------------|-------|-------|-------------|
| `embedding.weight` | BF16 | (201088, 2880) | 1,158,266,880 |
| `norm.scale` | BF16 | (2880,) | 5,760 |
| `unembedding.weight` | BF16 | (201088, 2880) | 1,158,266,880 |

### Per-Layer Tensors (× 24 layers)

**Attention (all BF16):**

| Tensor Name | Shape | Size (bytes) |
|-------------|-------|-------------|
| `block.{N}.attn.norm.scale` | (2880,) | 5,760 |
| `block.{N}.attn.qkv.weight` | (5120, 2880) | 29,491,200 |
| `block.{N}.attn.qkv.bias` | (5120,) | 10,240 |
| `block.{N}.attn.sinks` | (64,) | 128 |
| `block.{N}.attn.out.weight` | (2880, 4096) | 23,592,960 |
| `block.{N}.attn.out.bias` | (2880,) | 5,760 |

**MoE (mixed precision):**

| Tensor Name | Dtype | Shape | Size (bytes) |
|-------------|-------|-------|-------------|
| `block.{N}.mlp.norm.scale` | BF16 | (2880,) | 5,760 |
| `block.{N}.mlp.gate.weight` | BF16 | (32, 2880) | 184,320 |
| `block.{N}.mlp.gate.bias` | BF16 | (32,) | 64 |
| `block.{N}.mlp.mlp1_weight.blocks` | **FP4** | (32, 5760, 1440) | 265,420,800 |
| `block.{N}.mlp.mlp1_weight.scales` | **UE8** | (32, 5760) | 184,320 |
| `block.{N}.mlp.mlp1_bias` | BF16 | (32, 5760) | 368,640 |
| `block.{N}.mlp.mlp2_weight.blocks` | **FP4** | (32, 2880, 1440) | 132,710,400 |
| `block.{N}.mlp.mlp2_weight.scales` | **UE8** | (32, 2880) | 92,160 |
| `block.{N}.mlp.mlp2_bias` | BF16 | (32, 2880) | 184,320 |

### Size Totals

```
Per layer:
  Attention BF16:   ~53.1 MB
  MoE MXFP4:       ~398.1 MB (blocks)
  MoE scales:      ~0.27 MB
  MoE biases BF16: ~0.55 MB
  MoE other BF16:  ~0.19 MB
  Total per layer:  ~452.2 MB

24 layers:            ~10.85 GB
Embedding:            ~1.16 GB (BF16)
Unembedding:          ~1.16 GB (BF16)
Final norm:           ~5.7 KB

Grand total:          ~13.17 GB (matches ~13.8 GB safetensors with overhead)
```

### Understanding the MoE Weight Shapes

For `mlp1_weight.blocks` shape `(32, 5760, 1440)`:
- **32** = number of experts
- **5760** = `intermediate_size * 2` = output dimension (interleaved gate+up for SwiGLU)
- **1440** = `hidden_size / 2` = 2880 FP4 values packed into 1440 bytes (2 nibbles per byte)

Each byte packs 2 FP4 values:
- Low nibble (bits 0-3): value at even index
- High nibble (bits 4-7): value at odd index

For `mlp1_weight.scales` shape `(32, 5760)`:
- One uint8 scale per row of 2880 packed values
- Scale is E8M0 format: exponent with bias 127
- Actual scale = 2^(uint8_value - 127)

---

## 7. Forward Pass — Step by Step

Based on the official PyTorch reference implementation (`gpt_oss/torch/model.py`):

### 1. Token Embedding

```python
x = embedding(tokens)   # (seq_len,) → (seq_len, 2880)
```

No post-embedding norm. No tied embeddings.

### 2. For each of 24 layers:

#### 2a. Pre-Attention RMSNorm (learnable)

```python
t = rmsnorm(x) * scale   # scale is a learned (2880,) vector
```

Unlike nano_hindi where RMSNorm has no params, GPT-OSS has learnable scale.

#### 2b. Fused QKV Projection

```python
qkv = t @ qkv_weight.T + qkv_bias   # (2880,) → (5120,)
q = qkv[:4096]                        # 64 heads × 64 dim
k = qkv[4096:4608]                    # 8 heads × 64 dim
v = qkv[4608:5120]                    # 8 heads × 64 dim
```

Note: single fused linear layer, not separate Q/K/V projections.

#### 2c. RoPE with YaRN Scaling

```python
# Half-split style (same as nano_hindi):
x1, x2 = chunk(x, 2, dim=-1)   # split into first half and second half
o1 = x1 * cos - x2 * sin
o2 = x2 * cos + x1 * sin
rotated = cat(o1, o2)
```

**YaRN** extends RoPE to 128K context via NTK-by-parts interpolation:
- `rope_theta` = 150000 (base frequency)
- `scaling_factor` = 32 (extends 4096 → 131072)
- Uses concentration factor and interpolation/extrapolation ramp
- Low-frequency dimensions: interpolated (compressed)
- High-frequency dimensions: extrapolated (unchanged)
- Middle dimensions: smooth interpolation between the two

For our C engine with short context (4096), standard RoPE with theta=150000 should
work fine without YaRN (YaRN only matters for >4096 tokens).

#### 2d. GQA Attention with Sinks

```python
# q shape: (seq, 8 kv_groups, 8 queries_per_group, 64 head_dim)
# k shape: (seq, 8 kv_heads, 64 head_dim)
# v shape: (seq, 8 kv_heads, 64 head_dim)

# Attention scores
QK = einsum("qhmd,khmd->hmqk", Q, K) * (1/sqrt(64))

# Add causal mask
QK += causal_mask

# Append learned sink scores (per-head scalar)
QK_with_sinks = cat([QK, sinks], dim=-1)

# Softmax over [keys + 1 sink]
weights = softmax(QK_with_sinks, dim=-1)

# Drop the sink weight (only used for normalization)
weights = weights[..., :-1]

# Weighted sum of values
output = einsum("hmqk,khmd->qhmd", weights, V)
```

**Attention sinks** explained: Each head has a learned scalar that gets appended
to the attention scores before softmax. This creates a "null attention" target —
if a head doesn't want to attend to anything, the softmax probability mass flows
to the sink instead of being forced onto some key. After softmax, the sink weight
is discarded (there's no corresponding value).

**Sliding window**: Even-indexed layers (0, 2, 4, ...) use a 128-token window.
Odd-indexed layers (1, 3, 5, ...) use full attention. This alternating pattern
gives every other layer access to the full context.

#### 2e. Output Projection + Residual

```python
t = o_proj(attention_output) + o_proj_bias   # (4096,) → (2880,)
x = x + t                                     # residual connection
```

#### 2f. Pre-MLP RMSNorm (learnable)

```python
t = rmsnorm(x) * mlp_norm_scale
```

#### 2g. Router — Expert Selection

```python
gate_logits = t @ router_weight.T + router_bias   # (2880,) → (32,)
# No activation on gate logits before topk (raw logits)
top4_values, top4_indices = topk(gate_logits, k=4)
expert_weights = softmax(top4_values)              # softmax only over selected 4
```

The router selects the top-4 experts (out of 32) and computes softmax weights
over only those 4 selected experts.

#### 2h. Expert MLP Computation (× 4 active experts)

For each of the 4 selected experts:

```python
# MLP1: up-projection with interleaved gate+linear
h = x @ mlp1_weight[expert_idx].T + mlp1_bias[expert_idx]
# h shape: (5760,) = interleaved [gate_0, linear_0, gate_1, linear_1, ...]

# SwiGLU activation (special variant)
gate_part = h[::2]     # even indices → gate
linear_part = h[1::2]  # odd indices → linear

gate_part = clamp(gate_part, max=7.0)
linear_part = clamp(linear_part, min=-7.0, max=7.0)

activated = (gate_part * sigmoid(1.702 * gate_part)) * (linear_part + 1.0)
# Note the +1.0 bias on the linear branch — this is unique to GPT-OSS

# MLP2: down-projection
expert_out = activated @ mlp2_weight[expert_idx].T + mlp2_bias[expert_idx]
# (2880,) → (2880,)
```

#### 2i. Weighted Expert Sum + Residual

```python
moe_output = sum(expert_weights[i] * expert_outputs[i] for i in range(4))
x = x + moe_output
```

### 3. Final RMSNorm

```python
x = rmsnorm(x) * final_norm_scale
```

### 4. Logits (Unembedding)

```python
logits = x @ unembedding_weight.T   # (2880,) → (201088,)
```

No logit softcap (unlike nano_hindi which uses 15 * tanh(x/15)).
No tied embeddings (separate embedding and unembedding matrices).

---

## 8. MXFP4 Dequantization — The LUT Method

From OpenAI's `gpt_oss/torch/weights.py`:

### The FP4 Lookup Table

```python
FP4_VALUES = [
    +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,   # indices 0-7
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,   # indices 8-15
]
```

Each FP4 value is 4 bits (1 sign, 2 exponent, 1 mantissa = E2M1).
But in practice, we just use a 16-entry LUT — no bit manipulation needed.

### Packing Format

Each byte stores 2 FP4 values:
```
Byte: [high_nibble | low_nibble]
       bits 7-4      bits 3-0

low_nibble  → LUT index for even position
high_nibble → LUT index for odd position
```

### Scale Format (E8M0)

One uint8 per row. The scale is a pure power of 2:
```
actual_scale = 2^(uint8_value - 127)
```

### Full Dequantization (per row of 2880 values)

```python
# blocks: (1440,) uint8 — packed FP4 values
# scale:  uint8 — shared exponent

idx_lo = blocks & 0x0F          # low nibbles → 1440 indices
idx_hi = blocks >> 4             # high nibbles → 1440 indices

values[0::2] = FP4_LUT[idx_lo]  # even positions
values[1::2] = FP4_LUT[idx_hi]  # odd positions

values *= 2^(scale - 127)        # apply shared scale (ldexp)
```

### C Implementation

```c
static const float FP4_LUT[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// Dot product of MXFP4-packed row with float32 vector
// blocks: 1440 bytes (2880 FP4 values), scale: 1 byte, x: 2880 floats
float mxfp4_dot(const uint8_t* blocks, uint8_t scale, const float* x, int n_packed) {
    float dot = 0.0f;
    for (int j = 0; j < n_packed; j++) {
        uint8_t byte = blocks[j];
        dot += FP4_LUT[byte & 0x0F] * x[2*j];
        dot += FP4_LUT[byte >> 4]   * x[2*j + 1];
    }
    return ldexpf(dot, (int)scale - 127);
}
```

This is remarkably simple — just a LUT lookup and a single `ldexpf` per row.

---

## 9. Tokenizer — o200k_harmony (tiktoken)

### Format

GPT-OSS uses OpenAI's **o200k_harmony** tokenizer, which is based on **tiktoken**
(byte-pair encoding with regex pre-tokenization). This is fundamentally different
from the SentencePiece tokenizer used by nano_hindi.

### Key Differences from SentencePiece

| Feature | SentencePiece (nano_hindi) | tiktoken (GPT-OSS) |
|---------|---------------------------|---------------------|
| Vocab size | 68,096 | 201,088 |
| BPE variant | Unigram / BPE with scores | Rank-based BPE |
| Pre-tokenization | Prepend ▁, replace spaces | Regex pattern split |
| Space handling | ▁ (U+2581) | Explicit space tokens |
| Byte fallback | `<0xNN>` tokens | Native byte tokens |
| Special tokens | `<s>` (1), `</s>` (2) | `<\|endoftext\|>` (200001), `<\|end\|>` (200002), etc. |
| Merge priority | Score-based (higher = merge first) | Rank-based (lower = merge first) |

### Special Token IDs

```
eos_token_id = 200002    (<|end|>)
pad_token_id = 199999
```

### Export Strategy

We export the tokenizer to our binary format (same as nano_hindi's tokenizer.bin)
using **negative rank as score** so our existing BPE merge logic works:
- Lower rank → merged first → needs higher score
- So: `score = -rank` (or `score = vocab_size - rank`)

The C engine's BPE encode function can work with rank-based priority too — we just
need to prefer the merge with the LOWEST rank (instead of highest score).

### Pre-tokenization Regex

tiktoken's o200k regex splits text into chunks before BPE. The pattern handles:
- Whitespace (spaces, newlines, tabs)
- Contractions (English: 's, 't, 're, etc.)
- Letters vs numbers vs punctuation
- Unicode word boundaries

For a first version, we can use simplified pre-tokenization in C (split on
whitespace + punctuation boundaries) which will work for most cases.

---

## 10. Our Binary Format (.bin)

### Header (64 bytes = 16 × int32)

```
Offset  Field                      Value (20B)
──────  ─────                      ───────────
0       magic                      0x474F5353 ("GOSS")
4       version                    1
8       hidden_size                2880
12      intermediate_size          2880
16      num_hidden_layers          24
20      num_attention_heads        64
24      num_key_value_heads        8
28      head_dim                   64
32      vocab_size                 201088
36      num_local_experts          32
40      experts_per_token          4
44      sliding_window             128
48      max_seq_len                4096 (inference limit)
52      rope_theta                 150000
56      eos_token_id               200002
60      reserved                   0
```

### Weight Layout (after 64-byte header)

All BF16 tensors are converted to **float16** during export.
All MXFP4 tensors are kept as **raw uint8** (blocks + scales).

```
Section 1: Embedding
  embedding.weight           (201088, 2880)      float16    ~1.10 GB

Section 2: Transformer Layers (× 24)
  For each layer N = 0..23:

  --- Attention ---
  attn.norm.scale            (2880,)             float16
  attn.qkv.weight            (5120, 2880)        float16
  attn.qkv.bias              (5120,)             float16
  attn.sinks                 (64,)               float16
  attn.out.weight            (2880, 4096)         float16
  attn.out.bias              (2880,)             float16

  --- MoE ---
  mlp.norm.scale             (2880,)             float16
  mlp.gate.weight            (32, 2880)          float16
  mlp.gate.bias              (32,)               float16
  mlp.mlp1.blocks            (32, 5760, 1440)    uint8 (MXFP4)
  mlp.mlp1.scales            (32, 5760)          uint8 (E8M0)
  mlp.mlp1.bias              (32, 5760)          float16
  mlp.mlp2.blocks            (32, 2880, 1440)    uint8 (MXFP4)
  mlp.mlp2.scales            (32, 2880)          uint8 (E8M0)
  mlp.mlp2.bias              (32, 2880)          float16

Section 3: Final Norm + Unembedding
  norm.scale                 (2880,)             float16
  unembedding.weight         (201088, 2880)      float16    ~1.10 GB
```

### Estimated File Size

```
Embedding:                    ~1.10 GB
24 × attention (float16):    ~1.22 GB
24 × MoE blocks (uint8):     ~9.55 GB
24 × MoE scales (uint8):     ~6.4 MB
24 × MoE biases (float16):   ~12.7 MB
24 × other (float16):        ~4.5 MB
Final norm + unembed:         ~1.10 GB
Header:                       64 bytes
───────────────────────────────────────
Total:                        ~12.97 GB
```

---

## 11. Comparison: nano_hindi vs GPT-OSS 20B

| Feature | nano_hindi | GPT-OSS 20B |
|---------|-----------|-------------|
| Total params | 254M | 20.9B (82×) |
| Active params | 254M (all) | 3.6B (14×) |
| Architecture | Dense | MoE (32 experts, top-4) |
| Layers | 32 | 24 |
| Hidden dim | 768 | 2,880 |
| Attention heads | 12 Q, 4 KV (GQA 3:1) | 64 Q, 8 KV (GQA 8:1) |
| Head dim | 64 | 64 |
| MLP type | ReLU² (2 weights) | SwiGLU (3 weights × 32 experts) |
| MLP activation | relu(x)² | x·σ(1.702x)·(x_lin+1) |
| RoPE | Half-split, θ=10000 | Half-split, θ=150000, YaRN |
| Context | 1024 | 131,072 (128K) |
| Window attn | SSSL (512/1024) | Alternating (128/full) |
| RMSNorm | No learnable params | Learnable scale |
| Bias | None | Yes (attn Q,K,V,O + MoE) |
| QK normalization | Yes (per-head RMSNorm) | No |
| Attention sinks | No | Yes (learned per-head) |
| Logit softcap | 15·tanh(x/15) | None |
| Tied embeddings | Yes | No (separate unembed) |
| Per-layer mixing | resid_λ, x0_λ | None |
| Vocab size | 68,096 (Sarvam) | 201,088 (o200k) |
| Tokenizer | SentencePiece | tiktoken |
| Weight precision | float32 | BF16 + MXFP4 |
| Checkpoint size | 968 MB | 12.8 GiB |
| Chat template | Hindi markers | Harmony format |

---

## 12. Cloud Export Pipeline

### Prerequisites

- Cloud GPU instance with **>30 GB storage** (28 GB needed: 13.8 download + 13 output)
- Python 3.10+
- Internet access for HuggingFace download

### Commands (run in order)

```bash
# 1. Install dependencies
pip install safetensors torch numpy huggingface_hub tiktoken

# 2. Login to HuggingFace
huggingface-cli login --token YOUR_HF_TOKEN

# 3. Download the model (original/ format only — much smaller)
huggingface-cli download openai/gpt-oss-20b \
    --include "original/*" \
    --local-dir gpt-oss-20b-raw

# 4. Export model weights to our flat binary
python export_model.py \
    --input gpt-oss-20b-raw/original \
    --output gpt_oss_20b.bin

# 5. Export tokenizer
python export_tokenizer.py \
    --output tokenizer_gptoss.bin

# 6. Create HuggingFace repo and upload
huggingface-cli repo create omunaman/gpt-oss-20b-c --type model
huggingface-cli upload omunaman/gpt-oss-20b-c gpt_oss_20b.bin
huggingface-cli upload omunaman/gpt-oss-20b-c tokenizer_gptoss.bin

# 7. Verify
ls -lh gpt_oss_20b.bin tokenizer_gptoss.bin
```

### Expected Timings

| Step | Duration |
|------|----------|
| Download (13.8 GB) | 5-15 min (depends on bandwidth) |
| Export model | 5-10 min (mostly I/O) |
| Export tokenizer | <1 min |
| Upload (13 GB) | 5-15 min |

---

## 13. C Engine Design Notes

### New Functions Needed (vs nano_hindi run.c)

1. **`mxfp4_matmul()`** — matrix multiply with MXFP4 dequantization on-the-fly
2. **`f16_to_f32()`** — float16 to float32 conversion
3. **`f16_matmul()`** — matrix multiply with float16 weights
4. **`moe_forward()`** — router + top-k expert selection + weighted sum
5. **`swiglu()`** — the GPT-OSS variant with clamping and +1 bias
6. **`rmsnorm_with_scale()`** — RMSNorm with learnable scale vector
7. **`rope_yarn()`** — YaRN-extended RoPE (or standard RoPE for short contexts)
8. **`attention_with_sinks()`** — attention with learned sink bias

### KV Cache (CRITICAL)

The OpenAI reference code does NOT use a KV cache — it recomputes the full
sequence on every token. This is because it's a reference implementation,
not optimized for inference.

Our C engine MUST implement a KV cache (like nano_hindi) or inference will be
impossibly slow (O(n²) per token instead of O(n)).

### Memory Estimates for Inference

```
Model weights (mmap'd):                 ~13 GB (not in RAM, paged from disk)
KV cache (4096 ctx, float32):
  per layer: 2 × 512 × 4096 × 4 =      16 MB
  24 layers:                             384 MB
Activation buffers:
  x, xb, xb2, etc. at dim=2880:         ~1 MB
  attention scores (64 heads × 4096):    ~1 MB
  expert buffers:                        ~5 MB
  logits (201088):                       ~800 KB
──────────────────────────────────────────────────
Total RAM needed:                        ~400 MB + page cache for mmap
```

---

## 14. Performance Estimates

### PC (CPU, single-threaded, no SIMD)

```
nano_hindi (254M, float32):     6.2 tok/s
GPT-OSS active compute:        3.6B params vs 254M = ~14× more
But weights are 4-bit:          ~2× less memory bandwidth
Net slowdown:                   ~7× slower
Estimated:                      ~0.9 tok/s

With OpenMP (8 threads):        ~4-5 tok/s (rough estimate)
```

### Mobile (Vivo T4 5G, 8 GB RAM)

```
nano_hindi:                     12.0 tok/s
GPT-OSS estimated:              ~1.0 tok/s (if mmap doesn't thrash)
Reality with 8 GB RAM:          Likely 0.1-0.5 tok/s (constant page faults)
Phones with 12+ GB:             More viable (~0.5-1.5 tok/s)
```

### Key Optimization Opportunities

1. **SIMD (NEON on ARM, SSE/AVX on x86)**: 4-8× speedup on matmul
2. **Multi-threading (OpenMP)**: 4-8× on multi-core
3. **Expert caching**: Only 4 of 32 experts active — precompute which ones
4. **Partial KV cache**: Short sliding window (128) means less cache for half the layers
5. **Quantized KV cache**: Store K,V in int8 instead of float32

---

## 15. Open Questions & Risks

### Must Resolve

1. **Tokenizer encoding accuracy**: tiktoken regex pre-tokenizer in C is complex.
   May need to simplify for first version.

2. **Chat template format**: GPT-OSS uses "Harmony" format for structured responses.
   Need to understand the exact prompt format for chat mode.

3. **YaRN RoPE**: For short context (≤4096), standard RoPE with theta=150000 should
   work. For longer contexts, need YaRN implementation in C.

4. **SwiGLU interleaving**: The gate and linear branches are interleaved in the
   mlp1 output (even indices = gate, odd = linear). Must handle correctly.

### Risks

1. **12.8 GB mmap on 8 GB phone**: Will cause severe page fault thrashing.
   MoE makes this worse because different tokens access different experts
   (random access patterns).

2. **Tokenizer mismatch**: If our C encoder doesn't match tiktoken exactly,
   the model may produce degraded outputs (wrong token boundaries).

3. **Numerical accuracy**: MXFP4 dequant + float16 attention weights mean
   we're doing mixed-precision math. Need to verify outputs match PyTorch.

4. **Missing KV cache in reference**: We're adding KV cache ourselves — need
   to verify it produces identical outputs to the cacheless reference.

---

## Appendix A: OpenAI's Reference Implementation Key Files

```
gpt-oss/
├── gpt_oss/
│   ├── torch/
│   │   ├── model.py        ← Full architecture (Transformer, AttentionBlock, MLPBlock)
│   │   └── weights.py      ← MXFP4 dequantization (Checkpoint class, FP4_VALUES LUT)
│   ├── triton/              ← Optimized Triton kernels (MoE, attention)
│   ├── metal/               ← Apple Silicon Metal implementation
│   │   └── scripts/create-local-model.py  ← Weight conversion for Metal
│   ├── chat.py              ← Terminal chat client
│   └── generate.py          ← Generation module
```

## Appendix B: SwiGLU Variant Details

GPT-OSS's SwiGLU is NOT standard. The formula is:

```
swiglu(h) = gate_act * (linear + 1.0)

where:
  gate_part   = h[::2]                              (even indices)
  linear_part = h[1::2]                              (odd indices)
  gate_act    = clamp(gate, max=7.0) * sigmoid(1.702 * clamp(gate, max=7.0))
  linear      = clamp(linear_part, -7.0, 7.0)
```

The `1.702` coefficient makes the gate activation similar to GELU.
The `+1.0` on the linear branch means the identity is passed through by default.
The clamping to `[-7, 7]` prevents overflow in the activation.

## Appendix C: Attention Sinks Details

Each attention head has a learned scalar parameter `sinks[h]` (64 values total).
Before softmax, this scalar is appended as an extra "key" in the attention scores:

```
attention_scores = [q·k₀, q·k₁, ..., q·kₙ, sinks[h]]
weights = softmax(attention_scores)
# Use only the first n weights (discard the sink weight)
output = Σ weights[i] * v[i]   (for i = 0..n, excluding sink)
```

This allows heads to "not attend" by routing probability mass to the sink.
The sink has no corresponding value — its weight is computed but discarded.

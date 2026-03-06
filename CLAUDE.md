# GPT-OSS 20B — C Inference Engine

## Project Overview

From-scratch single-file C inference engine for OpenAI's GPT-OSS 20B (MoE, 20.9B params, 3.6B active/token). Built in the llama2.c / nano_hindi tradition. Currently at **7.2 tok/s** (E8M0 LUT + f16 KV + P-core pinning), **67% faster than llama.cpp** at 4.30 tok/s.

## Working Directory & OS

`D:\Coding_Workspace\GPT_OSS` — Windows 11, PowerShell, MinGW/gcc

## Architecture

- **Model**: GPT-OSS 20B — MoE, 32 experts, top-4 active per token
- **Weights**: MXFP4 (MoE experts, 91.4% of params) + float16 (attention)
- **Dims**: hidden=2880, layers=24, heads=64Q/8KV (GQA 8:1), head_dim=64
- **Vocab**: 201,088 (o200k_harmony, tiktoken-based BPE)
- **Context**: 4096 inference limit, native 131K via YaRN
- **Checkpoint**: 12.82 GB custom .bin (mmap'd)
- **Runtime**: ~194 MB (192 MB float16 KV cache + ~2 MB activations)

## Key Files

```
run_gptoss.h                    — structs, constants, FP4_LUT, declarations
run_gptoss.c                    — full engine (~2000 lines)
run_gptoss_native.exe           — compiled binary (-march=native)
raw_bin/gpt_oss_20b.bin         — model weights (12.82 GB)
raw_bin/tokenizer_gptoss.bin    — tokenizer binary
export_model.py                 — safetensors → .bin converter
export_tokenizer.py             — tiktoken → binary exporter
docs/POA.md                     — implementation plan
docs/GPT_OSS_DEEP_DIVE.md       — full model research
docs/devlog_1_simd_optimization.md — SIMD optimization plan
docs/devlog_3_phase1_optimizations.md — Phase 1-4 optimization results
```

## Build

```bash
gcc -O3 -march=native -fopenmp -o run_gptoss_native.exe run_gptoss.c -lm -lshell32
```

`-march=native` auto-detects Raptor Lake i7-14700HX, enables AVX2/FMA/F16C/BMI2 with CPU-specific scheduling. +18% over explicit flags.

## Run

```powershell
# Benchmark (deterministic greedy)
./run_gptoss_native.exe --model raw_bin/gpt_oss_20b.bin --tokenizer raw_bin/tokenizer_gptoss.bin --prompt "Explain the halting problem in one paragraph." --max_tokens 256 --temp 0 --seed 42

# Chat
./run_gptoss_native.exe --model raw_bin/gpt_oss_20b.bin --tokenizer raw_bin/tokenizer_gptoss.bin --chat --reasoning high --show-thinking --temp 0.7 --max_tokens 512 --seed 42
```

## Performance History (REAL MEASUREMENTS)

**Hardware**: i7-14700HX (8P+12E, 28 threads), 16 GB DDR5, Windows 11, NVMe SSD

| Build | tok/s | ms/tok | Notes |
|-------|-------|--------|-------|
| Scalar baseline | 1.66 | 604 | No SIMD |
| -mavx2 -mfma -mf16c | 3.34 | 300 | Explicit SIMD flags |
| -march=native | 3.92 | 255 | +18% from CPU-specific scheduling |
| + vpshufb MXFP4 | ~4.01 (proj) | ~249 | +20% from parallel nibble LUT |
| **llama.cpp** | **4.30** | **232** | **Competitor** |
| + E8M0 LUT | 6.88 | 145 | +128%! ldexpf() was the bottleneck |
| + serial router | 7.12 | 141 | +3.6% from avoiding OMP on 32-row gate |
| + f16 KV cache | 7.19 | 139 | 384→192 MB, SIMD attention reads |
| + P-core pinning | **7.38** | **135** | **67% faster than llama.cpp** |

**Memory bandwidth ceiling**: ~16.7 tok/s (3.6 GB active params × DDR5 ~60 GB/s practical)

## Critical Finding: ldexpf() Was the Real Bottleneck

The previous "memory-bandwidth-bound" diagnosis was wrong. The E8M0 LUT optimization proved that `ldexpf()` in the MXFP4 hot path was consuming more time than the actual memory reads:
- **E8M0 LUT (replace ldexpf)**: **+128%** — replaced ~50M ldexpf() calls/token with array lookups
- F16 4× accumulator unrolling: **0% gain** (FMA unit not the bottleneck)
- SIMD attention (AVX2 QK/V): **0% gain** (GCC already auto-vectorized)
- Reduce threads to 8: **-34%** (threads hide page fault latency on 16GB system)
- vpshufb MXFP4: **+20%** (reduces instruction count per memory access)

**Updated implication**: With ldexpf() gone, the engine IS now truly memory-bandwidth-bound at 7.2 tok/s (43% of theoretical DDR5 ceiling). Remaining gains must come from speculative decoding or weight repacking.

## Optimization Roadmap (TO IMPLEMENT)

### Phase 1: Low-Risk Kernel Wins ~~(target: 4.0–4.5 tok/s)~~ DONE — achieved 7.38 tok/s

**1a. ✅ E8M0 LUT — replace ldexpf() in mxfp4_dot_row()** → +128% (3.01→6.88 tok/s)
- Replaced `ldexpf(1.0f, (int)scales[g] - 128)` with `e8m0_half_lut[scales[g]]`
- Two 256-entry float LUTs initialized at startup
- THE dominant win — ldexpf() was the real bottleneck, not memory bandwidth

**1b. ❌ Fused greedy unembedding+argmax** → SKIPPED (no measurable gain)
- 784 KB logits buffer is negligible vs 3.6 GB model reads
- OpenMP parallel reduction overhead offset any theoretical savings

**1c. ✅ Software prefetching in matmul loops** → neutral (kept, benign)
- Added `_mm_prefetch` in f16_matmul, f16_matmul_bias, mxfp4_matmul_bias, mxfp4_dot_row
- Raptor Lake hardware prefetcher already handles sequential access well

**1d. ✅ Serial router matmul** → +3.6% (6.87→7.12 tok/s)
- Created f16_matmul_bias_serial() for the 32×2880 gate GEMV
- Eliminates 6,144 OMP fork/join cycles per generation

### Phase 2: Memory Bandwidth Reduction ~~(target: 4.5–5.5 tok/s)~~ DONE

**2a. ✅ Float16 KV cache** → +1% (7.12→7.19 tok/s), 384→192 MB
- Changed key_cache/value_cache from float* to uint16_t*
- Added f32_to_f16() helper, SIMD attention reads via _mm256_cvtph_ps
- Modest speed gain at short sequences, bigger win at longer contexts

**2b. ✅ Fused RoPE + KV cache store** → neutral
- Fused K-head RoPE with f32→f16 cache write (1 pass instead of 3)
- Data was already in L1, so no measurable gain

**2c. Windows PrefetchVirtualMemory for warmup** — NOT YET IMPLEMENTED
- Use `PrefetchVirtualMemory()` API on the mmap'd model file at startup
- Tells Windows to fault in pages with large sequential I/Os instead of random 4KB faults

### Phase 3: Scheduling & Threading ~~(target: 5.0–6.0 tok/s)~~ DONE

**3a. ✅ P-core thread pinning** → +3.4% (7.14→7.38 tok/s)
- SetProcessAffinityMask(GetCurrentProcess(), 0x0000FFFF) + HIGH_PRIORITY_CLASS
- Eliminates E-core stragglers at OpenMP barriers

**3b. Persistent expert scheduler** — NOT YET IMPLEMENTED
- Current: OpenMP fork/join on every matmul, 192× per token
- Would create persistent thread teams to eliminate repeated fork/join overhead

### Phase 4: UX & Benchmarking — DONE

**4a. ✅ --quiet benchmark mode**
- Buffers output, prints after timing ends
- Use with `--quiet` flag

**4b. ✅ Persistent KV cache in chat mode**
- Tracks chat_pos across turns, only processes new tokens
- Falls back to full replay on history truncation
- Dramatically faster multi-turn chat

**4c. ✅ Fix duplicate print_metrics()** — No fix needed (no actual duplication)

### Phase 5: Advanced (target: 8+ tok/s, future)

**5a. 2-row or 4-row MXFP4 repacking**
- Offline repack expert weights so multiple output rows share one x-vector load
- Amortizes input bandwidth across multiple dot products
- Requires changes to export_model.py and weight layout

**5b. Speculative decoding (Prompt Lookup Decoding)**
- Zero extra models needed: match last N generated tokens against prompt n-grams
- If match found, batch-verify next K tokens in single forward() pass
- Amortizes 3.6 GB memory read over K tokens instead of 1
- Only way to break the memory bandwidth ceiling
- Reported 2–3× speedup in llama.cpp on suitable workloads

**5c. Speculative decoding (self-speculative / Medusa-style)**
- Add lightweight prediction heads for multi-token speculation
- Medusa reports 2.2–3.6×, EAGLE reports 2.7–3.5×
- Requires model modification — heavier lift

## MXFP4 Quick Reference

- 4-bit E2M1, 2 per byte (low nibble=even, high nibble=odd)
- LUT: [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
- E8M0 scale per group of 32: actual_scale = 2^(uint8 - 127)
- vpshufb kernel: int8 scaled LUT [0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12], fold 0.5× into scale

## Competitors

1. **llama.cpp** — 4.30 tok/s, >100K LOC, MXFP4 GGUF native (**we beat it by 67%**)
2. **gpt-oss.java** (Amazon) — disqualified (garbage output, not benchmark-ready)

## Code Style

- Single .c + .h (no file splitting)
- Scalar reference + SIMD overlay via #ifdef / USE_SIMD
- OpenMP for parallelism
- mmap (Windows: CreateFileMapping/MapViewOfFile)
- Zero dynamic allocation during generation
- Always benchmark after each change: rebuild → 1 warmup + 5 measured runs
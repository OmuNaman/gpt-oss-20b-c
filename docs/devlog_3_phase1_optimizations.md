# Devlog 3: Phase 1–4 Optimization Results

**Date**: 2026-03-06
**Hardware**: i7-14700HX (8P+12E, 28 threads), 16 GB DDR5, Windows 11, NVMe SSD
**Build**: `gcc -O3 -march=native -fopenmp -o run_gptoss_native.exe run_gptoss.c -lm -lshell32`
**Benchmark**: `--prompt "Explain the halting problem in one paragraph." --max_tokens 256 --temp 0 --seed 42`

## Summary

Starting from 3.01 tok/s baseline, achieved **7.13–7.38 tok/s** through a series of optimizations. The dominant win was replacing `ldexpf()` with a precomputed LUT in the MXFP4 hot path (+128%). This revealed that the previous bottleneck was not memory bandwidth but a pathologically slow scalar math function called 90 times per row, 192 matmul calls per token.

## Results Table

| # | Optimization | tok/s | ms/tok | vs Prev | vs Baseline | Notes |
|---|-------------|-------|--------|---------|-------------|-------|
| 0 | **Baseline** (pre-optimization) | 3.01 | 332 | — | — | vpshufb MXFP4, -march=native |
| 1 | **E8M0 LUT** (replace ldexpf) | 6.88 | 145 | +128% | +128% | Single biggest win |
| 2 | Fused greedy argmax | — | — | — | — | **SKIPPED**: 784 KB negligible vs 3.6 GB model |
| 3 | Software prefetching | 6.87 | 146 | ~0% | +128% | Neutral — HW prefetcher already effective |
| 4 | **Serial router matmul** | 7.12 | 141 | +3.6% | +137% | Eliminates OpenMP overhead on 32-row gate GEMV |
| 5 | **Float16 KV cache** | 7.19 | 139 | +1% | +139% | 384→192 MB, SIMD attention reads with cvtph_ps |
| 6 | Fused RoPE + KV store | 7.14 | 140 | ~0% | +137% | Neutral — data was already in L1 |
| 7 | **P-core thread pinning** | 7.38 | 135 | +3.4% | +145% | SetProcessAffinityMask(0x0000FFFF) |
| 8 | --quiet benchmark mode | 7.13–7.19 | 140 | ~0% | +137% | UX feature, not a perf change |
| 9 | Duplicate print_metrics fix | — | — | — | — | Not needed — no actual duplication |
| 10 | Persistent KV cache (chat) | N/A | N/A | — | — | Chat-only speedup, no benchmark change |

**Final sustained**: **~7.2 tok/s** (135–140 ms/tok)

## Detailed Analysis

### 1. E8M0 LUT — The Monster Win (+128%)

The `ldexpf(1.0f, (int)scales[g] - 128)` call in the MXFP4 hot path was catastrophically expensive. This function computes 2^x, which on MinGW/GCC involves a full C library call with floating-point state management. It was called:
- 90 groups × 5760 rows (MLP1) = 518,400 times per expert per layer
- × 4 experts × 24 layers = ~49.8 million calls per token

Replacing it with a 256-entry float array lookup (`e8m0_half_lut[scales[g]]`) eliminated all this overhead. The LUT is initialized once at startup with the same `ldexpf()` values, so correctness is exact — output is bit-identical to baseline.

**Lesson**: Always profile scalar function calls in hot loops. A "simple" math function can dominate runtime when called millions of times per token.

### 2. Fused Greedy Argmax — No Gain (Skipped)

The fused unembedding+argmax was designed to skip materializing 201,088 logits (784 KB). In practice, this bandwidth saving is negligible compared to the 3.6 GB model weight reads per token. The OpenMP parallel reduction pattern also added overhead that offset any theoretical gain.

### 3. Software Prefetching — Neutral

Added `_mm_prefetch` 1 row ahead in f16_matmul, f16_matmul_bias, mxfp4_matmul_bias, and 2 groups ahead in mxfp4_dot_row. No measurable impact — the Raptor Lake hardware prefetcher already handles sequential access patterns well. Kept the prefetch hints as they're benign.

### 4. Serial Router Matmul — Small Win (+3.6%)

The router matmul is only 32 × 2880 = 92K multiply-adds. OpenMP fork/join overhead on 28 threads exceeds the compute for this size. Created `f16_matmul_bias_serial()` without the `#pragma omp parallel for`. This saves ~6,144 fork/join cycles per token (24 layers × 256 tokens).

### 5. Float16 KV Cache — Small Win (+1%, but huge memory reduction)

Changed `float*` KV cache to `uint16_t*`, halving from 384 MB to 192 MB. Added `f32_to_f16()` conversion helper. SIMD attention reads use `_mm256_cvtph_ps` to load 8 KV values at once.

The performance gain was modest (+1%) because at position 256, the KV cache read is only ~256 × 512 × 2 bytes = 256 KB per layer — small compared to the 3.6 GB model reads. The real win comes at longer sequences where attention reads dominate.

Output changes slightly due to f16 quantization of K/V values — expert utilization shifts by 2 counts on the top experts (1185→1183 for expert 24). Text remains coherent and topically identical.

### 6. Fused RoPE + KV Cache Store — Neutral

Fused the K-head RoPE application with the f32→f16 KV cache write (1 pass instead of 3). No measurable gain because K (512 floats = 2 KB) was already in L1 cache from the QKV projection, so the "extra passes" were hitting L1, not main memory.

### 7. P-Core Thread Pinning — Small Win (+3.4%)

Used `SetProcessAffinityMask(GetCurrentProcess(), 0x0000FFFF)` to restrict to the first 16 logical processors (8 P-cores with HT at 4.9+ GHz). This eliminates E-core stragglers (3.6 GHz) that cause the fastest threads to wait at OpenMP barriers.

Also set `HIGH_PRIORITY_CLASS` to reduce scheduling jitter.

### 8. --quiet Benchmark Mode

Buffers all generated text and prints after timing ends. Removes console I/O (`printf`/`fflush`) from the critical path. The performance impact was negligible (Windows console I/O is not as slow as expected with small outputs), but the flag is useful for clean benchmark automation.

### 10. Persistent KV Cache (Chat Mode)

Tracks `chat_pos` across turns. On subsequent turns, only new tokens (user message + assistant prefix) are processed through the model. Falls back to full replay if history truncation occurs. This doesn't affect single-prompt benchmarks but dramatically speeds up multi-turn chat by avoiding redundant prefill.

## vs llama.cpp

| Engine | tok/s | ms/tok | Status |
|--------|-------|--------|--------|
| **GPT-OSS C** (optimized) | **7.2** | **140** | **Winner** |
| llama.cpp | 4.30 | 232 | |
| GPT-OSS C (baseline) | 3.01 | 332 | Pre-optimization |

We now **exceed llama.cpp by 67%** on the same model and hardware. The key was that our baseline had a catastrophic `ldexpf()` bottleneck masking the true performance potential of the vpshufb MXFP4 kernel.

## Memory Usage

| Component | Before | After |
|-----------|--------|-------|
| KV cache | 384.0 MB | 192.0 MB |
| Model (mmap'd) | 12.82 GB | 12.82 GB |
| Activations | 1.9 MB | 1.9 MB |
| **Total RAM** | **385.9 MB** | **193.9 MB** |

## What's Next (Phase 5)

With `ldexpf()` gone, the engine is now truly memory-bandwidth-bound. The remaining optimizations must target:

1. **Speculative decoding** — amortize the 3.6 GB model read over multiple tokens
2. **MXFP4 weight repacking** — amortize input vector loads across multiple output rows
3. **Windows PrefetchVirtualMemory** — improve cold-start page fault pattern

The memory bandwidth ceiling at ~60 GB/s DDR5 practical throughput gives a theoretical maximum of ~16.7 tok/s (60 GB/s ÷ 3.6 GB per token). Current 7.2 tok/s = 43% of theoretical maximum — good, but there's room for speculative decoding to break past it.

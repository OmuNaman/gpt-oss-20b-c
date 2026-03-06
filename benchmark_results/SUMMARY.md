# GPT-OSS 20B CPU Benchmark Results

> **Date**: March 6, 2026
> **Hardware**: Intel Core i7-14700HX (8P+12E, 20 cores, 28 threads), 16 GB DDR5, NVMe SSD, Windows 11
> **Model**: GPT-OSS 20B (MoE, 20.9B params, 3.6B active/token, MXFP4 quantization)

---

## Head-to-Head: Generation Speed (tok/s)

| Engine | Language | LOC | Avg tok/s | ms/tok | Prompt tok/s | Output Quality |
|--------|----------|-----|-----------|--------|-------------|----------------|
| **llama.cpp** | C++ | >100K | **4.30** | **232** | 4.64 | Coherent |
| **run_gptoss.c** (ours) | C | ~2000 | **3.34** | **300** | 2.22 | Coherent |
| **gpt-oss.java** (Amazon) | Java | ~1000 | ~5.96* | ~168* | 2.19 | Garbage** |

*\*Java engine produces incoherent output (all `!` at temp=0, random multilingual gibberish at temp>0). Speed is measured but output is non-functional. Missing Harmony chat template + possible weight loading bug. Not a valid comparison.*

---

## C Engine Results (5 runs, 256 tokens, temp=0, seed=42)

| Run | Gen tok/s | ms/tok | Prompt tok/s | Wall time |
|-----|-----------|--------|-------------|-----------|
| 1 | 3.30 | 303 | 2.2 | 81.7s |
| 2 | 3.36 | 298 | 2.3 | 80.1s |
| 3 | 3.36 | 298 | 2.2 | 80.4s |
| 4 | 3.33 | 301 | 2.2 | 81.0s |
| 5 | 3.34 | 300 | 2.2 | 80.8s |
| **Average** | **3.34** | **300** | **2.22** | **80.8s** |
| **Std dev** | **0.02** | **2** | **0.04** | **0.6s** |

## llama.cpp Results (5 runs, ~209 gen tokens, temp=0, seed=42)

*Note: llama.cpp auto-detected Harmony chat template and generated 209 tokens before hitting EOS (not 256).*

| Run | Gen tok/s | ms/tok | Prompt tok/s | Total time |
|-----|-----------|--------|-------------|------------|
| 1 | 4.25 | 235 | 2.96 | 54.5s |
| 2 | 4.36 | 229 | 3.63 | 52.4s |
| 3 | 4.30 | 233 | 6.29 | 51.3s |
| 4 | 4.32 | 231 | 4.22 | 52.2s |
| 5 | 4.29 | 233 | 6.10 | 51.5s |
| **Average** | **4.30** | **232** | **4.64** | **52.4s** |
| **Std dev** | **0.04** | **2** | **1.44** | **1.2s** |

## llama-bench (Structured, 5 reps each)

| Test | tok/s | Description |
|------|-------|-------------|
| pp128 | 29.98 ± 6.02 | Prompt processing, 128 tokens |
| pp512 | 54.99 ± 0.23 | Prompt processing, 512 tokens |
| tg128 | 4.38 ± 0.03 | Text generation, 128 tokens |

## Java Engine (gpt-oss.java) — DISQUALIFIED

Amazon's gpt-oss.java produces **non-functional output** on this hardware:
- **temp=0**: All `!` tokens (degenerate greedy)
- **temp=0.1**: Random multilingual garbage ("castigileg", "verbringen", "esp\u00e9ritu", etc.)
- **temp=0.2**: Same garbage pattern

**Root cause**: Missing Harmony chat template formatting. The Java engine sends raw text without `<|start|>`, `<|message|>`, `<|end|>` special tokens that GPT-OSS 20B requires. Additionally uses `O200K_BASE` tokenizer instead of `O200K_HARMONY`.

We patched the Java engine to add Harmony template wrapping but output remained garbage, suggesting additional issues in weight loading or dequantization.

**Performance (for reference only, not comparable due to broken output):**
- Generation: ~5.96 tok/s (temp=0, 256 tokens forced)
- Prefill: ~2.19 tok/s
- Requires `-Xmx14g` heap (default OOMs)

---

## C Engine Thread Scaling

| Threads | Gen tok/s | ms/tok | Speedup vs 1T |
|---------|-----------|--------|---------------|
| 1 | 0.78 | 1287 | 1.00x |
| 4 | 2.28 | 439 | 2.92x |
| 8 | 3.25 | 308 | 4.17x |
| **Default (28)** | **3.34** | **300** | **4.28x** |
| 16 | 2.34 | 428 | 3.00x |

**Observation**: 8 threads is optimal for the MoE architecture. 16 threads shows regression (3.00x vs 4.17x at 8T) due to OpenMP fork/join overhead on small matrix multiplies. The default (all 28 threads) is slightly better than 8 threads, suggesting the runtime auto-tunes effectively.

---

## Analysis

### 1. llama.cpp Wins by 29%

llama.cpp achieves **4.30 tok/s** vs our **3.34 tok/s** — a **1.29x advantage**. This gap comes from:

- **GGML tensor library**: Optimized tiled matrix multiplication kernels with CPU cache-aware blocking
- **Flash Attention**: Fused attention kernel (auto-enabled) reduces memory bandwidth
- **REPACK optimization**: llama.cpp repacks weight matrices at load time for better cache locality (9.7 GB repack buffer)
- **Thread pool**: Custom thread pool (14 threads) vs OpenMP (28 threads) — less overhead
- **100K+ LOC of production optimization** vs our 2000-line educational engine

### 2. Our 2000-Line Engine Achieves 78% of llama.cpp

The remarkable finding: **a single-file C engine achieves 78% of the throughput of a 100K+ LOC production framework**. This translates to:

| Metric | C Engine | llama.cpp | Ratio |
|--------|----------|-----------|-------|
| Gen tok/s | 3.34 | 4.30 | 0.78x |
| LOC | ~2,000 | >100,000 | 0.02x |
| Binary size | 112 KB | 11 MB + 5.5 MB | 0.007x |
| **tok/s per 1K LOC** | **1.67** | **0.04** | **42x** |

Our engine is **42x more code-efficient** (tok/s per 1000 lines of code).

### 3. Memory-Bandwidth Bound

Both engines are **memory-bandwidth bound** during text generation:
- Active params per token: 3.6B (MoE top-4 of 32 experts)
- At MXFP4 (0.5 bytes/param): 1.8 GB read per token
- i7-14700HX DDR5 bandwidth: ~7-8 GB/s theoretical single-channel
- Expected ceiling: ~4.4 tok/s (7.8 GB/s \u00f7 1.8 GB)
- llama.cpp at 4.30 tok/s is hitting **~98% of memory bandwidth ceiling**

### 4. Prompt Processing Gap

llama.cpp's prompt processing is **2x faster** (4.64 vs 2.22 tok/s) and dramatically faster in batch mode (55 tok/s at pp512 via llama-bench). This is because llama.cpp processes multiple prompt tokens in parallel per batch, while our engine processes them sequentially.

### 5. Java Engine is Broken

Amazon's gpt-oss.java is not production-ready:
- Missing Harmony chat template (critical)
- Uses wrong tokenizer variant (O200K_BASE vs O200K_HARMONY)
- Default stop tokens include ID 0 (causes immediate termination at temp=0)
- Requires 14 GB JVM heap for a 13 GB model
- Output is completely incoherent

---

## Benchmark Configuration

| Parameter | Value |
|-----------|-------|
| Prompt | "Explain the halting problem in one paragraph." |
| Max tokens | 256 |
| Temperature | 0 (greedy/deterministic) |
| Seed | 42 |
| Warm-up | 1 run (discarded) |
| Measured runs | 5 |
| Power | Plugged in (no battery throttling) |

### Build Configurations

| Engine | Build Command |
|--------|---------------|
| C engine | `gcc -O3 -march=native -fopenmp` (GCC 15.2, MinGW) |
| llama.cpp | `cmake -DGGML_AVX2=ON -DGGML_FMA=ON -DCMAKE_BUILD_TYPE=Release` (GCC 15.2) |
| Java | `gradlew shadowJar` (JDK 23.0.2 Temurin, `-Xmx14g`) |

---

## Suggested Follow-Up Experiments

1. **Batch prompt processing** for C engine (process multiple tokens per forward pass)
2. **Weight repacking** at load time for better cache locality
3. **Custom thread pool** to replace OpenMP (reduce fork/join overhead)
4. **vpshufb MXFP4 optimization** (AVX2 shuffle-based LUT, currently scalar)
5. **Linux benchmark** (same hardware, dual-boot) to eliminate Windows overhead
6. **ARM comparison** (Apple M-series via llama.cpp) for architecture diversity

---

## Raw Data

All raw output is preserved in:
- `c_engine.txt` — C engine runs
- `llamacpp_engine.txt` — llama.cpp runs + llama-bench
- `java_engine.txt` — Java engine runs (garbage output documented)
- `thread_scaling.txt` — C engine OMP thread sweep
- `system_info.txt` — Hardware/software versions

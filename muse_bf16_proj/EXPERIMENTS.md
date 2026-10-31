# muse_bf16_proj experiment log

Times are µs per call on the H200 pod (`okc-musespark-kernels-h200`) with `timing.py`: CUDA graphs,
cold L2, launch overhead subtracted. They track the judge within ~5%. "x" is SGLang ÷ mine.

## Reference points

| weight | size | ideal floor | pure-read kernel (bwprobe.py) | SGLang m=1 |
|---|---|---|---|---|
| unpooler slice / router | 8.4 MB | 1.75 | 3.6 | 7.2 / 10.5 |
| o | 16.8 MB | 3.5 | 5.4 | 6.7 |
| qkv_gate | 37.7 MB | 7.9 | 10.1 | 13.7 |
| pooler / unpooler | 67 MB | 14.0 | 18.7 | 21.4 |

Pure-read best configs use ~2000 short CTAs. The pure-read time is the practical ceiling for m ≤ 32.

## Log

### E0 — Triton split-K skinny GEMM v1 (2026-10-25)
- W as dot's left operand (swap-AB), BM = 16/32 token pad, split-K partials to an FP32 workspace,
  last CTA per tile sums them (counter, self-resetting). Router = BF16 hi + lo, FP32 accumulate.
  m > 32 → cuBLAS (same as SGLang).
- `okc check` job dd8d2d8273f14c13: **9/9 tests, 1.058×** (95%: 1.043–1.074). Off the board.
- Wins: router at m ≤ 32 (1.4–3.7×). Losses: m = 32 all layers (0.55–0.88×), `o` and sliced unpooler (~0.9×).

### E1 — mask workspace rows to m (2026-10-25)
- Split-K partials were written/read for all 16 padded rows even at m = 1. Masked to rows < m.
- Checks: 0 failures on pod.

### E2 — tile config sweep (sweep.py, 2026-10-25)
- Grid: BN 16–128, BK 64–256, 264–4224 CTAs, stages 3/4. Results in sweep_results.txt.
- Best per shape vs the v1 default: small gains (o 7.7→6.6, sliced unpooler 7.2→6.2, m=32 qkv
  26.9→15.7). More CTAs did *not* help: the split-K fixup (store → atomic → last CTA re-reads)
  adds ~2–3 µs of serial L2 round trips at the end of the kernel.
- qkv m=1 stuck at 13.1 vs 10.1 pure read; router at 6.8 vs 3.6.

### E3 — CUDA kernel, cluster split-K (submission_cuda.py, 2026-10-25)
- load_inline CUDA. Each warp: 16·RT rows × 32·NSTEPS k, all 16-byte loads issued up front, then
  m16n8k16 BF16 MMAs (W = A operand, tokens = the 8-wide N side). Same k-permutation for A and B
  inside each 32-wide block, so 16-byte loads map straight onto MMA fragments.
- Split-K across the S CTAs of a thread-block cluster, summed through distributed shared memory
  (no workspace, atomics or counters).
- E3a, row-major W: correct, but geomean 0.83× (small-m loads). Sliced unpooler 5.7 (best so far),
  big weights slow (qkv 18, pooler 27).
- E3b, W repacked in prepare() so every warp's loads are one contiguous run (lossless permutation):
  geomean 0.92×. Sliced unpooler 5.3, router 6.5.
- E3c, config sweep: best configs are ~one wave of CTAs with more work each. qkv m=1 12.1
  (SGLang 14.2, Triton 13.1); pooler still 21 (Triton 19.1). Many short CTAs lose: each wave ends
  with cluster syncs while no loads are in flight.

### E4 — chunk loop per warp (CH), 2026-10-25
- Each warp runs CH chunks of NSTEPS steps in a loop, so one wave of CTAs can cover all of W
  without more registers. Other resident warps hide each chunk's load latency.
- Checks: 0 failures. Best per shape (µs, CUDA / Triton E2 / SGLang): qkv m=1 12.1/13.1/14.2,
  pooler m=1 19.0/19.1/21.4, sliced unpooler 5.5/6.2/7.2, router m=1 6.4/6.8/10.6. `o` and the
  whole unpooler stay best in cuBLAS.

### E5 — hybrid: CUDA kernel where it wins, cuBLAS elsewhere (submission.py, 2026-10-25)
- Config table `CONFIGS[(layer, MT)]`; C++ dispatch generated from it (10 instantiations, ~1 min
  compile). Triton v1 kept as submission_triton.py.
- Pod: 1.152×, 0 check failures. `okc check` 12e78e707e334d8a: 1.143× (1.126–1.160).
- **Submitted: job ef865f5c5d0d4eea, 1.154× (1.136–1.171), rank 743 / 1001.** Leader 1.610×.
- Remaining gaps: m = 128 / 256 loads at ~1.0× (cuBLAS); m = 16 / 32 at ~1.0–1.1×; small-m
  shapes still 1.2–1.8× above the pure-read time.

### E6 — split-K Triton GEMM for m = 128 / 256 (sweep_gemm.py)
- BF16 layers: always slower than cuBLAS (e.g. qkv m=256 31 vs 20 µs, pooler 37.7 vs 32.3).
  Split-K configs lost to S = 1. cuBLAS is good here; beating it needs a TMA / warp-specialized
  kernel. Parked.
- Router (hi + lo in one kernel): **12.3 vs 17.9 µs at m=128, 13.0 vs 19.2 at m=256** — SGLang
  runs two GEMMs + an add. Best (BM 64, BN 64, BK 128, S 8). Wider splits (16–64) did not help.

### E7 — DeepGEMM bf16_gemm_nt (probe_deepgemm.py)
- Ties cuBLAS on every BF16 shape (±1%); router via two calls with c=out ~7–11% faster than
  SGLang's, but slower than E6. Not used.

### E8 — router split-K Triton GEMM above 32 tokens (submission.py)
- Extra correctness: router m ∈ {33, 64, 100, 128, 200, 256, 300, 777}, normal + outliers: all pass
  (rel. L2 ~3.4e-6, tolerance 1e-4).
- Pod: 1.173×. **Submitted: job 4ed4a00394264da9, 1.175× (1.158–1.193).**

### Judge breakdown of E8 (job 4ed4a00394264da9), the starting point on 2026-10-29
- Loads: decode 1 1.21×, decode 16 1.18×, decode 32 1.12×, verify 1 (m 8) 1.48×, verify 16 (m 128) 1.05×,
  verify 32 (m 256) 1.06×. Every BF16 shape at m = 128 / 256 is cuBLAS (0.98–1.01×); the router there is the
  E6 Triton kernel (1.41–1.45×). Board 2026-10-29: 1342 entries, leader 1.637×, median 1.230×, us 1.175× (rank 951).

### E9 — cuBLASLt algorithm search (ltprobe.py), 2026-10-29
- Every candidate of cuBLASLt's heuristic (it returns 8 for these shapes) timed like the judge, per (layer, m) at
  m = 128 / 256: best vs SGLang's call 0.98–1.02×, pooler m = 128 1.07× (25.0 vs 26.7 µs). cuBLAS is already at its
  own best; nothing to take. (nvcc's front end crashed on std::map<std::tuple> in host code: build host-only
  cuBLASLt code as a C++ source, `load_inline(..., with_cuda=True)`, not as a .cu.)

### E10 — split-K TMA + wgmma GEMM for m = 32–256 (wgemm_probe.py), 2026-10-29 (not used)
- Producer warp (TMA, 128 B swizzle, W repacked tile-major so a 128 x 64 box is one 16 KB run) + 2 consumer
  warpgroups (wgmma m64nTOKk16, W as A, tokens as N, both from shared memory), split-K partials reduced through a
  cluster's DSMEM, or (no cluster) through a global workspace with a per-tile spin barrier and each CTA reducing
  1/S. Correct at the first compile (rel. err 2e-3).
- Best m = 128 times vs cuBLAS: pooler 26.6 / 26.7, qkv_gate 17.7 / 18.1, o 10.9 / 8.1, unpooler 23.0 / 20.6.
- Why: 384 threads x 154 registers = 1 CTA per SM; clusters of >= 3 fit only 120 CTAs at once (2 waves at
  S = 4); the last-CTA global reduction is serial at ~70 GB/s per SM (o 10.9 -> 17.6 µs); and even copying W alone
  (no x, no MMA, 128 CTAs) takes 24.7 µs for pooler (2.7 TB/s): one TMA ring per SM keeps ~80–190 KB in flight
  against ~2–3 µs of loaded latency. Beating cuBLAS on these shapes needs several CTAs (or rings) per SM and less
  x / partial traffic (x re-reads per 128-row tile equal W's bytes at m = 128) — not tuning.

### E11 — non-blocking split-K epilogue in the skinny kernel (not used)
- Measured first: with the cluster reduction removed (wrong output), unpooler m=1 5.7 -> 4.1 µs, pooler m=1 19.3
  -> 18.2: the cluster.sync tail is ~1–1.6 µs on the small weights (plain read of 8.4 MB: 3.7 µs).
- Replacing it by red.global.add into a per-weight FP32 accumulator + last-CTA write-out and re-zero (no clusters):
  0 check failures but slower everywhere (m = 32 pooler 27.6 vs 22.8, router 9.4 vs 7.9; small-m geomean 1.138 vs
  E8's 1.227). Reverted, not kept.

### E12 — lossless 12-bit weights (submission_c12.py; correct, not used), 2026-10-29
- Idea: at <= 32 tokens the time is weight bytes / bandwidth, and the rules allow lossless compression. Gaussian BF16
  weights use a narrow exponent range: keep each weight's low byte (exponent LSB + mantissa) and code the high byte
  (sign + exponent >> 1) in 4 bits relative to the tensor's maximum (16 binades): 12 bits, 25 % fewer bytes. Decode
  = 2 PRMT + ~5 integer ops per BF16 pair into the MMA A fragments. Out-of-window weights: 1.0-1.9e-4 of them
  (router's lo plane 1.8e-3) decode to the window's bottom and a per-CTA exception list (row, k, true and decoded
  BF16) corrects them in FP32 (x * true - x * decoded) before the cluster reduce. Probe columns bit-exact.
- Timing (pod, µs, E8 at its configs -> 12-bit with no exception handling -> 12-bit exact):
  m=1 qkv 12.6 -> 10.8 -> 11.6, pooler 19.3 -> 16.8 -> 17.6, unpooler slice 5.5 -> 5.2 -> 6.2, router 6.5 -> 6.0 -> 7.4;
  m=8 qkv 13.0 -> 11.4 -> 13.2, pooler 19.6 -> 17.2 -> 18.8; at MT >= 2 (16-32 tokens) 12-bit is slower (decode).
  The exception tail (dependent loads at the end of each CTA; prefetching them blew pooler up to 25 µs) eats most of
  the gain: per-shape best ~+1 % on the geomean, inside the judge's noise. Not submitted.
- Free finding: at MT = 1 the BF16 config (1, 16, 1, 4) beats E8's (2, 8, 1, 8) for qkv (12.0 vs 12.7 µs at m=1).
### E13 — 13-bit (one more code bit, 32 binades, no exceptions): dead
- Slower than 12-bit (qkv m=1 13.4, pooler 20.6: the extra 2-byte loads + ops), and still not lossless on these
  weights (a few qkv / pooler weights below 32 binades), so every layer would fall back.

### E14 — tested with `okc check` only (no pod), 2026-10-31
- Configs (two checks): qkv at MT = 1 as (1, 16, 1, 4) (E12's finding): m = 1 12.25 -> ~11.7–12.0 µs, kept. The skinny
  kernel for `o` (2, 8, 1, 1) / (1, 8, 1, 1) / (2, 4, 1, 2) and for the whole unpooler at m = 16 / 32 is far slower
  than cuBLAS (o m = 16: 10.2 vs 6.5 µs; unpooler m = 32: 27.2 vs 18.5): cuBLAS stays.
- Cluster tail pushed: each CTA stores its partial of element i straight into its reducer's shared memory
  (st.shared::cluster), one cluster barrier, then local sums (no remote reads, no second barrier). For every MT:
  1.1874×; MT = 1 only (the old pull above): 1.1790× — both inside the ±1.5 % interval of each other; small-m
  shapes gain 0.2–0.9 µs (unpooler m = 8 5.7 -> 4.8–5.2, router m = 8 6.8 -> 6.0–6.1). Kept: push at MT = 1.
- **Submitted: job fef2017d39ec4cc2, 1.190× (1.173–1.208)** (from 1.175×). submission_e14.py = this.

## Next
- The small-m loads (4 of 6) are where the geomean moves: the tail (~1–1.6 µs per small weight) and `o` (cuBLAS
  6.8 µs vs 5.5 plain read). Ideas not yet tried: a cheaper cluster tail (one cluster.sync + barrier.cluster.arrive
  release / wait split so loads of the next tile overlap; S = 2–4 clusters with more row tiles), and the skinny
  kernel for `o` with a config sweep under the current tail.
- m = 128 / 256 needs a multi-ring or multicast design (see E10); the router there (Triton, 11.5 / 12.0 µs vs floor
  ~2.5) is the cheapest target but its split-K reduction volume (S x M x 256 x 4 B) dominates at large S.

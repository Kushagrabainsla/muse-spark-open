# muse_moe_w4a16 experiment log

Times are µs per call on the H200 pod (`okc-musespark-kernels-h200`) with `timing.py` (CUDA graphs,
cold L2, launch overhead subtracted), for `bench.py`'s seeds. "floor~" is reference.py's bound for
that seed's routing (75% of its routes and touched experts).

## Baseline (SGLang Marlin MoE), 2026-10-25

| bs | local routes | experts touched | SGLang µs | floor~ |
|---|---|---|---|---|
| 1 | 3 | 3 | 87.8 | 13.3 |
| 8 | 13 | 8 | 147.7 | 35.4 |
| 16 | 23 | 6 | 142.2 | 26.6 |
| 32 | 48 | 12 | 162.2 | 53.2 |
| 128 | 176 | 24 | 477.9 | 106.6 |
| 256 | 425 | 24 | 648.6 | 107.0 |

- Each expert is 3 × 4096 × 4096 NVFP4 values = 25.2 MB packed + 3.1 MB scales. Reading only the
  touched experts at ~3.7 TB/s (what the bf16 problem's pure-read probe reached) is 2.5–3.5×
  faster than SGLang on every load.
- Leaderboard: leader 2.523×, median 1.532× (784 entries).

## Log

### M1 — fused NVFP4 grouped GEMMs, CUDA (submission.py)
- K0 routing: one CTA, a warp per local expert ballots over the (token, slot) pairs → per-expert
  pair lists ordered by pair id, plus the active-expert list.
- K1 gate/up and K2 down: persistent thread-block clusters loop over (active expert, row tile);
  the S CTAs of a cluster split K and sum through distributed shared memory. Weights repacked in
  prepare() (lossless nibble/byte permutation) so every warp load is a contiguous 512 B and each
  16-byte load feeds eight m16n8k16 MMAs. E2M1 → BF16 by bit placement (exponent bits e, mantissa
  m: exact up to a uniform 2^-126, the code-1 value 0.5 becomes a BF16 subnormal); x is scaled by
  2^100 in registers (exact) so products stay normal in FP32. Each MMA covers one 16-wide scale
  block, so the E4M3 block scale is applied to its FP32 result. SiLU·up fused into K1's epilogue.
- K3: per token, the post-expert RMSNorm of each local slot, weighted and summed in slot order.
- M1 result (cluster split-K, 4 warps): correct (rel. L2 3e-3, tol 1.5e-2; BF16-subnormal E2M1 trick
  works on the tensor cores), but 0.42×. Each cluster ran ~6 tiny tiles serially with two cluster
  syncs each, and x was re-read from L2 per tile (3.5× the weight traffic at bs=32).

### M2 — x staged in shared memory, next-unit prefetch, 8 warps, one cluster sync per tile
- Correct, 0.36×. Occupancy-sized grid had only 30–124 clusters; MT=4/8 variants at 255 registers.
  Per-tile cluster reduction latency still dominates.

### M3 — no clusters: FP32 atomic split-K into persistent zeroed accumulators (current submission.py)
- 256-row CTA tiles (warps split rows), S=4, x[tokens, K/S] staged once per (rank, expert) run;
  partials via red.global.add.f32 into state["h1"] / state["y"], which norm_kernel re-zeroes after use.
  silu(gate)·up applied when gemm2 stages its input (rounded to BF16, as SGLang).
- 0.674× overall; bs=1 1.19×, bs=8 1.13×, bs=16 1.06×, bs=32 0.56×, bs=128 0.83×, bs=256 0.14×.

### M4 — cheaper decode: bit-permuted E2M1 words (4 BF16 pairs from 3 shifts + masks), E4M3→FP32 by
  bfe + imad (prepare() checks all scales are positive normal, else the exact hardware path)
- Correct, 0.709×: bs=1 1.29×, bs=8 1.24×, bs=16 1.15×, bs=32 0.57×, bs=128 0.86×, bs=256 0.14×.
- nsys at bs=1: gemm1 35.6 µs, gemm2 19.8 µs, norm 8.5 µs (!), route 1.6 µs. Targets: ~14 / ~7 / <2.

### M5 — norm kernel: all slots at once, zeroing spread over 4 extra CTAs per token (2026-10-26)
- norm 8.5 → 4.3 µs at bs=1. 0.728× overall.
- Pod note: the first pod's host filled up while stopped; replaced by `okc-musespark-kernels-h200-b`
  (cvlfne3nu4np4z), old one terminated. ncu is blocked on RunPod (ERR_NVGPUCTRPERM), so the rest
  was diagnosed with a debug switch (skip math / skip atomics) and nsys kernel times.

### M6 — 3-stage cp.async weight ring per warp; RT / split sweeps
- No gain by itself (bytes in flight weren't the limit). RT 1/2/4 and splits 2–16 within ±10%.
- Debug breakdown, bs=1: gate/up 37.8 µs full vs 21.4 loads-only; bs=32: 186.7 vs 77.0. So the
  math was not hidden behind the loads, and loads alone ran at only ~2.5 TB/s.

### M7 — block scale applied to the weights in BF16 (hmul2), MMAs accumulate directly; x unscaled
- Removes 4 FMAs per MMA (cost no longer grows with the token tile). E2M1 × E4M3 × 2^100 is exact
  in BF16 and stays normal. 0.825× overall.

### M8 — division-free unit iterator, token group outermost within a (rank, expert) run
- Per-step integer divisions removed; x staged once per run and group (was once per tile when an
  expert had > 32 tokens). **1.043×** overall (bs=1 1.76×, bs=128 1.28×), bs=256 still 0.29×.

### M9 — units count token groups (route kernel publishes a groups prefix per expert)
- bs=256 has one hot expert with 204 tokens (7 groups); balancing by units ignored that, so ~11
  CTAs ran 7× longer. Now each unit is one tile of one token group. bs=256 2236 → 532 µs.
  1.299× overall.
- Sweep (fewer splits = fewer atomics) showed atomics were not the bottleneck: split 8 is best.
- Token tile sweep: MT=1 (8 tokens) up to bs=16, MT=2 (16) above. Pod: **1.432×**.
- `okc check` 8ed31c368da34f94: 1.371× (1.351–1.396), 5/5 tests.
- **Submitted: job 8770b04e6afa4157, 1.367× (1.347–1.388).** Leader 2.523×, median 1.532×.

### M10 — programmatic dependent launch (route → gemm1 → gemm2 → norm), 2026-10-26
- Each kernel launches early; gemm1 waits for route, gemm2 prefetches its first weight stages
  (it needs only the routing) before waiting for gemm1, norm waits for gemm2. Removes ~1–2 µs gaps
  and fills gemm1's tail with gemm2. Instantiations trimmed to MT 1/2/4 × RT 2.
- Pod: 1.497× (bs=1 1.89×, bs=8 1.55×, bs=16 1.61×, bs=32 1.11×, bs=128 1.55×, bs=256 1.39×).
- Pipeline depth: STG=2 helps bs=1 (43 µs, more CTAs fit) but costs bs=8/16 ~4 µs; 4 no better.
  Kept 3.
- `okc check` 561fb8dad29843aa: 1.419× (1.398–1.440).
- **Submitted: job 9e56c692773140b1, 1.424×, rank 463 / 796.** Overall kernel board rank 626.

### M11 — compact scale ring (one 64 B copy per slot, q = 0 lanes + __syncwarp), incremental slot address
- 3 CTAs/SM now fit, but no gain: pod 1.478× (M10 1.497×). Kept (less smem). Saved as submission_m11.py.
- SASS + counts: the gemm is instruction-issue bound (~2.8 warp-instr/clk/SM): ~520 instr per 2 KB
  step, of which ~150 E2M1 decode, ~60 scale building, 64 HMUL2, 16 HMMA, ~200 iterator/issue/address.
  (`mkdbg.py` variants: math-only ≈ loads-only ≈ 70–75 µs at bs=32; loads-only ≈ 3.0 TB/s.)

### M12 — mixed-block BF16 pairs, 16-byte x reads, compile-time KC (2026-10-26, pod okc-musespark-kernels-h200-c)
- Each BF16 pair holds position 4q+i of blocks 2w and 2w+1 (bit map: output = r, half = b), so one
  scale register (the two blocks' scales) serves a whole word: half the scale building.
- x staged interleaved to match (per (w, q): x[16(2w)+4q+i], x[16(2w+1)+4q+i]), so one 16-byte
  shared read feeds both MMAs of a word (half the x reads). KC = 512 a template constant, XS = KC + 32.
- Pod: **1.602×**, 0 check failures (bs=1 2.15×, bs=8 1.62×, bs=16 1.69×, bs=32 1.18×, bs=128 1.66×,
  bs=256 1.47×).

### M13 — two 128-wide steps per loop iteration (SUB = 2), ring of STG = 2 two-step stages
- Halves the iterator / wait / issue overhead per weight byte. STG = 3 with SUB = 2 is worse (the
  bigger ring costs occupancy): 1.492×.
- Pod: **1.632×**, 0 check failures (bs=1 2.09×, bs=8 1.72×, bs=16 1.79×, bs=32 1.20×, bs=128 1.67×,
  bs=256 1.47×).
- `okc check` 41b7dab4282141cb: 1.558× (1.534–1.581), 5/5 tests.
- **Submitted: job 61b7d5bdbc144f5c, 1.555× (1.531–1.578), rank 457 / 847.** Overall kernel board 593.
  Leader 2.576×.

### Decode layout search (no code change)
- Searched E2M1 → BF16 bit layouts: no rotation / funnel-shift tiling of 32, 64 or 128 bits lets every
  output be one shift + mask (each output needs a contiguous 3-bit field), and no layout gives the 4th
  output in 4 ops. ~11 ops per 8 values is the floor for this encoding.

### M14 — split-K 16 for both GEMMs up to 16 tokens (KC = 256 kernel instance), 8 above
- At bs=1 the down GEMM had 384 units for 264 CTAs (1–2 each). Pod: 1.648×, 0 check failures
  (bs=1 2.14×, bs=8 1.77×, bs=16 1.81×, bs=32 1.19×, bs=128 1.67×, bs=256 1.48×). +1% over M13, not
  submitted alone. Saved as submission_m14.py (= current submission.py).

### M15 — routing fused into the GEMMs (reverted)
- Every CTA counted tokens per expert (shared atomics) and built its run's pairs with a warp ballot over
  the ids. Pod 1.469×: the ballot scan is a serial chain of dependent global loads (~65 iterations at
  bs=256) on every CTA's critical path. Reverted to the route kernel.

### M14 submitted (2026-10-29)
- **Job 1de976a74db64bfc, 1.574× (1.551–1.598), rank 590** (board was M13 at 1.555×). Pod 1.654× on this run.

### M16 — one 16-byte scale copy per lane, x fragments loaded once per word (2026-10-29)
- The SUB·RT·2 = 8 slots of a stage have contiguous 64 B scale blocks, so the stage's scales are one
  `cp.async.cg 16` per lane instead of eight predicated 8-byte copies (q = 0 lanes). The x fragments of a word
  (uint4 per token sub-tile) are loaded before the row-tile loop instead of inside it.
- Pod **1.730×** (M14 1.654×), 0 check failures: bs=1 2.21×, bs=8 1.83×, bs=16 1.87×, bs=32 1.26×, bs=128 1.79×,
  bs=256 1.59×. **Submitted: job de1b1c2076e04c41, 1.642× (1.617–1.666), rank 565** (from 590). Overall board 656.
- Rejected on this kernel (sweeps, same pod): MT = 4 / 8 token tiles (slower at every bs, even with a per-run
  templated "live sub-tiles" path that removes the padding work: bs32 ties, bs128/256 lose 8–12 %),
  SUB = 4 (two steps per iteration stays best), `__launch_bounds__(256, 3)` (no change, a few spills),
  split-K 16 above 16 tokens. Registers: MT=2 108–112 → 2 CTAs/SM; MT=1 78 → 3.
- Why bigger token tiles lose: `mma.sync` m16n8k16 BF16 runs at half the tensor rate on Hopper; at MT = 2 the
  32 HMMAs per 2 KB step already take ~65 % of the step's issue time per SM sub-partition, and MT = 4 makes the
  step tensor-bound. The kernel is now bound by the tensor pipe plus the decode issue, not by decode alone.

### M17 — wgmma inner loop (submission_wgmma.py; correct, not faster), 2026-10-29
- A = the decoded BF16 weights from registers (a warp's mma.m16n8k16 A fragments are its 16-row slice of the
  warpgroup's m64 tile; rows repacked at prepare time as 64 (2 rt + warp / 4) + 16 (warp % 4) + 8 h + g), B = x
  from shared memory by descriptor (no swizzle, core matrices [step][word][k-step][k-half][token][16 B]; LBO = TK·16
  between the two k-halves, SBO = 128 between 8-token groups) with the kernel's permuted k order, m64n(8·MT)k16,
  FP32 D = the same acc[rt][mt][4] registers. Two fragment buffers, one commit per word, wait_group 1 before
  reuse; fence.proxy.async after the x staging. Correct at the first compile (0 failures).
- Pod 1.640× (M16 1.730×): bs=1 42 vs 40 µs, bs=32 135 vs 129, bs=256 426 vs 409. Registers fall 112 → 88 (MT=2,
  KC=512) but still 2 CTAs/SM. No help from: MT = 4 (408 µs at bs=256, worse elsewhere), one wgmma group per two
  words (half the fences/commits/waits: 1.639×), `__launch_bounds__(256, 3)` (1.601×).
- Conclusion: the inner loop is bound by the E2M1 decode + scale issue, not by the mma.sync tensor rate (the M16
  reading of the MT = 4 result was wrong). wgmma only helps once the decode is cheaper or the MMA count per
  decoded word rises without padding waste.

### M18 — where the small-batch time goes (H100 SXM: no H200 in stock on 2026-10-29 evening), 2026-10-29
- Debug variants (mkdbg2.py: dbg & 1 no decode / MMA, & 8 no weight loads, & 2 no atomics, & 16 / 32 / 64 skip the
  norm / gemm2 / gemm1 launch). H100 µs, M16 (full / no math / no loads / neither):
  bs=1 45.3 / 42.3 / 33.5 / 18.0; bs=8 104.3 / 93.3 / 68.0 / 26.8; bs=32 159.1 / 143.3 / 113.8 / 41.3.
  The "neither" skeleton at bs=1 splits into route 2.6, gemm1 4.8, gemm2 6.6, norm 4.2 µs; atomics are 2.0 (bs=1)
  to 13.3 µs (bs=32) of it. On the H100 the full kernel is close to the loads-only one (memory-bound there).
- Trap: skipping the atomics also lets the compiler delete the decode + MMAs feeding them, so "no atomics" looked
  like a 9-24 % gain; it is really "no math + no atomics". Measure atomics only in variants that keep the math.
- Norm kernel's exposed cost in the full run (launch skipped): 2.5 (bs=1), 3.4 (bs=8/16), 3.0 (bs=32), 6.9 (bs=128),
  11.3 µs (bs=256): folding it into gemm2 is worth at most ~+3 % on the geomean.

### M19 — 16-byte vector atomics for the split-K partials (correct; dead on the H200, see M21)
- Lane (g, q) holds rows g, g + 8 x tokens 2q, 2q + 1; the 4 lanes sharing (g / 4, q) transpose their 4 x 4 values with
  4 rotation shuffles, so each adds 4 consecutive rows of one token with one red.global.add.v4.f32 (4x fewer atomics).
- H100: geomean 1.431 -> 1.453 (+1.5 %), 0 check failures: bs=1 45.4 -> 46.6, bs=8 104.5 -> 102.9, bs=16 93.9 -> 91.8,
  bs=32 159.4 -> 155.6, bs=128 354.3 -> 341.8, bs=256 498.8 -> 485.6. Below the 3 % submit bar alone.

### M20 — gemm1 prefetching gemm2's weights into L2 (dead)
- gemm1 bulk-prefetches the active experts' w2 weights + scales (cp.async.bulk.prefetch.L2, evict_last) and streams
  its own evict_first, so gemm2 would read L2. H100: slower at every bs (bs=1 45.4 -> 50.5, geomean 1.431 -> 1.322):
  gemm1 already uses the HBM bandwidth, and evict_first breaks the per-token-group weight reuse at large bs.

### M21 — the H100 results do not carry over: v4 atomics and a fused norm are slower on an H200, 2026-10-29
- Norm fused into gemm2: after its last unit every CTA of the resident grid meets at a growing counter, then token t
  is normed by CTA t mod grid (two passes over y) and h1 is re-zeroed across the grid; norm kernel dropped. Correct.
- H200 (judge GPU), bench.py geomean: M16 1.735 (bs=1 40.0, bs=8 80.5, bs=16 76.4, bs=32 128.7, bs=128 266.1,
  bs=256 407.3 µs); + vector atomics (M19) 1.596 (44.5 / 90.0 / 83.3 / 137.5 / 285.9 / 432.6); + fused norm 1.507
  (48.5 / 95.1 / 87.6 / 143.7 / 302.4 / 450.3). On the H200 the kernel is issue-bound, so M19's shuffles + selects cost
  more than the atomics they save, and the grid barrier + serial norm tail cost more than the norm kernel's launch.
- Lesson: an H100 is not a proxy for this kernel (memory-bound there, issue-bound on the H200). Test on an H200 or
  with `okc check`.

### M22 — routing inside the gemm CTAs; norm reads its inputs before waiting (2026-10-29, pod okc-musespark-moe5)
- Diagnosis on the H200 (mkdbg2.py, µs at bs 1 / 8 / 16 / 32 / 128 / 256): full 39.7 / 80.4 / 76.0 / 128.4 / 265.5 /
  407.6; no math 35.6 / 75.0 / 70.6 / 122.4 / 253.9 / 366.4; no weight loads 33.7 / 69.4 / 64.8 / 113.4 / 235.0 /
  357.4; neither 17.9 / 26.2 / 31.6 / 40.0 / 85.2 / 159.9. Loads and math overlap well; the fixed "skeleton" is
  31–45 % of the time. Skeleton split: atomics 1.7–77 µs, norm 4–13 µs, route kernel 2.6 µs (bs 1) to 13 µs (bs 256:
  one CTA, a warp per expert scanning all 2048 ids from global).
- Inner-loop SASS (moe_gemm<1,2,0,false,256>, per 8-value word): decode 11 ops (6 LOP3, 3 IMAD.SHL, 2 SHF), scale
  2, HMUL2 4, HMMA 1, LDS ~0.8; ~150 iterator / issue / epilogue ops per 32 words. Timing-only probes: a decode
  4 ALU ops cheaper (o3 = (q * 2) & M) is only +1.5 %; dropping the per-unit atomics +3.5 % (both upper bounds).
  No cheaper exact bit layout exists with shifts / rotations only (searched: E2M1 -> BF16 {6,7,8,15} does not
  tile Z16; best found is 9 ops vs 11, via 4 rotations + 5 masks or 3 rotations + a carry-free multiply).
- Change: the route kernel is gone. Every gemm1 CTA routes in shared memory from ids (one token per thread, a
  warp ballot per local expert present: per-expert counts, offsets, active list, token groups, pairs in pair order,
  identical in every CTA); CTA 0 publishes it to a global buffer before its griddepcontrol.launch_dependents, and
  gemm2 copies it (L2 reads) instead of routing. A fully unrolled routing (all 32 experts, every warp) cost
  ~1,700 instructions per warp and made bs 1 slower (47.7 µs); looping only over present experts fixed it. The
  norm kernel reads ids / topk weights / norm weight before its griddepcontrol.wait.
- Pod (H200): **1.797×** (M16 1.736×): 37.7 / 79.3 / 74.3 / 125.5 / 255.2 / 389.8 µs. 0 check failures; a graph-replay
  stress (stress.py: 40 new inputs x 2 calls per graph, bs 1 / 8 / 16 skewed / 32 / 128 skewed / 257) 0 failures.
- **Submitted: job 069f543bb3604714, 1.704× (1.678–1.729), rank 581** (from 1.642×). Local submission.py = M22;
  submission_m16.py = M16. Intermediates: submission_r.py (routing in both gemms), submission_r2.py (gemm1 publishes).

## Next
00. (M22) The remaining skeleton: split-K atomics (up to +3.5 % if free; v4 atomics lost on the H200), the norm
   kernel (4–13 µs), x staging + per-run syncs. A 9-op decode layout (vs 11) is worth ~+0.7 % at most.
0. (2026-10-29) Vector atomics (M19), a fused norm (M21) and an L2 prefetch (M20) all lose on the H200, where the
   kernel is issue-bound: an idea must remove instructions from the inner loop or the epilogue, not add them.
1. Decode cost: ~10 ops per 8 values + 4 HMUL2 for the block scale + the scale build; the PRMT table decode costs
   16 ops. Reconsider the encoding: a prepare-time layout where the scale multiply is folded into the decode (e.g.
   store each weight nibble pre-multiplied is not allowed — the weights must stay W4), or decode two words with
   shared masks (lop3 with three inputs may merge the `& mask` of two outputs).
2. bs=32 (1.26×) remains the weakest load: 12 experts' 336 MB in 129 µs = 2.6 TB/s, decode-issue bound.
3. The wgmma variant (submission_wgmma.py) is the base if the MMA side ever becomes the limit (it frees 24
   registers per thread).

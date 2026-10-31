# muse_attention_decode experiment log

`bench.py` = judge-like timing over the 12 scored shapes (6 loads; sliding weight 46, full 16) and the 16
tests (uses `reference.collect`). `sweep.py` tunes the Triton attention's TUNE; `prof.py` for nsys.

## Baseline (SGLang: FusedQKNorm, k/v casts, store_cache, q cast, FA3, norm-gate), µs on the pod
| load | sliding | full (floor) |
|---|---|---|
| decode 1 | 22.4 | 33.5 (4.3) |
| decode 16 | 28.7 | 202.8 (63.2) |
| decode 32 | 33.9 | 505.1 (126.4) |
| verify 1 | 25.3 | 39.3 (4.3) |
| verify 16 | 32.4 | 206.3 (63.4) |
| verify 32 | 38.2 | 386.5 (126.8) |
Sliding layers are all launch/overhead (floors ≤ 3.7 µs). Full layers: FA3 at ~25% of bandwidth (page
size 1 → every key a scattered 64-byte gather per head). Board: leader 3.03×, median 1.56×.

### A1 — split-KV Triton attention with everything fused (submission_a1.py), 2026-10-26
- Chunk kernel (request, KV head, key chunk): QK-norm, RoPE, scale fold, new k/v quantized + stored by
  the program covering them, paged FP8 reads, online softmax; combine kernel + norm-gate.
- First version: NaNs (combine multiplied unwritten partials by 0) and byte-strided key loads
  ([even|odd] layout). Fixed (natural order via tl.split/join, masked combine): correct, 0.56×.
- Tuning (chunk 64–1024, block 32–128, warps, stages): ≤ 10%. nsys: sliding bs=1 chunk kernel 16.7 µs vs
  FA3 7.3; full bs=1 combine 38.5 µs (2 programs walking 80 chunks serially).

### A2 — both KV heads per program, per-row combine (submission_a2_triton_attn.py)
- Full layers improved (decode 32 full 523 µs ≈ FA3 504) but sliding got worse (fewer programs):
  0.54×. Beating FA3 needs a hand-tuned CUDA kernel. Parked.

### A3 — SGLang's FA3 path with the five pre-FA3 kernels fused into one Triton kernel (submission.py)
- One program per token: per-head RMSNorm of q and k, interleaved RoPE, query → E4M3 (before the
  query factor, as SGLang), k / v ÷ scales → E4M3 stored at the cache row. FA3 + norm-gate unchanged.
- Pod: **1.131×**, 16/16 tests + 12 scored pass (sliding 1.23–1.34×, full 1.01–1.21×).
- **Submitted: job 2c45705ab18b4339, 1.106×, rank 790 / 816.** Overall board 577.

### A4 — CUDA split-KV attention (submission_a4.py), 2026-10-26
- Prep (Triton) stores the new k/v first and writes FP16 queries pre-scaled (softmax scale × query factor
  × log2 e). CUDA kernel per (request, KV-head pair, key chunk): 32-key stages via a 4-stage cp.async ring,
  per key one page-table entry + 128 B of keys (both heads) + 128 B of values, zero-fill past the length;
  S = QKᵀ and PV as m16n8k16 FP16 MMAs with FP8 → FP16 by cvt.f16x2.e4m3x2 straight into fragments
  (lane-permuted head dims / output columns so shared reads are contiguous, swizzled tiles). Combine:
  A2's per-row Triton kernel.
- Correct first compile (0 failures). decode 32 full 445 µs vs FA3 505; L = 8 0.32× (8 warps each
  re-convert the same tile); small batches lose (page-table load stalls each stage's issue).

### A5 — page rows prefetched a stage ahead; shape-based dispatch (submission.py)
- Chunk sweep (q_len 1): small chunks cost per-CTA overhead; full bs=32 best at 2048 keys (370 µs vs
  FA3 499), bs=16 ties FA3 at 2048, bs=1 loses (43 vs 29); sliding bs=1 wins at 32 keys (15.0 vs 18.4).
- Dispatch: CUDA for full layers with batch ≥ 24 and sliding with batch ≤ 4 (q_len 1); FA3 + fused prep
  otherwise. Pod 1.204×; 0 failures on 16 tests + 12 scored + 36 extra (batch 1–33, page 1/64, scaled).
- **Submitted: job 62e3d04b55e14fa8, 1.129× (1.112–1.146).** On the judge's inputs FA3 runs decode 32
  full in 383 µs (507 on our seed), so the CUDA lead there is small (372 vs 383).

### A6 — multi-draw benchmark (bench_multi.py: mean over 5 length draws, seeds 1000–1004), 2026-10-26
- `kernel_fn_cuda` = the forced CUDA path (for benchmarking); `kernel_fn` dispatches. µs, FA3 path (A3) vs
  CUDA at the given chunk:

| shape | FA3 | CUDA | CUDA / FA3 |
|---|---|---|---|
| full bs1 L1 | 28.6 | 87.8 (1024) | 0.33× |
| full bs16 L1 | 194.8 | 194.1 (2048) | 1.00× |
| full bs32 L1 | 376.3 | 370.7 (2048) | 1.02× |
| full L8 bs1/16/32 | 32.4 / 198.5 / 382.7 | 92.7 / 533.7 / 990.8 | 0.35–0.39× |
| sliding bs1 L1 | 18.7 | 15.1 (32) | **1.24×** |
| sliding bs16/32 L1 | 22.6 / 27.4 | 26.2 / 34.2 (128) | 0.80–0.87× |
| sliding L8 bs1/16/32 | 19.0 / 25.4 / 31.2 | 21.9 / 51.7 / 90.0 | 0.13–0.87× |

- Conclusion: averaged over length draws, the CUDA kernel only clearly wins sliding bs=1 (q_len 1); full
  bs≥16 is a tie (the seed-906 win was a lucky draw). The submitted dispatch (A5) is still right; the
  kernel needs real speedups before it helps more shapes.

### A7 — page-table rows staged in shared memory, 64-key stages, CUDA on every decode step (submission.py), 2026-10-28
- A6's step 1 alone (64-key stages, masks only on edge stages) made the kernel *slower* (full bs16 at 2048 keys:
  194 → 286 µs at 64, 314 at 32 with edge-only masks). Two causes, found with nsys and variants:
  (a) the per-stage page-table loads (`rows_of` a stage ahead) were the real stall: with every key's cache row
  resolved up front into shared memory (one coalesced pass per CTA, ≤ 8 KB) the kernel streams at the random
  128 B gather rate (a microbenchmark gives 3.7 TB/s random 128 B rows, 4.0 for 256 B); (b) unmasked padding
  rows (rows 8–15 at q_len 1) with huge finite scores slow the softmax ~1.6×, so they are masked on every
  stage; only the edge stages run the full masks. Register counts unchanged (96–107 at 32 keys, 141–159 at 64).
- Attention kernel alone (nsys): full bs1 56.7 → 15.6 µs (chunk 512), full bs16 at 2048: 187 → 107.5 µs.
- bench_multi.py (5 draws), CUDA / FA3: full bs1 1.03× (512), bs16 1.69× (2048/64), bs32 1.77× (2048/64);
  sliding bs1 1.36× (128), bs16 1.20× (256), bs32 1.26× (512). Verify (q_len 8) still 0.45–0.9×.
- Dispatch: CUDA on every q_len 1 shape; chunks by shape (`_chunk`: sliding ≈ 128 CTAs, full 512/1024/2048
  by batch); FA3 for verify. bench.py: 1.207 → **1.415×**, 0 failures (28 judge-style + 50 forced-CUDA cases,
  check_cuda.py). **Submitted: job 4dfff989c4eb4bce, 1.332× (1.312–1.352), rank 752.**

### A8 — E4M3 MMAs (attn8_kernel) and the verify blocks, 2026-10-28
- Verify blocks (A6 step 2): m16n8k32 E4M3 MMAs take the E4M3 queries (the FA3 prep) and keys as they are, no
  conversion; P is rounded to E4M3 for P·V (as FA3's FP8 kernel; errors match FA3's, e.g. 5.2e-2 vs the FP16
  path's 3.9e-3, all within tolerance). Values are transposed once per stage into a shared tile (all threads,
  byte_perm), warps read 8-byte fragments. With `__launch_bounds__(256, 2)` for q_len 8.
- Per-stage cycle counts (clock64): at q_len 8 the S MMAs, softmax and P·V each cost ~1000–1800 cycles per
  64-key stage, more than the copy wait (~550–1100): the verify kernel is instruction-bound on mma.sync, not on
  memory. A no-compute variant of the pipeline runs full bs16 L8 in 176 µs vs 285–338 with compute; FA3 does
  it in 198 (wgmma). Beating FA3 on full verify needs a wgmma (warpgroup) kernel, not more tuning here.
- Results (5 draws): sliding verify bs1 **1.29×** (64-key chunks), bs16/32 0.7–0.9×; full verify 0.46–0.65×.
  At q_len 1 the E4M3 kernel is ~3% slower than the FP16 one (full bs32 219 vs 213 µs), so `MMA8` picks
  FP16 MMAs at q_len 1 and E4M3 at q_len 8.
- Dispatch: + sliding verify with batch ≤ 2 (`CUDA_VERIFY_MAX_BATCH`). bench.py **1.450×**, 0 failures.
  **Submitted: job a2c76ca68c5148d2, 1.361× (1.341–1.386), rank 740.** Overall board 688.

### A9 — wgmma (warpgroup) FP8 kernel for verify blocks (attnwg_kernel), 2026-10-29
- One warpgroup per KV head: the head's 64 query rows (8 tokens × 8 heads) are one m64 tile. Per stage of 64 keys:
  S = Q·Kᵀ as two m64n64k32 E4M3 wgmmas (Q in registers, K from the swizzled smem tile by descriptor), softmax in
  registers, V transposed once by all threads into a swizzled K-major tile (keys in the P-fragment slot order), P
  rounded to E4M3 as the register A operand of two m64n64k32 wgmmas. Layouts validated first with a standalone
  test (`tools/wg_test.py`): descriptor = start>>4 | SBO>>4 << 32 | layout << 62 (1 = 128 B swizzle,
  2 = 64 B), K-step = +32 B on the start address; accumulator rows 16w + l/4 (+8), cols 8i + 2(l%4) (+1).
- Hazards found: (1) `fence.proxy.async.shared::cta` after cp.async / st.shared writes before wgmma reads;
  (2) the register A operands are read asynchronously: with Q held in registers across the loop the compiler
  reused them under the in-flight wgmma (stages ≥ 1 read garbage) whatever the asm constraints — Q is reloaded
  from global (L1) every stage; P is consumed before anything else touches registers. (3) Accumulator reads are
  pinned around wait_group with empty asm ("+f") as CUTLASS does.
- Per-stage phases (clock64, cycles): copy wait + issue ~700, V transpose + barrier ~800, S issue→wait ~550,
  softmax ~800 (MUFU exp2 + cvt bound), P·V ~180: ~3000 serial per 64-key stage after: one barrier per stage (V of
  stage it + 1 transposed into a second VT buffer during stage it; 5-stage ring, wait 2 ahead), the chunk's
  page-table rows staged in two parts (first 4 stages' before the prologue copies, the rest while they fly; eight
  loads in flight per thread — the serialized loop cost 10K cycles per CTA), the score scale folded into the exp2 FMA.
- Structure sweep: two heads per 256-thread CTA (128 B rows, 2 CTAs/SM) 200–250 µs on full bs16 L8 vs FA3 198;
  one head per 128-thread CTA (64 B rows, 4 CTAs/SM) 227–248: its copy-only floor is 159 µs (64 B random rows
  gather at 2.3 TB/s vs 3.7 for 128 B, measured with `gather.py`), deeper rings (6, 8 stages) do not lower it.
  mma.sync ablations earlier: the no-compute floor of the 2-head 128-thread decode kernel is 115 µs.
- Results (3 draws, one head per CTA, 64-key stages): sliding verify bs1 **1.40×** (64-key chunks), bs16
  **1.04×** (256), bs32 0.85× (512); full verify 0.80–0.92× — FA3 keeps full verify and sliding bs32.
- Dispatch: sliding verify with batch ≤ 16 → wgmma kernel (`CUDA_VERIFY_MAX_BATCH`, `VERIFY_HEADS`, chunk rule in
  `_chunk`). bench.py 1.450 → **1.466×**, 0 failures (28 judge-style + 50 forced-CUDA). **Submitted: job e3e7b06cb3d14e80, 1.378× (1.357–1.399), rank 736.** Overall board 668.

### A10 — warp-specialized wgmma verify kernel (attnpc_kernel), 2026-10-29
- Same math as A9's attnwg_kernel (one KV head per CTA, 64-key stages) restructured as a producer warpgroup
  (cp.async copies, page-table rows, the value transposes into three VT buffers) and a consumer warpgroup (S of
  stage it + 1 issued before the softmax of stage it, two S accumulators, P·V async until the next stage), one
  __syncthreads per stage; 2 CTAs/SM (57 KB smem, 127 registers, no spills). `setmaxnreg` register moves faulted
  with cudaErrorInvalidAddressSpace and were not needed.
- Steps on the way (full bs16 L8, attention kernel alone, nsys): serial A9 237 µs → overlapped single-warpgroup
  (3 CTAs/SM, 147 regs) 237 → + transpose lanes remapped against 4-way bank conflicts + 2048-key chunks 221 →
  warp-specialized 186–195 (FA3 198). Over 5 draws the whole path (prep + attention + combine) ties FA3 on full
  verify (0.96–0.99× at bs16/32, 1.02–1.05× at bs1): the kill rule (< 170 µs) was not met, so full verify stays
  on FA3. ptxas still reports C7513 (wgmma serialized by non-wgmma definitions of its inputs inside the fence…
  commit region) after moving descriptors and A fragments before the fence; cause not found.
- Bugs met: the `rows` staging loop strided by 256 threads but ran on the producer's 128 (half the rows of
  stages ≥ 4 missing); "+r" constraints on the A fragments make ptxas serialize consecutive wgmmas (plain "r"
  inputs + pins after wait_group instead).
- Sliding verify (5 draws): bs1 **1.45×** (64-key chunks), bs16 **1.08×** (512), bs32 0.92× (FA3 keeps it).
  Dispatch: `VERIFY_PC` → attnpc for sliding verify ≤ 16 requests, chunk 64 (batch ≤ 4) / 512. bench.py 1.466 →
  **1.475×**, 0 failures (28 + 50 cases). **Submitted: job 8d8e1bdca2c440de, 1.391× (1.370–1.412), rank 727.** Overall board 672.
- Phase profile of the overlapped single-warpgroup loop (cycles per 64-key stage): barrier + copy wait ~150,
  copy issue + S issue + transpose ~1470, softmax ~880 (32 MUFU exp2 + 16 F2FP per thread, both 16/clk/SM),
  P·V issue ~240; prologue ~16K per CTA (page-table rows + first stage's DRAM latency + first transpose).

### A11 — sliding decode steps in one launch: prep + attention + merge in a cluster (attnc_kernel), 2026-10-30
- Why: sliding decode was 3 kernels (Triton prep 2.5 µs, attention 5.9, Triton combine 2.5 at bs1) for < 4 µs of
  real work; the judge weighs sliding shapes 46 vs 16.
- First try, attnf ("atomic"): one launch, partials to a workspace, the last CTA of each group (atomic counter it
  resets) merges. bs1 13.7 -> 12.4 µs only. Per-CTA clocks (clock64 / globaltimer per phase): the
  owner of the new token waited ~1.8 µs on a __threadfence after storing it, and the last CTA's merge took ~4 µs
  (16 chunks read from L2 four at a time). On full layers it is much slower (bs1 27.6 -> 37.4 µs: one CTA merges
  157 chunks); kept in the file, unused.
- attnc: the chunks of each (request, KV-head pair) are one cluster (NCH | 16 CTAs, non-portable size 16 at bs1).
  Every CTA normalizes / rotates its 16 query heads into shared memory; the owner of the new token computes its
  keys / values (same RoPE angles as its queries), puts them straight into its stage tile at issue time and stores
  them to the cache without waiting; each key-half warp writes its partial (m, l, o) to shared memory; after a
  cluster barrier each thread reads one (row, chunk) partial over DSMEM into the free ring, a relaxed arrive lets
  the other CTAs go, and CTA c merges rows 16c / NCH .. with the gated RMSNorm.
- Fixes found with the clocks (`mkcclk2.py` + `cclk2.py`): the compiler sank the `vscale` load to its use at the
  very end (a cold DRAM read: ~5500 cycles in the output phase) -> all scalars loaded up front and pinned with an
  empty asm; DSMEM reads done one chunk at a time behind branches (~9800 cycles) -> one read per thread; the
  key-half merge with two barriers -> both halves written as partials; sigmoid with __expf / __frcp_rn.
- Phases now (bs1, cycles): prep 4500 (seqlens -> page table: two dependent DRAM reads), loop 4500, merge 235,
  cluster barrier ~1500 (waiting for the owner), DSMEM 1800, merge 1000, output 1400; end ~8.1 µs.
- bench_multi (5 draws, µs) bs1 / 16 / 32: 13.7 / 19.0 / 22.0 (A10) -> 8.9 / 12.1 / ~16. Stages of 32 keys for
  chunks <= 128 (bs1). bench.py 1.475 -> **1.577×**, 0 failures (28 + 50 cases), 2400 graph replays bitwise equal
  (`tools/astress.py`).
- **Submitted: job f71bd3dd29a94d3b, 1.473× (1.451–1.495), was 1.391×.** Judge sliding decode 13.3 / 18.7 / 21.1 ->
  9.1 / 12.3 / 16.6 µs.

### A12 — full verify with both KV heads per CTA (attnpc2), 2026-10-30 — no gain, removed
- attnpc with 128 B rows (both heads: the faster gather), a producer warpgroup and one consumer warpgroup per head
  (384 threads, 1 CTA/SM; K tile in the 128 B swizzle, head at + 64 as attnwg HEADS = 2). Correct (50 cases).
- bench_multi full verify bs16 / 32: 226.8 / 439.7 µs (chunk 2048), 252.6 / 417.0 (4096) vs attnpc 200.4 and FA3
  199 / 382. Ablations at bs16: no value transposes 198.5 (the one producer transposing two heads is on the
  critical path), no P V 212.7; so even with free transposes it only ties: the consumer chain is ~2400 cycles per
  64-key stage (5 waves x 32 stages in 198 µs). 128-key stages (STG 4, 226 KB): 325 / 618 µs.
- Kill rule met (A10 + A12): full verify stays on FA3. Also tried: attnf (A11's atomic merge) on full decode:
  bs1 27.6 -> 37.4, bs16 115 -> 139 µs. Both kernels removed from the file.
- Note: bench_multi's first CUDA column after a fresh build can read high (attnpc 271.8 once, 200.4 rerun); rerun
  or reorder before trusting a single column.

### A13 — sliding verify blocks in one launch (attnpcc_kernel); cold-L2 instruction fetch, 2026-10-30
- attnpc's CTA (one KV head, producer + consumer warpgroups) with each (request, KV head)'s chunks one cluster
  (<= 16: chunk 136 at <= 4 requests, 512 above). Every CTA puts its 64 query rows (RMSNorm, RoPE, E4M3) in shared
  memory (the consumer's per-stage Q reload reads them there); the CTAs holding new tokens compute their keys /
  values, which the producer copies into the stage tiles from shared memory (cache stores without waiting).
- First version (partials on the idle ring, pulled over DSMEM): bs1 15.1 µs, bs16 27.2 vs the 3-kernel path 13.0 /
  23.2. Clocks (`tools/mkvclk.py` + `tools/vclk.py`): merge ~8000 cycles, prep ~5500 (8 sincosf per thread).
- Fixes: partials pushed into the merging CTAs' shared memory (st.shared::cluster, no wait) so the merge reads only
  local memory; RoPE (sin, cos) once per (token, pair) in a 256-entry table; sigmoid / weights with ex2.approx /
  rcp.approx. bs1 12.2 -> 11.5, bs16 19.4 -> 16.8.
- **Finding: the merge's ~6600 cycles were instruction fetch.** Run twice without the L2 flush (`vclk2.py`): merge
  6928 -> 2298 cycles, prep 3514 -> 2553, whole kernel 10.8 -> 7.8 µs. The judge flushes L2 before every replay, so
  code that runs once per CTA (prologue, epilogue) is fetched from DRAM on every call. Straight-line (unrolled)
  merge code was worse (8758): bytes matter, not branches. Fix: the merge is a __noinline__ function the idle
  producer runs once (nothing stored) after its last copy, so the real merge finds its code in the instruction
  cache: merge 6928 -> 2524 cycles, kernel 10.8 -> 8.9 µs. (Likely matters for every kernel here, MoE included.)
- bench_multi: vslide1 13.0 -> 10.2 µs, vslide16 23.2 -> 17.6 (bs32 36.6 vs FA3 31: stays on FA3). bench.py 1.577
  -> **1.629×**, 0 failures, 2100 replays bitwise equal. **Submitted: job 161f6859011940a9, 1.499× (1.476–1.528).**
  Judge verify sliding bs1 / 16: 12.3 / 21.6 -> 9.9 / 17.2 µs.

### A14 — decode (attnc) with pushed partials and a per-lane merge, 2026-10-30
- As A13: each key-half warp pushes its partial row to the merging CTA (st.shared::cluster v4), empty chunks push
  zeros, one cluster barrier, then each (row, 8 dims) lane loops over 2 NCH partials from local memory; the gated
  norm tail shared with attnpcc (ex2 / rcp approx). bench_multi slide1 / 16 / 32: 8.9 / 12.1 / 16.6 -> 8.5 / 11.4 /
  15.8 µs; bench.py 1.629 -> 1.647×, 0 failures, 2400 + 2100 replays equal.
- **Submitted: job 09b49ea4276e4306, 1.525× (1.502–1.548)**, rank 753 / 1152 (from 1.499×).
- Phases now (cycles, bs1 / 16 / 32): prep 4200 / 4800 / 4400, loop 4800 / 7350 / 14450, push ~650, barrier wait
  ~1500 (the owner's longer prep), merge 2700 / 1900 / 1700.

### A15 — full-layer decode steps: one launch + merge kernel, or a CUDA merge after attn_kernel — no gain
- attnc for full layers (no cluster: partials to a workspace, a second kernel merges, launched with programmatic
  serialization): its main loop is slower than attn_kernel's (bs16 121 vs 110 µs: 168 registers vs 102, and its
  13 KB of extra shared memory left 2 CTAs / SM); after trimming the layout bs16 / 32 still 145 / 247 vs 115 / 214.
- attn_kernel + a CUDA merge (attn_merge_kernel, one CTA per query row, two passes of independent loads, PDL):
  bs1 28.4 vs 27.6, bs16 119.8 vs 115.4, bs32 223 vs 214 µs: no better than the Triton combine. Both reverted
  (kept as submission_a15_fullattnc.py / submission_a15_cudamerge.py).

## Next
- Full verify ties FA3 now (kernel 186–195 µs vs 198 at bs16; the one-head copy floor is ~159 µs). What is left:
  the consumer chain (~1500 cycles/stage: softmax on MUFU/F2FP, two wgmma issues, barrier); options are exp2 for
  half the elements on FMA pipes (FA4's polynomial), 128-key stages (halves the fixed costs; needs the N = 128 S
  tile → registers), the prologue (~16K cycles per 32-stage CTA), and the C7513 serialization. Two-head CTAs
  (128 B rows, 2.3 → 3.7 TB/s gather) with 2 consumer warpgroups + 1 producer would raise the floor but need
  1 CTA/SM. Kill rule stays: not worth more than one more session.
- Decode steps: the FP16-MMA kernel is at ~2.8 TB/s on full bs32; per-CTA fixed costs at small batches.
- Weakest loads now (judge, A11): verify 16 x1.105 and verify 32 x1.051, 76-82 % of them full verify on FA3
  (199 / 384 µs; full decode does the same keys in 117 / 217). Full verify at full-decode speed would be ~x1.15 on
  the problem; at 150 / 290 µs ~x1.075.
- Cold-L2 instruction fetch (A13): warm the code of once-per-CTA phases with idle warps (a __noinline__ function run
  early with its stores off). attnc has no idle warp: a 5th "warmer" warp could do it for its merge (~1 µs?).
- Tools: `bench_multi.py --chunks=keys/stage/heads/pc`, `check_cuda.py`; in `tools/` (run on the pod from this
  folder): `astress.py` (graph-replay stress), `mkcclk2.py` + `cclk2.py` / `cclk3.py` (attnc phase clocks; cclk3
  adds a warm second run), `mkvclk.py` + `vclk.py` / `vclk2.py` (attnpcc phase clocks), `wg_test.py` (wgmma layout
  test), `wgclk.py` (phase cycles of a clock-instrumented variant), `prof2.py` + `nsk.sh` (nsys; args: file layer
  batch L chunk sk [dist heads pc]), `gather.py` (random-row gather bandwidth).

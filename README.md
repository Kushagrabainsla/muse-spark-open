# Muse Spark Open Kernel Challenge — our kernels

Custom CUDA / Triton kernels for the [Open Kernel Challenge](https://openkernelchallenge.dev) (Muse Spark Open,
one H200 per run). Each problem is one piece of a Muse decode step; a kernel is scored by how much faster it is
than SGLang's on the same inputs. The kernel track closes **2026-11-04, 11:59 pm PT**; the serving track runs
2026-11-05 → 11-19.

- Current state and what to do next: [RESUME.md](RESUME.md)
- Full experiment history: each problem's `EXPERIMENTS.md`

## Results (board, 2026-10-30)

| Problem | What it is | Our speedup | Rank | Median | #1 | Job |
|---|---|---|---|---|---|---|
| [muse_attention_decode](muse_attention_decode) | gated attention decode over a paged FP8 cache | **1.525×** | 754 / 1152 | 1.79× | 3.22× | 09b49ea4276e4306 |
| [muse_moe_w4a16](muse_moe_w4a16) | routed experts, NVFP4 weights | **1.704×** | 688 / 1206 | 1.81× | 2.76× | 069f543bb3604714 |
| [muse_decode_layer](muse_decode_layer) | the whole decoder layer | **1.425×** | 703 / 1072 | 1.66× | 2.42× | 66246e98c53c47a8 |
| [muse_bf16_proj](muse_bf16_proj) | dense BF16 projections (skinny GEMMs) | **1.175×** | 1096 / 1502 | 1.24× | 1.65× | 4ed4a00394264da9 |
| overall kernel board | the 4 problems as one decode step | | 722 / 1722 | | | |

Speedup = geometric mean over the problem's loads (decode / verify at batch 1, 16, 32 and so on) of SGLang's
time over ours. Ranks, medians and #1 as of 2026-10-30 23:59 UTC. `muse_prefill_layer` is not entered.

## Layout

Each problem folder was created by `okc init` and has the same shape:

| File | What |
|---|---|
| `TASK.md`, `AGENTS.md` | the problem statement, shapes, rules (from okc) |
| `reference.py`, `checks.py` | the judge's reference and correctness checks (local copies) |
| `baseline.py` | SGLang's kernel, the one we are timed against |
| `submission.py` | **our kernel** = what is on the board |
| `submission_*.py` | earlier versions and experiments, named by experiment id (A10, M16, L7, ...) |
| `EXPERIMENTS.md` | every experiment: idea, numbers, result, submitted jobs, open ideas ("## Next") |
| `bench.py` | judge-like timing of all scored shapes vs SGLang (`--check` runs the tests first) |
| `timing.py` | CUDA-graph timing with a cold L2, as the judge does |
| other `*.py`, `tools/` | sweeps, probes, profilers and stress tests for that problem (run on the pod) |
| `profiles/` | the judge's output for each submitted job (`okc` saves it) |

`muse_decode_layer/submission.py` is generated: `python3 muse_decode_layer/build_layer.py` splices the
attention, MoE and projection submissions into SGLang's layer. Rerun it after changing any of them.

## How we work

1. **GPU:** a RunPod H200 SXM pod named `okc-musespark-*` (image `lmsysorg/sglang:dev-muse-spark-open`, 120 GB disk,
   SSH). Budget: at most $70 per UTC day; check `date -u` and the bill every hour, stop the pod when idle.
   An H100 is not a valid stand-in (MoE results did not carry over).
2. **Copy** the problem folder to the pod (`/root/okc_<problem>`), run `bench.py --check` there. Local numbers track
   the judge within a few percent.
3. **Submit** with `~/.okc/bin/okc submit <problem>/submission.py` when the pod shows a clear gain (~3 %) and 0 check
   failures; `okc check` for risky changes. Limits: 10 submits and 20 checks per problem per day.
4. **Log** every experiment in that problem's `EXPERIMENTS.md` and keep `RESUME.md` current.

## What our kernels do

### Attention (`muse_attention_decode`)
- **Sliding-window decode** (one launch, `attnc_kernel`): one CTA per (request, KV-head pair, key chunk); the chunks
  of a request form one thread-block cluster. Each CTA normalizes and rotates its queries itself, the CTA holding
  the new token writes it to the cache and into its own tile, keys stream through a `cp.async` ring into FP16
  MMAs, and partial results are pushed into the merging CTA's shared memory (DSMEM), which applies the gated RMSNorm.
- **Sliding-window verify** (8-token blocks, `attnpcc_kernel`): the same one-launch idea around a warp-specialized
  wgmma kernel (a producer warpgroup copies and transposes, a consumer warpgroup runs FP8 wgmmas).
- **Full-attention decode**: a fused Triton prep, a CUDA split-KV kernel, a Triton combine. **Full verify** stays
  on SGLang's FA3 (our wgmma kernels only tie it).

### MoE (`muse_moe_w4a16`)
NVFP4 grouped GEMMs in CUDA. Routing happens inside the GEMM CTAs; only (token, local expert) pairs are computed,
each touched expert's weights are streamed once; weights are repacked once so each warp load is a contiguous 512 B;
E2M1 codes become BF16 by bit moves (exact); split-K partials go to FP32 accumulators with `red.global.add`; a
norm kernel applies the post-expert norm and the weighted sum. Kernels are chained with programmatic dependent launch.

### Projections (`muse_bf16_proj`)
Up to 32 tokens: a cluster split-K skinny GEMM (weights streamed by 16-byte loads straight into MMAs, partials summed
over DSMEM). Above that, cuBLAS (as SGLang), plus a split-K Triton GEMM for the router.

### Decode layer (`muse_decode_layer`)
SGLang's layer with the three kernels above swapped in.

## Lessons worth keeping

- **Cold L2 means cold code.** The judge flushes L2 before every replay, so code that runs once per CTA (prologue,
  epilogue) is fetched from DRAM on every call. In the attention verify kernel the merge took ~6900 cycles cold vs
  ~2300 warm; having the idle producer warps run the merge code early (stores off) fixed most of it. Smaller code
  helps; unrolled straight-line code hurt. (MoE is only 0–4 % affected.)
- **Fixed costs dominate small batches.** Removing kernel launches (prep + attention + combine → one kernel) gave the
  biggest attention gains; the remaining time is dependent DRAM reads (lengths → page table → keys).
- **Watch what the compiler moves.** It sank a scale load to the end of a kernel (a cold DRAM read on the critical
  path); loading all scalars up front and pinning them with an empty `asm` fixed it.
- **Pushing beats pulling over DSMEM:** remote stores before a cluster barrier, local reads after it.
- **Measure phases in the kernel** (`clock64` / `%globaltimer` per CTA): every big win came from finding the one slow
  phase. `ncu` is not available on RunPod.
- Ideas that did not pay off are in each `EXPERIMENTS.md` (e.g. 2-head full-verify CTAs, vector atomics in MoE,
  12-bit lossless projection weights).

## Open levers (by expected value)

1. MoE: split-K atomics cost up to ~77 µs at batch 256; accumulating over k in registers would remove most of them.
2. Attention: full verify at batch 16 / 32 (on FA3) is the weakest load (~1.05–1.2×).
3. Attention decode: ~2 µs of prologue (two dependent DRAM reads) and the merge phase.
4. Projections: small-batch tails and the `o` projection on cuBLAS.

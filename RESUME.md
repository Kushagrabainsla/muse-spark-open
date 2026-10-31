# Resume notes — Open Kernel Challenge (updated 2026-10-30, 23:50 UTC)

Project overview, layout and lessons: [README.md](README.md). This file is the working state.

## Where things stand
| Problem | Board | Job | Local file |
|---|---|---|---|
| muse_bf16_proj | 1.190× (2026-10-31) | fef2017d39ec4cc2 | muse_bf16_proj/submission.py (= submitted E14) |
| muse_moe_w4a16 | **1.704× (rank 688 of 1206)** | 069f543bb3604714 | muse_moe_w4a16/submission.py (= submitted M22) |
| muse_decode_layer | **1.425× (rank 703 of 1072)** | 66246e98c53c47a8 | muse_decode_layer/submission.py (L8 = A14 + M22) |
| muse_attention_decode | **1.525× (rank 754 of 1152)** | 09b49ea4276e4306 | muse_attention_decode/submission.py (A14 = submitted) |
| overall kernel board | rank 722 of 1722 (2026-10-30 23:59 UTC) | | |

Kernel track closes 2026-11-04 11:59 pm PT. Serving track (prizes) 2026-11-05 → 11-19.
Full experiment history: `muse_bf16_proj/EXPERIMENTS.md` (E0–E13), `muse_moe_w4a16/EXPERIMENTS.md` (M1–M22),
`muse_decode_layer/EXPERIMENTS.md` (L1–L6), `muse_attention_decode/EXPERIMENTS.md` (A1–A15). The layer is
generated from the three kernel files (attention, proj, moe): rerun `python3 muse_decode_layer/build_layer.py`
after changing any of them. muse_prefill_layer is not entered.

## 2026-10-31: no pod (user: do NOT use RunPod); every test is an `okc check`
- Tools: `okc check <problem> <file> --no-wait`, then `okc status <job> --wait --json`; per-shape times and the
  per-kernel profile come back. A local compare script printed SGLang / ours per shape against a base job.
- Proj E14: cluster tail pushed at MT = 1 + qkv config: board 1.175 -> 1.190×. Attention A16 (FA3 splits, page-table
  batching, chunk 256, CUDA full verify bs1): nothing kept. Tuning is exhausted; big levers need new kernels.

## 2026-10-30 (00:07–23:45 UTC): attention
- A11: sliding decode steps in one launch (attnc: prep + attention + cluster/DSMEM merge): board 1.391 -> 1.473×.
- A12: full verify with 2-head CTAs (attnpc2): no gain, removed (full verify stays on FA3: kill rule met).
- A13: sliding verify (<= 16 requests) in one launch (attnpcc) + **the cold-L2 instruction-fetch finding**: the
  judge's L2 flush makes once-per-CTA code (prologue / epilogue) come from DRAM each replay (merge 6928 -> 2298
  cycles when warm); fixed by having the idle producer run the merge code once early: board 1.499×.
  MoE checked the same way (`muse_moe_w4a16/tools/icache.py`): warm code saves only 0–4 %, not its bottleneck.
- A14: decode merge with pushed partials: board 1.525×. Layer rebuilt with A11 (L7, 1.418×) and A14 (L8, 1.425×).
- A15: full-layer decode in one launch / a CUDA merge: no gain (reverted). Full verify (A12) and full decode stay as
  they are; the weakest attention loads are verify 16 / 32 (x1.2 / x1.06), 80 % of them FA3 on full layers.
- Pod okc-musespark-attn1 (id pltjs1umqekro3) ran 00:07 -> 23:45 UTC: **~$125 on 2026-10-30, over the $70 cap**
  (time was misjudged; see memory runpod-clock-check). STOPPED at 23:45 UTC.

## Where the work paused (2026-10-29, ~19:30 UTC)
- muse_bf16_proj (the weakest board) was worked on 2026-10-29 afternoon with no gain: a cuBLASLt algorithm search
  (E9: cuBLAS is already at its best for m = 128 / 256), a split-K TMA + wgmma GEMM for m = 32–256 (E10: correct,
  at best ties cuBLAS — one TMA ring per SM can't stream fast enough, and x / partial re-reads double the traffic)
  a non-blocking skinny-kernel epilogue (E11: slower than the cluster barrier), and lossless 12-bit weights (E12:
  10-12 % faster streaming at <= 8 tokens, but the exact exception fix-up eats most of it: ~+1 %, not submitted;
  kept as submission_c12.py) / 13-bit (E13: slower). Local submission.py = E8 = submitted. The measured levers (small-m tails of ~1–1.6 µs, `o` on cuBLAS) and untried ideas are in its
  EXPERIMENTS.md "## Next".
- MoE (~half of a decode step): M16 on the board (1.642×). 2026-10-29: small-batch time split measured on an H100
  (M18); vector atomics (M19) and a norm fused into gemm2 (M21) are correct but slower on the H200 (geomean 1.735 ->
  1.596 -> 1.507; an H100 is not a proxy for this kernel); L2 prefetch dead (M20). Attention A10 1.391× (parked).
- MoE M22 (evening): H200 debug split showed a fixed "skeleton" (route kernel, atomics, norm, staging) of 31–45 % of
  the time; routing moved into the gemm CTAs (route kernel gone) and the norm reads its inputs before waiting:
  pod 1.736 -> 1.797×, board 1.642 -> 1.704×; layer rebuilt (L6) 1.391 -> 1.404×. Config re-sweep: no gain.
  Decode-op cuts are worth little (a 4-op cheaper decode probe: +1.5 %).
- Local files = what is on the board for all four problems. Backups: muse_moe_w4a16/submission_m14.py,
  muse_attention_decode/submission_a9.py / _a5 / _a3 (bench_multi's FA3 reference) / _a4 / _a1 / _a2.
- Tools: proj `bench.py [file] --check --m=...`, `ltprobe.py` (cuBLASLt search), `wgemm_probe.py` (E10 kernel);
  MoE `sweep.py bs 's1,s2,mt;...'`, `bench.py --check`; attention `bench_multi.py --chunks=keys/stage/heads/pc`,
  `check_cuda.py`; profiling / stress scripts in each problem's `tools/` (see its EXPERIMENTS.md "## Next").

## Next session (user said "Yes, tomorrow" on 2026-10-29)
- Restart/recreate the H200 pod after 00:00 UTC (new budget day), re-push muse_moe_w4a16 + muse_decode_layer + scratch tools.
- Work the MoE skeleton (EXPERIMENTS.md "Next" item 00): split-K atomics (≤ +3.5 %), norm kernel (4–13 µs), x staging / syncs.
- After that: attention small-batch fixed costs, attention full verify, proj 12-bit.

## muse_moe_w4a16 — exact state of the local file
- `submission.py` = M22 (= submitted, 1.704×; pod 1.797×), 0 check failures; the decode_layer build uses it.
  `submission_m16.py` = M16, `submission_r.py` / `_r2.py` = M22 intermediates, `submission_m14.py` = M14, `submission_m11.py` = pre-M12, `submission_wgmma.py` = M17 (wgmma, slower).
- Loads and math are each ~90 % of the time and overlap; the rest is the skeleton (M22). Earlier "issue-bound" reading
  was only partly right: cutting 4 decode ops per word gives +1.5 %. No ncu on RunPod (ERR_NVGPUCTRPERM): diagnose with
  `mkdbg.py` variants + nsys + `cuobjdump -sass`.

## Pod / tooling
- Pods: `okc-musespark-attn1` (id `pltjs1umqekro3`, H200 SXM, US-CO-1, ssh root@152.236.142.240 -p 12992; files in
  /root/okc_attn, /root/okc_moe, /root/okc_layer), created 2026-10-30 00:07 UTC, STOPPED 23:45 UTC (its GPU may be
  gone on start: then terminate and recreate). moe5 and older are terminated.
  H200 stock comes and goes; an H100 SXM is NOT a valid proxy for the MoE kernel (M21). A stopped pod keeps
  no files and its GPU is usually taken: try `start`, else terminate and recreate (a new host can take ~5–10 min).
- Spend 2026-10-30 UTC ≈ $125 (over the cap). Spend 2026-10-29 UTC for this project ≈ $58 by 19:30 (RunPod's day total also includes the other session's pods: count
  only okc-musespark-* records toward the $70/day cap).
- Local helper scripts (not in the repo; fill in the pod's ip / port):
  `pod.sh`:  exec ssh -i "$HOME/.ssh/runpod_ed25519" -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p PORT root@IP "$@"
  `push.sh`: d=$1; shift; exec scp -i "$HOME/.ssh/runpod_ed25519" -o BatchMode=yes -o StrictHostKeyChecking=accept-new -P PORT -q "$@" root@IP:$d/
- To recreate a pod: H200 SXM (`NVIDIA H200`, SECURE, minCudaVersion 12.8), image
  `lmsysorg/sglang:dev-muse-spark-open`, disk 120, ports 22/tcp, startSsh, entrypoint `/bin/bash -c` with cmd:
  apt-get install openssh-server if missing, `echo "$PUBLIC_KEY" > /root/.ssh/authorized_keys`, ssh-keygen -A,
  start sshd, `sleep infinity`. ~$5.29/hr; budget ≤ $70/day (raised from $50 on 2026-10-29; ask the user before going over). Then
  `echo "export PATH=/opt/sglang/bin:/usr/local/cuda/bin:\$PATH" > /root/.okcenv` and copy each problem's *.py.
- Tools in each problem dir: `bench.py` (judge-like timing vs SGLang, `--check`), `timing.py`, `sweep.py`,
  `prof.py` (nsys), `mkdbg.py` (debug variants). okc CLI: `~/.okc/bin/okc` (signed in as Kushagrabainsla);
  daily caps 10 submits / 20 checks per problem.

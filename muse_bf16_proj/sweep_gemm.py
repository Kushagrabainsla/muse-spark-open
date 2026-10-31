"""E6: split-K Triton GEMM for m = 128 / 256. python sweep_gemm.py [--m=128,256] [--layer=pooler]"""

import itertools
import sys

import torch
import triton
import triton.language as tl

HERE = __file__.rsplit("/", 1)[0]
sys.path.insert(0, HERE)
import reference as ref  # noqa: E402
from timing import kernel_us  # noqa: E402


@triton.jit
def _gemm_splitk(x_ptr, w_ptr, wlo_ptr, out_ptr, ws_ptr, cnt_ptr, M, N, K,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, S: tl.constexpr,
                 HAS_LO: tl.constexpr, OUT_F32: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_s = tl.program_id(2)
    kblocks = K // BK
    kb0 = pid_s * kblocks // S
    kb1 = (pid_s + 1) * kblocks // S
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = kb0 * BK + tl.arange(0, BK)
    m_mask = offs_m[:, None] < M
    x_ptrs = x_ptr + offs_m[:, None] * K + offs_k[None, :]
    w_ptrs = w_ptr + offs_n[None, :] * K + offs_k[:, None]
    wlo_ptrs = wlo_ptr + offs_n[None, :] * K + offs_k[:, None]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(kb0, kb1):
        a = tl.load(x_ptrs, mask=m_mask, other=0.0)
        b = tl.load(w_ptrs)
        acc = tl.dot(a, b, acc)
        if HAS_LO:
            acc = tl.dot(a, tl.load(wlo_ptrs), acc)
            wlo_ptrs += BK
        x_ptrs += BK
        w_ptrs += BK
    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    if S == 1:
        if OUT_F32:
            tl.store(out_ptrs, acc, mask=m_mask)
        else:
            tl.store(out_ptrs, acc.to(tl.bfloat16), mask=m_mask)
    else:
        tile = pid_m * tl.num_programs(1) + pid_n
        ws = ws_ptr + offs_m[:, None] * N + offs_n[None, :]
        tl.store(ws + pid_s * M * N, acc, mask=m_mask)
        tl.debug_barrier()
        arrived = tl.atomic_add(cnt_ptr + tile, 1, sem="acq_rel")
        if arrived == S - 1:
            tl.debug_barrier()
            total = tl.zeros((BM, BN), dtype=tl.float32)
            for s in range(S):
                total += tl.load(ws + s * M * N, mask=m_mask, other=0.0, cache_modifier=".cg")
            if OUT_F32:
                tl.store(out_ptrs, total, mask=m_mask)
            else:
                tl.store(out_ptrs, total.to(tl.bfloat16), mask=m_mask)
            tl.atomic_xchg(cnt_ptr + tile, 0)


WS = torch.empty(1 << 24, dtype=torch.float32, device="cuda")
CNT = torch.zeros(4096, dtype=torch.int32, device="cuda")


def run(x, w, wlo, out, cfg):
    bm, bn, bk, s, warps, stages = cfg
    m, k = x.shape
    n = w.shape[0]
    grid = (triton.cdiv(m, bm), n // bn, s)
    _gemm_splitk[grid](x, w, w if wlo is None else wlo, out, WS, CNT, m, n, k, BM=bm, BN=bn, BK=bk, S=s,
                       HAS_LO=wlo is not None, OUT_F32=out.dtype == torch.float32,
                       num_warps=warps, num_stages=stages)
    return out


SPLITS = (1, 2, 3, 4, 6, 8)


def main():
    global SPLITS
    if '--router-wide' in sys.argv:
        SPLITS = (4, 8, 16, 32, 64)
    args = sys.argv[1:]
    ms = [int(v) for v in next((a.split("=", 1)[1] for a in args if a.startswith("--m=")), "128,256").split(",")]
    layers = next((a.split("=", 1)[1] for a in args if a.startswith("--layer=")),
                  "qkv_gate,o,pooler,unpooler,router").split(",")
    weights = ref.generate_weights(ref.WEIGHTS_SEED)
    hi = weights["router"].to(torch.bfloat16)
    lo = (weights["router"] - hi.float()).to(torch.bfloat16)
    for m, layer in itertools.product(ms, layers):
        data = ref.generate_input(m, layer, seed=7)
        n = data["config"]["n"]
        is_r = layer == "router"
        w, wlo = (hi, lo) if is_r else (weights[layer][:n], None)
        out = torch.empty((m, n), dtype=torch.float32 if is_r else torch.bfloat16, device="cuda")
        res = []
        for bm, bn, bk, s, warps, stages in itertools.product((64, 128), (64, 128, 256), (64, 128),
                                                              SPLITS, (4, 8), (3, 4)):
            if n % bn or (bm * bk + bn * bk * (2 if is_r else 1)) * 2 * stages > 220_000:
                continue
            if bm * bn // (32 * warps) > 128 or s * m * n > WS.numel():
                continue
            cfg = (bm, bn, bk, s, warps, stages)
            fn = lambda d, st, c=cfg: run(d["x"], w, wlo, out, c)
            try:
                ok, msg = ref.check_implementation(data, fn(data, None).clone(), weights)
                if not ok:
                    print("BAD", cfg, msg, flush=True)
                    continue
                res.append((kernel_us(fn, data, None), cfg, triton.cdiv(m, bm) * (n // bn) * s))
            except Exception as e:  # noqa: BLE001
                print("ERR", cfg, str(e).splitlines()[0][:100], flush=True)
        res.sort()
        floor = ref.speed_of_light_ns(m, layer, {"hbm_bytes_per_s": 4.8e12, "bf16_dense_flops_per_s": 989e12}) / 1e3
        print(f"== m={m} {layer} floor {floor:.1f}: " + "  ".join(f"{t:.2f}us {c} ctas={nc}" for t, c, nc in res[:4]),
              flush=True)


if __name__ == "__main__":
    main()

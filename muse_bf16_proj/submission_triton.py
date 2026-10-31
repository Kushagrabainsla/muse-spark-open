# okc: problem=muse_bf16_proj
"""Dense projections as a split-K skinny GEMM in Triton.

Up to 32 tokens the projections are bound by streaming the weights, so each program reads a
[BN, K/S] slab of W once and multiplies it by the matching slice of x (W is the dot's
left operand, so the token count only has to fill the dot's narrow side). Split-K partials
go to an FP32 workspace; the last program of each column tile (a counter, reset by that
program, so CUDA graph replays stay correct) sums them in a fixed order and writes the
output. The router runs as a BF16 hi + lo pair of its FP32 weights with FP32 accumulation.
Above 32 tokens it falls back to cuBLAS, as SGLang does.
"""

from typing import TypedDict

import torch
import triton
import triton.language as tl


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int      # output width; the unpooler's is 1024 up to 8 tokens (this GPU's slice), else 8192
    k: int
    layer: str  # which weight: "qkv_gate", "o", "pooler", "unpooler" or "router"


class Inputs(TypedDict):
    x: torch.Tensor  # bfloat16 [m, k]
    config: Config


SKINNY_MAX_M = 32
NUM_SMS = 132
WS_FLOATS = 1 << 24
MAX_TILES = 1024


@triton.jit
def _skinny_kernel(x_ptr, w_ptr, wlo_ptr, out_ptr, ws_ptr, cnt_ptr, M, N, K,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, S: tl.constexpr,
                   HAS_LO: tl.constexpr, OUT_F32: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    kblocks = K // BK
    kb0 = pid_s * kblocks // S
    kb1 = (pid_s + 1) * kblocks // S

    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_m = tl.arange(0, BM)
    offs_k = kb0 * BK + tl.arange(0, BK)
    w_ptrs = w_ptr + offs_n[:, None] * K + offs_k[None, :]
    wlo_ptrs = wlo_ptr + offs_n[:, None] * K + offs_k[None, :]
    x_ptrs = x_ptr + offs_m[None, :] * K + offs_k[:, None]
    m_mask = offs_m[None, :] < M

    acc = tl.zeros((BN, BM), dtype=tl.float32)
    for _ in range(kb0, kb1):
        xt = tl.load(x_ptrs, mask=m_mask, other=0.0)
        w = tl.load(w_ptrs, eviction_policy="evict_first")
        acc = tl.dot(w, xt, acc)
        if HAS_LO:
            wl = tl.load(wlo_ptrs, eviction_policy="evict_first")
            acc = tl.dot(wl, xt, acc)
            wlo_ptrs += BK
        w_ptrs += BK
        x_ptrs += BK

    # acc[j, i] is out[i, pid_n * BN + j].
    out_ptrs = out_ptr + offs_m[None, :] * N + offs_n[:, None]
    if S == 1:
        if OUT_F32:
            tl.store(out_ptrs, acc, mask=m_mask)
        else:
            tl.store(out_ptrs, acc.to(tl.bfloat16), mask=m_mask)
    else:
        tile_ws = offs_m[None, :] * N + offs_n[:, None]
        tl.store(ws_ptr + pid_s * BM * N + tile_ws, acc, mask=m_mask)
        tl.debug_barrier()
        arrived = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel")
        if arrived == S - 1:
            tl.debug_barrier()
            total = tl.zeros((BN, BM), dtype=tl.float32)
            for s in range(S):
                total += tl.load(ws_ptr + s * BM * N + tile_ws, mask=m_mask, other=0.0, cache_modifier=".cg")
            if OUT_F32:
                tl.store(out_ptrs, total, mask=m_mask)
            else:
                tl.store(out_ptrs, total.to(tl.bfloat16), mask=m_mask)
            tl.atomic_xchg(cnt_ptr + pid_n, 0)


def _config(n: int, k: int, m: int, has_lo: bool):
    """(BM, BN, BK, S, num_warps, num_stages) for an [m, k] x [n, k]^T product, m <= 32."""
    bm = 16 if m <= 16 else 32
    bn = 64 if m <= 16 else 128
    while bn > 32 and n % bn:
        bn //= 2
    bk = 128 if bn <= 64 else 64
    if has_lo:
        bn = 32
        bk = 128
    tiles = n // bn
    kblocks = k // bk
    # Aim for two programs per SM, each with at least two K blocks.
    s = max(1, min(kblocks // 2, round(2 * NUM_SMS / tiles)))
    return bm, bn, bk, s, 4, 4


def _launch(x, w, wlo, out, state):
    m, k = x.shape
    n = w.shape[0]
    bm, bn, bk, s, warps, stages = _config(n, k, m, wlo is not None)
    grid = (n // bn, s)
    _skinny_kernel[grid](x, w, w if wlo is None else wlo, out, state["ws"], state["cnt"], m, n, k,
                         BM=bm, BN=bn, BK=bk, S=s, HAS_LO=wlo is not None,
                         OUT_F32=out.dtype == torch.float32, num_warps=warps, num_stages=stages)
    return out


def prepare(weights: Weights):
    state = {name: w.contiguous() for name, w in weights.items()}
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hi"] = hi.contiguous()
    state["router_lo"] = lo.contiguous()
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    dev = router.device
    state["ws"] = torch.empty(WS_FLOATS, dtype=torch.float32, device=dev)
    state["cnt"] = torch.zeros(MAX_TILES, dtype=torch.int32, device=dev)
    # Compile every configuration up front, outside any graph capture.
    for layer, w in weights.items():
        k = w.shape[1]
        for m in (1, 5, 8, 9, 16, 32):
            n = 1024 if layer == "unpooler" and m <= 8 else w.shape[0]
            x = torch.zeros((m, k), dtype=torch.bfloat16, device=dev)
            kernel_fn({"x": x, "config": {"m": m, "n": n, "k": k, "layer": layer}}, state)
    torch.cuda.synchronize()
    return state


def kernel_fn(data: Inputs, state) -> torch.Tensor:
    x, cfg = data["x"], data["config"]
    m, n, layer = cfg["m"], cfg["n"], cfg["layer"]
    if layer == "router":
        if m <= SKINNY_MAX_M:
            out = torch.empty((m, n), dtype=torch.float32, device=x.device)
            return _launch(x, state["router_hi"], state["router_lo"], out, state)
        acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
        acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
        return acc
    w = state[layer][:n]
    if m <= SKINNY_MAX_M:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        return _launch(x, w, None, out, state)
    return x @ w.T

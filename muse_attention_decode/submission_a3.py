# okc: problem=muse_attention_decode
"""SGLang's attention for Muse Spark Open (FA3 backend, FP8 cache), with the five kernels before FA3 fused into
one Triton kernel.

SGLang runs, before flash_attn_with_kvcache: the fused QK-norm (with RoPE on sliding layers), k and v casts to
E4M3, the cache store, and the query cast to E4M3. Here one program per token does all of it: per-head RMSNorm
of q and k, interleaved RoPE (pairs (2i, 2i + 1)) on sliding layers, the query rounded to E4M3 (normalized,
before the query factor, which FA3 applies through the softmax scale, as SGLang does), and k and v divided by
the cache scales, rounded to E4M3 and stored at the token's cache row. FA3 and the gated output norm are
SGLang's.
"""

import math
from typing import TypedDict

import torch
import triton
import triton.language as tl
from sgl_kernel.flash_attn import flash_attn_with_kvcache

from sglang.kernels.ops.muse_spark_v12.norm_gate import rms_norm_gate


class Config(TypedDict):
    batch: int            # B
    q_len: int            # L: 1 for decode, 8 for a speculative verify block (causal inside it)
    num_q_heads: int      # Hq: 16 per GPU under TP8
    num_kv_heads: int     # Hkv: 2; each serves Hq / Hkv query heads
    head_dim: int         # 64
    page_size: int        # 1 in the scored cases (as SGLang serves Muse on H200), 64 in some tests
    window: int           # 2048 on sliding-window layers (a token and the 2,047 before it); 0 = all
    rope_theta: float     # 500000 on sliding-window layers; 0 = no RoPE
    qk_scale_factor: float
    sm_scale: float       # 1/8
    kv_scale: float       # the cache scales' value: 1 in the scored cases (SGLang leaves them unset)


class Inputs(TypedDict):
    qkv: torch.Tensor          # bfloat16 [B, L, (2*Hq + 2*Hkv)*64], 2304 under tp8: q (Hq*64), keys (Hkv*64),
                               # values (Hkv*64), the output gate's logits (Hq*64)
    positions: torch.Tensor    # int64 [B*L], each new token's position
    cache_loc: torch.Tensor    # int64 [B*L], each new token's flat cache row
    cache_seqlens: torch.Tensor  # int32 [B], seq_lens + L
    cu_seqlens_q: torch.Tensor   # int32 [B+1], 0, L, 2L, ...
    scheduler_metadata: torch.Tensor | None  # int32, FA3's tile schedule on full-attention decode steps
    k_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, Hkv, 64], written in place
    v_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, Hkv, 64], written in place
    k_scale: torch.Tensor     # float32 [], key = e4m3 * k_scale
    v_scale: torch.Tensor     # float32 [], value = e4m3 * v_scale
    page_table: torch.Tensor  # int32 [B, max_pages], physical page of each logical page (-1 unused)
    seq_lens: torch.Tensor    # int32 [B], cached tokens per request; the block starts at this position
    config: Config


RMS_EPS = 1e-5
FP8 = torch.float8_e4m3fn


@triton.jit
def _rms_rope(x, pos, theta_log2, HAS_ROPE: tl.constexpr, R: tl.constexpr, D: tl.constexpr):
    """x [R, D] FP32: RMSNorm over each row (head), then interleaved RoPE on (2i, 2i + 1)."""
    x = x * tl.rsqrt(tl.sum(x * x, axis=1) / D + 1e-5)[:, None]
    if HAS_ROPE:
        xe, xo = tl.split(tl.reshape(x, (R, D // 2, 2)))
        i = tl.arange(0, D // 2).to(tl.float32)
        ang = pos.to(tl.float32) * tl.exp2(-(2.0 * i / D) * theta_log2)
        c = tl.cos(ang)
        s = tl.sin(ang)
        x = tl.reshape(tl.join(xe * c[None, :] - xo * s[None, :], xe * s[None, :] + xo * c[None, :]), (R, D))
    return x


@triton.jit
def _qkv_prep(qkv_ptr, pos_ptr, loc_ptr, ks_ptr, vs_ptr, q8_ptr, kc_ptr, vc_ptr, theta_log2,
              HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, QKV_W: tl.constexpr, HAS_ROPE: tl.constexpr):
    t = tl.program_id(0)
    pos = tl.load(pos_ptr + t)
    dn = tl.arange(0, D)
    hq = tl.arange(0, HQ)
    q = tl.load(qkv_ptr + t * QKV_W + hq[:, None] * D + dn[None, :]).to(tl.float32)
    q = _rms_rope(q, pos, theta_log2, HAS_ROPE, HQ, D)
    tl.store(q8_ptr + t * HQ * D + hq[:, None] * D + dn[None, :], q.to(tl.float8e4nv))
    hk = tl.arange(0, HKV)
    k = tl.load(qkv_ptr + t * QKV_W + (HQ + hk)[:, None] * D + dn[None, :]).to(tl.float32)
    k = _rms_rope(k, pos, theta_log2, HAS_ROPE, HKV, D)
    v = tl.load(qkv_ptr + t * QKV_W + (HQ + HKV + hk)[:, None] * D + dn[None, :]).to(tl.float32)
    row = tl.load(loc_ptr + t) * HKV
    tl.store(kc_ptr + (row + hk)[:, None] * D + dn[None, :], (k / tl.load(ks_ptr)).to(tl.float8e4nv))
    tl.store(vc_ptr + (row + hk)[:, None] * D + dn[None, :], (v / tl.load(vs_ptr)).to(tl.float8e4nv))


def kernel_fn(data: Inputs) -> torch.Tensor:
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    tokens = batch * length
    qkv = data["qkv"].view(tokens, -1)
    theta = cfg["rope_theta"]
    q8 = torch.empty((tokens, hq, d), dtype=FP8, device=qkv.device)
    k_cache, v_cache = data["k_cache"], data["v_cache"]
    _qkv_prep[(tokens,)](qkv, data["positions"], data["cache_loc"], data["k_scale"], data["v_scale"], q8, k_cache,
                         v_cache, math.log2(theta) if theta else 0.0, HQ=hq, HKV=hkv, D=d, QKV_W=qkv.shape[1],
                         HAS_ROPE=bool(theta), num_warps=4)
    window = cfg["window"]
    descale = {}
    if cfg["kv_scale"] != 1.0:
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, hkv),
                       v_descale=data["v_scale"].reshape(1, 1).expand(batch, hkv))
    out = flash_attn_with_kvcache(
        q=q8,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=data["page_table"],
        cache_seqlens=data["cache_seqlens"],
        cu_seqlens_q=data["cu_seqlens_q"],
        max_seqlen_q=length,
        # The 5.473 query factor is folded into the softmax scale, as SGLang does.
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"],
        causal=True,
        window_size=(window - 1, 0) if window else (-1, -1),
        scheduler_metadata=data["scheduler_metadata"],
        **descale,
    )
    if isinstance(out, tuple):
        out = out[0]
    gate = qkv[:, (hq + 2 * hkv) * d:]
    o = rms_norm_gate(out.reshape(tokens, hq * d).to(torch.bfloat16), gate, None, head_dim=d, eps=RMS_EPS,
                      has_norm=True)
    return o.view(batch, length, hq * d)

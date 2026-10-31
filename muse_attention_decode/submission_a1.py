# okc: problem=muse_attention_decode
"""Gated attention decode as two Triton kernels (SGLang runs six).

  attn_chunk   grid (request, KV head, key chunk). Each program normalizes its query group (the Hq / Hkv
               query heads of the KV head, for every token of the block) from the fused projection, applies
               RoPE on sliding layers, and folds in the softmax and query scale factors (queries stay BF16).
               It then runs an online softmax over its chunk of the visible keys: past keys are read from
               the paged FP8 cache; the block's new keys and values are normalized, rotated and rounded to
               FP8 in place of a cache read, and the program that covers them also writes them to the cache.
               Partial (max, sum, unnormalized output) go to a workspace.
  attn_combine grid (request, KV head): merges the chunks and applies the per-head RMSNorm times
               sigmoid(gate).

RoPE pairs are (2i, 2i + 1): split and re-joined in registers, so cache rows load as contiguous 64 bytes.
Every visible key is attended (the whole window on sliding
layers, the whole context on full ones).
"""

from typing import TypedDict

import torch
import triton
import triton.language as tl


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
LOG2E = 1.4426950408889634
# Per layer kind: keys per program (chunk), keys per inner step, warps, pipeline stages.
TUNE = {"sliding": (512, 64, 4, 2), "full": (1024, 64, 4, 2)}


@triton.jit
def _rms_rope(x, pos, theta_log2, HAS_ROPE: tl.constexpr, R: tl.constexpr, D: tl.constexpr):
    """x [R, D] FP32 in natural column order: RMSNorm over the head, then interleaved RoPE on (2i, 2i + 1)."""
    r = tl.rsqrt(tl.sum(x * x, axis=1) / D + 1e-5)
    x = x * r[:, None]
    if HAS_ROPE:
        xe, xo = tl.split(tl.reshape(x, (R, D // 2, 2)))
        i = tl.arange(0, D // 2).to(tl.float32)
        inv_freq = tl.exp2(-(2.0 * i / D) * theta_log2)
        ang = pos.to(tl.float32)[:, None] * inv_freq[None, :]
        c = tl.cos(ang)
        s = tl.sin(ang)
        x = tl.reshape(tl.join(xe * c - xo * s, xe * s + xo * c), (R, D))
    return x


@triton.jit
def _attn_chunk(qkv_ptr, kc_ptr, vc_ptr, pt_ptr, seq_ptr, loc_ptr, ks_ptr, vs_ptr,
                ws_o, ws_m, ws_l, q_mult, theta_log2,
                HKV: tl.constexpr, G: tl.constexpr, L: tl.constexpr, D: tl.constexpr, ROWS: tl.constexpr,
                QKV_W: tl.constexpr, PAGE: tl.constexpr, MAX_PAGES, WINDOW: tl.constexpr,
                HAS_ROPE: tl.constexpr, CH: tl.constexpr, BN: tl.constexpr, NCH):
    bh = tl.program_id(0)
    c = tl.program_id(1)
    b = bh // HKV
    h = bh % HKV
    seq = tl.load(seq_ptr + b)
    hi = seq + L  # keys 0 .. hi - 1 exist; the block's tokens are seq .. hi - 1
    if WINDOW > 0:
        lo = tl.maximum(seq - (WINDOW - 1), 0)
    else:
        lo = 0
    cs = lo + c * CH
    ce = tl.minimum(cs + CH, hi)
    row_base = (bh * NCH + c) * ROWS

    rows = tl.arange(0, ROWS)
    j = rows // G  # block token of each query row
    g = rows % G
    rvalid = rows < G * L
    dn = tl.arange(0, D)
    if cs >= hi:
        tl.store(ws_m + row_base + rows, tl.full((ROWS,), float("-inf"), tl.float32))
        tl.store(ws_l + row_base + rows, tl.zeros((ROWS,), tl.float32))
        return

    # Queries [ROWS, D]: normalized, rotated, scaled (softmax scale, query factor and log2 e folded in).
    tok = b * L + j
    qp = qkv_ptr + tok[:, None] * QKV_W + (h * G + g)[:, None] * D
    q = tl.load(qp + dn[None, :], mask=rvalid[:, None], other=0.0).to(tl.float32)
    q = (_rms_rope(q, seq + j, theta_log2, HAS_ROPE, ROWS, D) * q_mult).to(tl.bfloat16)
    qpos = seq + j

    k_scale = tl.load(ks_ptr)
    v_scale = tl.load(vs_ptr)
    m_i = tl.full((ROWS,), float("-inf"), tl.float32)
    l_i = tl.zeros((ROWS,), tl.float32)
    acc = tl.zeros((ROWS, D), tl.float32)
    for kb in range(cs, ce, BN):
        pos = kb + tl.arange(0, BN)
        kvalid = pos < ce
        past = kvalid & (pos < seq)
        page = tl.load(pt_ptr + b * MAX_PAGES + pos // PAGE, mask=past, other=0)
        krow = (page.to(tl.int64) * PAGE + pos % PAGE) * HKV + h
        kk = tl.load(kc_ptr + krow[:, None] * D + dn[None, :], mask=past[:, None], other=0.0).to(tl.float32)
        vv = tl.load(vc_ptr + krow[:, None] * D + dn[None, :], mask=past[:, None], other=0.0).to(tl.float32)
        if kb + BN > seq:
            # The block's own keys and values: from the projection, as they are stored (FP8), and stored.
            new = kvalid & (pos >= seq)
            nt = b * L + (pos - seq)
            nk = tl.load(qkv_ptr + nt[:, None] * QKV_W + (HKV * G + h) * D + dn[None, :],
                         mask=new[:, None], other=0.0).to(tl.float32)
            nk8 = (_rms_rope(nk, pos, theta_log2, HAS_ROPE, BN, D) / k_scale).to(tl.float8e4nv)
            nv = tl.load(qkv_ptr + nt[:, None] * QKV_W + (HKV * G + HKV + h) * D + dn[None, :],
                         mask=new[:, None], other=0.0).to(tl.float32)
            nv8 = (nv / v_scale).to(tl.float8e4nv)
            loc = tl.load(loc_ptr + nt, mask=new, other=0)
            crow = loc * HKV + h
            tl.store(kc_ptr + crow[:, None] * D + dn[None, :], nk8, mask=new[:, None])
            tl.store(vc_ptr + crow[:, None] * D + dn[None, :], nv8, mask=new[:, None])
            kk = tl.where(new[:, None], nk8.to(tl.float32), kk)
            vv = tl.where(new[:, None], nv8.to(tl.float32), vv)
        kk = (kk * k_scale).to(tl.bfloat16)
        vv = (vv * v_scale).to(tl.bfloat16)
        s = tl.dot(q, tl.trans(kk))  # [ROWS, BN], log2 units
        vis = kvalid[None, :] & (pos[None, :] <= qpos[:, None]) & rvalid[:, None]
        if WINDOW > 0:
            vis = vis & (pos[None, :] > qpos[:, None] - WINDOW)
        s = tl.where(vis, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp2(s - m_safe[:, None])
        alpha = tl.exp2(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vv)
        m_i = m_new
    tl.store(ws_m + row_base + rows, m_i)
    tl.store(ws_l + row_base + rows, l_i)
    tl.store(ws_o + (row_base + rows)[:, None] * D + dn[None, :], acc)


@triton.jit
def _attn_combine(qkv_ptr, out_ptr, ws_o, ws_m, ws_l, NCH,
                  HKV: tl.constexpr, G: tl.constexpr, L: tl.constexpr, D: tl.constexpr, ROWS: tl.constexpr,
                  QKV_W: tl.constexpr, HQ: tl.constexpr):
    bh = tl.program_id(0)
    b = bh // HKV
    h = bh % HKV
    rows = tl.arange(0, ROWS)
    dn = tl.arange(0, D)
    rvalid = rows < G * L
    m = tl.full((ROWS,), float("-inf"), tl.float32)
    for c in range(0, NCH):
        m = tl.maximum(m, tl.load(ws_m + (bh * NCH + c) * ROWS + rows))
    m_safe = tl.where(m == float("-inf"), 0.0, m)
    l = tl.zeros((ROWS,), tl.float32)
    acc = tl.zeros((ROWS, D), tl.float32)
    for c in range(0, NCH):
        base = (bh * NCH + c) * ROWS
        mc = tl.load(ws_m + base + rows)
        w = tl.exp2(mc - m_safe)
        live = mc != float("-inf")  # chunks past a request's keys left their outputs unwritten
        w = tl.where(live, w, 0.0)
        l += w * tl.where(live, tl.load(ws_l + base + rows), 0.0)
        acc += w[:, None] * tl.load(ws_o + (base + rows)[:, None] * D + dn[None, :], mask=live[:, None], other=0.0)
    o = acc / tl.where(l > 0, l, 1.0)[:, None]
    o = o.to(tl.bfloat16).to(tl.float32)  # the attention output is BF16 before the norm, as in SGLang
    r = tl.rsqrt(tl.sum(o * o, axis=1) / D + 1e-5)
    j = rows // G
    g = rows % G
    tok = b * L + j
    head = h * G + g
    gate = tl.load(qkv_ptr + tok[:, None] * QKV_W + (HQ + 2 * HKV) * D + head[:, None] * D + dn[None, :],
                   mask=rvalid[:, None], other=0.0).to(tl.float32)
    y = o * r[:, None] * tl.sigmoid(gate)
    tl.store(out_ptr + tok[:, None] * (HQ * D) + head[:, None] * D + dn[None, :], y.to(tl.bfloat16),
             mask=rvalid[:, None])


def kernel_fn(data: Inputs) -> torch.Tensor:
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    group = hq // hkv
    rows = max(16, triton.next_power_of_2(group * length))
    qkv = data["qkv"]
    qkv_w = qkv.shape[-1]
    page = cfg["page_size"]
    pt = data["page_table"]
    window = cfg["window"]
    ch, bn, warps, stages = TUNE["sliding" if window else "full"]
    span = (window - 1 + length) if window else pt.shape[1] * page  # keys a block can see, at most
    nch = triton.cdiv(span, ch)
    dev = qkv.device
    ws_o = torch.empty((batch * hkv * nch * rows, d), dtype=torch.float32, device=dev)
    ws_m = torch.empty((batch * hkv * nch * rows,), dtype=torch.float32, device=dev)
    ws_l = torch.empty_like(ws_m)
    theta = cfg["rope_theta"]
    q_mult = cfg["qk_scale_factor"] * cfg["sm_scale"] * LOG2E
    import math
    theta_log2 = math.log2(theta) if theta else 0.0
    _attn_chunk[(batch * hkv, nch)](
        qkv, data["k_cache"], data["v_cache"], pt, data["seq_lens"], data["cache_loc"], data["k_scale"],
        data["v_scale"], ws_o, ws_m, ws_l, q_mult, theta_log2,
        HKV=hkv, G=group, L=length, D=d, ROWS=rows, QKV_W=qkv_w, PAGE=page, MAX_PAGES=pt.shape[1],
        WINDOW=window, HAS_ROPE=bool(theta), CH=ch, BN=bn, NCH=nch, num_warps=warps, num_stages=stages)
    out = torch.empty((batch, length, hq * d), dtype=torch.bfloat16, device=dev)
    _attn_combine[(batch * hkv,)](qkv, out, ws_o, ws_m, ws_l, nch, HKV=hkv, G=group, L=length, D=d, ROWS=rows,
                                  QKV_W=qkv_w, HQ=hq, num_warps=4)
    return out

# okc: problem=muse_attention_decode
"""Gated attention decode as two Triton kernels (SGLang runs six).

  attn_chunk   grid (request, pair of KV heads, key chunk): both heads' cache rows are adjacent, so a key
               costs one page-table entry and one 128-byte read. Each program normalizes its queries (the
               query heads of its KV heads, for every token of the block) from the fused projection, applies
               RoPE on sliding layers, and folds in the softmax and query scale factors (queries stay BF16).
               It then runs an online softmax over its chunk of the visible keys: past keys are read from
               the paged FP8 cache; the block's new keys and values are normalized, rotated and rounded to
               FP8 in place of a cache read, and the program that covers them also writes them to the cache.
               Partial (max, sum, unnormalized output) go to a workspace.
  attn_combine one program per query row: merges the chunks and applies the per-head RMSNorm times
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
                HKV: tl.constexpr, KVPC: tl.constexpr, G: tl.constexpr, L: tl.constexpr, D: tl.constexpr,
                RH: tl.constexpr, QKV_W: tl.constexpr, PAGE: tl.constexpr, MAX_PAGES, WINDOW: tl.constexpr,
                HAS_ROPE: tl.constexpr, CH: tl.constexpr, BN: tl.constexpr, NCH):
    """One (request, group of KVPC KV heads, key chunk). Query rows: KVPC blocks of RH rows (head hh, then
    token j = r // G and query head g = r % G of the group). The KVPC heads' cache rows are adjacent, so each
    key costs one page-table entry and one contiguous KVPC * 64-byte read of keys (and of values)."""
    R: tl.constexpr = KVPC * RH
    bg = tl.program_id(0)
    c = tl.program_id(1)
    b = bg // (HKV // KVPC)
    h0 = (bg % (HKV // KVPC)) * KVPC
    seq = tl.load(seq_ptr + b)
    hi = seq + L  # keys 0 .. hi - 1 exist; the block's tokens are seq .. hi - 1
    if WINDOW > 0:
        lo = tl.maximum(seq - (WINDOW - 1), 0)
    else:
        lo = 0
    cs = lo + c * CH
    ce = tl.minimum(cs + CH, hi)
    row_base = (bg * NCH + c) * R

    rows = tl.arange(0, R)
    hh = rows // RH  # KV head of the row, within the group
    r = rows % RH
    j = r // G  # block token
    g = r % G
    rvalid = r < G * L
    dn = tl.arange(0, D)
    if cs >= hi:
        tl.store(ws_m + row_base + rows, tl.full((R,), float("-inf"), tl.float32))
        tl.store(ws_l + row_base + rows, tl.zeros((R,), tl.float32))
        return

    # Queries [R, D]: normalized, rotated, scaled (softmax scale, query factor and log2 e folded in).
    tok = b * L + j
    qp = qkv_ptr + tok[:, None] * QKV_W + ((h0 + hh) * G + g)[:, None] * D
    q = tl.load(qp + dn[None, :], mask=rvalid[:, None], other=0.0).to(tl.float32)
    q = (_rms_rope(q, seq + j, theta_log2, HAS_ROPE, R, D) * q_mult).to(tl.bfloat16)
    qpos = seq + j

    k_scale = tl.load(ks_ptr)
    v_scale = tl.load(vs_ptr)
    m_i = tl.full((R,), float("-inf"), tl.float32)
    l_i = tl.zeros((R,), tl.float32)
    acc = tl.zeros((R, D), tl.float32)
    for kb in range(cs, ce, BN):
        pos = kb + tl.arange(0, BN)
        kvalid = pos < ce
        past = kvalid & (pos < seq)
        page = tl.load(pt_ptr + b * MAX_PAGES + pos // PAGE, mask=past, other=0)
        prow = page.to(tl.int64) * PAGE + pos % PAGE
        new = kvalid & (pos >= seq)
        nt = b * L + (pos - seq)
        loc = tl.load(loc_ptr + nt, mask=new, other=0)
        vis = kvalid[None, :] & (pos[None, :] <= qpos[:, None]) & rvalid[:, None]
        if WINDOW > 0:
            vis = vis & (pos[None, :] > qpos[:, None] - WINDOW)
        for x in tl.static_range(KVPC):
            # Head h0 + x: its rows' online-softmax update (the other rows see p = 0, alpha = 1).
            h = h0 + x
            krow = prow * HKV + h
            kk = tl.load(kc_ptr + krow[:, None] * D + dn[None, :], mask=past[:, None], other=0.0).to(tl.float32)
            vv = tl.load(vc_ptr + krow[:, None] * D + dn[None, :], mask=past[:, None], other=0.0).to(tl.float32)
            if kb + BN > seq:
                # The block's own keys and values: from the projection, as they are stored (FP8), and stored.
                nk = tl.load(qkv_ptr + nt[:, None] * QKV_W + (HKV * G + h) * D + dn[None, :],
                             mask=new[:, None], other=0.0).to(tl.float32)
                nk8 = (_rms_rope(nk, pos, theta_log2, HAS_ROPE, BN, D) / k_scale).to(tl.float8e4nv)
                nv = tl.load(qkv_ptr + nt[:, None] * QKV_W + (HKV * G + HKV + h) * D + dn[None, :],
                             mask=new[:, None], other=0.0).to(tl.float32)
                nv8 = (nv / v_scale).to(tl.float8e4nv)
                crow = loc * HKV + h
                tl.store(kc_ptr + crow[:, None] * D + dn[None, :], nk8, mask=new[:, None])
                tl.store(vc_ptr + crow[:, None] * D + dn[None, :], nv8, mask=new[:, None])
                kk = tl.where(new[:, None], nk8.to(tl.float32), kk)
                vv = tl.where(new[:, None], nv8.to(tl.float32), vv)
            kk = (kk * k_scale).to(tl.bfloat16)
            vv = (vv * v_scale).to(tl.bfloat16)
            s = tl.dot(q, tl.trans(kk))  # [R, BN], log2 units
            s = tl.where(vis & (hh == x)[:, None], s, float("-inf"))
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
                  HKV: tl.constexpr, KVPC: tl.constexpr, G: tl.constexpr, L: tl.constexpr, D: tl.constexpr,
                  RH: tl.constexpr, QKV_W: tl.constexpr, HQ: tl.constexpr, CT: tl.constexpr):
    """One query row: merges its chunks (CT at a time), then RMSNorm over the head times sigmoid(gate)."""
    R: tl.constexpr = KVPC * RH
    pid = tl.program_id(0)
    bg = pid // R
    row = pid % R
    r = row % RH
    if r >= G * L:
        return
    b = bg // (HKV // KVPC)
    h = (bg % (HKV // KVPC)) * KVPC + row // RH
    j = r // G
    g = r % G
    dn = tl.arange(0, D)
    cc = tl.arange(0, CT)
    m = float("-inf")
    for c0 in range(0, NCH, CT):
        idx = ((bg * NCH + c0 + cc) * R + row)
        m = tl.maximum(m, tl.max(tl.load(ws_m + idx, mask=c0 + cc < NCH, other=float("-inf"))))
    m_safe = tl.where(m == float("-inf"), 0.0, m)
    l = 0.0
    acc = tl.zeros((D,), tl.float32)
    for c0 in range(0, NCH, CT):
        idx = ((bg * NCH + c0 + cc) * R + row)
        mc = tl.load(ws_m + idx, mask=c0 + cc < NCH, other=float("-inf"))
        live = mc != float("-inf")  # chunks past a request's keys left their outputs unwritten
        w = tl.where(live, tl.exp2(mc - m_safe), 0.0)
        l += tl.sum(w * tl.load(ws_l + idx, mask=live, other=0.0))
        acc += tl.sum(w[:, None] * tl.load(ws_o + idx[:, None] * D + dn[None, :], mask=live[:, None], other=0.0),
                      axis=0)
    o = acc / tl.where(l > 0, l, 1.0)
    o = o.to(tl.bfloat16).to(tl.float32)  # the attention output is BF16 before the norm, as in SGLang
    rr = tl.rsqrt(tl.sum(o * o) / D + 1e-5)
    tok = b * L + j
    head = h * G + g
    gate = tl.load(qkv_ptr + tok * QKV_W + (HQ + 2 * HKV) * D + head * D + dn).to(tl.float32)
    tl.store(out_ptr + tok * (HQ * D) + head * D + dn, (o * rr * tl.sigmoid(gate)).to(tl.bfloat16))


def kernel_fn(data: Inputs) -> torch.Tensor:
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    group = hq // hkv
    rh = max(16, triton.next_power_of_2(group * length))  # query rows per KV head
    kvpc = 2 if hkv % 2 == 0 and hkv <= 2 else 1  # KV heads per program
    rows = kvpc * rh
    qkv = data["qkv"]
    qkv_w = qkv.shape[-1]
    page = cfg["page_size"]
    pt = data["page_table"]
    window = cfg["window"]
    ch, bn, warps, stages = TUNE["sliding" if window else "full"]
    span = (window - 1 + length) if window else pt.shape[1] * page  # keys a block can see, at most
    nch = triton.cdiv(span, ch)
    dev = qkv.device
    groups = batch * (hkv // kvpc)
    ws_o = torch.empty((groups * nch * rows, d), dtype=torch.float32, device=dev)
    ws_m = torch.empty((groups * nch * rows,), dtype=torch.float32, device=dev)
    ws_l = torch.empty_like(ws_m)
    theta = cfg["rope_theta"]
    q_mult = cfg["qk_scale_factor"] * cfg["sm_scale"] * LOG2E
    import math
    theta_log2 = math.log2(theta) if theta else 0.0
    _attn_chunk[(groups, nch)](
        qkv, data["k_cache"], data["v_cache"], pt, data["seq_lens"], data["cache_loc"], data["k_scale"],
        data["v_scale"], ws_o, ws_m, ws_l, q_mult, theta_log2,
        HKV=hkv, KVPC=kvpc, G=group, L=length, D=d, RH=rh, QKV_W=qkv_w, PAGE=page, MAX_PAGES=pt.shape[1],
        WINDOW=window, HAS_ROPE=bool(theta), CH=ch, BN=bn, NCH=nch, num_warps=warps, num_stages=stages)
    out = torch.empty((batch, length, hq * d), dtype=torch.bfloat16, device=dev)
    _attn_combine[(groups * rows,)](qkv, out, ws_o, ws_m, ws_l, nch, HKV=hkv, KVPC=kvpc, G=group, L=length, D=d,
                                    RH=rh, QKV_W=qkv_w, HQ=hq, CT=64, num_warps=2)
    return out

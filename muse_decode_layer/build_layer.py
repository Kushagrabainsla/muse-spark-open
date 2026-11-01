"""Build muse_decode_layer/submission.py from SGLang's layer (baseline.py) with our kernels swapped in:

  experts                                   muse_moe_w4a16/submission.py (NVFP4 grouped GEMMs)
  qkv_gate, o, pooler, unpooler, router>8   muse_bf16_proj/submission.py (skinny GEMMs, cuBLAS fallback)

Each kernel file's top-level names get a prefix (_proj_, _moe_) so they can live in one file.
python build_layer.py
"""

import re

ROOT = __file__.rsplit("/", 2)[0]
HERE = ROOT + "/muse_decode_layer"


def body(path):
    """A kernel file without its okc header, docstring, TypedDicts and its own imports."""
    s = open(path).read()
    s = s.split("\n", 1)[1]  # okc header
    s = re.sub(r'^""".*?"""\n', "", s, count=1, flags=re.S)
    s = re.sub(r"^from typing import TypedDict\n", "", s, flags=re.M)
    s = re.sub(r"^class (Weights|Config|Inputs)\(TypedDict\):\n(    .*\n)+\n*", "", s, flags=re.M)
    return s


def prefix(s, names, pre):
    for n in sorted(names, key=len, reverse=True):
        s = re.sub(r"(?<![\w.\"'])" + re.escape(n) + r"\b", pre + n.lstrip("_"), s)
    return s


def top_names(s):
    names = set(re.findall(r"^(?:def|class) (\w+)", s, flags=re.M))
    names |= set(re.findall(r"^(\w+) *=", s, flags=re.M))
    return names - {"torch", "triton", "tl"}


proj = body(ROOT + "/muse_bf16_proj/submission.py")
attn = body(ROOT + "/muse_attention_decode/submission.py")
attn = prefix(attn, top_names(attn), "_attn_")
attn = re.sub(r'"muse_attn_decode\w*"', '"muse_layer_attn_decode"', attn)
assert '"muse_layer_attn_decode"' in attn
moe = body(ROOT + "/muse_moe_w4a16/submission.py")
proj = prefix(proj, top_names(proj), "_proj_")
moe = prefix(moe, top_names(moe), "_moe_")
# load_inline module names must differ, and stay distinct from the standalone problems' builds.
proj = proj.replace('"muse_bf16_skinny"', '"muse_layer_bf16_skinny"')
moe = moe.replace('"muse_moe_nvfp4"', '"muse_layer_moe_nvfp4"')

base = open(HERE + "/baseline.py").read()
doc_end = base.index('"""', 3) + 3
header = '''# okc: problem=muse_decode_layer
"""SGLang's decoder layer for Muse Spark Open (see SGLang's MuseSpark1xDecoderLayer.forward) with two of its
module groups replaced by our kernels; everything else is SGLang's, as in baseline.py:

  experts      NVFP4 grouped GEMMs in CUDA (as submitted to muse_moe_w4a16): only the routed (token, local
               expert) pairs, each touched expert's weights streamed once, SiLU and the post-expert norm
               and weighted sum fused, FP32 split-K accumulation.
  projections  qkv_gate, o, pooler and unpooler (and the router above 8 tokens, on the BF16 hi + lo split
               SGLang uses there) as cluster split-K skinny GEMMs in CUDA up to 32 tokens (as submitted to
               muse_bf16_proj), cuBLAS above, as SGLang.
  attention    as submitted to muse_attention_decode: on sliding decode steps and sliding verify blocks of up to
               16 requests one CUDA launch (prep, split-KV attention over the paged FP8 cache, the chunks merged
               in a cluster, the gated output norm); on full decode steps a fused Triton prep, the CUDA attention
               and a Triton combine with the gated norm; elsewhere the fused prep, SGLang's FA3 and norm-gate.
Up to 8 tokens the router keeps SGLang's FP32 input and weights.
"""
'''
imports = "\nfrom __future__ import annotations\n" + base[doc_end:base.index("class Weights(TypedDict):")]
imports = imports.replace("from typing import TypedDict\n", "import math\nfrom typing import TypedDict\n\nimport triton\nimport triton.language as tl\nfrom torch.utils.cpp_extension import load_inline\n")
rest = base[base.index("class Weights(TypedDict):"):]

# prepare(): build both kernels' states.
rest = rest.replace('''    del freqs
    torch.cuda.synchronize()
    return state''', '''    del freqs
    state["proj"] = _proj_prepare({name: weights[name] for name in ("qkv_gate", "o", "pooler", "unpooler", "router")})
    state["moe"] = _moe_prepare({"w13": weights["w13"], "w13_sf": weights["w13_sf"], "w13_gs": weights["w13_gs"],
                                 "w2": weights["w2"], "w2_sf": weights["w2_sf"], "w2_gs": weights["w2_gs"],
                                 "post_norm_w": weights["post_expert_norm"]})
    # The pre-expert RMSNorm folded into the pooler and gemm1 (up to SKINNY_MAX_M tokens): per-token sums of squares
    # (zero between calls: the MoE's norm kernel re-zeroes them) and the FP32 (1 + w).
    state["pre_ss"] = torch.zeros(256, dtype=torch.float32, device=device)
    state["pre_w"] = (1.0 + weights["pre_expert_norm"].float()).contiguous()
    # Up to SLICED_UNPOOL_MAX_TOKENS the unpooler writes its slice into these rows; the rest stays zero (the
    # all-gather's other GPUs' columns), so no pad kernel runs.
    state["unpool_pad"] = torch.zeros((SLICED_UNPOOL_MAX_TOKENS, HIDDEN), dtype=torch.bfloat16, device=device)
    torch.cuda.synchronize()
    return state''')
assert "_proj_prepare" in rest
# The Marlin bank is no longer used.
rest = rest.replace('        "bank": _marlin_bank(weights),\n', "")


def proj_call(x, name, n, state_expr):
    return (f'_proj_kernel_fn({{"x": {x}, "config": {{"m": {x}.shape[0], "n": {n}, "k": {x}.shape[1], '
            f'"layer": "{name}"}}}}, {state_expr})')


repl = [
    ('''    # This GPU's experts start at local_start: as SGLang's expert_map on that EP rank.
    local_ids = (topk_ids - data["local_start"]) % N_EXPERTS''',
     '''    # This GPU's experts start at local_start (as SGLang's expert_map on that EP rank): the MoE kernels read it
    # and take the global ids (no remap kernel).'''),
    ('    qkv = _gemm(data["attn_input"], dense["qkv_gate"])',
     '    qkv = ' + proj_call('data["attn_input"]', "qkv_gate", 'dense["qkv_gate"].shape[0]', 'state["proj"]')),
    ('''    if cfg["rope_theta"]:
        q, k = state["qk_norm"](qkv, data["positions"], state["cos_sin"])
    else:
        q, k = state["qk_norm"](qkv)
    v = qkv[:, hq * d + hkv * d: hq * d + 2 * hkv * d]
    gate = qkv[:, hq * d + 2 * hkv * d:]
    k_cache, v_cache = data["k_cache"], data["v_cache"]
    scaled = cfg["kv_scale"] != 1.0
    k = k.reshape(tokens, hkv * d)
    if scaled:
        k, v = k / data["k_scale"], v / data["v_scale"]
    # The pool keeps FP8 rows as bytes; slot 0 is a live row here (SGLang reserves it for padding).
    store_cache(k.to(FP8).view(torch.uint8), v.to(FP8).view(torch.uint8),
                k_cache.view(-1, hkv * d).view(torch.uint8), v_cache.view(-1, hkv * d).view(torch.uint8),
                data["cache_loc"], reserved_skip_index=-1)
    window = cfg["window"]
    descale = {}
    if scaled:
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, hkv),
                       v_descale=data["v_scale"].reshape(1, 1).expand(batch, hkv))
    out = flash_attn_with_kvcache(
        q=q.view(tokens, hq, d).to(FP8),
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=data["page_table"],
        cache_seqlens=data["cache_seqlens"],
        cu_seqlens_q=data["cu_seqlens_q"],
        max_seqlen_q=length,
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"],
        causal=True,
        window_size=(window - 1, 0) if window else (-1, -1),
        scheduler_metadata=data["scheduler_metadata"],
        **descale,
    )
    if isinstance(out, tuple):
        out = out[0]
    o = state["o_norm_gate"](out.reshape(tokens, hq * d).to(torch.bfloat16), gate)''',
     '''    # QK-norm, RoPE, the cache store, attention and the gated output norm, as submitted to muse_attention_decode.
    o = _attn_kernel_fn({"qkv": qkv.view(batch, length, -1), "positions": data["positions"],
                         "cache_loc": data["cache_loc"], "cache_seqlens": data["cache_seqlens"],
                         "cu_seqlens_q": data["cu_seqlens_q"], "scheduler_metadata": data["scheduler_metadata"],
                         "k_cache": data["k_cache"], "v_cache": data["v_cache"], "k_scale": data["k_scale"],
                         "v_scale": data["v_scale"], "page_table": data["page_table"], "seq_lens": data["seq_lens"],
                         "config": cfg}).view(tokens, hq * d)'''),
    ('    attn_output = _gemm(o, dense["o"])  # all-reduce: identity',
     '    attn_output = ' + proj_call("o", "o", 'dense["o"].shape[0]', 'state["proj"]') + '  # all-reduce: identity'),
    ('    expert_input = _gemm(ffn_input, dense["pooler"])',
     '''    # Where the pooler is the skinny kernel, it also sums each token's squares and gemm1 applies the pre-expert
    # RMSNorm as it stages x (one kernel fewer); elsewhere the norm runs on its own, as SGLang does.
    fold_norm = _proj_config("pooler", tokens) is not None
    expert_input = _proj_kernel_fn({"x": ffn_input, "config": {"m": ffn_input.shape[0], "n": dense["pooler"].shape[0],
                                    "k": ffn_input.shape[1], "layer": "pooler"},
                                    **({"ssq": state["pre_ss"]} if fold_norm else {})}, state["proj"])'''),
    ('''            logits = torch.mm(ffn_input, state["router_hi_t"], out_dtype=torch.float32)
            logits += torch.mm(ffn_input, state["router_lo_t"], out_dtype=torch.float32)''',
     '''            logits = ''' + proj_call("ffn_input", "router", "256", 'state["proj"]')),
    ('''    expert_output = _experts(state["pre_expert_norm"](expert_input), topk_weights, local_ids, state)''',
     '''    expert_output = _moe_kernel_fn({
        "hidden_states": expert_input if fold_norm else state["pre_expert_norm"](expert_input).contiguous(),
        "pre_norm": (state["pre_ss"], state["pre_w"], RMS_EPS) if fold_norm else None,
        "topk_ids": topk_ids.to(torch.int32).contiguous(),
        "local_start_dev": data["local_start"].to(torch.int32),
        "topk_weights": topk_weights.to(torch.float32).contiguous(),
        "config": {"bs": tokens, "d_hidden": D_EXPERT, "d_expert": D_EXPERT, "n_experts": N_EXPERTS,
                   "local_start": 0, "local_count": LOCAL_EXPERTS, "top_k": TOP_K, "sharding": "ep8"}}, state["moe"])'''),
    ('''        unpooled = _gemm(expert_output, dense["unpooler"][:UNPOOL_SLICE])
        ffn_output = F.pad(unpooled, (0, HIDDEN - UNPOOL_SLICE))  # all-gather: identity''',
     '''        ffn_output = state["unpool_pad"][:tokens]  # all-gather: identity (the other columns stay zero)
        _proj_kernel_fn({"x": expert_output, "config": {"m": expert_output.shape[0], "n": UNPOOL_SLICE,
                         "k": expert_output.shape[1], "layer": "unpooler"}, "out": ffn_output[:, :UNPOOL_SLICE]},
                        state["proj"])'''),
    ('''        ffn_output = _gemm(expert_output, dense["unpooler"])''',
     '''        ffn_output = ''' + proj_call("expert_output", "unpooler", 'dense["unpooler"].shape[0]', 'state["proj"]')),
]
for a, b in repl:
    assert a in rest, a
    rest = rest.replace(a, b)

out = header + imports + "\n# ---- attention (muse_attention_decode) ----\n" + attn + \
    "\n# ---- projections (muse_bf16_proj) ----\n" + proj + \
    "\n# ---- experts (muse_moe_w4a16) ----\n" + moe + "\n# ---- the layer (SGLang's, with the above swapped in) ----\n" + rest
open(HERE + "/submission.py", "w").write(out)
print("wrote", HERE + "/submission.py", len(out.splitlines()), "lines")

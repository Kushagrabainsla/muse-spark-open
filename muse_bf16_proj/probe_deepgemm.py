"""E7: DeepGEMM bf16_gemm_nt vs cuBLAS at the m >= 128 shapes (and the router, FP32 out, hi + lo)."""
import sys
import torch
import deep_gemm
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import reference as ref  # noqa: E402
from timing import kernel_us  # noqa: E402

weights = ref.generate_weights(ref.WEIGHTS_SEED)
hi = weights["router"].to(torch.bfloat16)
lo = (weights["router"] - hi.float()).to(torch.bfloat16)
for m in (16, 32, 128, 256):
    for layer in ("qkv_gate", "o", "pooler", "unpooler", "router"):
        data = ref.generate_input(m, layer, seed=3)
        n = data["config"]["n"]
        x = data["x"]
        if layer == "router":
            out = torch.empty((m, n), dtype=torch.float32, device="cuda")
            def dg(d, s):
                deep_gemm.bf16_gemm_nt(x, hi, out)
                deep_gemm.bf16_gemm_nt(x, lo, out, c=out)
                return out
            def cb(d, s):
                acc = torch.mm(x, hi.t(), out_dtype=torch.float32)
                acc += torch.mm(x, lo.t(), out_dtype=torch.float32)
                return acc
        else:
            w = weights[layer][:n].contiguous()
            out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
            def dg(d, s):
                deep_gemm.bf16_gemm_nt(x, w, out)
                return out
            def cb(d, s):
                return x @ w.T
        try:
            ok, msg = ref.check_implementation(data, dg(None, None).clone(), weights)
            t_dg = kernel_us(dg, None, None)
        except Exception as e:  # noqa: BLE001
            ok, msg, t_dg = False, str(e).splitlines()[0][:120], float("nan")
        t_cb = kernel_us(cb, None, None)
        print(f"m={m:4d} {layer:9s} cublas {t_cb:6.2f}  deepgemm {t_dg:6.2f}  x{t_cb / t_dg:5.3f}  {'ok' if ok else 'BAD ' + msg}", flush=True)

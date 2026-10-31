"""Sweep the CUDA kernel's chunk per (layer, batch) at q_len 1: python sweep_cuda.py"""
import importlib.util, sys
HERE = __file__.rsplit("/", 1)[0]
sys.path.insert(0, HERE)
import reference as ref  # noqa: E402
from timing import kernel_us  # noqa: E402
spec = importlib.util.spec_from_file_location("mine", HERE + "/submission.py")
mine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mine)
base = importlib.util.module_from_spec(importlib.util.spec_from_file_location("b", HERE + "/submission_a3.py"))
importlib.util.spec_from_file_location("b", HERE + "/submission_a3.py").loader.exec_module(base)
for layer in ("full", "sliding"):
    for b in ((16, 32) if layer == "full" else ()):
        d = ref.generate_input(batch=b, ctx=80000, heads="tp8", layer=layer, page=1, seed=5,
                               spread="agentic" if b > 1 else "fixed", dist="coherent" if layer == "full" else "soft")
        line = f"{layer:7s} b={b:2d}  fa3+prep {kernel_us(lambda x, s: base.kernel_fn(x), d, None):6.1f}  cuda:"
        for ch in ((1024, 2048, 4096, 8192) if layer == "full" else ()):
            mine.CHUNK[layer] = ch
            line += f"  {ch}:{kernel_us(lambda x, s: mine.kernel_fn(x), d, None):6.1f}"
        print(line, flush=True)

"""Sweep the skinny kernel's tile configs per shape: python sweep.py [--m=1,16,32] [--layer=o]"""

import importlib.util
import itertools
import sys

import torch

HERE = __file__.rsplit("/", 1)[0]
sys.path.insert(0, HERE)
import reference as ref  # noqa: E402
from timing import kernel_us  # noqa: E402

spec = importlib.util.spec_from_file_location("mine", HERE + "/submission.py")
mine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mine)

args = sys.argv[1:]
ms = [int(v) for v in next((a.split("=", 1)[1] for a in args if a.startswith("--m=")), "1,8,16,32").split(",")]
layers = next((a.split("=", 1)[1] for a in args if a.startswith("--layer=")), "qkv_gate,o,pooler,unpooler,router").split(",")

weights = ref.generate_weights(ref.WEIGHTS_SEED)
orig_config = mine._config
mine._config = lambda n, k, m, lo: orig_config(n, k, m, lo)
state = mine.prepare(weights)

for m, layer in itertools.product(ms, layers):
    data = ref.generate_input(m, layer, seed=7)
    n, k = data["config"]["n"], data["config"]["k"]
    lo = layer == "router"
    bm = 16 if m <= 16 else 32
    results = []
    default = orig_config(n, k, m, lo)
    for bn, bk, ctas, warps, stages in itertools.product((16, 32, 64, 128), (64, 128, 256),
                                                         (264, 528, 1056, 2112, 4224), (4,), (3, 4)):
        if n % bn or k % bk:
            continue
        tiles, kb = n // bn, k // bk
        s = max(1, min(kb, round(ctas / tiles)))
        if bn * bk * 2 * (2 if lo else 1) * stages + bm * bk * 2 * stages > 220_000:
            continue
        if bn * bm // (32 * warps) > 128:
            continue
        cfg = (bm, bn, bk, s, warps, stages)
        if any(r[1] == cfg for r in results):
            continue
        mine._config = lambda *_a, c=cfg: c
        try:
            out = mine.kernel_fn(data, state)
            ok, msg = ref.check_implementation(data, out, weights)
            if not ok:
                print("BAD", cfg, msg, flush=True)
                continue
            t = kernel_us(mine.kernel_fn, data, state)
        except Exception as e:  # noqa: BLE001
            print("ERR", cfg, str(e).splitlines()[0][:120], flush=True)
            continue
        results.append((t, cfg))
    results.sort()
    mine._config = lambda *_a, c=default: c
    td = kernel_us(mine.kernel_fn, data, state)
    floor = ref.speed_of_light_ns(m, layer, {"hbm_bytes_per_s": 4.8e12, "bf16_dense_flops_per_s": 989e12}) / 1e3
    print(f"== m={m} {layer} n={n} k={k} floor {floor:.1f}  default {default} {td:.2f}us", flush=True)
    for t, cfg in results[:6]:
        print(f"   {t:6.2f}us  {cfg}", flush=True)

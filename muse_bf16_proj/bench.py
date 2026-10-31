"""Local judge-like timing, mine vs SGLang's: python bench.py [file.py] [--check] [--m=1,16]"""

import importlib.util
import sys

import torch

HERE = __file__.rsplit("/", 1)[0]
sys.path.insert(0, HERE)
import reference as ref  # noqa: E402
from timing import kernel_us  # noqa: E402

SCORED = [(m, l) for m in (1, 16, 32, 8, 128, 256) for l in ("qkv_gate", "o", "pooler", "unpooler", "router")]
TESTS = [(1, "qkv_gate", "normal"), (7, "pooler", "normal"), (33, "o", "normal"), (130, "unpooler", "normal"),
         (5, "unpooler", "normal"), (8, "router", "normal"), (9, "router", "outliers"),
         (512, "qkv_gate", "outliers"), (1000, "unpooler", "outliers")]
HW = {"hbm_bytes_per_s": 4.8e12, "bf16_dense_flops_per_s": 989e12}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    args = sys.argv[1:]
    path = next((a for a in args if a.endswith(".py")), HERE + "/submission.py")
    mine = load(path, "mine")
    base = load(HERE + "/baseline.py", "base")
    weights = ref.generate_weights(ref.WEIGHTS_SEED)
    st_m = mine.prepare({k: v.clone() for k, v in weights.items()})
    st_b = base.prepare({k: v.clone() for k, v in weights.items()})

    if "--check" in args or "--only-check" in args:
        bad = 0
        for m, layer, dist in TESTS + [(m, l, "normal") for m, l in SCORED]:
            for seed in (1, 2):
                data = ref.generate_input(m, layer, seed=1000 * seed + m, dist=dist)
                ok, msg = ref.check_implementation(data, mine.kernel_fn(data, st_m), weights)
                if not ok:
                    bad += 1
                    print(f"FAIL m={m} {layer} {dist}: {msg}")
        print(f"checks done, {bad} failures")
        if "--only-check" in args:
            return

    shapes = SCORED
    sel = next((a.split("=", 1)[1] for a in args if a.startswith("--m=")), None)
    if sel:
        ms = {int(v) for v in sel.split(",")}
        shapes = [s for s in SCORED if s[0] in ms]
    loads = {}
    for m, layer in shapes:
        data = ref.generate_input(m, layer, seed=99)
        tb = kernel_us(base.kernel_fn, data, st_b)
        tm = kernel_us(mine.kernel_fn, data, st_m)
        floor = ref.speed_of_light_ns(m, layer, HW) / 1e3
        print(f"m={m:4d} {layer:9s} sglang {tb:6.1f}  mine {tm:6.1f}  x{tb / tm:5.3f}  floor {floor:5.1f}", flush=True)
        lb, lm = loads.get(m, (0, 0))
        loads[m] = (lb + tb, lm + tm)
    geo = 1.0
    for m, (lb, lm) in loads.items():
        print(f"  load m={m}: x{lb / lm:.3f}")
        geo *= lb / lm
    print(f"geomean over loads x{geo ** (1 / len(loads)):.3f}")


if __name__ == "__main__":
    main()

"""Local judge-like timing for muse_decode_layer, mine vs SGLang's: python bench.py [file.py] [--check] [--loads=decode 1,...]"""

import importlib.util
import sys

import torch

HERE = __file__.rsplit("/", 1)[0]
sys.path.insert(0, HERE)
import reference as ref  # noqa: E402
from timing import kernel_us  # noqa: E402

SCORED = [  # (load, weight, kwargs)
    ("decode 1", 46, dict(batch=1, ctx=80000, layer="sliding", page=1, dist="soft", seed=801)),
    ("decode 1", 16, dict(batch=1, ctx=80000, layer="full", page=1, dist="soft", seed=802)),
    ("decode 16", 46, dict(batch=16, ctx=80000, layer="sliding", page=1, spread="agentic", dist="soft", seed=803)),
    ("decode 16", 16, dict(batch=16, ctx=80000, layer="full", page=1, spread="agentic", dist="soft", seed=804)),
    ("decode 32", 46, dict(batch=32, ctx=80000, layer="sliding", page=1, spread="agentic", dist="soft", seed=805)),
    ("decode 32", 16, dict(batch=32, ctx=80000, layer="full", page=1, spread="agentic", dist="soft", seed=806)),
    ("verify 1", 46, dict(batch=1, ctx=80000, layer="sliding", page=1, q_len=8, dist="soft", seed=807)),
    ("verify 1", 16, dict(batch=1, ctx=80000, layer="full", page=1, q_len=8, dist="soft", seed=808)),
    ("verify 16", 46, dict(batch=16, ctx=80000, layer="sliding", page=1, q_len=8, spread="agentic", dist="soft", seed=809)),
    ("verify 16", 16, dict(batch=16, ctx=80000, layer="full", page=1, q_len=8, spread="agentic", dist="soft", seed=810)),
    ("verify 32", 46, dict(batch=32, ctx=80000, layer="sliding", page=1, q_len=8, spread="agentic", dist="soft", seed=811)),
    ("verify 32", 16, dict(batch=32, ctx=80000, layer="full", page=1, q_len=8, spread="agentic", dist="soft", seed=812)),
]
TESTS = [
    dict(batch=1, ctx=10, layer="sliding"), dict(batch=1, ctx=3000, layer="full"),
    dict(batch=8, ctx=2047, layer="sliding"), dict(batch=9, ctx=5000, layer="sliding", spread="mixed", page=1),
    dict(batch=3, ctx=6000, layer="full", q_len=8, dist="outliers"),
    dict(batch=5, ctx=5000, layer="sliding", q_len=8, spread="mixed", page=1),
    dict(batch=33, ctx=20000, layer="full", spread="mixed", page=1), dict(batch=2, ctx=300000, layer="sliding", q_len=8),
    dict(batch=1, ctx=1048576, layer="full", q_len=8), dict(batch=4, ctx=7000, layer="sliding", q_len=8, page=1, kv_scale=0.015625),
    dict(batch=1, ctx=80000, layer="full", dist="soft", page=1), dict(batch=4, ctx=6000, layer="sliding", q_len=8, dist="soft"),
    dict(batch=16, ctx=20000, layer="full", spread="mixed", dist="soft", page=1),
]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    args = sys.argv[1:]
    path = next((a for a in args if a.endswith(".py")), HERE + "/submission.py")
    weights = ref.generate_weights(ref.WEIGHTS_SEED)
    base = load(HERE + "/baseline.py", "base")
    st_b = base.prepare(weights)
    mine = load(path, "mine")
    st_m = mine.prepare(weights)

    if "--check" in args or "--only-check" in args:
        bad = 0
        for kw in TESTS + [k for _, _, k in SCORED]:
            kw = dict(kw)
            kw.setdefault("seed", 77)
            data = ref.generate_input(**kw)
            ok, msg = ref.check_implementation(data, ref.collect(data, mine.kernel_fn(data, st_m)), weights)
            if not ok:
                bad += 1
                print(f"FAIL {kw}: {msg}", flush=True)
            del data
        print(f"checks done, {bad} failures", flush=True)
        if "--only-check" in args:
            return

    sel = next((a.split("=", 1)[1].split(",") for a in args if a.startswith("--loads=")), None)
    loads = {}
    for name, weight, kw in SCORED:
        if sel and name not in sel:
            continue
        data = ref.generate_input(**kw)
        tb = kernel_us(base.kernel_fn, data, st_b)
        tm = kernel_us(mine.kernel_fn, data, st_m)
        print(f"{name:10s} {kw['layer']:7s} sglang {tb:7.1f}  mine {tm:7.1f}  x{tb / tm:5.3f}", flush=True)
        lb, lm = loads.get(name, (0.0, 0.0))
        loads[name] = (lb + weight * tb, lm + weight * tm)
        del data
    geo = 1.0
    for name, (lb, lm) in loads.items():
        print(f"  load {name}: x{lb / lm:.3f}")
        geo *= lb / lm
    print(f"geomean over loads x{geo ** (1 / len(loads)):.3f}")


if __name__ == "__main__":
    main()

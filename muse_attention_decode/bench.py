"""Local judge-like timing for muse_attention_decode, mine vs SGLang's: python bench.py [file.py] [--check] [--base-only]"""

import importlib.util
import sys

HERE = __file__.rsplit("/", 1)[0]
sys.path.insert(0, HERE)
import reference as ref  # noqa: E402
from timing import kernel_us  # noqa: E402

C = dict(ctx=80000, heads="tp8", page=1)
SCORED = [  # (load, weight, kwargs)
    ("decode 1", 46, dict(batch=1, layer="sliding", dist="soft", seed=901, **C)),
    ("decode 1", 16, dict(batch=1, layer="full", dist="coherent", seed=902, **C)),
    ("decode 16", 46, dict(batch=16, layer="sliding", spread="agentic", dist="soft", seed=903, **C)),
    ("decode 16", 16, dict(batch=16, layer="full", spread="agentic", dist="coherent", seed=904, **C)),
    ("decode 32", 46, dict(batch=32, layer="sliding", spread="agentic", dist="soft", seed=905, **C)),
    ("decode 32", 16, dict(batch=32, layer="full", spread="agentic", dist="coherent", seed=906, **C)),
    ("verify 1", 46, dict(batch=1, layer="sliding", q_len=8, dist="soft", seed=907, **C)),
    ("verify 1", 16, dict(batch=1, layer="full", q_len=8, dist="coherent", seed=908, **C)),
    ("verify 16", 46, dict(batch=16, layer="sliding", q_len=8, spread="agentic", dist="soft", seed=909, **C)),
    ("verify 16", 16, dict(batch=16, layer="full", q_len=8, spread="agentic", dist="coherent", seed=910, **C)),
    ("verify 32", 46, dict(batch=32, layer="sliding", q_len=8, spread="agentic", dist="soft", seed=911, **C)),
    ("verify 32", 16, dict(batch=32, layer="full", q_len=8, spread="agentic", dist="coherent", seed=912, **C)),
]
TESTS = [
    dict(batch=1, ctx=10, heads="tp8", layer="sliding"), dict(batch=3, ctx=2047, heads="tp8", layer="sliding"),
    dict(batch=5, ctx=5000, heads="tp8", layer="sliding", q_len=8, spread="mixed", page=1),
    dict(batch=4, ctx=3000, heads="tp8", layer="full"), dict(batch=8, ctx=8192, heads="dp", layer="full", spread="mixed"),
    dict(batch=2, ctx=100000, heads="tp8", layer="full", q_len=8, page=1),
    dict(batch=4, ctx=6000, heads="tp8", layer="sliding", dist="edges", page=1),
    dict(batch=4, ctx=6000, heads="tp8", layer="full", q_len=8, dist="edges"),
    dict(batch=1, ctx=1048576, heads="tp8", layer="full", q_len=8),
    dict(batch=3, ctx=4000, heads="tp8", layer="sliding", q_len=8, page=1, kv_scale=0.015625),
    dict(batch=2, ctx=9000, heads="tp8", layer="full", kv_scale=0.015625),
    dict(batch=4, ctx=80000, heads="tp8", layer="full", spread="mixed", dist="soft"),
    dict(batch=4, ctx=6000, heads="tp8", layer="sliding", q_len=8, dist="soft", page=1),
    dict(batch=2, ctx=20000, heads="dp", layer="full", q_len=8, dist="soft"),
    dict(batch=4, ctx=80000, heads="tp8", layer="full", spread="mixed", dist="coherent", page=1),
    dict(batch=2, ctx=20000, heads="dp", layer="full", q_len=8, dist="coherent"),
]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    args = sys.argv[1:]
    path = next((a for a in args if a.endswith(".py")), HERE + "/submission.py")
    base = load(HERE + "/baseline.py", "base")
    mine = None if "--base-only" in args else load(path, "mine")
    if mine and ("--check" in args or "--only-check" in args):
        bad = 0
        for kw in TESTS + [k for _, _, k in SCORED]:
            kw = dict(kw)
            kw.setdefault("seed", 77)
            data = ref.generate_input(**kw)
            ok, msg = ref.check_implementation(data, ref.collect(data, mine.kernel_fn(data)))
            if not ok:
                bad += 1
                print(f"FAIL {kw}: {msg}", flush=True)
        print(f"checks done, {bad} failures", flush=True)
        if "--only-check" in args:
            return
    loads = {}
    for name, weight, kw in SCORED:
        data = ref.generate_input(**kw)
        tb = kernel_us(lambda d, s: base.kernel_fn(d), data, None)
        line = f"{name:10s} {kw['layer']:7s} sglang {tb:7.1f}"
        tm = tb
        if mine:
            tm = kernel_us(lambda d, s: mine.kernel_fn(d), data, None)
            line += f"  mine {tm:7.1f}  x{tb / tm:5.3f}"
        floor = ref.speed_of_light_ns(hw={"hbm_bytes_per_s": 4.8e12, "bf16_dense_flops_per_s": 989e12}, **kw) / 1e3
        print(line + f"  floor {floor:6.1f}", flush=True)
        lb, lm = loads.get(name, (0.0, 0.0))
        loads[name] = (lb + weight * tb, lm + weight * tm)
    geo = 1.0
    for name, (lb, lm) in loads.items():
        print(f"  load {name}: x{lb / lm:.3f}")
        geo *= lb / lm
    print(f"geomean over loads x{geo ** (1 / len(loads)):.3f}")


if __name__ == "__main__":
    main()

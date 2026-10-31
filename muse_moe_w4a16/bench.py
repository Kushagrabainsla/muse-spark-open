"""Local judge-like timing for muse_moe_w4a16, mine vs SGLang's: python bench.py [file.py] [--check] [--bs=1,16]
Also prints SGLang's per-stage breakdown with --stages."""

import importlib.util
import sys

import torch

HERE = __file__.rsplit("/", 1)[0]
sys.path.insert(0, HERE)
import reference as ref  # noqa: E402
from timing import kernel_us  # noqa: E402

SCORED = [1, 16, 32, 8, 128, 256]
TESTS = [(4, "normal"), (33, "normal"), (257, "normal"), (128, "skewed"), (1, "normal")]
HW = {"hbm_bytes_per_s": 4.8e12, "bf16_dense_flops_per_s": 989e12}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def routing_stats(data):
    ids = data["topk_ids"]
    local = ids[ids < 32]
    return local.numel(), torch.unique(local).numel()


def main():
    args = sys.argv[1:]
    path = next((a for a in args if a.endswith(".py")), HERE + "/submission.py")
    weights = ref.generate_weights(ref.WEIGHTS_SEED)
    base = load(HERE + "/baseline.py", "base")
    st_b = base.prepare(weights)
    mine = None
    if not path.endswith("baseline.py"):
        mine = load(path, "mine")
        st_m = mine.prepare(weights)

    if mine and ("--check" in args or "--only-check" in args):
        bad = 0
        for bs, dist in TESTS + [(b, "normal") for b in SCORED]:
            for seed in (11, 12):
                data = ref.generate_input(bs, "ep8", seed=seed * 1000 + bs, dist=dist)
                ok, msg = ref.check_implementation(data, mine.kernel_fn(data, st_m), weights)
                if not ok:
                    bad += 1
                    print(f"FAIL bs={bs} {dist} seed={seed}: {msg}")
        print(f"checks done, {bad} failures", flush=True)
        if "--only-check" in args:
            return

    sel = next((a.split("=", 1)[1] for a in args if a.startswith("--bs=")), None)
    shapes = [int(v) for v in sel.split(",")] if sel else SCORED
    geo, n = 1.0, 0
    for bs in shapes:
        data = ref.generate_input(bs, "ep8", seed=400 + bs)
        routes, distinct = routing_stats(data)
        tb = kernel_us(base.kernel_fn, data, st_b)
        line = f"bs={bs:4d} routes={routes:4d} experts={distinct:2d}  sglang {tb:7.1f}"
        if mine:
            tm = kernel_us(mine.kernel_fn, data, st_m)
            line += f"  mine {tm:7.1f}  x{tb / tm:5.3f}"
            geo *= tb / tm
            n += 1
        floor = ref.speed_of_light_ns(bs, "ep8", HW, router_load=routes, router_distinct=distinct) / 1e3
        print(line + f"  floor~{floor:6.1f}", flush=True)
    if n:
        print(f"geomean x{geo ** (1 / n):.3f}")


if __name__ == "__main__":
    main()

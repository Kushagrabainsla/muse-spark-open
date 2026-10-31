"""kernel_us of a submission at one bs, plus a profiled replay of the timing graph: python tprof.py <bs> <file>"""
import importlib.util, sys, torch
sys.path.insert(0, "/root/okc_moe")
import reference as ref
import timing
bs, path = int(sys.argv[1]), sys.argv[2]
spec = importlib.util.spec_from_file_location("mine", path)
mine = importlib.util.module_from_spec(spec); spec.loader.exec_module(mine)
w = ref.generate_weights(ref.WEIGHTS_SEED); st = mine.prepare(w)
d = ref.generate_input(bs, "ep8", seed=400 + bs)
print("kernel_us", timing.kernel_us(mine.kernel_fn, d, st))
timing.REPS = 3
def both():
    for _ in range(timing.REPS):
        timing._read_flush(); mine.kernel_fn(d, st)
g = timing._capture(both)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
g.replay(); torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()

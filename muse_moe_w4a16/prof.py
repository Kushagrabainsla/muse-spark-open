"""Per-kernel profile of submission.py at one bs: nsys profile --stats=true python prof.py <bs>"""
import importlib.util
import sys
import torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import reference as ref  # noqa: E402

bs = int(sys.argv[1])
path = sys.argv[2] if len(sys.argv) > 2 else __file__.rsplit("/", 1)[0] + "/submission.py"
spec = importlib.util.spec_from_file_location("mine", path)
mine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mine)
w = ref.generate_weights(ref.WEIGHTS_SEED)
st = mine.prepare(w)
d = ref.generate_input(bs, "ep8", seed=400 + bs)
for _ in range(3):
    mine.kernel_fn(d, st)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
for _ in range(20):
    mine.kernel_fn(d, st)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()

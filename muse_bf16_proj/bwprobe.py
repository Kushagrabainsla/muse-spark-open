"""How fast can one kernel stream N MB from a cold L2? Read-only Triton kernels, varied shapes."""
import sys
import torch
import triton
import triton.language as tl
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from timing import kernel_us


@triton.jit
def _read(src, dst, n_elems, BLOCK: tl.constexpr, ITERS: tl.constexpr):
    pid = tl.program_id(0)
    acc = tl.zeros((BLOCK,), tl.float32)
    base = pid * BLOCK * ITERS
    for i in range(ITERS):
        offs = base + i * BLOCK + tl.arange(0, BLOCK)
        acc += tl.load(src + offs, mask=offs < n_elems, other=0).to(tl.float32)
    tl.store(dst + pid, tl.sum(acc))


for mb in (8.4, 16.8, 37.7, 67.1):
    n = int(mb * 1e6 / 2)
    src = torch.randn(n, device="cuda").to(torch.bfloat16)
    dst = torch.empty(1 << 20, device="cuda")
    best = []
    for block in (1024, 2048, 4096, 8192):
        for warps in (4, 8):
            for ctas in (132, 264, 528, 1056, 2112):
                iters = triton.cdiv(n, block * ctas)
                grid = (triton.cdiv(n, block * iters),)
                fn = lambda d, s, b=block, it=iters, g=grid, w=warps: _read[g](src, dst, n, BLOCK=b, ITERS=it, num_warps=w)
                t = kernel_us(fn, None, None)
                best.append((t, block, warps, grid[0]))
    best.sort()
    print(f"{mb:5.1f} MB  floor {mb / 4.8:5.2f}us  best {best[0][0]:.2f}us ({mb / best[0][0] / 1e0 * 1e0:.2f} TB/s)  cfg {best[0][1:]}  top3 {[round(b[0], 2) for b in best[:3]]}", flush=True)

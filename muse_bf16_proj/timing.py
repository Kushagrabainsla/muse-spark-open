"""Kernel time under CUDA graphs with a cold L2, without the replay's launch overhead.

One graph holds REPS x (read a buffer larger than L2, call fn); another holds only the reads.
The kernel's time is the difference over REPS.
"""

import statistics

import torch

REPS = 20
_flush = None
_sink = None


def _read_flush():
    _flush.sum()


def _capture(body):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        body()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        body()
    return g


def _time(g, n=7):
    ts = []
    for _ in range(n):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000)
    return statistics.median(ts)


def kernel_us(fn, data, state):
    """Median µs per call of fn(data, state), cold L2, launch overhead excluded."""
    global _flush, _sink
    if _flush is None:
        _flush = torch.ones(128 << 20, dtype=torch.float16, device="cuda")  # 256 MB
        _sink = torch.empty((), dtype=torch.float16, device="cuda")

    def both():
        for _ in range(REPS):
            _read_flush()
            fn(data, state)

    def only_flush():
        for _ in range(REPS):
            _read_flush()

    gk, gf = _capture(both), _capture(only_flush)
    # Interleave to cancel drift.
    tk, tf = [], []
    for _ in range(3):
        tk.append(_time(gk))
        tf.append(_time(gf))
    return (statistics.median(tk) - statistics.median(tf)) / REPS

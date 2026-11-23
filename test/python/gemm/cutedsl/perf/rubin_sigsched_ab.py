# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""wip/sigsched-acquire-once: stock dynamic scheduler against the streaming signal scheduler (every M tile ready at
launch), MXFP8 GLU and dGLU (stock SF atoms), dsv3 and qwen35 EP4 per-rank shapes.  The signal-before numbers come
from ``rubin_sigsched_diag`` (DIAG_ONLY=stock_dynamic,signal) at the commit before the fix.  From test/python:

    ncu --metrics gpu__time_duration.sum -k regex:kernel_kernel --csv python -m gemm.cutedsl.perf.rubin_sigsched_ab ncu
    python -m gemm.cutedsl.perf.rubin_sigsched_ab            # CUDA events (include launch latency)

``ncu`` as first argument: one launch per variant, in VARIANTS order per (kind, shape), for ncu.
DIAG_KINDS=<glu,dglu> / DIAG_SHAPES=<shape,...> in the environment restrict the run.
"""

import os
import statistics
import sys

import torch

from gemm.cutedsl import _streaming_signal_blockscaled as bs
from gemm.cutedsl import _streaming_signal_dglu as dg

SHAPES = {"deepseek_v3": (64, 1024, 7168, 2048), "qwen35_397b": (128, 768, 4096, 1024)}
VARIANTS = {"stock_dynamic": (False, {}), "signal": (True, {})}
KNOBS = ()


def build(kind, shape, name, problem):
    signal, env = VARIANTS[name]
    for knob in KNOBS:
        os.environ.pop(knob, None)
    os.environ.update(env)
    experts = SHAPES[shape][0]
    options = dict(use_dynamic_sched=True)
    if signal:
        options["streaming_signal_capacity"] = problem["rows"] // 256
    kernel = bs.make_glu_kernel(experts, **options) if kind == "glu" else dg.make_dglu_kernel("mxfp8", experts, **options)
    launcher = bs.Launcher(kernel, problem) if kind == "glu" else dg.Launcher(kernel, problem)
    for knob in KNOBS:
        os.environ.pop(knob, None)
    return launcher, launcher.buffers(problem), bs.signal_ready(kernel) if signal else None


def main():
    ncu = "ncu" in sys.argv[1:]
    rounds = int(os.environ.get("ROUNDS", "8"))
    kinds = os.environ.get("DIAG_KINDS", "glu,dglu").split(",")
    shapes = os.environ.get("DIAG_SHAPES", ",".join(SHAPES)).split(",")
    for kind in kinds:
        for shape in shapes:
            experts, rows, hidden, inter = SHAPES[shape]
            if kind == "glu":
                problem = bs.make_problem([rows] * experts, 2 * inter, hidden, seed=0)
            else:
                dg.HIDDEN, dg.INTERMEDIATE = hidden, inter
                problem = dg.make_problem("mxfp8", [rows] * experts, seed=0)
            runs = {name: build(kind, shape, name, problem) for name in VARIANTS}
            if ncu:
                for name, (launcher, buffers, ready) in runs.items():
                    if ready is not None:
                        ready(buffers[-1])
                    launcher.launch(problem, buffers)
                    torch.cuda.synchronize()
                    print(f"NCU_ORDER {kind} {shape} {name}", flush=True)
                continue
            times = {name: [] for name in VARIANTS}
            for rnd in range(rounds):
                for name in list(VARIANTS) if rnd % 2 == 0 else list(VARIANTS)[::-1]:
                    launcher, buffers, ready = runs[name]
                    for rep in range(10):
                        if ready is not None:
                            ready(buffers[-1])
                        if kind == "dglu":
                            buffers[1].zero_()
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        launcher.launch(problem, buffers)
                        end.record()
                        torch.cuda.synchronize()
                        if rep >= 2:
                            times[name].append(start.elapsed_time(end))
            base = statistics.median(times["stock_dynamic"]) if "stock_dynamic" in times else statistics.median(next(iter(times.values())))
            for name in VARIANTS:
                med = statistics.median(times[name])
                print(f"{kind:4} {shape:12} {name:24} median {med * 1000:8.1f} us  vs stock {med / base - 1:+6.1%}", flush=True)


if __name__ == "__main__":
    main()

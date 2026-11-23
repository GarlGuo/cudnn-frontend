# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rubin (SM107) compact-SFA prototype timing: the MXFP8 GLU (FC1 forward) and dGLU (backward) Rubin twins, stock
(256-deep K tile, SF atoms) against compact_sfa_sf_warp (128-deep K tile, compact rows) with the cluster-scope and the
CTA-scope SF release, at the EP4 per-rank shapes of deepseek_v3 and qwen35_397b.  Every M tile is ready at launch (the
signal scheduler waits for nothing).  CUDA events, interleaved rounds, median of the launches.  From test/python:

    python -m gemm.cutedsl.perf.rubin_compact_sfa_perf            # both kernels, both shapes
    python -m gemm.cutedsl.perf.rubin_compact_sfa_perf glu qwen35_397b
"""

import statistics
import sys

import torch

from gemm.cutedsl import _streaming_signal_blockscaled as bs
from gemm.cutedsl import _streaming_signal_dglu as dg

# EP4 per rank: (local experts, rows per expert (256-aligned), hidden, intermediate)
SHAPES = {"deepseek_v3": (64, 1024, 7168, 2048), "qwen35_397b": (128, 768, 4096, 1024)}
VARIANTS = {
    "stock_dynamic": dict(use_dynamic_sched=True),
    "signal_atoms": dict(use_dynamic_sched=True, signal=True),
    "signal_compact_cluster": dict(use_dynamic_sched=True, signal=True, compact_sfa_sf_warp=True),
    "signal_compact_cta": dict(use_dynamic_sched=True, signal=True, compact_sfa_sf_warp=True, compact_sfa_cta_release=True),
}
import os

# ROUNDS=<n> in the environment: more interleaved rounds (each REPS launches per variant).
ROUNDS, REPS, WARMUP = int(os.environ.get("ROUNDS", "6")), 10, 2


def build(kind, shape, variant, problem):
    experts, _, hidden, inter = SHAPES[shape]
    options = dict(VARIANTS[variant])
    if options.pop("signal", False):
        options["streaming_signal_capacity"] = problem["rows"] // 256
    if kind == "glu":
        kernel = bs.make_glu_kernel(experts, **options)
        launcher = bs.Launcher(kernel, problem)
        ready = bs.signal_ready(kernel) if "streaming_signal_capacity" in options else None
    else:
        kernel = dg.make_dglu_kernel("mxfp8", experts, **options)
        launcher = dg.Launcher(kernel, problem)
        ready = bs.signal_ready(kernel) if "streaming_signal_capacity" in options else None
    return launcher, launcher.buffers(problem), ready


def main():
    kinds = [a for a in sys.argv[1:] if a in ("glu", "dglu")] or ["glu", "dglu"]
    shapes = [a for a in sys.argv[1:] if a in SHAPES] or list(SHAPES)
    print(f"device {torch.cuda.get_device_name()} cc {torch.cuda.get_device_capability()}", flush=True)
    for kind in kinds:
        for shape in shapes:
            experts, rows, hidden, inter = SHAPES[shape]
            if kind == "glu":
                problem = bs.make_problem([rows] * experts, 2 * inter, hidden, seed=0)
            else:
                dg.HIDDEN, dg.INTERMEDIATE = hidden, inter
                problem = dg.make_problem("mxfp8", [rows] * experts, seed=0)
            runs = {name: build(kind, shape, name, problem) for name in VARIANTS}
            times = {name: [] for name in VARIANTS}
            for rnd in range(ROUNDS):
                order = list(VARIANTS) if rnd % 2 == 0 else list(VARIANTS)[::-1]
                for name in order:
                    launcher, buffers, ready = runs[name]
                    for rep in range(REPS):
                        if ready is not None:
                            ready(buffers[-1])
                        if kind == "dglu":
                            buffers[1].zero_()
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        launcher.launch(problem, buffers)
                        end.record()
                        torch.cuda.synchronize()
                        if rep >= WARMUP:
                            times[name].append(start.elapsed_time(end))
            base = statistics.median(times["stock_dynamic"])
            for name in VARIANTS:
                med = statistics.median(times[name])
                print(f"{kind:4} {shape:12} {name:24} median {med * 1000:8.1f} us  vs stock_dynamic {med / base - 1:+6.1%}  n={len(times[name])}", flush=True)


if __name__ == "__main__":
    main()

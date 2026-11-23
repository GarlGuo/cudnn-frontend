# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DIAGNOSTIC (wip/sigsched-rubin-diag): in-kernel counters for the Rubin MXFP8 GLU twin, stock dynamic scheduler vs the
streaming signal scheduler, every M tile ready, dsv3 and qwen35 EP4 shapes (SIGDIAG_COUNTERS=1, set by this script).

Per variant (after warm-up launches): batch openings and batches opened (signal), claims and the scheduler leader's
average cycles per claim (signal), the TMA warp's average cycles per CTA spent waiting for tile_info (both: scheduler
starvation of the mainloop), and the kernel time (CUDA events, launch latency included; use ncu for exact times).
From test/python:

    python -m gemm.cutedsl.perf.rubin_sigsched_counters
"""

import os

os.environ["SIGDIAG_COUNTERS"] = "1"

import statistics

import torch

from gemm.cutedsl import _streaming_signal_blockscaled as bs

SHAPES = {"deepseek_v3": (64, 1024, 7168, 2048), "qwen35_397b": (128, 768, 4096, 1024)}


def counters(kernel, workspace):
    words = workspace.view(torch.int32)[kernel.get_workspace_bytes() // 4 - 16 :].tolist()
    return words


def main():
    print(f"device {torch.cuda.get_device_name()}", flush=True)
    for shape, (experts, rows, hidden, inter) in SHAPES.items():
        problem = bs.make_problem([rows] * experts, 2 * inter, hidden, seed=0)
        for name in ("stock_dynamic", "signal"):
            options = dict(use_dynamic_sched=True)
            if name == "signal":
                options["streaming_signal_capacity"] = problem["rows"] // 256
            kernel = bs.make_glu_kernel(experts, **options)
            kernel.test_compile_key = (kernel.test_compile_key, "counters")
            launcher = bs.Launcher(kernel, problem)
            buffers = launcher.buffers(problem)
            ready = bs.signal_ready(kernel) if name == "signal" else None
            times, samples = [], []
            for rep in range(12):
                buffers[-1].zero_()
                if ready is not None:
                    ready(buffers[-1])
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                launcher.launch(problem, buffers)
                end.record()
                torch.cuda.synchronize()
                if rep >= 2:
                    times.append(start.elapsed_time(end))
                    samples.append(counters(kernel, buffers[-1]))
            med = lambda i: statistics.median(s[i] for s in samples)
            openings, claim_cyc, claims, wait_cyc, ctas, batches = (med(i) for i in range(6))
            print(
                f"{shape:12} {name:14} time {statistics.median(times) * 1000:8.1f} us | TMA tile_info wait / CTA "
                f"{wait_cyc * 256 / max(ctas, 1):10.0f} cyc ({ctas:.0f} CTAs) | claims {claims:.0f}, leader cycles / claim "
                f"{claim_cyc * 256 / max(claims, 1):8.0f} | openings {openings:.0f}, batches {batches:.0f}",
                flush=True,
            )


if __name__ == "__main__":
    main()

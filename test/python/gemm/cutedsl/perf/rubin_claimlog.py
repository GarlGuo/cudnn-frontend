# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DIAGNOSTIC (wip/sigsched-claimlog-diag): the per-claim tile sequence and claim times of the stock dynamic scheduler
and the streaming signal scheduler (every M tile ready), MXFP8 GLU, dsv3 and qwen35 EP4 per-rank shapes.  Prints
whether both claim the same tiles in the same order, experts / B tiles in flight, and the claim-time spread of the
claims sharing a B tile and an A tile.  From test/python:

    python -m gemm.cutedsl.perf.rubin_claimlog
"""

import os
import tempfile

os.environ["CUTE_DSL_CACHE_DIR"] = tempfile.mkdtemp(prefix="claimlog_cute_")  # the log address is baked in

import numpy as np
import torch

from gemm.cutedsl import _streaming_signal_blockscaled as bs

SHAPES = {"deepseek_v3": (64, 1024, 7168, 2048), "qwen35_397b": (128, 768, 4096, 1024)}


def run(shape):
    experts, rows, hidden, inter = SHAPES[shape]
    problem = bs.make_problem([rows] * experts, 2 * inter, hidden, seed=0)
    log = torch.zeros(1 << 22, dtype=torch.int32, device="cuda")
    os.environ["CLAIMLOG_ADDR"] = str(log.data_ptr())
    out = {}
    for name in ("stock", "signal"):
        options = dict(use_dynamic_sched=True)
        if name == "signal":
            options["streaming_signal_capacity"] = problem["rows"] // 256
        kernel = bs.make_glu_kernel(experts, **options)
        launcher = bs.Launcher(kernel, problem)
        buffers = launcher.buffers(problem)
        ready = bs.signal_ready(kernel) if name == "signal" else None
        for _ in range(3):
            log.zero_()
            if ready is not None:
                ready(buffers[-1])
            launcher.launch(problem, buffers)
            torch.cuda.synchronize()
        out[name] = log.view(-1, 4).cpu().numpy().copy()
    os.environ.pop("CLAIMLOG_ADDR")
    m_tiles, n_tiles = rows // 256, (2 * inter) // 256
    claims = experts * m_tiles * n_tiles
    st, sg = out["stock"][:claims], out["signal"][:claims]
    expert, local = st[:, 0] // (m_tiles * n_tiles), st[:, 0] % (m_tiles * n_tiles)
    stock_linear = (expert * m_tiles + local % m_tiles) * n_tiles + local // m_tiles
    print(f"{shape}: {claims} claims; stock and signal claim the same tile at every claim index: {bool((stock_linear == sg[:, 0]).all())}")
    for name, t in (("stock", st), ("signal", sg)):
        ts = t[:, 1].astype(np.uint32).astype(np.int64)
        ts -= ts.min()
        cluster = t[:, 2]
        end = np.full(claims, ts.max() + 1)
        for c in np.unique(cluster):
            idx = np.nonzero(cluster == c)[0]
            idx = idx[np.argsort(ts[idx])]
            end[idx[:-1]] = ts[idx[1:]]
        lin = t[:, 0] if name == "signal" else stock_linear
        ex = lin // (m_tiles * n_tiles)
        btile = ex * n_tiles + lin % n_tiles
        samples = np.linspace(0, np.percentile(ts, 95), 200)
        in_flight = [len(np.unique(ex[(ts <= s) & (end > s)])) for s in samples]
        b_in_flight = [len(np.unique(btile[(ts <= s) & (end > s)])) for s in samples]
        b_groups = ts.reshape(-1, m_tiles)
        a_groups = ts.reshape(experts, n_tiles, m_tiles)
        print(
            f"  {name:6} span {ts.max() / 1e3:8.1f} us, experts in flight median {np.median(in_flight):.0f} max {max(in_flight)}, "
            f"B tiles in flight median {np.median(b_in_flight):.0f}, claim spread per B tile {np.median(np.ptp(b_groups, 1)) / 1e3:.2f} us, "
            f"per A tile {np.median(np.ptp(a_groups, 1)) / 1e3:.2f} us"
        )


if __name__ == "__main__":
    for shape in SHAPES:
        run(shape)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DIAGNOSTIC (wip/sigsched-rubin-bisect): where the signal scheduler's Rubin GLU overhead comes from.  MXFP8 GLU (stock SF
atoms), every M tile ready, dsv3 and qwen35 EP4 per-rank shapes.  From test/python:

    # (3) bisect: one launch per variant, NCU_ORDER lines give the order
    ncu --metrics gpu__time_duration.sum -k regex:kernel_kernel --csv python -m gemm.cutedsl.perf.rubin_sigsched_bisect ncu
    # (1) + (2) per-tile log, stock vs signal (no ncu)
    python -m gemm.cutedsl.perf.rubin_sigsched_bisect tilelog

Variants (grouped/_sigdiag.py): stock_dynamic; signal; signal_stock_sched (the signal build running the stock
scheduler); signal_hyb_claim (no batches: claim index -> tile); signal_hyb_bcast (stock broadcast / slot release);
signal_hyb_both.  DIAG_ONLY=<variant,...> / DIAG_SHAPES=<shape,...> restrict the run.
"""

import os
import statistics
import sys
import tempfile

os.environ.setdefault("CUTE_DSL_CACHE_DIR", tempfile.mkdtemp(prefix="sigbisect_cute_"))  # TILELOG_ADDR is baked in

import numpy as np
import torch

from gemm.cutedsl import _streaming_signal_blockscaled as bs

SHAPES = {"deepseek_v3": (64, 1024, 7168, 2048), "qwen35_397b": (128, 768, 4096, 1024)}
VARIANTS = {
    "stock_dynamic": (False, {}),
    "signal": (True, {}),
    "signal_stock_sched": (True, {"SIGDIAG_STOCK_SCHED": "1"}),
    "signal_hyb_claim": (True, {"SIGDIAG_HYBRID_CLAIM": "1"}),
    "signal_hyb_bcast": (True, {"SIGDIAG_HYBRID_BCAST": "1"}),
    "signal_hyb_both": (True, {"SIGDIAG_HYBRID_CLAIM": "1", "SIGDIAG_HYBRID_BCAST": "1"}),
}
KNOBS = ("SIGDIAG_STOCK_SCHED", "SIGDIAG_HYBRID_CLAIM", "SIGDIAG_HYBRID_BCAST", "TILELOG_ADDR")
if os.environ.get("DIAG_ONLY"):
    VARIANTS = {name: VARIANTS[name] for name in os.environ["DIAG_ONLY"].split(",")}


def build(shape, name, problem, extra_env=None, tag=""):
    signal, env = VARIANTS[name]
    for knob in KNOBS:
        os.environ.pop(knob, None)
    os.environ.update(env)
    os.environ.update(extra_env or {})
    options = dict(use_dynamic_sched=True)
    if signal:
        options["streaming_signal_capacity"] = problem["rows"] // 256
    kernel = bs.make_glu_kernel(SHAPES[shape][0], **options)
    kernel.test_compile_key = (kernel.test_compile_key, name, tag)  # the knobs are not constructor options
    launcher = bs.Launcher(kernel, problem)
    for knob in KNOBS:
        os.environ.pop(knob, None)
    return launcher, launcher.buffers(problem), bs.signal_ready(kernel) if signal else None


def launch(problem, run):
    launcher, buffers, ready = run
    if ready is not None:
        ready(buffers[-1])
    launcher.launch(problem, buffers)
    torch.cuda.synchronize()


def bisect(shapes):
    for shape in shapes:
        experts, rows, hidden, inter = SHAPES[shape]
        problem = bs.make_problem([rows] * experts, 2 * inter, hidden, seed=0)
        for name in VARIANTS:
            launch(problem, build(shape, name, problem))
            print(f"NCU_ORDER glu {shape} {name}", flush=True)


def tilelog(shapes):
    log = torch.zeros(1 << 22, dtype=torch.int32, device="cuda")
    for shape in shapes:
        experts, rows, hidden, inter = SHAPES[shape]
        problem = bs.make_problem([rows] * experts, 2 * inter, hidden, seed=0)
        recs = {}
        for name in ("stock_dynamic", "signal"):
            run = build(shape, name, problem, {"TILELOG_ADDR": str(log.data_ptr())}, tag="tilelog")
            for _ in range(3):
                log.zero_()
                launch(problem, run)
            words = log.cpu().numpy()
            counts = words[:65536].reshape(-1, 4)
            ctas = int((counts[:, 0] > 0).sum())
            r = words[65536:].reshape(-1, 256, 16)[:ctas]
            recs[name] = (r, counts[:ctas])
        print(f"== {shape}: CTAs {ctas}")
        tiles = {}
        for name, (r, counts) in recs.items():
            rows_ = []
            for cta in range(r.shape[0]):
                for seq in range(int(counts[cta, 0])):
                    rec = r[cta, seq]
                    rows_.append((cta, seq, int(rec[0]), int(rec[1]), int(rec[2]), int(rec[3]), *[int(np.uint32(rec[4 + k])) for k in range(4)]))
            a = np.array(rows_, dtype=np.int64)
            t0 = a[:, 6:10][a[:, 6:10] > 0].min()
            a[:, 6:10] -= t0
            tiles[name] = a
            sched, tma, mma, epi = a[:, 6], a[:, 7], a[:, 8], a[:, 9]
            ok = (tma > 0) & (mma > 0) & (epi > 0)
            # per-CTA steady-state: time between consecutive tiles' epilogue ends and MMA starts
            d_epi, d_mma, gap = [], [], []
            for cta in np.unique(a[:, 0]):
                c = a[a[:, 0] == cta]
                c = c[np.argsort(c[:, 1])]
                d_epi += list(np.diff(c[:, 9]))
                d_mma += list(np.diff(c[:, 8]))
                gap += list(c[1:, 7] - c[:-1, 6])
            q = lambda x: f"{np.median(x) / 1e3:6.2f}/{np.percentile(x, 90) / 1e3:6.2f}"
            print(
                f"  {name:13} tiles {len(a)} span {epi.max() / 1e3:7.1f} us | us median/p90: publish->TMA first load {q((tma - sched)[ok])}, "
                f"TMA->MMA start {q((mma - tma)[ok])}, MMA start->epilogue end {q((epi - mma)[ok])} | per-CTA tile period: MMA {q(d_mma)}, "
                f"epilogue {q(d_epi)} | publish lead over previous TMA {q(gap)}"
            )
        st, sg = tiles["stock_dynamic"], tiles["signal"]
        key = lambda a: {(int(x[2]), int(x[3]), int(x[4])): int(x[5]) for x in a}
        ks, kg = key(st), key(sg)
        common = set(ks) & set(kg)
        same_sm = np.mean([ks[k] == kg[k] for k in common])
        sms = max(st[:, 5].max(), sg[:, 5].max()) + 1
        half = lambda sm: sm >= sms // 2
        for name, kk in (("stock_dynamic", ks), ("signal", kg)):
            groups = {}
            for (e, m, n), sm in kk.items():
                groups.setdefault((e, n), set()).add(half(sm))  # the CTAs sharing a B tile: SM halves used
            mixed = np.mean([len(v) > 1 for v in groups.values()])
            print(f"  {name:13} B tiles whose CTAs span both SM halves: {mixed:.2f}")
        print(f"  same SM for the same tile, stock vs signal: {same_sm:.2f} of {len(common)} tiles")


if __name__ == "__main__":
    shapes = os.environ.get("DIAG_SHAPES", ",".join(SHAPES)).split(",")
    if "tilelog" in sys.argv[1:]:
        tilelog(shapes)
    else:
        bisect(shapes)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bitwise determinism of N_CHUNK output notify on the SM100 MXFP8 block-scaled MoE quant (FC2) forward kernel.

N_CHUNK changes which cluster runs a tile and when, never the tile math, so the BF16 D must be raw-byte equal across
replays, extra clusters that stop early or never stop, chunk sizes, neighboring experts and the stock dynamic scheduler.

Directions: forward only.  This is a forward kernel; backward is unsupported by contract, not untested.  Inputs are
quantized once (E4M3 codes, E8M0 SF atoms) and reused unchanged across every packing.
"""

import pytest
import torch

from determinism_utils import assert_bitwise, assert_replays, snapshot
from gemm.cutedsl._streaming_signal import BIG_EXPERT, MANY_EXPERTS_64, ROWS_ALIGNMENT, padded
from gemm.cutedsl._streaming_signal_blockscaled import (  # noqa: F401  (require_sm100, require_sm10x: autouse fixture)
    Launcher,
    assemble_problem,
    cluster_id,
    make_problem,
    make_quant_kernel,
    notify_kernel,
    require_sm100,
    require_sm10x,
)

HIDDEN = 512
REPLAYS = 3


def run(launcher, problem):
    return launcher(problem)[0]


@pytest.mark.L0
@pytest.mark.parametrize("one_cta", [False, True], ids=lambda one_cta: f"fprop-mxfp8-{cluster_id(one_cta)}")
@pytest.mark.parametrize("n_and_chunk", [(768, 512), (1024, 256)], ids=["n768_c512", "n1024_c256"])
def test_mxfp8_output_notify_replays_match_dynamic(n_and_chunk, one_cta):
    """Replays on fresh buffers beside concurrent work give the same bytes, which equal the stock dynamic scheduler's,
    whether the extra clusters stop early (one cluster beside the consumer) or never stop."""
    n, chunk = n_and_chunk
    tokens = MANY_EXPERTS_64
    problem = make_problem(tokens, n, HIDDEN, seed=3)
    expected = snapshot(run(Launcher(make_quant_kernel(len(tokens), one_cta, use_dynamic_sched=True), problem), problem))
    results = {}
    for beside in (1, 1000):
        launcher = Launcher(notify_kernel(len(tokens), n, chunk, one_cta, beside), problem)
        results[beside] = assert_replays(lambda: run(launcher, problem), repeats=REPLAYS)
        assert_bitwise(results[beside], expected, f"N_CHUNK, {beside} cluster(s) beside the consumer, vs stock")
    assert_bitwise(results[1], results[1000], "early-stopping vs non-stopping extra clusters")


@pytest.mark.L0
def test_mxfp8_output_notify_chunk_size_invariance():
    """The chunk size reorders the tiles only: C = 256, 512 and 1024 (one chunk) give the same bytes."""
    n = 1024
    problem = make_problem(BIG_EXPERT, n, HIDDEN, seed=9)
    results = {chunk: snapshot(run(Launcher(notify_kernel(len(BIG_EXPERT), n, chunk), problem), problem)) for chunk in (256, 512, 1024)}
    assert_bitwise(results[512], results[256], "C = 512 vs 256")
    assert_bitwise(results[1024], results[256], "C = 1024 vs 256")


# Batch invariance: one target expert keeps its rows, codes, scales, B, alpha and prob while its neighbors change.  Every
# expert is quantized once, as one expert of a bank problem.  Four expert slots each (empty slots pad the count), so one
# compiled kernel serves every packing.  The packings are the BF16 suite's (test_grouped_gemm_streaming_signal_determinism).
BANK_TOKENS = {"target": 700, "n300": 300, "n513": 513, "r1": 1, "r255": 255, "r257": 257}
PACKINGS = {
    "alone": ["target", None, None, None],
    "neighbors": ["n300", "target", "n513", None],
    "empty_neighbors": [None, "target", None, None],
    "ragged": ["r1", "r255", "target", "r257"],
    "reordered": ["n513", "n300", None, "target"],
    "duplicated": ["target", "n300", "target", None],
}
SLOTS = 4
BATCH_ROWS = max(sum(padded(BANK_TOKENS[key]) for key in packing if key) for packing in PACKINGS.values())


def pack(bank, packing):
    """A problem of SLOTS experts in ``packing`` order over BATCH_ROWS rows, copied from ``bank``; returns it and the
    target's row ranges."""
    n, ends = bank["n"], bank["offsets"].tolist()
    a = torch.zeros(BATCH_ROWS, bank["k"], device="cuda", dtype=torch.float8_e4m3fn)
    a_exp = torch.full((BATCH_ROWS, bank["a_exp"].shape[1]), 127, device="cuda", dtype=torch.uint8)
    b, b_exp = torch.zeros_like(bank["b"][:SLOTS]), torch.full_like(bank["b_exp"][: SLOTS * n], 127)
    alpha, prob = torch.ones(SLOTS, device="cuda"), torch.ones(BATCH_ROWS, 1, 1, device="cuda")
    layout = torch.zeros(SLOTS + 1, device="cuda", dtype=torch.int32)
    begin, target_rows = 0, []
    for slot, key in enumerate(packing):
        if key is not None:
            e = list(BANK_TOKENS).index(key)
            rows = slice(ends[e - 1] if e else 0, ends[e])
            end = begin + rows.stop - rows.start
            a[begin:end], a_exp[begin:end], prob[begin:end] = bank["a"][rows], bank["a_exp"][rows], bank["prob"][rows]
            b[slot], b_exp[slot * n : (slot + 1) * n], alpha[slot] = bank["b"][e], bank["b_exp"][e * n : (e + 1) * n], bank["alpha"][e]
            if key == "target":
                target_rows.append(slice(begin, end))
            begin = end
        layout[slot + 1] = begin
    return assemble_problem(a, a_exp, b, b_exp, layout, alpha, prob), target_rows


@pytest.mark.L0
@pytest.mark.parametrize("one_cta", [False, True], ids=lambda one_cta: f"fprop-mxfp8-{cluster_id(one_cta)}")
def test_mxfp8_output_notify_target_expert_batch_invariance(one_cta):
    """The target expert's D bytes do not depend on what is packed around it: alone, with neighbors, behind empty
    experts, among ragged ones (1, 255, 257 rows), reordered, or duplicated."""
    n = 768
    bank = make_problem(list(BANK_TOKENS.values()), n, HIDDEN, seed=1234)
    problems = {name: pack(bank, packing) for name, packing in PACKINGS.items()}
    alone, alone_rows = problems["alone"]
    stock_out = run(Launcher(make_quant_kernel(SLOTS, one_cta, use_dynamic_sched=True), alone), alone)
    expected = snapshot({"d": stock_out["d"][alone_rows[0]]})
    launcher = Launcher(notify_kernel(SLOTS, n, 512, one_cta), alone)
    for name, (problem, target_rows) in problems.items():
        out = run(launcher, problem)
        for copy, rows in enumerate(target_rows):
            assert_bitwise(snapshot({"d": out["d"][rows]}), expected, f"target in {name} (copy {copy})")


@pytest.mark.L0
def test_mxfp8_output_notify_claim_slot_release_warm_stress():
    """The claim slot each cluster broadcasts is released only after every lane read it (python/cudnn/AGENTS.md Rule 11):
    back-to-back and shared-memory-poisoned N_CHUNK launches with many claims per cluster, every byte against stock."""
    from warm_stress import StressCase, assert_warm_stress

    n, clusters = 1024, 4
    problem = make_problem(BIG_EXPERT, n, HIDDEN, seed=13)
    assert problem["rows"] == sum(padded(t) for t in BIG_EXPERT), "every D row is written, so the sentinel fill is overwritten"
    stock, _ = Launcher(make_quant_kernel(len(BIG_EXPERT), use_dynamic_sched=True), problem)(problem)
    launcher = Launcher(notify_kernel(len(BIG_EXPERT), n, 256), problem, max_active_clusters=clusters)
    d, workspace = launcher.buffers(problem)
    claims = (problem["rows"] // ROWS_ALIGNMENT) * (n // 256)
    assert_warm_stress(
        StressCase(
            "MXFP8 N_CHUNK output notify, big expert, 4 clusters",
            launch=lambda: launcher.launch(problem, (d, workspace)),
            outputs={"d": d},
            thread_blocks=2 * clusters,
            refills_per_launch=claims // clusters,
            reference={"d": stock["d"]},
        ),
        warm=500,
        poisoned=500,
        budget_s=30.0,
    )

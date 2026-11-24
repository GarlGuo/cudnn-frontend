# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bitwise determinism of N_CHUNK output notify on the SM100 MXFP8 block-scaled MoE quant (FC2) forward kernel.

N_CHUNK changes which cluster runs a tile and when, never the tile math, so the BF16 D must be raw-byte equal across
replays, extra clusters that stop early or never stop, chunk sizes, neighboring experts and the stock dynamic scheduler.

Directions: forward only.  This is a forward kernel; backward is unsupported by contract, not untested.  Inputs are
quantized once (E4M3 codes, E8M0 SF atoms) and reused unchanged across every replay and packing.
"""

import pytest

from determinism_utils import assert_bitwise, assert_replays, snapshot
from gemm.cutedsl import _streaming_signal_blockscaled as bs
from gemm.cutedsl._streaming_signal_blockscaled import require_sm10x  # noqa: F401  (autouse fixture)

REPLAYS = 3


def run(launcher, problem):
    return launcher(problem)[0]


def case_ids(cases):
    return ["-".join([bs.KIND_IDS[kind], *map(str, rest)]) for kind, *rest in cases]


# ---- N_CHUNK output notify (FC2) ----


@pytest.mark.L0
@pytest.mark.parametrize("one_cta", [False, True], ids=lambda one_cta: f"fprop-mxfp8-{bs.cluster_id(one_cta)}")
@pytest.mark.parametrize("n_and_chunk", [(768, 512), (1024, 256)], ids=["n768_c512", "n1024_c256"])
def test_mxfp8_output_notify_replays_match_dynamic(n_and_chunk, one_cta):
    """Replays on fresh buffers beside concurrent work give the same bytes, which equal the stock dynamic scheduler's,
    whether the extra clusters stop early (one cluster beside the consumer) or never stop."""
    n, chunk = n_and_chunk
    tokens = bs.ROUTINGS["many_experts_64"]
    problem = bs.make_problem("quant", tokens, n, seed=3)
    expected = snapshot(run(bs.Launcher(bs.stock_kernel("quant", len(tokens), one_cta), problem), problem))
    for beside in (1, 1000):
        launcher = bs.Launcher(bs.notify_kernel(len(tokens), n, chunk, one_cta, beside), problem)
        assert_bitwise(assert_replays(lambda: run(launcher, problem), repeats=REPLAYS), expected, f"{beside} cluster(s) beside the consumer vs stock")


@pytest.mark.L0
def test_mxfp8_output_notify_chunk_size_invariance():
    """The chunk size reorders the tiles only: C = 256, 512 and 1024 (one chunk) give the same bytes."""
    tokens = bs.ROUTINGS["big_expert"]
    problem = bs.make_problem("quant", tokens, 1024, seed=9)
    results = {chunk: snapshot(run(bs.Launcher(bs.notify_kernel(len(tokens), 1024, chunk), problem), problem)) for chunk in (256, 512, 1024)}
    assert_bitwise(results[512], results[256], "C = 512 vs 256")
    assert_bitwise(results[1024], results[256], "C = 1024 vs 256")


# ---- Batch invariance and claim-slot warm stress ----

BATCH_CASES = [("quant", "cluster21"), ("quant", "cluster11")]


@pytest.mark.L0
@pytest.mark.parametrize("kind, variant", BATCH_CASES, ids=case_ids(BATCH_CASES))
def test_target_expert_batch_invariance(kind, variant):
    """The target expert's bytes do not depend on what is packed around it: alone, with neighbors, behind empty
    experts, among ragged ones (1, 255, 257 rows), reordered, or duplicated (N_CHUNK: N = 768, chunk 512)."""
    one_cta = variant == "cluster11"
    bank = bs.make_problem(kind, list(bs.BANK_TOKENS.values()), 768, seed=1234)
    problems = {name: bs.pack(bank, packing) for name, packing in bs.PACKINGS.items()}
    alone, alone_rows = problems["alone"]
    kernel = bs.notify_kernel(bs.SLOTS, 768, 512, one_cta)
    stock = bs.Launcher(bs.stock_kernel(kind, bs.SLOTS, one_cta), alone)(alone)[0]
    expected = snapshot({name: value[alone_rows[0]] for name, value in stock.items()})
    launcher = bs.Launcher(kernel, alone)
    for name, (problem, target_rows) in problems.items():
        out = run(launcher, problem)
        for copy, rows in enumerate(target_rows):
            assert_bitwise(snapshot({key: value[rows] for key, value in out.items()}), expected, f"target in {name} (copy {copy})")


WARM_CASES = [("quant", "n_chunk")]


@pytest.mark.L0
@pytest.mark.parametrize("kind, variant", WARM_CASES, ids=case_ids(WARM_CASES))
def test_claim_slot_release_warm_stress(kind, variant):
    """The claim slot each cluster broadcasts is released only after every lane read it (python/cudnn/AGENTS.md Rule
    11): back-to-back and shared-memory-poisoned launches of a 5000-row expert on 4 clusters (many claims per cluster),
    every byte against stock."""
    from warm_stress import StressCase, assert_warm_stress

    clusters, tokens = 4, bs.ROUTINGS["big_expert"]
    problem = bs.make_problem(kind, tokens, seed=13)
    assert problem["rows"] == sum(bs.padded(t) for t in tokens), "every output row is written, so the sentinel fill is overwritten"
    stock, _ = bs.Launcher(bs.stock_kernel(kind, len(tokens)), problem)(problem)
    kernel = bs.notify_kernel(len(tokens), problem["n"], 256)
    launcher = bs.Launcher(kernel, problem, max_active_clusters=clusters)
    buffers = launcher.buffers(problem)

    def launch():
        launcher.launch(problem, buffers)

    assert_warm_stress(
        StressCase(
            f"{bs.KIND_IDS[kind]} {variant}, 5000-row expert, {clusters} clusters",
            launch=launch,
            outputs=bs.outputs(kernel, problem, buffers),
            thread_blocks=2 * clusters,
            refills_per_launch=(problem["rows"] // bs.ROWS_ALIGNMENT) * (problem["n"] // 256) // clusters,
            reference=stock,
        ),
        warm=500,
        poisoned=500,
        budget_s=30.0,
    )

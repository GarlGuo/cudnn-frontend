# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bitwise determinism of N_CHUNK output notify on the MXFP8 block-scaled MoE quant (FC2) forward kernels (SM100; the
Rubin twin on SM107).

N_CHUNK changes which cluster runs a tile and when, never the tile math, so the BF16 D must be raw-byte equal across
replays, extra clusters that stop early or never stop, chunk sizes, neighboring experts and the stock dynamic scheduler.

Directions: forward only.  These are forward kernels; backward is unsupported by contract, not untested.  Inputs are
quantized once (E4M3 codes, E8M0 SF atoms) and reused unchanged across every packing.
"""

import pytest
import torch

from determinism_utils import assert_bitwise, assert_replays, snapshot
from gemm.cutedsl._streaming_signal import BIG_EXPERT, MANY_EXPERTS_64, ROWS_ALIGNMENT, padded
from gemm.cutedsl._streaming_signal_blockscaled import PACKINGS, SLOTS, Launcher, expert_bank, make_problem, make_quant_kernel, notify_config, pack, reference

HIDDEN = 512
REPLAYS = 3


@pytest.fixture(autouse=True)
def require_sm10x_cutedsl():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    from cudnn.tensor_adapter import has_tcgen05

    if major != 10 or not has_tcgen05(major):
        pytest.skip("an SM10x GPU (compute capability 10.x) is required")
    from cudnn.frost.buffers import cutedsl_arch_requirement_error, cutedsl_state, cutedsl_too_old

    installed, version = cutedsl_state()
    if not installed or cutedsl_too_old(version):
        pytest.skip("requires nvidia-cutlass-dsl at or above the library floor")
    reason = cutedsl_arch_requirement_error((major, minor))
    if reason:
        pytest.skip(reason)


def notify_kernel(experts, n, chunk, one_cta=False, clusters_beside_consumer=1):
    notify = notify_config(n, chunk, clusters_beside_consumer)
    return make_quant_kernel(experts, one_cta, use_dynamic_sched=True, streaming_output_notify=notify, launch_pdl_dependents_at_entry=True)


def run(launcher, problem):
    return launcher(problem)[0]


def cluster_id(one_cta):
    return "cluster11" if one_cta else "cluster21"


@pytest.mark.L0
@pytest.mark.parametrize("one_cta", [False, True], ids=lambda one_cta: f"fprop-mxfp8-{cluster_id(one_cta)}")
@pytest.mark.parametrize("n_and_chunk", [(768, 512), (1024, 256)], ids=["n768_c512", "n1024_c256"])
def test_mxfp8_output_notify_replays_match_dynamic(n_and_chunk, one_cta):
    """Replays on fresh buffers beside concurrent work give the same bytes, which equal the stock dynamic scheduler's,
    whether the extra clusters stop early (one cluster beside the consumer) or never stop."""
    n, chunk = n_and_chunk
    tokens = MANY_EXPERTS_64
    problem = make_problem(tokens, n, HIDDEN, seed=3)
    stock = Launcher(make_quant_kernel(len(tokens), one_cta, use_dynamic_sched=True), problem)
    expected = snapshot(run(stock, problem))
    torch.testing.assert_close(expected["d"].float(), reference(problem)[: expected["d"].shape[0]], atol=2e-2, rtol=2e-2)
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


@pytest.mark.L0
@pytest.mark.parametrize("one_cta", [False, True], ids=lambda one_cta: f"fprop-mxfp8-{cluster_id(one_cta)}")
def test_mxfp8_output_notify_target_expert_batch_invariance(one_cta):
    """The target expert's D bytes do not depend on what is packed around it: alone, with neighbors, behind empty
    experts, among ragged ones (1, 255, 257 rows), reordered, or duplicated."""
    n = 768
    bank = expert_bank(n, HIDDEN)
    problems = {name: pack(bank, packing, n, HIDDEN) for name, packing in PACKINGS.items()}
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

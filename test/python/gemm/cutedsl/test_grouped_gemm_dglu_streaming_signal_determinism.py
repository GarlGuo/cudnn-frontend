# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bitwise determinism of the streaming input signal on the dSwiGLU kernels: BF16 and MXFP8 on SM100 with
``deterministic=True`` (one dprob slot per N tile).

``streaming_signal_capacity`` changes which cluster runs a tile and when, never the tile math, so D and dprob must be
raw-byte equal across replays, producer release orders and the stock dynamic scheduler.

Directions: backward (the dGLU of FC1).  Inputs are quantized once and reused unchanged across every release order.
"""

import pytest
import torch

from determinism_utils import assert_bitwise, assert_replays, snapshot
from gemm.cutedsl._streaming_signal import BIG_EXPERT, MANY_EXPERTS_64, ROWS_ALIGNMENT, padded
from gemm.cutedsl._streaming_signal_input import signal_ready
from gemm.cutedsl._streaming_signal_dglu import (
    INTERMEDIATE,
    Launcher,
    assert_outputs_match,
    exact_outputs,
    is_rubin,
    make_dglu_kernel,
    make_problem,
    run_with_producer,
)

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


VARIANTS = {
    "signal": {},
    "signal_entry_trigger": dict(use_pdl=True, launch_pdl_dependents_at_entry=True),
    "compact_signal": dict(compact_sfa_sf_warp=True),
}
CASES = [
    ("bf16", "signal"),
    ("bf16", "signal_entry_trigger"),
    ("mxfp8", "signal"),
    ("mxfp8", "signal_entry_trigger"),
    ("mxfp8", "compact_signal"),
]


def skip_unsupported(recipe, variant):
    if recipe == "mxfp8" and is_rubin():
        pytest.skip("the SM100 MXFP8 kernel (the Rubin twin does not take a streaming input producer)")


def signal_kernel(recipe, problem, experts, variant):
    capacity = problem["rows"] // ROWS_ALIGNMENT
    return make_dglu_kernel(recipe, experts, use_dynamic_sched=True, streaming_signal_capacity=capacity, **VARIANTS[variant])


@pytest.mark.L0
@pytest.mark.parametrize("recipe, variant", CASES, ids=[f"{r}-{v}" for r, v in CASES])
def test_dglu_signal_replays_match_dynamic(recipe, variant):
    """Replays on fresh buffers beside concurrent work give the same D and dprob bytes, equal to the stock scheduler's."""
    skip_unsupported(recipe, variant)
    tokens = MANY_EXPERTS_64
    problem = make_problem(recipe, tokens, seed=5)
    expected = snapshot(Launcher(make_dglu_kernel(recipe, len(tokens), use_dynamic_sched=True), problem)(problem)[0])
    kernel = signal_kernel(recipe, problem, len(tokens), variant)
    launcher = Launcher(kernel, problem)
    replayed = assert_replays(lambda: exact_outputs(kernel, launcher(problem, signal_ready(kernel))[0]), repeats=REPLAYS)
    assert_bitwise(replayed, exact_outputs(kernel, expected), f"{recipe} {variant} vs stock")
    assert_outputs_match(kernel, launcher(problem, signal_ready(kernel))[0], expected, f"{recipe} {variant} vs stock")


RELEASE_CASES = [
    ("bf16", "signal", "a"),
    ("mxfp8", "signal", "a"),
    ("mxfp8", "signal", "sfa"),
    ("mxfp8", "compact_signal", "sfa"),
    ("mxfp8", "compact_signal", "both"),
]


@pytest.mark.L0
@pytest.mark.parametrize("recipe, variant, late", RELEASE_CASES, ids=[f"{r}-{v}-late_{l}" for r, v, l in RELEASE_CASES])
def test_dglu_producer_release_order_invariance(recipe, variant, late):
    """The GEMM starts while every M tile reads not-ready and the late operand holds NaN; a producer releases the tiles
    in order, in reverse and in a seeded random order.  Every order gives the stock scheduler's D and dprob bytes."""
    skip_unsupported(recipe, variant)
    tokens = MANY_EXPERTS_64
    problem = make_problem(recipe, tokens, seed=7)
    expected = snapshot(Launcher(make_dglu_kernel(recipe, len(tokens), use_dynamic_sched=True), problem)(problem)[0])
    kernel = signal_kernel(recipe, problem, len(tokens), variant)
    results = run_with_producer(kernel, problem, late, ["in_order", "reverse", "random"])
    for order, got in results.items():
        assert_bitwise(snapshot(exact_outputs(kernel, got)), exact_outputs(kernel, expected), f"{recipe} {variant}, {late} late, {order} release vs stock")
        assert_outputs_match(kernel, got, expected, f"{recipe} {variant}, {late} late, {order} release vs stock")
    for order in ("reverse", "random"):
        exact = snapshot(exact_outputs(kernel, results[order]))
        assert_bitwise(exact, snapshot(exact_outputs(kernel, results["in_order"])), f"{order} vs in-order release")


WARM_CASES = [("bf16", "signal"), ("mxfp8", "signal"), ("mxfp8", "compact_signal")]


@pytest.mark.L0
@pytest.mark.parametrize("recipe, variant", WARM_CASES, ids=[f"{r}-{v}" for r, v in WARM_CASES])
def test_dglu_signal_slot_release_warm_stress(recipe, variant):
    """The claim slot each cluster broadcasts, and with the SF warp its tile_info slot, are released only after every
    lane read them (python/cudnn/AGENTS.md Rule 11): back-to-back and shared-memory-poisoned launches with many claims
    per cluster, every byte against stock."""
    from warm_stress import StressCase, assert_warm_stress

    skip_unsupported(recipe, variant)
    clusters = 4
    problem = make_problem(recipe, BIG_EXPERT, seed=13)
    assert problem["rows"] == sum(padded(t) for t in BIG_EXPERT), "every D row is written, so the sentinel fill is overwritten"
    stock, _ = Launcher(make_dglu_kernel(recipe, len(BIG_EXPERT), use_dynamic_sched=True), problem)(problem)
    kernel = signal_kernel(recipe, problem, len(BIG_EXPERT), variant)
    launcher = Launcher(kernel, problem, max_active_clusters=clusters)
    d, dprob, workspace = launcher.buffers(problem)
    reset = signal_ready(kernel)

    def launch():
        # dprob slots are written once per launch (one writer each): zero them so every launch's sum starts equal.
        dprob.zero_()
        launcher.launch(problem, (d, dprob, workspace))

    claims = (problem["rows"] // ROWS_ALIGNMENT) * (INTERMEDIATE // 256)
    assert_warm_stress(
        StressCase(
            f"{recipe} dGLU streaming signal scheduler{', compact SFA' if variant.startswith('compact') else ''}, big expert, 4 clusters",
            launch=launch,
            outputs=exact_outputs(kernel, {"d": d, "dprob": dprob}),
            thread_blocks=2 * clusters,
            refills_per_launch=claims // clusters,
            reference=exact_outputs(kernel, {"d": stock["d"], "dprob": stock["dprob"]}),
            reset=lambda: reset(workspace),
        ),
        warm=500,
        poisoned=500,
        budget_s=30.0,
    )

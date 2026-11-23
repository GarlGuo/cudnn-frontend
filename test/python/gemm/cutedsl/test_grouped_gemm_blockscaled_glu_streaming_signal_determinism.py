# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bitwise determinism of the streaming input signal on the MXFP8 block-scaled SwiGLU (FC1) forward kernel (SM100; the
Rubin twin on SM107).

``streaming_signal_capacity`` (with or without ``device_layout``) changes which cluster runs a tile and when, never the
tile math, so the BF16 C and D must be raw-byte equal across replays, producer release orders, neighboring experts and
the stock dynamic scheduler.

Directions: forward only.  These are forward kernels; backward is unsupported by contract, not untested.  Inputs are
quantized once (E4M3 codes, E8M0 SF atoms) and reused unchanged across every packing and release order.
"""

import pytest
import torch

from determinism_utils import assert_bitwise, assert_replays, snapshot
from gemm.cutedsl._streaming_signal import BIG_EXPERT, MANY_EXPERTS_64, ROWS_ALIGNMENT, padded
from gemm.cutedsl._streaming_signal_blockscaled import (
    BATCH_ROWS,
    PACKINGS,
    SLOTS,
    Launcher,
    expert_bank,
    make_glu_kernel,
    make_problem,
    pack,
    poison_inputs,
    produce_tiles,
    release_order,
    signal_ready,
    signal_view,
    valid_rows,
)

HIDDEN = 512
N = 1024
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


def signal_options(capacity, variant):
    options = dict(use_dynamic_sched=True, streaming_signal_capacity=capacity)
    if variant.startswith("compact"):
        options["compact_sfa_sf_warp"] = True
    if variant.endswith("signal_layout_trigger"):
        options.update(device_layout=True, use_pdl=True, launch_pdl_dependents_at_all_streaming_signal_slots_ready=True)
    return options


ROUTINGS = {"many_experts_64": MANY_EXPERTS_64, "big_expert": BIG_EXPERT}


@pytest.mark.L0
@pytest.mark.parametrize("variant", ["signal", "signal_layout_trigger", "compact_signal", "compact_signal_layout_trigger"])
@pytest.mark.parametrize("routing", sorted(ROUTINGS))
def test_mxfp8_glu_signal_replays_match_dynamic(variant, routing):
    """Replays on fresh buffers beside concurrent work give the same C and D bytes, which equal the stock scheduler's."""
    tokens = ROUTINGS[routing]
    problem = make_problem(tokens, N, HIDDEN, seed=len(routing))
    expected = snapshot(Launcher(make_glu_kernel(len(tokens), use_dynamic_sched=True), problem)(problem)[0])
    kernel = make_glu_kernel(len(tokens), **signal_options(problem["rows"] // ROWS_ALIGNMENT, variant))
    launcher = Launcher(kernel, problem)
    replayed = assert_replays(lambda: launcher(problem, signal_ready(kernel))[0], repeats=REPLAYS)
    assert_bitwise(replayed, expected, "signal vs stock")


@pytest.mark.L0
@pytest.mark.parametrize("late, compact", [("a", False), ("sfa", False), ("sfa", True)], ids=["late_a", "late_sfa", "late_compact_sfa"])
def test_mxfp8_glu_producer_release_order_invariance(late, compact):
    """The GEMM starts while every M tile reads not-ready and the late operand holds NaN; a producer releases the tiles
    in order, in reverse and in a seeded random order.  Every order gives the stock scheduler's bytes."""
    from cudnn.gemm.cutedsl.grouped.moe_kernel_helpers import reset_streaming_signal_workspace

    tokens = MANY_EXPERTS_64
    problem = make_problem(tokens, N, HIDDEN, seed=7)
    capacity = problem["rows"] // ROWS_ALIGNMENT
    expected = snapshot(Launcher(make_glu_kernel(len(tokens), use_dynamic_sched=True), problem)(problem)[0])
    kernel = make_glu_kernel(len(tokens), use_dynamic_sched=True, streaming_signal_capacity=capacity, compact_sfa_sf_warp=compact)
    launcher = Launcher(kernel, problem, max_active_clusters=4)
    launcher(problem, signal_ready(kernel))
    torch.cuda.synchronize()
    gemm_stream, producer_stream = torch.cuda.Stream(), torch.cuda.Stream()
    results = {}
    for order in ("in_order", "reverse", "random"):
        buffers = launcher.buffers(problem)
        final = poison_inputs(problem, late)
        signal = signal_view(kernel, buffers[-1])
        gemm_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(gemm_stream):
            reset_streaming_signal_workspace(buffers[-1], kernel)
        producer_stream.wait_stream(gemm_stream)
        with torch.cuda.stream(producer_stream):
            torch.cuda._sleep(20_000_000)
            produce_tiles(problem, final, late, signal, release_order(order, capacity))
        launcher.launch(problem, buffers, gemm_stream)
        torch.cuda.synchronize()
        rows = valid_rows(problem)
        results[order] = snapshot({"c": buffers[0][:rows], "d": buffers[1][:rows]})
        assert_bitwise(results[order], expected, f"{late} late, {order} release vs stock")
    assert_bitwise(results["reverse"], results["in_order"], "reverse vs in-order release")
    assert_bitwise(results["random"], results["in_order"], "random vs in-order release")


@pytest.mark.L0
@pytest.mark.parametrize("device_layout", [False, True], ids=["host-layout", "device-layout"])
def test_mxfp8_glu_signal_target_expert_batch_invariance(device_layout):
    """The target expert's C and D bytes do not depend on what is packed around it: alone, with neighbors, behind empty
    experts, among ragged ones (1, 255, 257 rows), reordered, or duplicated."""
    bank = expert_bank(N, HIDDEN)
    problems = {name: pack(bank, packing, N, HIDDEN) for name, packing in PACKINGS.items()}
    alone, alone_rows = problems["alone"]
    stock_out = Launcher(make_glu_kernel(SLOTS, use_dynamic_sched=True), alone)(alone)[0]
    expected = snapshot({name: value[alone_rows[0]] for name, value in stock_out.items()})
    kernel = make_glu_kernel(SLOTS, use_dynamic_sched=True, streaming_signal_capacity=BATCH_ROWS // ROWS_ALIGNMENT, device_layout=device_layout)
    launcher = Launcher(kernel, alone)
    for name, (problem, target_rows) in problems.items():
        out = launcher(problem, signal_ready(kernel))[0]
        for copy, rows in enumerate(target_rows):
            assert_bitwise(snapshot({key: value[rows] for key, value in out.items()}), expected, f"target in {name} (copy {copy})")


@pytest.mark.L0
@pytest.mark.parametrize("compact", [False, True], ids=["sfa_atoms", "compact_sfa"])
def test_mxfp8_glu_signal_slot_release_warm_stress(compact):
    """The claim slot each cluster broadcasts, and with the SF warp its tile_info slot, are released only after every
    lane read them (python/cudnn/AGENTS.md Rule 11): back-to-back and shared-memory-poisoned launches with many claims
    per cluster, every byte against stock."""
    from warm_stress import StressCase, assert_warm_stress

    clusters = 4
    problem = make_problem(BIG_EXPERT, N, HIDDEN, seed=13)
    assert problem["rows"] == sum(padded(t) for t in BIG_EXPERT), "every C and D row is written, so the sentinel fill is overwritten"
    stock, _ = Launcher(make_glu_kernel(len(BIG_EXPERT), use_dynamic_sched=True), problem)(problem)
    kernel = make_glu_kernel(len(BIG_EXPERT), use_dynamic_sched=True, streaming_signal_capacity=problem["rows"] // ROWS_ALIGNMENT, compact_sfa_sf_warp=compact)
    launcher = Launcher(kernel, problem, max_active_clusters=clusters)
    c, d, workspace = launcher.buffers(problem)
    reset = signal_ready(kernel)
    claims = (problem["rows"] // ROWS_ALIGNMENT) * (N // 256)
    assert_warm_stress(
        StressCase(
            f"MXFP8 GLU streaming signal scheduler{', compact SFA' if compact else ''}, big expert, 4 clusters",
            launch=lambda: launcher.launch(problem, (c, d, workspace)),
            outputs={"c": c, "d": d},
            thread_blocks=2 * clusters,
            refills_per_launch=claims // clusters,
            reference={"c": stock["c"], "d": stock["d"]},
            reset=lambda: reset(workspace),
        ),
        warm=500,
        poisoned=500,
        budget_s=30.0,
    )

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming input signal of the MXFP8 block-scaled SwiGLU (FC1) forward kernel (SM100).

The producer writes A (E4M3 codes) and its scale factors as swizzled SF atoms, then releases ``signal[t] = t``; tile t
covers A rows ``[256 t, 256 (t + 1))`` and the SFA atom rows 2t and 2t + 1, the contiguous bytes
``[t * 256 * Kc, (t + 1) * 256 * Kc)``.  The signal scheduler only reorders tiles and waits, so C and D must be raw-byte
equal to the stock dynamic scheduler's.  A and SFA are poisoned independently (NaN codes, NaN scales) until their tile
is released: a tile loaded before its signal would carry NaN into D.

L0 keeps one case per behaviour; the remaining routings and release orders are L1.
"""

import pytest
import torch

from gemm.cutedsl._streaming_signal import BIG_EXPERT, MANY_EXPERTS_64, ROWS_ALIGNMENT
from gemm.cutedsl._streaming_signal_input import (
    Launcher,
    is_rubin,
    make_glu_kernel,
    make_problem,
    poison_inputs,
    produce_tiles,
    release_order,
    signal_ready,
    signal_view,
    valid_rows,
)

HIDDEN = 512
N = 1024  # 2 x 512 intermediate: gate and up


@pytest.fixture(autouse=True)
def require_sm10x():
    """An SM10x GPU with tcgen05 and a CuTe DSL that is new enough and targets the device."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from cudnn.frost.buffers import cutedsl_arch_requirement_error, cutedsl_state, cutedsl_too_old
    from cudnn.tensor_adapter import has_tcgen05

    cc = torch.cuda.get_device_capability()
    if cc[0] != 10 or not has_tcgen05(cc[0]):
        pytest.skip("an SM10x GPU (compute capability 10.x) is required")
    if cutedsl_too_old(cutedsl_state()[1]):
        pytest.skip(f"the CuTe DSL is older than the library floor: {cutedsl_state()[1]}")
    reason = cutedsl_arch_requirement_error(cc)
    if reason is not None:
        pytest.skip(reason)
    if is_rubin():
        pytest.skip("the SM100 GLU kernel (the Rubin twin does not take a streaming input producer)")


def levels(cases, l0, ids):
    """``cases`` as pytest params: L0 for those in ``l0``, L1 for the rest."""
    return [pytest.param(*case, marks=pytest.mark.L0 if case in l0 else pytest.mark.L1, id=ids(*case)) for case in cases]


# More than 32 experts take the batch opener several lane rounds; an expert of more than 16 M tiles needs several batches.
ROUTINGS = {"uneven": [700, 0, 256, 1], "many_experts_64": MANY_EXPERTS_64, "big_expert": BIG_EXPERT}


def stock(tokens):
    return make_glu_kernel(len(tokens), use_dynamic_sched=True)


def signal_kernel(tokens, capacity, variant="signal"):
    options = dict(use_dynamic_sched=True, streaming_signal_capacity=capacity)
    if variant == "signal_layout_trigger":
        options.update(device_layout=True, use_pdl=True, launch_pdl_dependents_at_all_streaming_signal_slots_ready=True)
    return make_glu_kernel(len(tokens), **options)


def assert_same_bytes(got, want, label=""):
    for name in want:
        assert torch.equal(got[name].view(torch.int16), want[name].view(torch.int16)), f"{label}: {name} differs from the stock scheduler"


MATCH_CASES = [(variant, routing) for variant in ("signal", "signal_layout_trigger") for routing in sorted(ROUTINGS)]
MATCH_L0 = {("signal", "many_experts_64"), ("signal_layout_trigger", "big_expert")}


@pytest.mark.parametrize("variant, routing", levels(MATCH_CASES, MATCH_L0, lambda variant, routing: f"{variant}-{routing}"))
def test_mxfp8_glu_signal_matches_dynamic(variant, routing):
    """Every M tile ready at launch: C and D equal the stock dynamic scheduler's bytes."""
    tokens = ROUTINGS[routing]
    problem = make_problem(tokens, N, HIDDEN, seed=len(routing))
    expected, _ = Launcher(stock(tokens), problem)(problem)
    kernel = signal_kernel(tokens, problem["rows"] // ROWS_ALIGNMENT, variant)
    got, _ = Launcher(kernel, problem)(problem, signal_ready(kernel))
    torch.cuda.synchronize()
    assert_same_bytes(got, expected)
    assert not torch.isnan(got["d"].float()).any()


@pytest.mark.L0
@pytest.mark.parametrize("late", ["a", "sfa"])
def test_mxfp8_glu_poison_reaches_d(late):
    """The detector works: with A codes or SFA scales poisoned, the stock kernel's D holds NaN."""
    tokens = ROUTINGS["uneven"]
    problem = make_problem(tokens, N, HIDDEN, seed=1)
    poison_inputs(problem, late)
    out, _ = Launcher(stock(tokens), problem)(problem)
    torch.cuda.synchronize()
    assert torch.isnan(out["d"].float()).any(), f"poisoned {late} left D finite: the producer tests could not see an early read"


def run_with_producer(tokens, problem, late, orders, kernel_options=None, device_layout=False):
    """The signal GEMM launched last on its stream, with every M tile not ready and the ``late`` operands poisoned; a
    producer stream writes them tile by tile in each order of ``orders``.  Returns {order: outputs}."""
    capacity = problem["rows"] // ROWS_ALIGNMENT
    kernel = make_glu_kernel(len(tokens), use_dynamic_sched=True, streaming_signal_capacity=capacity, device_layout=device_layout, **(kernel_options or {}))
    # A few clusters only, so the producer finds free SMs while the GEMM waits.
    launcher = Launcher(kernel, problem, max_active_clusters=4)
    launcher(problem, signal_ready(kernel))  # loads the module: under lazy loading a first launch waits for running work
    torch.cuda.synchronize()
    from cudnn.gemm.cutedsl.grouped.moe_kernel_helpers import reset_streaming_signal_workspace

    final_layout = problem["layout"].clone()
    gemm_stream, producer_stream = torch.cuda.Stream(), torch.cuda.Stream()
    results = {}
    for order in orders:
        buffers = launcher.buffers(problem)
        final = poison_inputs(problem, late)
        if device_layout:
            problem["layout"].fill_(-1)
        signal = signal_view(kernel, buffers[-1])
        gemm_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(gemm_stream):
            reset_streaming_signal_workspace(buffers[-1], kernel)
        producer_stream.wait_stream(gemm_stream)
        # Producer work goes before the GEMM's launch, so no host launch waits for the spinning GEMM; the sleep makes the
        # GEMM wait first.
        with torch.cuda.stream(producer_stream):
            torch.cuda._sleep(20_000_000)
            if device_layout:
                problem["layout"].copy_(final_layout)
            produce_tiles(problem, final, late, signal, release_order(order, capacity))
        launcher.launch(problem, buffers, gemm_stream)
        torch.cuda.synchronize()
        rows = valid_rows(problem)
        results[order] = {"c": buffers[0][:rows].clone(), "d": buffers[1][:rows].clone()}
    return results


PRODUCER_CASES = [
    (late, routing, order) for late in ("a", "sfa", "both") for routing in ("many_experts_64", "big_expert") for order in ("in_order", "reverse", "random")
]
PRODUCER_L0 = {("a", "many_experts_64", "reverse"), ("sfa", "many_experts_64", "random"), ("both", "big_expert", "in_order")}


@pytest.mark.parametrize("late, routing, order", levels(PRODUCER_CASES, PRODUCER_L0, lambda late, routing, order: f"late_{late}-{routing}-{order}"))
def test_mxfp8_glu_signal_waits_for_producer(late, routing, order):
    """The GEMM starts while every M tile reads not-ready and the ``late`` operands (A codes, SFA atoms, or both) hold
    NaN; a producer writes each tile's rows and SFA bytes, then releases it.  Reading any tile early carries NaN into D."""
    tokens = ROUTINGS[routing]
    problem = make_problem(tokens, N, HIDDEN, seed=7)
    expected, _ = Launcher(stock(tokens), problem)(problem)
    results = run_with_producer(tokens, problem, late, [order])
    assert_same_bytes(results[order], expected, f"{late} late, {order} release")


@pytest.mark.parametrize("late", levels([("both",), ("sfa",)], {("both",)}, lambda late: f"late_{late}"))
def test_mxfp8_glu_device_layout_waits_for_producer(late):
    """With device_layout the GEMM also starts before the producer writes the expert layout (-1 until then); the trigger
    warp lets a PDL dependent launch once every M tile is ready."""
    tokens = ROUTINGS["many_experts_64"]
    problem = make_problem(tokens, N, HIDDEN, seed=11)
    expected, _ = Launcher(stock(tokens), problem)(problem)
    options = dict(use_pdl=True, launch_pdl_dependents_at_all_streaming_signal_slots_ready=True)
    results = run_with_producer(tokens, problem, late, ["in_order"], kernel_options=options, device_layout=True)
    assert_same_bytes(results["in_order"], expected, "device layout")


@pytest.mark.parametrize("device_layout", levels([(False,), (True,)], {(True,)}, lambda device_layout: f"device_layout_{device_layout}"))
def test_mxfp8_glu_signal_with_no_rows_exits(device_layout):
    problem = make_problem([0, 0, 0, 0], N, HIDDEN)
    kernel = make_glu_kernel(4, use_dynamic_sched=True, streaming_signal_capacity=1, device_layout=device_layout)
    Launcher(kernel, problem)(problem, signal_ready(kernel))
    torch.cuda.synchronize()


@pytest.mark.L0
def test_mxfp8_glu_signal_rejects_invalid():
    from cudnn.gemm.cutedsl.grouped.moe_utils import MoEWeightMode

    with pytest.raises(ValueError, match="use_dynamic_sched"):
        make_glu_kernel(4, streaming_signal_capacity=4)
    with pytest.raises(ValueError, match="dense weights"):
        make_glu_kernel(4, use_dynamic_sched=True, streaming_signal_capacity=4, weight_mode=MoEWeightMode.DISCRETE)
    with pytest.raises(ValueError, match="device_layout"):
        make_glu_kernel(4, use_dynamic_sched=True, device_layout=True)
    with pytest.raises(ValueError, match="launch_pdl_dependents_at_all_streaming_signal_slots_ready"):
        make_glu_kernel(4, use_dynamic_sched=True, launch_pdl_dependents_at_all_streaming_signal_slots_ready=True)
    with pytest.raises(ValueError, match="cluster"):
        make_glu_kernel(4, use_dynamic_sched=True, streaming_signal_capacity=4, cluster_shape_mn=(4, 1))

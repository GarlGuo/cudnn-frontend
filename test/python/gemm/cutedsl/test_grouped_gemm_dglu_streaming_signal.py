# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming input signal of the dSwiGLU (backward FC1 activation) kernels: BF16 and MXFP8 on SM100 and, on SM107, the
MXFP8 Rubin twin.

The producer (an EP dispatch of the output gradient) writes dY's rows, and for MXFP8 their scales (SF atoms, or compact
rows with ``compact_sfa_sf_warp``), then releases ``signal[t] = t`` for the 256-row M tile t.  The signal scheduler only
reorders tiles and waits, so D and the deterministic dprob must be raw-byte equal to the stock dynamic scheduler's.  dY
and its scales are poisoned (NaN) until their tile is released: a tile loaded before its signal carries NaN into D.

L0 keeps one case per behaviour; the remaining routings and release orders are L1.
"""

import pytest
import torch

from gemm.cutedsl._streaming_signal import BIG_EXPERT, MANY_EXPERTS_64, ROWS_ALIGNMENT
from gemm.cutedsl._streaming_signal_blockscaled import signal_ready
from gemm.cutedsl._streaming_signal_dglu import (
    Launcher,
    is_rubin,
    make_dglu_kernel,
    make_problem,
    poison_inputs,
    run_with_producer,
)


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


def levels(cases, l0, ids):
    return [pytest.param(*case, marks=pytest.mark.L0 if case in l0 else pytest.mark.L1, id=ids(*case)) for case in cases]


ROUTINGS = {"uneven": [700, 0, 256, 1], "many_experts_64": MANY_EXPERTS_64, "big_expert": BIG_EXPERT}
SIGNAL_VARIANTS = {
    "signal": {},
    "signal_layout_trigger": dict(device_layout=True, use_pdl=True, launch_pdl_dependents_at_all_streaming_signal_slots_ready=True),
    "signal_entry_trigger": dict(use_pdl=True, launch_pdl_dependents_at_entry=True),
    # compact SFA (SM100 MXFP8 only): the SF warp builds the atoms from the dispatched compact rows.
    "compact_signal": dict(compact_sfa_sf_warp=True),
    "compact_cta_signal_layout_trigger": dict(
        compact_sfa_sf_warp=True, compact_sfa_cta_release=True, device_layout=True, use_pdl=True, launch_pdl_dependents_at_all_streaming_signal_slots_ready=True
    ),
}


def skip_unsupported(recipe, options):
    if options.get("compact_sfa_sf_warp") and (recipe != "mxfp8" or is_rubin()):
        pytest.skip("compact_sfa_sf_warp: the SM100 MXFP8 kernel only")


def stock(recipe, tokens):
    return make_dglu_kernel(recipe, len(tokens), use_dynamic_sched=True)


def assert_same_bytes(got, want, label=""):
    for name in want:
        assert torch.equal(got[name].view(torch.int16), want[name].view(torch.int16)), f"{label}: {name} differs from the stock scheduler"


MATCH_CASES = [(recipe, variant, routing) for recipe in ("bf16", "mxfp8") for variant in SIGNAL_VARIANTS for routing in sorted(ROUTINGS)]
MATCH_L0 = {
    ("bf16", "signal", "many_experts_64"),
    ("bf16", "signal_layout_trigger", "big_expert"),
    ("mxfp8", "signal", "big_expert"),
    ("mxfp8", "signal_entry_trigger", "uneven"),
    ("mxfp8", "compact_signal", "many_experts_64"),
    ("mxfp8", "compact_cta_signal_layout_trigger", "big_expert"),
}


@pytest.mark.parametrize("recipe, variant, routing", levels(MATCH_CASES, MATCH_L0, lambda r, v, g: f"{r}-{v}-{g}"))
def test_dglu_signal_matches_dynamic(recipe, variant, routing):
    """Every M tile ready at launch: D and dprob equal the stock dynamic scheduler's bytes."""
    options = SIGNAL_VARIANTS[variant]
    skip_unsupported(recipe, options)
    tokens = ROUTINGS[routing]
    problem = make_problem(recipe, tokens, seed=len(routing))
    expected, _ = Launcher(stock(recipe, tokens), problem)(problem)
    kernel = make_dglu_kernel(recipe, len(tokens), use_dynamic_sched=True, streaming_signal_capacity=problem["rows"] // ROWS_ALIGNMENT, **options)
    got, _ = Launcher(kernel, problem)(problem, signal_ready(kernel))
    torch.cuda.synchronize()
    assert_same_bytes(got, expected, f"{recipe} {variant}")
    assert not torch.isnan(got["d"].float()).any()


@pytest.mark.L0
@pytest.mark.parametrize("recipe, late", [("bf16", "a"), ("mxfp8", "a"), ("mxfp8", "sfa")])
def test_dglu_poison_reaches_d(recipe, late):
    """The detector works: with dY codes or scales poisoned, the stock kernel's D holds NaN."""
    tokens = ROUTINGS["uneven"]
    problem = make_problem(recipe, tokens, seed=1)
    poison_inputs(problem, late)
    out, _ = Launcher(stock(recipe, tokens), problem)(problem)
    torch.cuda.synchronize()
    assert torch.isnan(out["d"].float()).any(), f"poisoned {late} left D finite: the producer tests could not see an early read"


@pytest.mark.L0
def test_dglu_compact_sfa_poison_reaches_d():
    """The detector works for the compact form: with the compact rows poisoned, D holds NaN."""
    if is_rubin():
        pytest.skip("compact_sfa_sf_warp: the SM100 MXFP8 kernel only")
    tokens = ROUTINGS["uneven"]
    problem = make_problem("mxfp8", tokens, seed=1)
    poison_inputs(problem, "sfa")
    out, _ = Launcher(make_dglu_kernel("mxfp8", len(tokens), use_dynamic_sched=True, compact_sfa_sf_warp=True), problem)(problem)
    torch.cuda.synchronize()
    assert torch.isnan(out["d"].float()).any()


PRODUCER_CASES = [
    ("bf16", "signal", "a"),
    ("mxfp8", "signal", "a"),
    ("mxfp8", "signal", "sfa"),
    ("mxfp8", "signal", "both"),
    ("mxfp8", "compact_signal", "sfa"),
    ("mxfp8", "compact_signal", "both"),
    ("mxfp8", "compact_cta_signal", "both"),
]
PRODUCER_L0 = {("bf16", "signal", "a"), ("mxfp8", "signal", "both"), ("mxfp8", "compact_signal", "both"), ("mxfp8", "compact_cta_signal", "both")}
PRODUCER_OPTIONS = {
    "signal": {},
    "compact_signal": dict(compact_sfa_sf_warp=True),
    "compact_cta_signal": dict(compact_sfa_sf_warp=True, compact_sfa_cta_release=True),
}


@pytest.mark.parametrize("recipe, variant, late", levels(PRODUCER_CASES, PRODUCER_L0, lambda r, v, l: f"{r}-{v}-late_{l}"))
def test_dglu_signal_waits_for_producer(recipe, variant, late):
    """The GEMM starts while every M tile reads not-ready and the late operands hold NaN; a producer stream writes and
    releases the tiles in order, in reverse and in a seeded random order.  Every order gives the stock bytes."""
    options = PRODUCER_OPTIONS[variant]
    skip_unsupported(recipe, options)
    tokens = ROUTINGS["many_experts_64"]
    problem = make_problem(recipe, tokens, seed=7)
    expected, _ = Launcher(stock(recipe, tokens), problem)(problem)
    kernel = make_dglu_kernel(recipe, len(tokens), use_dynamic_sched=True, streaming_signal_capacity=problem["rows"] // ROWS_ALIGNMENT, **options)
    results = run_with_producer(kernel, problem, late, ["in_order", "reverse", "random"])
    for order, got in results.items():
        assert_same_bytes(got, expected, f"{recipe} {variant}, {late} late, {order} release")


@pytest.mark.L0
@pytest.mark.parametrize("recipe", ["bf16", "mxfp8"])
def test_dglu_signal_device_layout_waits_for_producer(recipe):
    """With ``device_layout`` the producer also writes the expert layout while the GEMM runs (the dispatch's case)."""
    tokens = ROUTINGS["big_expert"]
    problem = make_problem(recipe, tokens, seed=11)
    expected, _ = Launcher(stock(recipe, tokens), problem)(problem)
    options = dict(device_layout=True, use_pdl=True, launch_pdl_dependents_at_all_streaming_signal_slots_ready=True)
    kernel = make_dglu_kernel(recipe, len(tokens), use_dynamic_sched=True, streaming_signal_capacity=problem["rows"] // ROWS_ALIGNMENT, **options)
    results = run_with_producer(kernel, problem, "a", ["random"], device_layout=True)
    assert_same_bytes(results["random"], expected, f"{recipe} device layout")


@pytest.mark.L0
@pytest.mark.parametrize("recipe", ["bf16", "mxfp8"])
def test_dglu_launch_pdl_dependents_at_entry_matches_dynamic(recipe):
    """``launch_pdl_dependents_at_entry`` on the stock dynamic scheduler changes no byte."""
    tokens = ROUTINGS["uneven"]
    problem = make_problem(recipe, tokens, seed=3)
    expected, _ = Launcher(stock(recipe, tokens), problem)(problem)
    kernel = make_dglu_kernel(recipe, len(tokens), use_dynamic_sched=True, use_pdl=True, launch_pdl_dependents_at_entry=True)
    got, _ = Launcher(kernel, problem)(problem)
    torch.cuda.synchronize()
    assert_same_bytes(got, expected, f"{recipe} entry trigger")


@pytest.mark.L0
@pytest.mark.parametrize("recipe", ["bf16", "mxfp8"])
def test_dglu_signal_rejects_invalid(recipe):
    from cudnn.gemm.cutedsl.grouped.moe_utils import MoEWeightMode

    with pytest.raises(ValueError, match="use_dynamic_sched"):
        make_dglu_kernel(recipe, 4, streaming_signal_capacity=4)
    with pytest.raises(ValueError, match="device_layout requires"):
        make_dglu_kernel(recipe, 4, use_dynamic_sched=True, device_layout=True)
    with pytest.raises(ValueError, match="launch_pdl_dependents_at_all_streaming_signal_slots_ready requires"):
        make_dglu_kernel(recipe, 4, use_dynamic_sched=True, launch_pdl_dependents_at_all_streaming_signal_slots_ready=True)
    with pytest.raises(ValueError, match="cluster"):
        make_dglu_kernel(recipe, 4, use_dynamic_sched=True, streaming_signal_capacity=4, cluster_shape_mn=(4, 1))
    if recipe == "mxfp8":
        with pytest.raises(ValueError, match="dense weights"):
            make_dglu_kernel(recipe, 4, use_dynamic_sched=True, streaming_signal_capacity=4, weight_mode=MoEWeightMode.DISCRETE)


@pytest.mark.L0
def test_dglu_compact_sfa_rejects_invalid():
    if is_rubin():
        pytest.skip("compact_sfa_sf_warp: the SM100 MXFP8 kernel only")
    with pytest.raises(ValueError, match="compact_sfa_sf_warp"):
        make_dglu_kernel("mxfp8", 4, use_dynamic_sched=True, compact_sfa_sf_warp=True, use_2cta_instrs=False, mma_tiler_mn=(128, 256), cluster_shape_mn=(1, 1))
    with pytest.raises(ValueError, match="compact_sfa_cta_release requires compact_sfa_sf_warp"):
        make_dglu_kernel("mxfp8", 4, use_dynamic_sched=True, compact_sfa_cta_release=True)

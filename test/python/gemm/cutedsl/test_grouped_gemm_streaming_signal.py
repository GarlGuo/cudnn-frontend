# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming signal options of the BF16 grouped GEMM kernels (plain and GLU).

Each signal path must give the same bits as the stock dynamic scheduler: the tile math is the same, only the order
in which tiles run and the readiness waits differ.
"""

import pytest
import torch

pytestmark = pytest.mark.L0

EXPERTS = 4
HIDDEN = 512
FFN = 512
ROWS_ALIGNMENT = 256


def require_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("SM100 / SM103 is required")


def kernel_classes():
    from cudnn.gemm.cutedsl.grouped.unfused.moe_grouped_gemm import MoEGroupedGemmBf16Kernel
    from cudnn.gemm.cutedsl.grouped.glu.moe_grouped_gemm_glu_bias import MoEGroupedGemmGluBiasBf16Kernel

    return MoEGroupedGemmBf16Kernel, MoEGroupedGemmGluBiasBf16Kernel


def make_kernel(glu, one_cta=False, **options):
    import cutlass
    from cudnn.gemm.cutedsl.grouped.moe_utils import MoEWeightMode

    plain_class, glu_class = kernel_classes()
    common = dict(
        acc_dtype=cutlass.Float32,
        use_2cta_instrs=not one_cta,
        mma_tiler_mn=(128, 256) if one_cta else (256, 256),
        cluster_shape_mn=(1, 1) if one_cta else (2, 1),
        vectorized_f32=True,
        expert_cnt=EXPERTS,
        weight_mode=MoEWeightMode.DENSE,
    )
    if glu:
        return glu_class(generate_c=True, **common, **options)
    return plain_class(generate_c=True, enable_bias=False, **common, **options)


def make_problem(tokens_per_expert, n, k, seed=0):
    """Expert-major rows padded to ROWS_ALIGNMENT; padding rows hold data too (the kernel computes them)."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    padded = [-(-tokens // ROWS_ALIGNMENT) * ROWS_ALIGNMENT for tokens in tokens_per_expert]
    rows = max(sum(padded), ROWS_ALIGNMENT)
    a = torch.randn(rows, k, generator=generator, device="cuda", dtype=torch.bfloat16) * 0.25
    b = torch.randn(EXPERTS, n, k, generator=generator, device="cuda", dtype=torch.bfloat16) * 0.25
    # A device layout reads layout[0] (expert 0's start) before padded_offsets = layout[1:E+1].
    layout = torch.zeros(EXPERTS + 1, device="cuda", dtype=torch.int32)
    layout[1:] = torch.tensor(padded, device="cuda", dtype=torch.int64).cumsum(0).to(torch.int32)
    offsets = layout[1:]
    alpha = torch.rand(EXPERTS, generator=generator, device="cuda") + 0.5
    prob = torch.rand(rows, 1, 1, generator=generator, device="cuda")
    return {"a": a, "b": b, "layout": layout, "offsets": offsets, "alpha": alpha, "prob": prob, "rows": rows, "n": n, "k": k}


def as_mkl(x):
    m, k = x.shape
    return x.as_strided((m, k, 1), (k, 1, m * k))


def run(kernel, problem, glu, prepare_workspace=None, max_active_clusters=None, stream=None, defer_launch=False):
    """Compile and launch ``kernel`` on ``problem``; returns (C, D, workspace), plus the launch itself with
    ``defer_launch``.  ``prepare_workspace(ws)`` runs on the launch stream first."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.nvgpu import OperandMajorMode
    from cutlass.cute.runtime import from_dlpack
    from cutlass.utils import HardwareInfo

    rows, n, k = problem["rows"], problem["n"], problem["k"]
    d_cols = n // 2 if glu else n
    c = torch.full((rows, n), float("nan"), device="cuda", dtype=torch.bfloat16)
    d = torch.full((rows, d_cols), float("nan"), device="cuda", dtype=torch.bfloat16)
    workspace = torch.zeros(max(kernel.get_workspace_bytes(), 1), device="cuda", dtype=torch.uint8)
    stream = stream or torch.cuda.current_stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        if prepare_workspace is not None:
            prepare_workspace(workspace)
    if max_active_clusters is None:
        max_active_clusters = HardwareInfo().get_max_active_clusters(2)

    def tensor(x):
        return from_dlpack(x, assumed_align=16)

    b = problem["b"].permute(1, 2, 0)
    args = [
        tensor(as_mkl(problem["a"])),
        tensor(b),
        cutlass.Int32(n),
        cutlass.Int32(k),
        cutlass.Int64(k),
        OperandMajorMode.K,
        from_dlpack(workspace, assumed_align=128).iterator,
        tensor(as_mkl(c)),
        tensor(as_mkl(d)),
        from_dlpack(problem["offsets"], assumed_align=4),
        tensor(problem["alpha"]),
    ]
    prob = tensor(problem["prob"])
    args += [prob, None] if glu else [None, prob]
    cu_stream = cuda.CUstream(stream.cuda_stream)
    compiled = cute.compile(kernel, *args, max_active_clusters, cu_stream)
    runtime_args = [x for i, x in enumerate(args) if i != 5]

    def launch():
        compiled(*runtime_args, cu_stream)

    if defer_launch:
        return c, d, workspace, launch
    launch()
    return c, d, workspace


def plain_reference(problem):
    a, b, alpha, prob = problem["a"].float(), problem["b"].float(), problem["alpha"], problem["prob"][:, 0, 0]
    out = torch.zeros(problem["rows"], problem["n"], device="cuda")
    begin = 0
    for expert, end in enumerate(problem["offsets"].tolist()):
        out[begin:end] = alpha[expert] * (a[begin:end] @ b[expert].T)
        begin = end
    return out, out * prob[:, None]


def signal_ready(kernel):
    from cudnn.gemm.cutedsl.grouped.moe_signal_helpers import reset_streaming_signal_workspace

    return lambda workspace: reset_streaming_signal_workspace(workspace, kernel, all_complete=True)


def valid_rows(problem):
    return int(problem["offsets"][-1].item())


ROUTINGS = {
    "uneven": [700, 0, 256, 1],
    "one_expert": [1300, 0, 0, 0],
}


@pytest.mark.parametrize("routing", sorted(ROUTINGS))
@pytest.mark.parametrize("variant", ["signal", "signal_layout_pdl"])
@pytest.mark.parametrize("glu", [False, True], ids=["plain", "glu"])
def test_signal_consumer_matches_dynamic(glu, variant, routing):
    require_sm100()
    n = 2 * FFN if glu else FFN
    problem = make_problem(ROUTINGS[routing], n, HIDDEN, seed=len(routing))
    capacity = problem["rows"] // ROWS_ALIGNMENT
    options = dict(use_dynamic_sched=True, streaming_signal_capacity=capacity)
    if variant == "signal_layout_pdl":
        options.update(device_layout=True, use_pdl=True)
        trigger = "launch_pdl_dependents_at_all_streaming_signal_slots_ready" if glu else "launch_pdl_dependents_at_entry"
        options[trigger] = True
    stock_c, stock_d, _ = run(make_kernel(glu, use_dynamic_sched=True), problem, glu)
    signal_kernel = make_kernel(glu, **options)
    c, d, _ = run(signal_kernel, problem, glu, prepare_workspace=signal_ready(signal_kernel))
    torch.cuda.synchronize()
    rows = valid_rows(problem)
    assert torch.equal(c[:rows], stock_c[:rows])
    assert torch.equal(d[:rows], stock_d[:rows])
    if not glu:
        c_ref, d_ref = plain_reference(problem)
        torch.testing.assert_close(c[:rows].float(), c_ref[:rows], atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(d[:rows].float(), d_ref[:rows], atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("device_layout", [False, True])
def test_signal_consumer_with_no_rows_exits(device_layout):
    require_sm100()
    problem = make_problem([0] * EXPERTS, FFN, HIDDEN)
    kernel = make_kernel(False, use_dynamic_sched=True, streaming_signal_capacity=1, device_layout=device_layout)
    run(kernel, problem, False, prepare_workspace=signal_ready(kernel))
    torch.cuda.synchronize()


def test_signal_consumer_waits_for_producer():
    """The GEMM starts while every signal slot reads not-ready and A holds NaN; a later kernel on another stream writes A one
    M tile at a time and then marks it ready.  Reading any tile early would carry NaN into D."""
    require_sm100()
    problem = make_problem([700, 300, 256, 513], FFN, HIDDEN, seed=7)
    rows = problem["rows"]
    capacity = rows // ROWS_ALIGNMENT
    final_a = problem["a"].clone()
    problem["a"].fill_(float("nan"))
    kernel = make_kernel(False, use_dynamic_sched=True, streaming_signal_capacity=capacity)
    stock_c, stock_d, _ = run(make_kernel(False, use_dynamic_sched=True), dict(problem, a=final_a), False)
    torch.cuda.synchronize()

    from cudnn.gemm.cutedsl.grouped.moe_signal_helpers import reset_streaming_signal_workspace, streaming_signal_workspace_offsets

    gemm_stream, producer_stream = torch.cuda.Stream(), torch.cuda.Stream()
    holder = {}

    def prepare(workspace):
        reset_streaming_signal_workspace(workspace, kernel)
        signal_word = streaming_signal_workspace_offsets(kernel)["signal"] // 4
        holder["signal"] = workspace.view(torch.int32)[signal_word : signal_word + capacity]

    # A few clusters only, so the producer kernels find free SMs while the GEMM waits.
    c, d, workspace, launch = run(kernel, problem, False, max_active_clusters=4, stream=gemm_stream, defer_launch=True)
    # One launch with every signal slot ready loads the GEMM's module: under lazy loading a first launch waits for running work.
    with torch.cuda.stream(gemm_stream):
        reset_streaming_signal_workspace(workspace, kernel, all_complete=True)
    launch()
    torch.cuda.synchronize()
    c.fill_(float("nan"))
    d.fill_(float("nan"))
    gemm_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(gemm_stream):
        prepare(workspace)
    producer_stream.wait_stream(gemm_stream)
    # Producer launches go before the GEMM's, so no host launch has to wait for the spinning GEMM.  The sleep makes
    # the GEMM start waiting before the first row lands.
    with torch.cuda.stream(producer_stream):
        torch.cuda._sleep(50_000_000)
        for tile in reversed(range(capacity)):
            rows_of_tile = slice(tile * ROWS_ALIGNMENT, (tile + 1) * ROWS_ALIGNMENT)
            problem["a"][rows_of_tile].copy_(final_a[rows_of_tile])
            holder["signal"][tile].fill_(tile)
    launch()
    torch.cuda.synchronize()
    valid = valid_rows(problem)
    assert torch.equal(c[:valid], stock_c[:valid])
    assert torch.equal(d[:valid], stock_d[:valid])


@pytest.mark.parametrize("glu", [False, True], ids=["plain", "glu"])
def test_device_layout_waits_for_producer(glu):
    """The GEMM starts while the layout reads -1, every signal slot reads not-ready and A holds NaN; a later kernel on
    another stream writes the layout, then A one M tile at a time with its signal slot.  The GLU also triggers once every signal slot is ready."""
    require_sm100()
    n = 2 * FFN if glu else FFN
    problem = make_problem([700, 300, 256, 513], n, HIDDEN, seed=11)
    capacity = problem["rows"] // ROWS_ALIGNMENT
    final_a, final_layout = problem["a"].clone(), problem["layout"].clone()
    options = dict(use_dynamic_sched=True, streaming_signal_capacity=capacity, device_layout=True)
    if glu:
        options["launch_pdl_dependents_at_all_streaming_signal_slots_ready"] = True
    kernel = make_kernel(glu, **options)
    stock_c, stock_d, _ = run(make_kernel(glu, use_dynamic_sched=True), problem, glu)
    torch.cuda.synchronize()

    from cudnn.gemm.cutedsl.grouped.moe_signal_helpers import reset_streaming_signal_workspace, streaming_signal_workspace_offsets

    gemm_stream, producer_stream = torch.cuda.Stream(), torch.cuda.Stream()
    c, d, workspace, launch = run(kernel, problem, glu, max_active_clusters=4, stream=gemm_stream, defer_launch=True)
    # One launch with the layout written and every signal slot ready loads the module (lazy loading waits for running work).
    with torch.cuda.stream(gemm_stream):
        reset_streaming_signal_workspace(workspace, kernel, all_complete=True)
    launch()
    torch.cuda.synchronize()
    signal_word = streaming_signal_workspace_offsets(kernel)["signal"] // 4
    signal = workspace.view(torch.int32)[signal_word : signal_word + capacity]
    c.fill_(float("nan"))
    d.fill_(float("nan"))
    problem["a"].fill_(float("nan"))
    problem["layout"].fill_(-1)
    gemm_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(gemm_stream):
        reset_streaming_signal_workspace(workspace, kernel)
    producer_stream.wait_stream(gemm_stream)
    # Producer launches go before the GEMM's; the sleep makes the GEMM wait on the layout first.
    with torch.cuda.stream(producer_stream):
        torch.cuda._sleep(50_000_000)
        problem["layout"].copy_(final_layout)
        for tile in range(capacity):
            rows_of_tile = slice(tile * ROWS_ALIGNMENT, (tile + 1) * ROWS_ALIGNMENT)
            problem["a"][rows_of_tile].copy_(final_a[rows_of_tile])
            signal[tile].fill_(tile)
    launch()
    torch.cuda.synchronize()
    valid = valid_rows(problem)
    assert torch.equal(c[:valid], stock_c[:valid])
    assert torch.equal(d[:valid], stock_d[:valid])


@pytest.mark.parametrize("one_cta", [False, True], ids=["cluster21", "cluster11"])
@pytest.mark.parametrize("clusters_beside_consumer", [1, 1000])
def test_output_notify_publishes_every_chunk(clusters_beside_consumer, one_cta):
    require_sm100()
    from cudnn.gemm.cutedsl.grouped.moe_streaming_output_chunk_tile_scheduler import (
        StreamingOutputNotifyConfig,
        StreamingOutputNotifyMode,
        streaming_output_notify_capacity,
        streaming_output_notify_offsets,
    )

    n, chunk = 2 * HIDDEN, 256
    problem = make_problem([700, 0, 256, 1], n, FFN, seed=3)
    capacity = streaming_output_notify_capacity(chunk, n)
    notify = StreamingOutputNotifyConfig(StreamingOutputNotifyMode.N_CHUNK, chunk, capacity, clusters_beside_consumer=clusters_beside_consumer)
    stock_c, stock_d, _ = run(make_kernel(False, one_cta, use_dynamic_sched=True), problem, False)
    kernel = make_kernel(False, one_cta, use_dynamic_sched=True, streaming_output_notify=notify, launch_pdl_dependents_at_entry=True)
    c, d, workspace = run(kernel, problem, False)
    torch.cuda.synchronize()
    valid = valid_rows(problem)
    assert torch.equal(c[:valid], stock_c[:valid])
    assert torch.equal(d[:valid], stock_d[:valid])
    offset = streaming_output_notify_offsets(kernel)["signal"]
    signal = workspace[offset : offset + 4 * capacity].view(torch.int32)
    assert signal.tolist() == list(range(capacity))


def test_streaming_options_reject_invalid():
    from cudnn.gemm.cutedsl.grouped.moe_streaming_output_chunk_tile_scheduler import StreamingOutputNotifyConfig, StreamingOutputNotifyMode

    with pytest.raises(ValueError, match="use_dynamic_sched"):
        make_kernel(False, streaming_signal_capacity=4)
    for glu in (False, True):
        with pytest.raises(ValueError, match="one CTA"):
            make_kernel(glu, one_cta=True, use_dynamic_sched=True, streaming_signal_capacity=4)
    with pytest.raises(ValueError, match="device_layout"):
        make_kernel(False, use_dynamic_sched=True, device_layout=True)
    with pytest.raises(ValueError, match="launch_pdl_dependents_at_all_streaming_signal_slots_ready"):
        make_kernel(True, use_dynamic_sched=True, launch_pdl_dependents_at_all_streaming_signal_slots_ready=True)
    notify = StreamingOutputNotifyConfig(StreamingOutputNotifyMode.N_CHUNK, 256, 4, clusters_beside_consumer=1)
    with pytest.raises(ValueError, match="use_dynamic_sched"):
        make_kernel(False, streaming_output_notify=notify)
    with pytest.raises(AssertionError):
        make_kernel(False, use_dynamic_sched=True, streaming_signal_capacity=4, streaming_output_notify=notify)
    import cutlass

    plain_class, _ = kernel_classes()
    with pytest.raises(ValueError, match="cluster"):
        plain_class(
            cutlass.Float32,
            True,
            (256, 256),
            (2, 2),
            True,
            True,
            False,
            EXPERTS,
            use_dynamic_sched=True,
            streaming_signal_capacity=4,
        )

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


@pytest.fixture(autouse=True)
def require_sm10x():
    """Every test here: an SM10x GPU (compute capability 10.x) with tcgen05, and a CuTe DSL that is new enough and
    targets the device."""
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


def kernel_classes():
    from cudnn.gemm.cutedsl.grouped.unfused.moe_grouped_gemm import MoEGroupedGemmBf16Kernel
    from cudnn.gemm.cutedsl.grouped.glu.moe_grouped_gemm_glu_bias import MoEGroupedGemmGluBiasBf16Kernel

    return MoEGroupedGemmBf16Kernel, MoEGroupedGemmGluBiasBf16Kernel


def frozen(value):
    """A hashable, order-independent stand-in for a kernel option (tuples, enums, plain config objects)."""
    if isinstance(value, dict):
        return tuple(sorted((key, frozen(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(frozen(item) for item in value)
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return (type(value).__qualname__, frozen(vars(value)))
    return value


def make_kernel(glu, one_cta=False, experts=EXPERTS, **options):
    import cutlass
    from cudnn.gemm.cutedsl.grouped.moe_utils import MoEWeightMode

    plain_class, glu_class = kernel_classes()
    common = dict(
        acc_dtype=cutlass.Float32,
        use_2cta_instrs=not one_cta,
        mma_tiler_mn=(128, 256) if one_cta else (256, 256),
        cluster_shape_mn=(1, 1) if one_cta else (2, 1),
        vectorized_f32=True,
        expert_cnt=experts,
        weight_mode=MoEWeightMode.DENSE,
    )
    if glu:
        kernel = glu_class(generate_c=True, **common, **options)
    else:
        kernel = plain_class(generate_c=True, enable_bias=False, **common, **options)
    # Every constructor argument, so ``run`` reuses a compile only for an identically built kernel.
    kernel.test_compile_key = (type(kernel).__qualname__, frozen(common), frozen(options))
    return kernel


# One compile per (kernel construction, argument layout, max_active_clusters) in this process: tensors are passed with
# static layouts, so their shapes and strides are part of the key; their values and the launch buffers are not.
COMPILED = {}


def tensor_layout(x):
    return (tuple(x.shape), tuple(x.stride()), str(x.dtype))


def make_problem(tokens_per_expert, n, k, seed=0):
    """Expert-major rows padded to ROWS_ALIGNMENT; padding rows hold data too (the kernel computes them).  One expert
    per entry of ``tokens_per_expert``."""
    experts = len(tokens_per_expert)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    padded = [-(-tokens // ROWS_ALIGNMENT) * ROWS_ALIGNMENT for tokens in tokens_per_expert]
    rows = max(sum(padded), ROWS_ALIGNMENT)
    a = torch.randn(rows, k, generator=generator, device="cuda", dtype=torch.bfloat16) * 0.25
    b = torch.randn(experts, n, k, generator=generator, device="cuda", dtype=torch.bfloat16) * 0.25
    # A device layout reads layout[0] (expert 0's start) before padded_offsets = layout[1:E+1].
    layout = torch.zeros(experts + 1, device="cuda", dtype=torch.int32)
    layout[1:] = torch.tensor(padded, device="cuda", dtype=torch.int64).cumsum(0).to(torch.int32)
    offsets = layout[1:]
    alpha = torch.rand(experts, generator=generator, device="cuda") + 0.5
    prob = torch.rand(rows, 1, 1, generator=generator, device="cuda")
    return {"a": a, "b": b, "layout": layout, "offsets": offsets, "alpha": alpha, "prob": prob, "rows": rows, "n": n, "k": k}


def as_mkl(x):
    m, k = x.shape
    return x.as_strided((m, k, 1), (k, 1, m * k))


def run(kernel, problem, glu, prepare_workspace=None, max_active_clusters=None, stream=None, defer_launch=False):
    """Compile ``kernel`` on ``problem`` and launch it, or with ``defer_launch`` leave the launch to the caller; returns
    (C, D, workspace, launch), ``launch`` None once launched.  ``prepare_workspace(ws)`` runs on the launch stream first."""
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
    layouts = [as_mkl(problem["a"]), b, as_mkl(c), as_mkl(d), problem["offsets"], problem["alpha"], problem["prob"], workspace]
    key = (kernel.test_compile_key, glu, n, k, max_active_clusters, tuple(tensor_layout(x) for x in layouts))
    compiled = COMPILED.get(key)
    if compiled is None:
        compiled = COMPILED[key] = cute.compile(kernel, *args, max_active_clusters, cu_stream)
    runtime_args = [x for i, x in enumerate(args) if i != 5]

    def launch():
        compiled(*runtime_args, cu_stream)

    if defer_launch:
        return c, d, workspace, launch
    launch()
    return c, d, workspace, None


def plain_reference(problem):
    a, b, alpha, prob = problem["a"].float(), problem["b"].float(), problem["alpha"], problem["prob"][:, 0, 0]
    out = torch.zeros(problem["rows"], problem["n"], device="cuda")
    begin = 0
    for expert, end in enumerate(problem["offsets"].tolist()):
        out[begin:end] = alpha[expert] * (a[begin:end] @ b[expert].T)
        begin = end
    return out, out * prob[:, None]


def signal_ready(kernel):
    from cudnn.gemm.cutedsl.grouped.moe_kernel_helpers import reset_streaming_signal_workspace

    return lambda workspace: reset_streaming_signal_workspace(workspace, kernel, all_complete=True)


def valid_rows(problem):
    return int(problem["offsets"][-1].item())


# Rows per expert.  More than 32 experts take the batch opener several lane rounds; an expert of more than 16 M tiles
# (EXPERT_BATCH_MAX_M_TILES) needs several batches.
ROUTINGS = {
    "uneven": [700, 0, 256, 1],
    "one_expert": [1300, 0, 0, 0],
    "many_experts_64": [(0, 1, 255, 256, 257, 700, 1300)[i % 7] for i in range(64)],
    "many_experts_128_small": [(256, 300, 512, 600, 768)[i % 5] for i in range(128)],
    "big_expert": [256, 5000, 0, 513, 1],
    "mostly_empty": [700 if i == 5 else 300 if i == 31 else 0 for i in range(32)],
    "rows_not_aligned": [255, 257, 511, 1025],
}


def experts_of(routing):
    return len(ROUTINGS[routing])


@pytest.mark.parametrize("routing", sorted(ROUTINGS))
@pytest.mark.parametrize("variant", ["signal", "signal_layout_pdl"])
@pytest.mark.parametrize("glu", [False, True], ids=["plain", "glu"])
def test_signal_consumer_matches_dynamic(glu, variant, routing):
    n = 2 * FFN if glu else FFN
    problem = make_problem(ROUTINGS[routing], n, HIDDEN, seed=len(routing))
    capacity = problem["rows"] // ROWS_ALIGNMENT
    options = dict(use_dynamic_sched=True, streaming_signal_capacity=capacity)
    if variant == "signal_layout_pdl":
        options.update(device_layout=True, use_pdl=True)
        trigger = "launch_pdl_dependents_at_all_streaming_signal_slots_ready" if glu else "launch_pdl_dependents_at_entry"
        options[trigger] = True
    experts = experts_of(routing)
    stock_c, stock_d, _, _ = run(make_kernel(glu, experts=experts, use_dynamic_sched=True), problem, glu)
    signal_kernel = make_kernel(glu, experts=experts, **options)
    c, d, _, _ = run(signal_kernel, problem, glu, prepare_workspace=signal_ready(signal_kernel))
    torch.cuda.synchronize()
    rows = valid_rows(problem)
    assert torch.equal(c[:rows], stock_c[:rows])
    assert torch.equal(d[:rows], stock_d[:rows])
    if not glu:
        c_ref, d_ref = plain_reference(problem)
        torch.testing.assert_close(c[:rows].float(), c_ref[:rows], atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(d[:rows].float(), d_ref[:rows], atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("experts", [EXPERTS, 64])
@pytest.mark.parametrize("device_layout", [False, True])
def test_signal_consumer_with_no_rows_exits(device_layout, experts):
    problem = make_problem([0] * experts, FFN, HIDDEN)
    kernel = make_kernel(False, experts=experts, use_dynamic_sched=True, streaming_signal_capacity=1, device_layout=device_layout)
    run(kernel, problem, False, prepare_workspace=signal_ready(kernel))
    torch.cuda.synchronize()


PRODUCER_ROUTINGS = {"four_experts": [700, 300, 256, 513], "many_experts_64": ROUTINGS["many_experts_64"], "big_expert": ROUTINGS["big_expert"]}


def release_order(order, capacity):
    """The order in which the producer releases the M tiles."""
    if order == "in_order":
        return list(range(capacity))
    if order == "reverse":
        return list(reversed(range(capacity)))
    return torch.randperm(capacity, generator=torch.Generator().manual_seed(capacity)).tolist()


@pytest.mark.parametrize("order", ["in_order", "reverse", "random"])
@pytest.mark.parametrize("routing", sorted(PRODUCER_ROUTINGS))
@pytest.mark.parametrize("glu", [False, True], ids=["plain", "glu"])
def test_signal_consumer_waits_for_producer(glu, routing, order):
    """The GEMM starts while every signal[] entry reads not-ready and A holds NaN; a later kernel on another stream writes A one
    M tile at a time, in ``order``, and then marks it ready.  Reading any tile early would carry NaN into D."""
    n = 2 * FFN if glu else FFN
    tokens_per_expert = PRODUCER_ROUTINGS[routing]
    experts = len(tokens_per_expert)
    problem = make_problem(tokens_per_expert, n, HIDDEN, seed=7)
    rows = problem["rows"]
    capacity = rows // ROWS_ALIGNMENT
    final_a = problem["a"].clone()
    problem["a"].fill_(float("nan"))
    kernel = make_kernel(glu, experts=experts, use_dynamic_sched=True, streaming_signal_capacity=capacity)
    stock_c, stock_d, _, _ = run(make_kernel(glu, experts=experts, use_dynamic_sched=True), dict(problem, a=final_a), glu)
    torch.cuda.synchronize()

    from cudnn.gemm.cutedsl.grouped.moe_kernel_helpers import reset_streaming_signal_workspace, streaming_signal_workspace_offsets

    gemm_stream, producer_stream = torch.cuda.Stream(), torch.cuda.Stream()
    holder = {}

    def prepare(workspace):
        reset_streaming_signal_workspace(workspace, kernel)
        signal_word = streaming_signal_workspace_offsets(kernel)["signal"] // 4
        holder["signal"] = workspace.view(torch.int32)[signal_word : signal_word + capacity]

    # A few clusters only, so the producer kernels find free SMs while the GEMM waits.
    c, d, workspace, launch = run(kernel, problem, glu, max_active_clusters=4, stream=gemm_stream, defer_launch=True)
    # One launch with every M tile ready loads the GEMM's module: under lazy loading a first launch waits for running work.
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
        for tile in release_order(order, capacity):
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
    """The GEMM starts while the layout reads -1, every signal[] entry reads not-ready and A holds NaN; a later kernel on
    another stream writes the layout, then A one M tile at a time with its signal[] entry.  The GLU also triggers once every M tile is ready."""
    n = 2 * FFN if glu else FFN
    problem = make_problem([700, 300, 256, 513], n, HIDDEN, seed=11)
    capacity = problem["rows"] // ROWS_ALIGNMENT
    final_a, final_layout = problem["a"].clone(), problem["layout"].clone()
    options = dict(use_dynamic_sched=True, streaming_signal_capacity=capacity, device_layout=True)
    if glu:
        options["launch_pdl_dependents_at_all_streaming_signal_slots_ready"] = True
    kernel = make_kernel(glu, **options)
    stock_c, stock_d, _, _ = run(make_kernel(glu, use_dynamic_sched=True), problem, glu)
    torch.cuda.synchronize()

    from cudnn.gemm.cutedsl.grouped.moe_kernel_helpers import reset_streaming_signal_workspace, streaming_signal_workspace_offsets

    gemm_stream, producer_stream = torch.cuda.Stream(), torch.cuda.Stream()
    c, d, workspace, launch = run(kernel, problem, glu, max_active_clusters=4, stream=gemm_stream, defer_launch=True)
    # One launch with the layout written and every M tile ready loads the module (lazy loading waits for running work).
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


# (D columns N, N chunk C): whole chunks; N < C (one chunk); N == C; a partial last chunk (columns 512..767).
OUTPUT_CHUNKS = {"n1024_c256": (2 * HIDDEN, 256), "n768_c1024": (768, 1024), "n768_c768": (768, 768), "n768_c512": (768, 512)}
# (N and chunk, routing): every chunk shape on the uneven routing; many experts and a big expert on two of them.
OUTPUT_CASES = [(n_and_chunk, "uneven") for n_and_chunk in sorted(OUTPUT_CHUNKS)] + [
    (n_and_chunk, routing) for n_and_chunk in ("n1024_c256", "n768_c512") for routing in ("many_experts_64", "big_expert")
]


@pytest.mark.parametrize("n_and_chunk, routing", OUTPUT_CASES, ids=[f"{c}-{r}" for c, r in OUTPUT_CASES])
@pytest.mark.parametrize("one_cta", [False, True], ids=["cluster21", "cluster11"])
@pytest.mark.parametrize("clusters_beside_consumer", [1, 1000])
def test_output_notify_publishes_every_chunk(clusters_beside_consumer, one_cta, n_and_chunk, routing):
    from cudnn.gemm.cutedsl.grouped.moe_streaming_output_chunk_tile_scheduler import (
        StreamingOutputNotifyConfig,
        StreamingOutputNotifyMode,
        streaming_output_notify_capacity,
        streaming_output_notify_offsets,
    )

    n, chunk = OUTPUT_CHUNKS[n_and_chunk]
    experts = experts_of(routing)
    problem = make_problem(ROUTINGS[routing], n, FFN, seed=3)
    capacity = streaming_output_notify_capacity(chunk, n)
    assert capacity == len(range(0, n, chunk))
    notify = StreamingOutputNotifyConfig(StreamingOutputNotifyMode.N_CHUNK, chunk, capacity, clusters_beside_consumer=clusters_beside_consumer)
    stock_c, stock_d, _, _ = run(make_kernel(False, one_cta, experts=experts, use_dynamic_sched=True), problem, False)
    kernel = make_kernel(False, one_cta, experts=experts, use_dynamic_sched=True, streaming_output_notify=notify, launch_pdl_dependents_at_entry=True)
    c, d, workspace, _ = run(kernel, problem, False)
    torch.cuda.synchronize()
    valid = valid_rows(problem)
    assert torch.equal(c[:valid], stock_c[:valid])
    assert torch.equal(d[:valid], stock_d[:valid])
    offset = streaming_output_notify_offsets(kernel)["signal"]
    signal = workspace[offset : offset + 4 * capacity].view(torch.int32)
    assert signal.tolist() == list(range(capacity))


def test_signal_slot_release_warm_stress():
    """The claim slot each cluster broadcasts is released only after every lane read it (python/cudnn/AGENTS.md Rule 11):
    back-to-back and shared-memory-poisoned launches of a shape with many claims per cluster, every byte against stock."""
    from warm_stress import StressCase, assert_warm_stress

    routing = ROUTINGS["big_expert"]
    problem = make_problem(routing, FFN, HIDDEN, seed=13)
    clusters = 4
    stock_c, stock_d, _, _ = run(make_kernel(False, experts=len(routing), use_dynamic_sched=True), problem, False, max_active_clusters=clusters)
    kernel = make_kernel(False, experts=len(routing), use_dynamic_sched=True, streaming_signal_capacity=problem["rows"] // ROWS_ALIGNMENT)
    c, d, workspace, launch = run(kernel, problem, False, prepare_workspace=signal_ready(kernel), max_active_clusters=clusters, defer_launch=True)
    reset_workspace = signal_ready(kernel)
    claims = (problem["rows"] // ROWS_ALIGNMENT) * (FFN // 256)
    assert_warm_stress(
        StressCase(
            "streaming signal scheduler, big expert, 4 clusters",
            launch=launch,
            outputs={"c": c, "d": d},
            thread_blocks=2 * clusters,
            refills_per_launch=claims // clusters,
            reference={"c": stock_c, "d": stock_d},
            reset=lambda: reset_workspace(workspace),
        ),
        warm=500,
        poisoned=500,
        budget_s=30.0,
    )


def test_streaming_options_reject_invalid():
    from cudnn.gemm.cutedsl.grouped.moe_streaming_output_chunk_tile_scheduler import StreamingOutputNotifyConfig, StreamingOutputNotifyMode

    with pytest.raises(ValueError, match="use_dynamic_sched"):
        make_kernel(False, streaming_signal_capacity=4)
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

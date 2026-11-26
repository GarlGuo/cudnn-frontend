# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One MXFP8 wgrad split into two launches that share a caller-owned claim counter (``scheduler_counter``).

With ``atomic_counter_claim``, the kernel classes take a caller's int32 ``scheduler_counter`` in ``__call__``. Nothing
resets it but the caller, so two launches with their own persistent grids claim from one tile list and each tile runs
once. This is the shape of a backward pass that runs the wgrad as a "tail" on the SMs a communication kernel frees,
then as a "base" on the rest:

    producer of A -> tail  (use_pdl, pdl_mode="wait_then_trigger")          waits for the producer, then releases the base
                  -> base  (use_pdl, pdl_mode="trigger_then_wait_at_exit")  reads at once, waits for the tail at exit

The base reads A with no wait in front. That is correct only because A does not come from the base's own primary
(the tail): the tail let the base launch only after the producer had ended. The negative cases break that guarantee
on purpose (the producer is the base's primary, or the tail releases the base before its wait) and must read stale A
whenever the dependent really starts early.

The kernels are called through their classes (the API wrapper sizes the grid from the device). dW of every split
must be byte-equal to one default launch of the same kernel class.
"""

import pytest
import torch

M, N = 1024, 1024
# Uneven experts in whole 128-token K tiles, two of them empty.
GROUPS = (256, 0, 768, 2048, 0, 256, 512, 1024)
TILE, CLUSTER = (256, 256), (2, 1)
TAIL_CLUSTERS, BASE_CLUSTERS = 2, 6


@pytest.fixture(autouse=True)
def require_sm100():
    from cudnn.tensor_adapter import has_tcgen05

    if not torch.cuda.is_available() or not has_tcgen05(torch.cuda.get_device_capability()[0]):
        pytest.skip("Requires SM100+ for grouped GEMM wgrad")
    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor not in (100, 103):
        pytest.skip("SM100/SM103 kernels")


def _classes():
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad import BlockScaledMoEGroupedGemmWgradKernel
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad_dense import BlockScaledMoEGroupedGemmWgradDenseKernel
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad_reuse import BlockScaledMoEGroupedGemmWgradReuseKernel

    return {"stock": BlockScaledMoEGroupedGemmWgradKernel, "dense": BlockScaledMoEGroupedGemmWgradDenseKernel, "reuse": BlockScaledMoEGroupedGemmWgradReuseKernel}


_PROBLEM = {}


def _problem():
    if "p" not in _PROBLEM:
        from gemm.cutedsl.test_grouped_gemm_wgrad_atomic_counter import _make_mxfp8_problem

        _PROBLEM["p"] = _make_mxfp8_problem(GROUPS, M, N, seed=17)
    return _PROBLEM["p"]


def _units(name):
    tiles = len(GROUPS) * (M // TILE[0]) * (N // TILE[1])
    if name == "reuse":
        return tiles // _classes()["reuse"].n_group_for(N)  # the A-reuse kernel claims work units of N tiles
    return tiles


def _cute(t, align=16):
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(t, assumed_align=align)


class _Launch:
    """One compiled kernel class instance on ``clusters`` persistent clusters (None: the device's), called on caller
    tensors. ``counter``: compiled with a caller ``scheduler_counter``. Own workspace."""

    def __init__(self, name, clusters=None, counter=False, **options):
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.runtime import make_fake_stream
        from cudnn.gemm.cutedsl.grouped.moe_utils import MoEWeightMode, WGradInputOrder

        claim = options.pop("atomic_counter_claim", False) or counter
        kernel = _classes()[name](
            sf_vec_size=32,
            acc_dtype=cutlass.Float32,
            use_2cta_instrs=True,
            mma_tiler_mn=TILE,
            cluster_shape_mn=CLUSTER,
            expert_cnt=len(GROUPS),
            weight_mode=MoEWeightMode.DENSE,
            input_order=WGradInputOrder.Tensor2D,
            atomic_counter_claim=claim,
            **options,
        )
        if clusters is None:
            clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(CLUSTER[0] * CLUSTER[1])
        self.kernel, self.clusters, self.counter = kernel, clusters, counter
        self.workspace = torch.zeros(max(kernel.get_workspace_bytes(), 128), dtype=torch.uint8, device="cuda")
        p = _problem()
        args = self._args(p["a"], p["output"], torch.zeros(1, dtype=torch.int32, device="cuda") if counter else None)
        self.compiled = cute.compile(kernel, *args[:7], clusters, make_fake_stream(), *args[7:])

    def _args(self, a, out, counter):
        p = _problem()
        args = [_cute(a), _cute(p["b"]), _cute(p["sfa"]), _cute(p["sfb"]), _cute(out), _cute(p["offsets"], 4), _cute(self.workspace, 128)]
        tail = [None, None, None]
        if counter is not None:
            tail.append(_cute(counter, 4))
        return args + tail

    def __call__(self, a, out, counter=None):
        import cuda.bindings.driver as cuda

        assert (counter is not None) == self.counter
        args = self._args(a, out, counter)
        self.compiled(*args[:7], cuda.CUstream(torch.cuda.current_stream().cuda_stream), *args[7:])


_LAUNCHES = {}


def _launch(key, name, **kwargs):
    if (key, name) not in _LAUNCHES:
        _LAUNCHES[key, name] = _Launch(name, **kwargs)
    return _LAUNCHES[key, name]


def _reference(name):
    out = torch.full_like(_problem()["output"], float("nan"))
    _launch("single", name)(_problem()["a"], out)
    torch.cuda.synchronize()
    return out


def _tail(name):
    return _launch("tail", name, clusters=TAIL_CLUSTERS, counter=True, use_pdl=True, pdl_mode="wait_then_trigger")


def _base(name):
    return _launch("base", name, clusters=BASE_CLUSTERS, counter=True, use_pdl=True, pdl_mode="trigger_then_wait_at_exit")


def _same(x, y):
    return torch.equal(x.view(torch.int16), y.view(torch.int16))


def _graph(chain):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        chain()
    return graph


# ---------------------------------------------------------------------------
# Options (CPU only: the kernel constructors)
# ---------------------------------------------------------------------------


@pytest.mark.L0
def test_pdl_mode_rejections():
    import cutlass
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad_rubin import BlockScaledMoEGroupedGemmWgradRubinKernel

    classes = _classes()
    common = dict(sf_vec_size=32, acc_dtype=cutlass.Float32, use_2cta_instrs=True, mma_tiler_mn=TILE, cluster_shape_mn=CLUSTER, expert_cnt=4)
    for cls in classes.values():
        assert cls(**common).pdl_mode == "wait_at_start"
        with pytest.raises(ValueError, match="pdl_mode must be one of"):
            cls(**common, use_pdl=True, pdl_mode="exit")
        for mode in ("wait_then_trigger", "trigger_then_wait_at_exit"):
            with pytest.raises(ValueError, match="requires use_pdl"):
                cls(**common, pdl_mode=mode)
            with pytest.raises(ValueError, match="requires use_pdl"):
                cls(**common, use_pdl=True, launch_pdl_dependents_at_entry=True, pdl_mode=mode)
        assert cls(**common, use_pdl=True, pdl_mode="wait_then_trigger").pdl_mode == "wait_then_trigger"
    # The stock kernels always launch the descriptor helper in front of the GEMM.
    for cls in (classes["stock"], BlockScaledMoEGroupedGemmWgradRubinKernel):
        with pytest.raises(ValueError, match="helper kernel"):
            cls(**common, use_pdl=True, pdl_mode="trigger_then_wait_at_exit")
    for name in ("dense", "reuse"):
        assert classes[name](**common, use_pdl=True, pdl_mode="trigger_then_wait_at_exit").pdl_mode == "trigger_then_wait_at_exit"


@pytest.mark.L0
def test_exit_wait_needs_the_caller_counter():
    """Dense / A-reuse with atomic_counter_claim and no scheduler_counter launch the counter reset in front of the GEMM:
    an exit-wait GEMM would claim before it ends, so the trace rejects it."""
    with pytest.raises(Exception, match="needs a caller's scheduler_counter"):
        _Launch("dense", clusters=2, atomic_counter_claim=True, use_pdl=True, pdl_mode="trigger_then_wait_at_exit")


# ---------------------------------------------------------------------------
# On a device
# ---------------------------------------------------------------------------


@pytest.mark.L0
@pytest.mark.parametrize("name", ["dense", "reuse"])
def test_split_pair_matches_single_launch(name):
    """Tail then base on one caller-zeroed counter, eager and replayed from a CUDA graph (the reset inside it): dW
    byte-equal to one default launch, and the counter at one claim per work unit plus each cluster's final empty claim.
    The counter preset to every unit runs no tile at all."""
    expected = _reference(name)
    p = _problem()
    tail, base = _tail(name), _base(name)
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    out = torch.empty_like(expected)
    claims = _units(name) + TAIL_CLUSTERS + BASE_CLUSTERS

    def chain():
        counter.zero_()
        tail(p["a"], out, counter)
        base(p["a"], out, counter)

    results = []
    graph = None
    for run in ("eager", "eager", "graph", "graph"):
        out.fill_(float("nan"))
        torch.cuda.synchronize()
        if run == "graph":
            graph = graph or _graph(chain)
            graph.replay()
        else:
            chain()
        torch.cuda.synchronize()
        results.append((run, _same(out, expected), int(counter.item())))
    if graph is not None:
        graph.reset()
    assert all(ok and c == claims for _, ok, c in results), f"(run, dW bitwise equal, counter) per run, counter expected {claims}: {results}"

    # Preclaimed: every unit already taken, so neither launch writes dW.
    out.fill_(float("nan"))
    counter.fill_(_units(name))
    tail(p["a"], out, counter)
    base(p["a"], out, counter)
    torch.cuda.synchronize()
    assert bool(out.isnan().all()), "a launch ran a tile the counter had already given out"
    assert int(counter.item()) == claims


@pytest.mark.L0
def test_stock_kernel_shares_a_caller_counter_across_streams():
    """The stock kernel keeps its descriptor helper (per launch, own workspace) but leaves a caller's counter alone: two
    launches on two streams split one tile list."""
    name = "stock"
    expected = _reference(name)
    p = _problem()
    first = _launch("s1", name, clusters=TAIL_CLUSTERS, counter=True)
    second = _launch("s2", name, clusters=BASE_CLUSTERS, counter=True)
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    out = torch.full_like(expected, float("nan"))
    side = torch.cuda.Stream()
    torch.cuda.synchronize()
    first(p["a"], out, counter)
    with torch.cuda.stream(side):
        second(p["a"], out, counter)
    torch.cuda.synchronize()
    assert _same(out, expected)
    assert int(counter.item()) == _units(name) + TAIL_CLUSTERS + BASE_CLUSTERS


ARRANGEMENTS = {
    # The producer is the tail's primary; the tail releases the base only after its wait: A is final for the base.
    "producer_tail_base": "correct",
    # The producer is the base's own primary: the base reads A in front of its exit wait.
    "producer_base": "unsafe",
    # The tail waits at start but releases the base at entry, before that wait.
    "producer_tail_at_entry_base": "unsafe",
}


@pytest.mark.L0
@pytest.mark.parametrize("name", ["dense", "reuse"])
def test_exit_wait_base_behind_early_release_producer(name):
    """An early-releasing producer writes A (it lets its dependents start, then writes slowly). In the arrangement with
    the guarantee, dW is the single launch's every time, eager and from a CUDA graph. Without it (the producer is the
    exit-wait GEMM's own primary, or the tail releases it before waiting), the GEMM reads A while the producer still
    writes it; the test requires stale reads there whenever the positive control shows a dependent starting early."""
    from pdl_dependent_copy import DependentCopy, EarlyReleaseWriter

    expected = _reference(name)
    p = _problem()
    tail, base = _tail(name), _base(name)
    entry_tail = _launch("tail_at_entry", name, clusters=TAIL_CLUSTERS, counter=True, use_pdl=True, launch_pdl_dependents_at_entry=True)
    a_final = p["a"].clone()
    a_buf = torch.empty_like(a_final)
    stale = torch.empty_like(a_final)
    stale.view(torch.uint8).copy_(~a_final.view(torch.uint8))  # zero-free and far from A
    writer = EarlyReleaseWriter(torch.device("cuda"))
    unwaited = DependentCopy(torch.device("cuda"), wait=False)
    seen = torch.empty_like(a_final)
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    out = torch.empty_like(expected)

    def chain_for(arrangement):
        def chain():
            counter.zero_()
            writer(a_final, a_buf)
            if arrangement == "producer_tail_base":
                tail(a_buf, out, counter)
            elif arrangement == "producer_tail_at_entry_base":
                entry_tail(a_buf, out, counter)
            base(a_buf, out, counter)

        return chain

    def control():
        writer(a_final, a_buf)
        unwaited(a_buf, seen)

    for arrangement in ARRANGEMENTS:  # warm every launch
        chain_for(arrangement)()
    control()
    torch.cuda.synchronize()

    def run(launch, check):
        a_buf.copy_(stale)
        out.fill_(float("nan"))
        seen.copy_(stale)
        torch.cuda.synchronize()
        launch()
        torch.cuda.synchronize()
        return check()

    early_starts = sum(run(control, lambda: not torch.equal(seen.view(torch.uint8), a_final.view(torch.uint8))) for _ in range(3))
    report = {}
    for arrangement in ARRANGEMENTS:
        chain = chain_for(arrangement)
        graph = _graph(chain)
        try:
            report[arrangement] = [run(launch, lambda: _same(out, expected)) for launch in (chain, chain, chain, graph.replay, graph.replay)]
        finally:
            graph.reset()
    assert all(report["producer_tail_base"]), f"the guaranteed arrangement differs from the single launch: {report}"
    if early_starts == 0:
        pytest.skip(f"no early start reproduced on this device (coverage gap); runs: {report}")
    for arrangement, kind in ARRANGEMENTS.items():
        if kind == "unsafe":
            assert not all(report[arrangement]), f"{arrangement}: no stale read although dependents start early: {report}"

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming input signal (``streaming_signal_capacity``) of the SM100/SM107 grouped GEMM wgrad kernels.

The TMA warp waits for the token tiles of each K block before it loads the block. Every output tile is still one CTA
(pair) over its expert's whole token range in the same order, so each case compares the signal path with the
atomic-counter path without the signal byte for byte, and with an independent reference.
"""

import pytest
import torch

from determinism_utils import assert_bitwise, snapshot
from gemm.cutedsl.test_grouped_gemm_wgrad_bf16_utils import assert_grouped_gemm_wgrad_close, grouped_gemm_wgrad_bf16_reference
from gemm.cutedsl.test_grouped_gemm_wgrad_utils import check_ref_grouped_gemm_wgrad
from gemm.cutedsl.test_grouped_gemm_wgrad_streaming_signal_utils import (
    TOKEN_TILE,
    WgradLauncher,
    capacity_for,
    make_problem,
    pad_groups,
    poison_operands,
    reset_signal,
    run_ready,
    run_with_delayed_producer,
    signal_view,
    split_tokens,
    token_rows,
)


@pytest.fixture(autouse=True)
def require_sm100():
    from cudnn.tensor_adapter import has_tcgen05

    if not torch.cuda.is_available() or not has_tcgen05(torch.cuda.get_device_capability()[0]):
        pytest.skip("Requires SM100+ for grouped GEMM wgrad")


EXPERTS = 8
EMPTY = (1, 4)  # experts that receive no tokens
M, N = 1024, 1024
# (mma_tiler_mn, cluster_shape_mn)
L0_CONFIGS = [((256, 256), (2, 1)), ((128, 256), (1, 1))]
L1_CONFIGS = [((256, 256), (4, 1)), ((128, 128), (1, 2))]
CONFIGS = [pytest.param(c, marks=pytest.mark.L0) for c in L0_CONFIGS] + [pytest.param(c, marks=pytest.mark.L1) for c in L1_CONFIGS]
MXFP8_CONFIGS = [pytest.param(((256, 256), (2, 1)), marks=pytest.mark.L0), pytest.param(((128, 128), (1, 1)), marks=pytest.mark.L0)] + [
    pytest.param(c, marks=pytest.mark.L1) for c in L1_CONFIGS
]
TOKENS = [1, 257, 10000]


def _config_id(config):
    (tm, tn), (cm, cn) = config
    return f"tile{tm}x{tn}-cluster{cm}x{cn}"


def _groups(kind, tokens, experts=EXPERTS, empty=EMPTY):
    return pad_groups(split_tokens(tokens, experts, empty, seed=tokens), kind)


def _launchers(kind, problem, config, **options):
    tile, cluster = config
    common = dict(tile=tile, cluster=cluster, **options)
    return (
        WgradLauncher(kind, problem["experts"], capacity=capacity_for(problem), **common),
        WgradLauncher(kind, problem["experts"], **common),
    )


def _check_reference(problem, output, accumulate=False):
    if problem["kind"] == "bf16":
        assert_grouped_gemm_wgrad_close(output, grouped_gemm_wgrad_bf16_reference(problem, accumulate=accumulate))
    else:
        check_ref_grouped_gemm_wgrad(output, problem["reference"])


def _signal_vs_counter(kind, problem, config, initial_value=None, **options):
    signal, counter = _launchers(kind, problem, config, **options)
    expected = run_ready(counter, problem, initial_value=initial_value)
    actual = run_ready(signal, problem, initial_value=initial_value)
    assert_bitwise({"wgrad": actual}, snapshot({"wgrad": expected}), "signal vs counter")
    return actual


# ---------------------------------------------------------------------------
# Every token tile ready: same bytes as the counter path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tokens", TOKENS)
@pytest.mark.parametrize("config", CONFIGS, ids=_config_id)
def test_signal_matches_counter_bf16(config, tokens):
    problem = make_problem("bf16", _groups("bf16", tokens), M, N)
    output = _signal_vs_counter("bf16", problem, config)
    _check_reference(problem, output)


@pytest.mark.L0
@pytest.mark.parametrize("input_order", ["tensor2d", "tensor_ragged"])
def test_signal_matches_counter_bf16_discrete_accumulate(input_order):
    """Discrete output and accumulate_on_output, both input orders."""
    problem = make_problem("bf16", _groups("bf16", 10000), M, N, input_order=input_order, discrete=True, initial_value=0.25)
    output = _signal_vs_counter("bf16", problem, L0_CONFIGS[0], initial_value=0.25, accumulate=True, discrete=True, input_order=input_order)
    _check_reference(problem, output, accumulate=True)


@pytest.mark.parametrize("tokens", TOKENS)
@pytest.mark.parametrize("config", MXFP8_CONFIGS, ids=_config_id)
def test_signal_matches_counter_mxfp8(config, tokens):
    problem = make_problem("mxfp8", _groups("mxfp8", tokens), M, N, seed=tokens)
    output = _signal_vs_counter("mxfp8", problem, config)
    _check_reference(problem, output)


@pytest.mark.L0
def test_signal_matches_counter_mxfp8_discrete_accumulate():
    problem = make_problem("mxfp8", _groups("mxfp8", 10000), M, N, seed=5, initial_value=0.25)
    output = _signal_vs_counter("mxfp8", problem, L0_CONFIGS[0], initial_value=0.25, accumulate=True, discrete=True)
    _check_reference(problem, output, accumulate=True)


@pytest.mark.L0
@pytest.mark.parametrize("config", L0_CONFIGS, ids=_config_id)
def test_signal_matches_counter_bf16_unpadded_groups(config):
    """1- and 255-token experts are off the BF16 contract: a K block reads past its expert into the next one's rows, so
    the gate must cover those rows too (up to the last valid token). Equality only, as in the counter tests."""
    groups = (1, 0, 255, 256, 0, 512)
    tokens = sum(groups)
    generator = torch.Generator(device="cuda").manual_seed(20261124)
    problem = {
        "kind": "bf16",
        "a": (torch.randn((tokens, M), device="cuda", generator=generator) * 0.125).to(torch.bfloat16).t(),
        "b": (torch.randn((tokens, N), device="cuda", generator=generator) * 0.125).to(torch.bfloat16),
        "offsets": torch.tensor(groups, dtype=torch.int32, device="cuda").cumsum(0, dtype=torch.int32),
        "output": torch.empty((len(groups), M, N), dtype=torch.bfloat16, device="cuda"),
        "group_k_list": groups,
        "experts": len(groups),
    }
    tile, cluster = config
    signal = WgradLauncher("bf16", len(groups), tile=tile, cluster=cluster, capacity=capacity_for(problem), dynamic_tokens=False)
    counter = WgradLauncher("bf16", len(groups), tile=tile, cluster=cluster, dynamic_tokens=False)
    expected = run_ready(counter, problem)
    assert_bitwise({"wgrad": run_ready(signal, problem)}, snapshot({"wgrad": expected}), "signal vs counter")


# ---------------------------------------------------------------------------
# A producer that lands token tiles late and out of order
# ---------------------------------------------------------------------------


@pytest.mark.L0
@pytest.mark.parametrize("order", ["reversed", "shuffled"])
@pytest.mark.parametrize("kind", ["bf16", "mxfp8"])
def test_signal_waits_for_delayed_producer(kind, order):
    """Operands start as NaN and every signal[] entry -1; the producer writes token tiles in ``order`` and publishes each
    late. A K block loaded before its tiles landed would carry NaN into dW."""
    problem = make_problem(kind, _groups(kind, 10000), M, N, seed=11)
    signal, counter = _launchers(kind, problem, L0_CONFIGS[0], max_active_clusters=4)
    expected = run_ready(counter, problem)
    final = poison_operands(problem)
    actual = run_with_delayed_producer(signal, problem, final, order=order, seed=3)
    assert_bitwise({"wgrad": actual}, snapshot({"wgrad": expected}), f"delayed {order} producer vs counter")


@pytest.mark.L0
@pytest.mark.parametrize("kind", ["bf16", "mxfp8"])
def test_signal_waits_with_one_cta_clusters(kind):
    """Cluster (1, 1): a claimed tile's single CTA gates its own loads."""
    problem = make_problem(kind, _groups(kind, 257), M, N, seed=13)
    signal, counter = _launchers(kind, problem, L0_CONFIGS[1], max_active_clusters=8)
    expected = run_ready(counter, problem)
    final = poison_operands(problem)
    actual = run_with_delayed_producer(signal, problem, final, order="reversed")
    assert_bitwise({"wgrad": actual}, snapshot({"wgrad": expected}), "delayed producer vs counter")


# ---------------------------------------------------------------------------
# Empty experts and unpublished tiles
# ---------------------------------------------------------------------------


@pytest.mark.L0
@pytest.mark.parametrize("kind", ["bf16", "mxfp8"])
def test_signal_all_experts_empty_waits_on_nothing(kind):
    """No routed tokens in an allocated receive buffer: no K block, so the kernel finishes with every signal[] entry -1
    and writes zeros."""
    problem = make_problem(kind, (TOKEN_TILE,) + (0,) * (EXPERTS - 1), M, N)
    problem["offsets"].zero_()
    signal = WgradLauncher(kind, EXPERTS, capacity=4)
    output = run_ready(signal, problem, all_complete=False)
    assert torch.count_nonzero(output.float()) == 0


@pytest.mark.L0
@pytest.mark.parametrize("kind", ["bf16", "mxfp8"])
def test_signal_waits_only_on_tiles_with_tokens(kind):
    """One routed token: only token tile 0 is published; the entries past the last valid row stay -1 and no K block
    waits on them."""
    problem = make_problem(kind, _groups(kind, 1), M, N, seed=1)
    assert token_rows(problem) <= TOKEN_TILE
    capacity = 8
    signal = WgradLauncher(kind, EXPERTS, capacity=capacity)
    counter = WgradLauncher(kind, EXPERTS)
    expected = run_ready(counter, problem)
    output = torch.full_like(problem["output"], float("nan"))
    workspace = signal.workspace()
    reset_signal(signal, workspace, all_complete=False)
    signal_view(signal, workspace)[0].fill_(0)
    signal.run(problem, output, workspace)
    torch.cuda.synchronize()
    assert_bitwise({"wgrad": output}, snapshot({"wgrad": expected}), "one tile vs counter")


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------


def _kernel_classes():
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad import BlockScaledMoEGroupedGemmWgradKernel
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad_rubin import BlockScaledMoEGroupedGemmWgradRubinKernel
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_grouped_gemm_wgrad import MoEGroupedGemmWgradBF16Kernel

    return [
        (MoEGroupedGemmWgradBF16Kernel, {}),
        (BlockScaledMoEGroupedGemmWgradKernel, {"sf_vec_size": 32}),
        (BlockScaledMoEGroupedGemmWgradRubinKernel, {"sf_vec_size": 32}),
    ]


@pytest.mark.L0
def test_signal_workspace_layout():
    from cudnn.gemm.cutedsl.grouped.moe_utils import MoEWeightMode
    from cudnn.gemm.cutedsl.grouped.wgrad.streaming_signal import (
        STREAMING_SIGNAL_TOKEN_TILE,
        reset_streaming_signal_workspace,
        streaming_signal_capacity_for_tokens,
        streaming_signal_workspace_offsets,
    )

    assert STREAMING_SIGNAL_TOKEN_TILE == TOKEN_TILE
    assert [streaming_signal_capacity_for_tokens(t) for t in (0, 1, 256, 257, 10000)] == [1, 1, 1, 2, 40]
    for cls, extra in _kernel_classes():
        for weight_mode in (MoEWeightMode.DENSE, MoEWeightMode.DISCRETE):
            base = cls(expert_cnt=EXPERTS, weight_mode=weight_mode, atomic_counter_claim=True, **extra)
            kernel = cls(expert_cnt=EXPERTS, weight_mode=weight_mode, atomic_counter_claim=True, streaming_signal_capacity=7, **extra)
            assert base.streaming_signal_capacity is None
            desc = kernel.get_desc_workspace_bytes()
            assert desc == base.get_desc_workspace_bytes()
            assert kernel.get_workspace_bytes() == base.get_workspace_bytes() + 4 * 7 == desc + 4 + 4 * 7
            offsets = streaming_signal_workspace_offsets(kernel)
            assert offsets == {"counter": desc, "signal": desc + 4, "streaming_signal_capacity": 7, "total_bytes": desc + 32}
            workspace = torch.full((kernel.get_workspace_bytes(),), 0x55, dtype=torch.uint8, device="cuda")
            reset_streaming_signal_workspace(workspace, kernel)
            words = workspace.view(torch.int32)
            assert torch.equal(words[(desc + 4) // 4 :].cpu(), torch.full((7,), -1, dtype=torch.int32))
            assert int(words[desc // 4]) == 0x55555555, "the reset leaves the claim counter to the kernel's helper"
            reset_streaming_signal_workspace(workspace, kernel, all_complete=True)
            assert torch.equal(words[(desc + 4) // 4 :].cpu(), torch.arange(7, dtype=torch.int32))


@pytest.mark.L0
def test_signal_options_reject_invalid():
    for cls, extra in _kernel_classes():
        with pytest.raises(ValueError, match="requires atomic_counter_claim"):
            cls(expert_cnt=EXPERTS, streaming_signal_capacity=4, **extra)
        for bad in (0, -1, True, 2.0, "4"):
            with pytest.raises(ValueError, match="positive int"):
                cls(expert_cnt=EXPERTS, atomic_counter_claim=True, streaming_signal_capacity=bad, **extra)
        assert cls(expert_cnt=EXPERTS, **extra).get_workspace_bytes() == cls(expert_cnt=EXPERTS, streaming_signal_capacity=None, **extra).get_workspace_bytes()


@pytest.mark.L0
def test_signal_capacity_below_static_tokens_rejected():
    """With a static token extent the build checks that signal[] covers every allocated token row."""
    problem = make_problem("bf16", (512, 256, 0), 256, 256)
    signal = WgradLauncher("bf16", 3, capacity=2, dynamic_tokens=False)  # 768 rows need 3 tiles
    with pytest.raises(ValueError, match="streaming_signal_capacity 2 < 3"):
        run_ready(signal, problem)


# ---------------------------------------------------------------------------
# More persistent clusters than output tiles
# ---------------------------------------------------------------------------


@pytest.mark.L0
@pytest.mark.parametrize("signal", [False, True], ids=["counter", "signal"])
@pytest.mark.parametrize("kind", ["bf16", "mxfp8"])
def test_more_clusters_than_tiles(kind, signal):
    """4 experts x 2 x 2 output tiles of 256 x 256 = 16 tiles on every cluster the GPU holds: most clusters' first
    claim is already past the last tile. Such a cluster must publish only the sentinel and exit (it used to publish
    its invalid first claim ahead of the sentinel, whose stage was then never released: a hang). Bytes equal CLC's."""
    groups = (512, 0, 256, 768) if kind == "bf16" else (384, 0, 128, 640)
    problem = make_problem(kind, groups, 512, 512, seed=17)
    capacity = capacity_for(problem) if signal else None
    launcher = WgradLauncher(kind, len(groups), capacity=capacity)
    clc = WgradLauncher(kind, len(groups), counter=False)
    expected = run_ready(clc, problem)
    assert_bitwise({"wgrad": run_ready(launcher, problem)}, snapshot({"wgrad": expected}), "more clusters than tiles vs CLC")
    _check_reference(problem, expected)

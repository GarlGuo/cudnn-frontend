# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Wentao Guo, Phuong Nguyen

"""Bitwise determinism of the SM90 BF16 grouped GEMM wgrad: one CTA owns each output tile and there is no split-K,
so neither the counter's claim order nor the other experts may change an expert's dW bytes."""

import cutlass
import pytest
import torch
import cudnn

from cudnn.gemm.cutedsl.grouped.wgrad import api as wgrad_api
from determinism_utils import assert_bitwise, assert_golden, assert_replays

pytestmark = pytest.mark.L0

M, N = 512, 384


@pytest.fixture(autouse=True)
def require_sm90():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("The SM90 grouped GEMM wgrad needs a cc 9.0 device")


def expert_slices(group_k_list, seed=7):
    """Per-expert logical (dY^T slice (M, k), X slice (k, N)), fixed by the expert's index and length."""
    slices = []
    for expert, tokens in enumerate(group_k_list):
        g = torch.Generator(device="cuda").manual_seed(seed * 1000 + expert * 37 + tokens)
        a = torch.randn((M, tokens), generator=g, device="cuda", dtype=torch.bfloat16) * 0.125
        b = torch.randn((tokens, N), generator=g, device="cuda", dtype=torch.bfloat16) * 0.125
        slices.append((a, b))
    return slices


def run(slices, order=None, **schedule):
    """dW for the experts in ``order`` (default all), from row-major dY (tokens, M) and X (tokens, N)."""
    order = list(range(len(slices))) if order is None else order
    dy = torch.cat([slices[e][0].t() for e in order]).contiguous()
    x = torch.cat([slices[e][1] for e in order]).contiguous()
    counts = torch.tensor([slices[e][0].shape[1] for e in order], dtype=torch.int32, device="cuda")
    out = cudnn.grouped_gemm_wgrad_wrapper_sm100(
        a_tensor=dy.t(),
        b_tensor=x,
        sfa_tensor=None,
        sfb_tensor=None,
        offsets_tensor=counts.cumsum(0, dtype=torch.int32),
        acc_dtype=torch.float32,
        wgrad_dtype=torch.bfloat16,
        **schedule,
    )["wgrad_tensor"]
    return {"wgrad": out}


GROUPS = (512, 0, 1536, 256, 768, 0, 256, 2048)


def test_sm90_wgrad_replays_bitwise():
    slices = expert_slices(GROUPS)
    assert_replays(lambda: run(slices), repeats=6)


@pytest.mark.parametrize("margin", ["33", "62"], ids=["half-clusters", "four-clusters"])
def test_sm90_wgrad_claim_order_invariance(monkeypatch, margin):
    # Fewer persistent clusters (H100: 66 of (2, 1)) claim the same tiles in another order.
    slices = expert_slices(GROUPS)
    expected = run(slices)
    monkeypatch.setenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", margin)
    assert_bitwise(run(slices), expected, f"margin {margin}")


def test_sm90_wgrad_expert_batch_invariance():
    # An expert's dW must not depend on its position, its neighbours or the token total.
    slices = expert_slices(GROUPS)
    full = run(slices)["wgrad"]
    order = [6, 2, 0, 7, 4]
    reordered = run(slices, order)["wgrad"]
    for position, expert in enumerate(order):
        assert_bitwise({"wgrad": reordered[position]}, {"wgrad": full[expert]}, f"expert {expert} reordered")
    alone = run(slices, [2])["wgrad"]
    assert_bitwise({"wgrad": alone[0]}, {"wgrad": full[2]}, "expert 2 alone")


def test_sm90_wgrad_fresh_cache_matches():
    # A cold compile (fresh caches) produces the same bytes as the warm one.
    slices = expert_slices(GROUPS)
    expected = run(slices)
    saved = dict(wgrad_api._cache_of_GroupedGemmWgradSm100Objects), dict(wgrad_api._wgrad_wrapper_memo)
    wgrad_api._cache_of_GroupedGemmWgradSm100Objects.clear()
    wgrad_api._wgrad_wrapper_memo.clear()
    try:
        assert_bitwise(run(slices), expected, "cold compile")
    finally:
        wgrad_api._cache_of_GroupedGemmWgradSm100Objects.update(saved[0])
        wgrad_api._wgrad_wrapper_memo.update(saved[1])


def test_sm90_wgrad_golden():
    slices = expert_slices(GROUPS)
    inputs = {f"dy_{e}": a for e, (a, _) in enumerate(slices)} | {f"x_{e}": b for e, (_, b) in enumerate(slices)}
    assert_golden("grouped_gemm_wgrad_sm90_bf16", inputs, run(slices))


PINGPONG = dict(kernel_schedule="pingpong")


def test_sm90_wgrad_pingpong_replays_bitwise():
    slices = expert_slices(GROUPS)
    assert_replays(lambda: run(slices, **PINGPONG), repeats=6)


@pytest.mark.parametrize("tile", [(128, 128), (128, 160), (128, 192)], ids=["128x128", "128x160", "128x192"])
def test_sm90_wgrad_pingpong_matches_cooperative_bytes(tile):
    # Both schedules reduce each output element with the same WGMMA sequence, so at one tile the bytes agree.
    slices = expert_slices(GROUPS)
    cooperative = run(slices, mma_tiler_mn=tile, kernel_schedule="cooperative")
    assert_bitwise(run(slices, mma_tiler_mn=tile, **PINGPONG), cooperative, f"pingpong vs cooperative {tile}")


@pytest.mark.parametrize("remaining", ["half", "one"], ids=["half-clusters", "one-cluster"])
def test_sm90_wgrad_pingpong_claim_order_invariance(monkeypatch, remaining):
    # Fewer (2, 1) clusters claim more tiles each, handed in turn to the two MMA warp groups.
    slices = expert_slices(GROUPS)
    expected = run(slices, **PINGPONG)
    clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(2)
    margin = str(clusters - (clusters // 2 if remaining == "half" else 1))
    monkeypatch.setenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", margin)
    assert_bitwise(run(slices, **PINGPONG), expected, f"margin {margin}")


def test_sm90_wgrad_pingpong_expert_batch_invariance():
    slices = expert_slices(GROUPS)
    full = run(slices, **PINGPONG)["wgrad"]
    order = [6, 2, 0, 7, 4]
    reordered = run(slices, order, **PINGPONG)["wgrad"]
    for position, expert in enumerate(order):
        assert_bitwise({"wgrad": reordered[position]}, {"wgrad": full[expert]}, f"expert {expert} reordered")

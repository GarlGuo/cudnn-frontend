# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Wentao Guo, Phuong Nguyen

"""SM90 (Hopper) coverage for the unfused BF16 grouped GEMM wrapper: forward (K-major B) and dgrad (N-major B)."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.L0


@pytest.fixture(autouse=True)
def require_sm90():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 is required")


def make_problem(counts, n, k, *, direction="fwd", seed=0, out_dtype=torch.bfloat16):
    """Padded (256-row) expert segments; ``direction="dgrad"`` passes B as the canonical N-major (E, K, N) buffer."""
    assert all(count % 256 == 0 for count in counts)
    generator = torch.Generator(device="cuda").manual_seed(20261121 + seed)
    experts, m = len(counts), max(sum(counts), 256)
    a = torch.randn((m, k), generator=generator, device="cuda", dtype=torch.bfloat16) * 0.125
    # fwd: weights (E, N, K) row-major (K-major B). dgrad: the forward weights (E, n_out=K, k_in=N), read N-major.
    storage_shape = (experts, n, k) if direction == "fwd" else (experts, k, n)
    weights = torch.randn(storage_shape, generator=generator, device="cuda", dtype=torch.bfloat16) * 0.125
    offsets = torch.tensor(counts, dtype=torch.int64).cumsum(0).to(torch.int32).cuda()
    alpha = torch.linspace(0.5, 1.5, experts, dtype=torch.float32, device="cuda")
    prob = torch.rand((m,), generator=generator, device="cuda", dtype=torch.float32)
    return dict(a=a, weights=weights, offsets=offsets, alpha=alpha, prob=prob, n=n, k=k, counts=counts, direction=direction, out_dtype=out_dtype)


def reference(problem):
    """FP32 per-expert C = alpha * A @ B^T and D = prob * C."""
    a, weights = problem["a"].float(), problem["weights"].float()
    c = torch.zeros((a.shape[0], problem["n"]), dtype=torch.float32, device="cuda")
    begin = 0
    for expert, count in enumerate(problem["counts"]):
        w = weights[expert] if problem["direction"] == "fwd" else weights[expert].t()
        c[begin : begin + count] = problem["alpha"][expert] * (a[begin : begin + count] @ w.t())
        begin += count
    return c, c * problem["prob"][:, None]


def run(problem, **kwargs):
    import cudnn

    return cudnn.grouped_gemm_wrapper_sm100(
        a_tensor=problem["a"],
        padded_offsets=problem["offsets"],
        alpha_tensor=problem["alpha"],
        b_tensor=problem["weights"],
        b_major="k" if problem["direction"] == "fwd" else "n",
        prob_tensor=problem["prob"],
        c_dtype=problem["out_dtype"],
        d_dtype=problem["out_dtype"],
        generate_c=True,
        **kwargs,
    )


def check(problem, result):
    expected_c, expected_d = reference(problem)
    rows = sum(problem["counts"])
    for name, actual, expected in (("c", result["c_tensor"], expected_c), ("d", result["d_tensor"], expected_d)):
        assert not torch.isnan(actual[:rows].float()).any(), name
        torch.testing.assert_close(actual[:rows].float(), expected[:rows].to(actual.dtype).float(), rtol=2e-2, atol=3e-2, msg=name)


@pytest.mark.parametrize("direction", ["fwd", "dgrad"])
@pytest.mark.parametrize(
    "tile,cluster",
    [(None, None), ((128, 256), (1, 1)), ((128, 128), (1, 1)), ((128, 192), (2, 1)), ((64, 256), (1, 2)), ((64, 64), (2, 2))],
    ids=["default", "128x256-c11", "128x128-c11", "128x192-c21", "64x256-c12", "64x64-c22"],
)
def test_sm90_grouped_gemm_tiles(direction, tile, cluster):
    problem = make_problem((256, 0, 768, 512), n=1024, k=512, direction=direction)
    check(problem, run(problem, mma_tiler_mn=tile, cluster_shape_mn=cluster))


@pytest.mark.parametrize("direction", ["fwd", "dgrad"])
@pytest.mark.parametrize("n,k", [(96, 256), (416, 288), (7168, 2048), (2048, 7168)], ids=["N96", "N416-K288", "N7168-K2048", "N2048-K7168"])
def test_sm90_grouped_gemm_edges(direction, n, k):
    # N below one tile and off the 256 grid, K off the 64 grid, and the deepseek_v3 FC2 / dgrad shapes.
    problem = make_problem((512, 256, 0, 256), n=n, k=k, direction=direction, seed=n + k)
    check(problem, run(problem))


@pytest.mark.parametrize("experts", [1, 8, 32, 256])
def test_sm90_grouped_gemm_experts(experts):
    counts = [256 * ((i * 7) % 4) for i in range(experts)]
    counts[-1] = 256  # the last expert's end is the token count
    problem = make_problem(tuple(counts), n=512, k=256, seed=experts)
    check(problem, run(problem))


def test_sm90_grouped_gemm_all_empty():
    problem = make_problem((0, 0, 0), n=256, k=256)
    result = run(problem)
    torch.cuda.synchronize()
    assert result["d_tensor"].shape == (256, 256)


def test_sm90_grouped_gemm_fp16_and_legacy_layout():
    import cudnn

    problem = make_problem((256, 512), n=512, k=256, out_dtype=torch.float16)
    check(problem, run(problem))
    # The kernel-facing (m, k, 1) A, (n, k, l) B and (m, 1, 1) prob views give the same bytes.
    m, k = problem["a"].shape
    legacy = cudnn.grouped_gemm_wrapper_sm100(
        a_tensor=problem["a"].as_strided((m, k, 1), (k, 1, m * k)),
        padded_offsets=problem["offsets"],
        alpha_tensor=problem["alpha"],
        b_tensor=problem["weights"].permute(1, 2, 0),
        prob_tensor=problem["prob"].view(m, 1, 1),
        c_dtype=torch.float16,
        d_dtype=torch.float16,
    )
    assert torch.equal(legacy["d_tensor"].view(m, -1), run(problem)["d_tensor"])


def test_sm90_grouped_gemm_static_request_runs_dynamic():
    # SM90 always runs the dynamic scheduler; use_dynamic_sched=False is accepted.
    problem = make_problem((256, 256), n=256, k=256)
    check(problem, run(problem, use_dynamic_sched=False))


def test_sm90_grouped_gemm_rejects_unsupported():
    problem = make_problem((256, 256), n=256, k=256)
    with pytest.raises(NotImplementedError, match="bias"):
        run(problem, bias_tensor=torch.zeros((2, 256), dtype=torch.bfloat16, device="cuda").t())
    fp32 = dict(problem, out_dtype=torch.float32)
    with pytest.raises((NotImplementedError, ValueError)):
        run(fp32)
    with pytest.raises((NotImplementedError, ValueError, AssertionError)):
        run(problem, mma_tiler_mn=(256, 256))
    import cudnn

    ptrs = torch.tensor([w.data_ptr() for w in problem["weights"]], dtype=torch.int64, device="cuda")
    with pytest.raises((NotImplementedError, AssertionError)):
        cudnn.grouped_gemm_wrapper_sm100(
            a_tensor=problem["a"],
            padded_offsets=problem["offsets"],
            alpha_tensor=problem["alpha"],
            b_ptrs=ptrs,
            n=256,
            b_dtype=torch.bfloat16,
            prob_tensor=problem["prob"],
        )


@pytest.mark.parametrize("tile,cluster", [((128, 256), (4, 1)), ((64, 128), (8, 1)), ((128, 128), (4, 2))], ids=["128-c41", "64-c81", "128-c42"])
def test_sm90_grouped_gemm_rejects_cluster_m_over_expert_pad(tile, cluster):
    # A cluster M tile over the 256-row expert pad would write the next expert's rows.
    problem = make_problem((256, 256), n=256, k=256)
    with pytest.raises(ValueError):
        run(problem, mma_tiler_mn=tile, cluster_shape_mn=cluster)

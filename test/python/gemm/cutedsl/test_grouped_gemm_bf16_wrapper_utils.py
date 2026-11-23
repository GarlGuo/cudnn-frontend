# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Wentao Guo, Phuong Nguyen

"""Ragged-expert problems, wrapper calls and FP32 references for the BF16 grouped GEMM, GLU and dGLU wrappers."""

from __future__ import annotations

import torch

import cudnn

BLOCK = 32  # interleaved_32: accumulator column block 2q is gate, block 2q + 1 is up


def make_gemm_problem(counts, n, k, *, direction="fwd", seed=0, out_dtype=torch.bfloat16):
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


def run_gemm(problem, **kwargs):
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


def gemm_reference(problem):
    """FP32 per-expert C = alpha * A @ B^T and D = prob * C."""
    a, weights = problem["a"].float(), problem["weights"].float()
    c = torch.zeros((a.shape[0], problem["n"]), dtype=torch.float32, device="cuda")
    begin = 0
    for expert, count in enumerate(problem["counts"]):
        w = weights[expert] if problem["direction"] == "fwd" else weights[expert].t()
        c[begin : begin + count] = problem["alpha"][expert] * (a[begin : begin + count] @ w.t())
        begin += count
    return c, c * problem["prob"][:, None]


def check_gemm(problem, result):
    expected_c, expected_d = gemm_reference(problem)
    rows = sum(problem["counts"])
    for name, actual, expected in (("c", result["c_tensor"], expected_c), ("d", result["d_tensor"], expected_d)):
        assert not torch.isnan(actual[:rows].float()).any(), name
        torch.testing.assert_close(actual[:rows].float(), expected[:rows].to(actual.dtype).float(), rtol=2e-2, atol=3e-2, msg=name)


def make_glu_problem(counts, n, k, *, seed=0, out_dtype=torch.bfloat16, b_major="k"):
    """Padded (256-row) expert segments; ``n`` is the GEMM N (the GLU's C width, the dGLU's gate/up width)."""
    assert all(count % 256 == 0 for count in counts)
    generator = torch.Generator(device="cuda").manual_seed(20261122 + seed)
    experts, m = len(counts), max(sum(counts), 256)
    a = torch.randn((m, k), generator=generator, device="cuda", dtype=torch.bfloat16) * 0.25
    storage = (experts, n, k) if b_major == "k" else (experts, k, n)
    weights = torch.randn(storage, generator=generator, device="cuda", dtype=torch.bfloat16) * 0.125
    offsets = torch.tensor(counts, dtype=torch.int64).cumsum(0).to(torch.int32).cuda()
    alpha = torch.linspace(0.5, 1.5, experts, dtype=torch.float32, device="cuda")
    beta = torch.linspace(1.25, 0.75, experts, dtype=torch.float32, device="cuda")
    prob = torch.rand((m,), generator=generator, device="cuda", dtype=torch.float32)
    # The forward's saved pre-activation C (gate | up), wide enough to cross the GeGLU clamps.
    c = (torch.randn((m, 2 * n), generator=generator, device="cuda", dtype=torch.float32) * 4).to(out_dtype)
    return dict(a=a, weights=weights, offsets=offsets, alpha=alpha, beta=beta, prob=prob, c=c, n=n, k=k, counts=counts, out_dtype=out_dtype, b_major=b_major)


def run_glu(problem, act="swiglu", **kwargs):
    return cudnn.grouped_gemm_glu_wrapper_sm100(
        a_tensor=problem["a"],
        sfa_tensor=None,
        padded_offsets=problem["offsets"],
        alpha_tensor=problem["alpha"],
        b_tensor=problem["weights"],
        prob_tensor=problem["prob"],
        c_dtype=problem["out_dtype"],
        d_dtype=problem["out_dtype"],
        act_func=act,
        generate_c=True,
        **kwargs,
    )


def run_dglu(problem, act="dswiglu", dprob=None, **kwargs):
    dprob = torch.zeros_like(problem["prob"]) if dprob is None else dprob
    return cudnn.grouped_gemm_dglu_wrapper_sm100(
        a_tensor=problem["a"],
        c_tensor=problem["c"],
        sfa_tensor=None,
        padded_offsets=problem["offsets"],
        alpha_tensor=problem["alpha"],
        beta_tensor=problem["beta"],
        prob_tensor=problem["prob"],
        dprob_tensor=dprob,
        b_tensor=problem["weights"],
        b_major=problem["b_major"],
        d_dtype=problem["out_dtype"],
        act_func=act,
        **kwargs,
    )


def split_gate_up(x):
    """(m, 2n) interleaved_32 columns -> (gate, up), each (m, n)."""
    m, two_n = x.shape
    blocks = x.view(m, two_n // (2 * BLOCK), 2, BLOCK)
    return blocks[:, :, 0].reshape(m, -1), blocks[:, :, 1].reshape(m, -1)


def join_gate_up(gate, up):
    m, n = gate.shape
    return torch.stack((gate.view(m, n // BLOCK, BLOCK), up.view(m, n // BLOCK, BLOCK)), dim=2).reshape(m, 2 * n)


def expert_slices(counts):
    begin = 0
    for expert, count in enumerate(counts):
        yield expert, slice(begin, begin + count)
        begin += count


def glu_gemm_reference(problem, scale):
    """FP32 per-expert scale[e] * A @ B^T, (m, n)."""
    a, weights = problem["a"].float(), problem["weights"].float()
    acc = torch.zeros((a.shape[0], problem["n"]), dtype=torch.float32, device="cuda")
    for expert, rows in expert_slices(problem["counts"]):
        w = weights[expert] if problem["b_major"] == "k" else weights[expert].t()
        acc[rows] = scale[expert] * (a[rows] @ w.t())
    return acc


def expert_rows(problem, per_expert):
    out = torch.zeros((problem["a"].shape[0], 1), dtype=torch.float32, device="cuda")
    for expert, rows in expert_slices(problem["counts"]):
        out[rows] = per_expert[expert]
    return out


def glu_reference(problem, act, linear_offset):
    c = glu_gemm_reference(problem, problem["alpha"])
    gate, up = split_gate_up(c)
    if act == "geglu":
        gate, up = gate.clamp(max=7.0), up.clamp(-7.0, 7.0)
        d = (up + linear_offset) * gate * torch.sigmoid(1.702 * gate)
    else:
        d = up * gate * torch.sigmoid(gate)
    return c, d * problem["prob"][:, None]


def check_glu(problem, result, act="swiglu", linear_offset=None):
    linear_offset = (1.0 if act == "geglu" else 0.0) if linear_offset is None else linear_offset
    expected_c, expected_d = glu_reference(problem, act, linear_offset)
    rows = sum(problem["counts"])
    for name, actual, expected in (("c", result["c_tensor"], expected_c), ("d", result["d_tensor"], expected_d)):
        assert not torch.isnan(actual[:rows].float()).any(), name
        torch.testing.assert_close(actual[:rows].float(), expected[:rows].to(actual.dtype).float(), rtol=2e-2, atol=3e-2, msg=name)


def dglu_reference(problem, act, linear_offset, round_dgrad=False):
    """D (dgate | dup) and dprob of dGLU, from dY @ W^T scaled by alpha^2 and the saved C scaled by beta."""
    g = glu_gemm_reference(problem, problem["alpha"] ** 2)
    if round_dgrad:
        g = g.to(torch.bfloat16).float()
    beta = expert_rows(problem, problem["beta"])
    x1, x2 = (half.float() * beta for half in split_gate_up(problem["c"]))
    prob = problem["prob"][:, None]
    if act == "dgeglu":
        y1, y2 = x1.clamp(max=7.0), x2.clamp(-7.0, 7.0) + linear_offset
        s = torch.sigmoid(1.702 * y1)
        dprob = (y1 * s * y2 * g).sum(dim=1)
        d1 = (x1 <= 7.0) * g * prob * s * (1 + 1.702 * y1 * (1 - s)) * y2
        d2 = ((x2 >= -7.0) & (x2 <= 7.0)) * g * prob * y1 * s
    else:
        s = torch.sigmoid(x1)
        dprob = (g * x2 * x1 * s).sum(dim=1)
        d1 = g * prob * x2 * s * (1 + x1 * (1 - s))
        d2 = g * prob * x1 * s
    return join_gate_up(d1, d2), dprob


def check_dglu(problem, result, act="dswiglu", linear_offset=None, round_dgrad=False):
    linear_offset = (1.0 if act == "dgeglu" else 0.0) if linear_offset is None else linear_offset
    expected_d, expected_dprob = dglu_reference(problem, act, linear_offset, round_dgrad)
    rows = sum(problem["counts"])
    actual_d = result["d_row_tensor"].view(result["d_row_tensor"].shape[0], -1)
    assert not torch.isnan(actual_d[:rows].float()).any()
    torch.testing.assert_close(actual_d[:rows].float(), expected_d[:rows].to(actual_d.dtype).float(), rtol=2e-2, atol=5e-2, msg="d")
    dprob = result["dprob_tensor"].view(-1)
    torch.testing.assert_close(dprob[:rows], expected_dprob[:rows], rtol=2e-2, atol=2e-1 * (problem["n"] / 256) ** 0.5, msg="dprob")


def gemm_segment_problem(direction, segments, n=768, k=640):
    """Pack the given (count, seed) segments; each segment's A rows, weights, alpha and prob come from its seed alone."""
    parts = [make_gemm_problem((count,), n=n, k=k, direction=direction, seed=seed) for count, seed in segments]
    return dict(
        parts[0],
        a=torch.cat([p["a"][: p["counts"][0]] for p in parts]),
        weights=torch.cat([p["weights"] for p in parts]),
        offsets=torch.tensor([c for c, _ in segments], dtype=torch.int64).cumsum(0).to(torch.int32).cuda(),
        alpha=torch.tensor([float(1 + seed % 5) * 0.25 for _, seed in segments], device="cuda"),
        prob=torch.cat([p["prob"][: p["counts"][0]] for p in parts]),
        counts=tuple(c for c, _ in segments),
    )


def glu_segment_problem(segments, n=768, k=640):
    """Pack the given (count, seed) segments; each segment's rows, weights, alpha, beta and prob come from its seed alone."""
    parts = [make_glu_problem((count,), n=n, k=k, seed=seed) for count, seed in segments]
    return dict(
        parts[0],
        a=torch.cat([p["a"][: p["counts"][0]] for p in parts]),
        c=torch.cat([p["c"][: p["counts"][0]] for p in parts]),
        weights=torch.cat([p["weights"] for p in parts]),
        offsets=torch.tensor([c for c, _ in segments], dtype=torch.int64).cumsum(0).to(torch.int32).cuda(),
        alpha=torch.tensor([float(1 + seed % 5) * 0.25 for _, seed in segments], device="cuda"),
        beta=torch.tensor([float(1 + seed % 3) * 0.5 for _, seed in segments], device="cuda"),
        prob=torch.cat([p["prob"][: p["counts"][0]] for p in parts]),
        counts=tuple(c for c, _ in segments),
    )


def rows_of(result, segments, index):
    begin = sum(count for count, _ in segments[:index])
    return {name: value[begin : begin + segments[index][0]] for name, value in result.items()}

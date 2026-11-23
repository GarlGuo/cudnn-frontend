# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Wentao Guo, Phuong Nguyen

"""Ragged-expert problems and wrapper calls for the BF16 grouped GEMM, GLU and dGLU wrappers."""

from __future__ import annotations

import torch

import cudnn


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

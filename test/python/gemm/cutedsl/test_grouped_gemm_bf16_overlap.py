# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Wentao Guo, Phuong Nguyen

"""Two same-signature BF16 grouped GEMM wrapper calls overlapping on two streams match their serial results bitwise.

A cached plan's workspace holds the dynamic scheduler's tile counter and the discrete-mode TMA descriptors; overlapping
calls that shared it stole each other's tiles or descriptors. 32 experts x 512 rows at N=4096, K=1024 overlapped on GB200 every run.
"""

from __future__ import annotations

import pytest
import torch

import cudnn
from determinism_utils import assert_bitwise, snapshot
from gemm.cutedsl import test_grouped_gemm_bf16_wrapper_utils as problems
from gemm.cutedsl.test_grouped_gemm_wgrad_bf16_utils import make_grouped_gemm_wgrad_bf16_problem

pytestmark = pytest.mark.L0

COUNTS = (512,) * 32
N, K = 4096, 1024
ITERATIONS = 40


def _capability():
    return torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)


def _make(op, seed, discrete=False):
    if op == "wgrad":
        # M = N, N = K over 32 experts of 512 tokens; a per-seed scale keeps the two problems' bytes apart.
        problem = make_grouped_gemm_wgrad_bf16_problem(group_k_list=COUNTS, m=N, n=K, a_major="m", b_major="n", discrete=discrete)
        problem["a"].mul_(0.5 * seed)
        return problem
    if op in ("fwd", "dgrad"):
        problem = problems.make_gemm_problem(COUNTS, n=N, k=K, direction=op, seed=seed)
    else:
        problem = problems.make_glu_problem(COUNTS, n=N, k=K, seed=seed)
    if discrete:
        problem["b_ptrs"] = torch.tensor([expert.data_ptr() for expert in problem["weights"]], dtype=torch.int64, device="cuda")
    return problem


def _launch_discrete(op, problem, kwargs):
    common = dict(
        a_tensor=problem["a"],
        padded_offsets=problem["offsets"],
        alpha_tensor=problem["alpha"],
        b_ptrs=problem["b_ptrs"],
        n=N,
        b_dtype=torch.bfloat16,
        prob_tensor=problem["prob"],
        generate_c=True,
        **kwargs,
    )
    if op == "fwd":
        result = cudnn.grouped_gemm_wrapper_sm100(**common)
    else:
        result = cudnn.grouped_gemm_glu_wrapper_sm100(sfa_tensor=None, **common)
    return {"c": result["c_tensor"], "d": result["d_tensor"]}


def _launch(op, problem, sched):
    """One wrapper call on the current stream; returns its outputs (fresh buffers)."""
    kwargs = {} if sched is None else {"use_dynamic_sched": sched == "dynamic"}
    if "b_ptrs" in problem:
        return _launch_discrete(op, problem, kwargs)
    if op in ("fwd", "dgrad"):
        result = problems.run_gemm(problem, **kwargs)
        return {"c": result["c_tensor"], "d": result["d_tensor"]}
    if op == "glu":
        result = problems.run_glu(problem, **kwargs)
        return {"c": result["c_tensor"], "d": result["d_tensor"]}
    if op == "dglu":
        # dprob is an atomic sum across N tiles, so only D is compared bitwise.
        return {"d": problems.run_dglu(problem, **kwargs)["d_row_tensor"]}
    output = torch.empty_like(problem["output"])
    discrete = problem["discrete"]
    ptrs = torch.tensor([expert.data_ptr() for expert in output], dtype=torch.int64, device="cuda") if discrete else None
    cudnn.grouped_gemm_wgrad_wrapper_sm100(
        a_tensor=problem["a"],
        b_tensor=problem["b"],
        sfa_tensor=None,
        sfb_tensor=None,
        offsets_tensor=problem["offsets"],
        output_mode="discrete" if discrete else "dense",
        wgrad_tensor=output,
        wgrad_ptrs=ptrs,
        acc_dtype=torch.float32,
        wgrad_dtype=problem["output_dtype"],
    )
    return {"wgrad": output}


def _assert_overlap_matches_serial(op, sched, discrete=False):
    pair = [_make(op, seed, discrete) for seed in (1, 2)]
    expected = [snapshot(_launch(op, problem, sched)) for problem in pair]
    torch.cuda.synchronize()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for iteration in range(ITERATIONS):
        torch.cuda.synchronize()
        results = []
        for stream, problem in zip(streams, pair):
            with torch.cuda.stream(stream):
                results.append(_launch(op, problem, sched))
        torch.cuda.synchronize()
        for index, (result, reference) in enumerate(zip(results, expected)):
            assert_bitwise(snapshot(result), reference, f"{op} {sched} iteration {iteration} call {index}")


SM100_CASES = [
    *[(op, sched) for op in ("fwd", "dgrad", "glu", "dglu") for sched in ("static", "dynamic")],
    ("wgrad", None),
    *[(op, sched) for op in ("fwd-discrete", "glu-discrete") for sched in ("static", "dynamic")],
    ("wgrad-discrete", None),
]


@pytest.mark.parametrize("op,sched", SM100_CASES, ids=[f"{op}-{sched or 'default'}" for op, sched in SM100_CASES])
def test_sm100_bf16_overlapping_streams(op, sched):
    if _capability()[0] != 10:
        pytest.skip("SM100/103 is required")
    _assert_overlap_matches_serial(op.split("-")[0], sched, discrete=op.endswith("discrete"))

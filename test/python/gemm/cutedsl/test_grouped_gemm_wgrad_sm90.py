# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Wentao Guo, Phuong Nguyen

"""SM90 (Hopper) BF16 grouped GEMM wgrad behind grouped_gemm_wgrad_wrapper_sm100 / GroupedGemmWgradSm100."""

import pytest
import torch
import cudnn

from gemm.cutedsl.test_grouped_gemm_wgrad_bf16_utils import (
    assert_grouped_gemm_wgrad_close,
    grouped_gemm_wgrad_bf16_reference,
    make_grouped_gemm_wgrad_bf16_problem,
)

pytestmark = pytest.mark.L0


@pytest.fixture(autouse=True)
def require_sm90():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("The SM90 grouped GEMM wgrad needs a cc 9.0 device")


def run_wrapper(problem, **overrides):
    kwargs = dict(
        a_tensor=problem["a"],
        b_tensor=problem["b"],
        sfa_tensor=None,
        sfb_tensor=None,
        offsets_tensor=problem["offsets"],
        wgrad_tensor=problem["output"],
        acc_dtype=torch.float32,
        wgrad_dtype=problem["output_dtype"],
    )
    kwargs.update(overrides)
    return cudnn.grouped_gemm_wgrad_wrapper_sm100(**kwargs)["wgrad_tensor"]


# id: (group_k_list, m, n, a_major, b_major, wgrad dtype, mma_tiler_mn, cluster_shape_mn); None takes the SM90 default.
CASES = {
    "default": ((256, 0, 256), 256, 256, "m", "n", torch.bfloat16, None, None),
    "kmajor-fp32": ((512, 256), 256, 512, "k", "k", torch.float32, None, None),
    "edge-n192": ((256, 512, 0, 256), 384, 192, "m", "n", torch.bfloat16, (128, 192), (2, 1)),
    "fp16-128x128-c11": ((0, 256, 768), 512, 256, "m", "n", torch.float16, (128, 128), (1, 1)),
    "64x256-c12": ((256,) * 8, 320, 512, "m", "n", torch.bfloat16, (64, 256), (1, 2)),
    "experts32": ((0, 256, 512, 1024) * 8, 512, 256, "m", "k", torch.bfloat16, None, None),
    "one-tile-cluster": ((256, 256), 128, 128, "k", "n", torch.bfloat16, None, (2, 1)),
    "long-k": ((4096, 256), 256, 256, "m", "n", torch.bfloat16, None, None),
}


@pytest.mark.parametrize("case", list(CASES))
def test_sm90_wgrad_matches_reference(case):
    group_k_list, m, n, a_major, b_major, dtype, tile, cluster = CASES[case]
    # A nonzero initial value checks that every tile, including empty experts' zeros, is overwritten.
    problem = make_grouped_gemm_wgrad_bf16_problem(group_k_list=group_k_list, m=m, n=n, a_major=a_major, b_major=b_major, output_dtype=dtype, initial_value=3.0)
    actual = run_wrapper(problem, mma_tiler_mn=tile, cluster_shape_mn=cluster)
    torch.cuda.synchronize()
    assert actual is problem["output"]
    assert_grouped_gemm_wgrad_close(actual, grouped_gemm_wgrad_bf16_reference(problem))


def test_sm90_wgrad_class_api_reuses_compile_across_token_counts():
    sample = make_grouped_gemm_wgrad_bf16_problem(group_k_list=(256, 256), m=256, n=256, a_major="m", b_major="n")
    op = cudnn.GroupedGemmWgradSm100(
        sample_a=sample["a"],
        sample_b=sample["b"],
        sample_sfa=None,
        sample_sfb=None,
        sample_offsets=sample["offsets"],
        sample_wgrad=sample["output"],
        acc_dtype=torch.float32,
    )
    assert op.check_support()
    op.compile()
    for group_k_list in ((256, 256), (0, 1024), (768, 0)):
        problem = make_grouped_gemm_wgrad_bf16_problem(group_k_list=group_k_list, m=256, n=256, a_major="m", b_major="n", initial_value=-1.0)
        op.execute(
            a_tensor=problem["a"],
            b_tensor=problem["b"],
            sfa_tensor=None,
            sfb_tensor=None,
            offsets_tensor=problem["offsets"],
            wgrad_tensor=problem["output"],
        )
        torch.cuda.synchronize()
        assert_grouped_gemm_wgrad_close(problem["output"], grouped_gemm_wgrad_bf16_reference(problem))


@pytest.mark.parametrize(
    "overrides,error,match",
    [
        (dict(accumulate_on_output=True), NotImplementedError, "accumulate_on_output"),
        (dict(output_mode="discrete"), NotImplementedError, "dense output"),
        (dict(input_order="tensor_ragged"), NotImplementedError, "Tensor2D"),
        (dict(mma_tiler_mn=(256, 128)), ValueError, "Unsupported BF16 grouped GEMM wgrad configuration"),
        (dict(cluster_shape_mn=(4, 4)), ValueError, "Unsupported BF16 grouped GEMM wgrad configuration"),
    ],
    ids=["accumulate", "discrete", "ragged", "tile-m-256", "cluster-16"],
)
def test_sm90_wgrad_rejects_unsupported(overrides, error, match):
    problem = make_grouped_gemm_wgrad_bf16_problem(group_k_list=(256, 256), m=256, n=256)
    if overrides.get("output_mode") == "discrete":
        ptrs = torch.tensor([t.data_ptr() for t in problem["output"]], dtype=torch.int64, device="cuda")
        overrides = dict(overrides, wgrad_tensor=None, wgrad_ptrs=ptrs)
    with pytest.raises(error, match=match):
        run_wrapper(problem, **overrides)

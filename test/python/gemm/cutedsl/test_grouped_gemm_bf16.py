# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BF16 coverage for the unfused SM100 grouped GEMM wrapper."""

from __future__ import annotations

import pytest
import torch

from test_grouped_gemm_bf16_utils import (
    assert_grouped_gemm_close,
    grouped_gemm_bf16_reference,
    make_grouped_gemm_bf16_problem,
)


@pytest.fixture(autouse=True)
def require_sm100():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    from cudnn.tensor_adapter import has_tcgen05

    if not has_tcgen05(major):
        pytest.skip("SM100 is required")


@pytest.mark.L0
@pytest.mark.parametrize("discrete", [False, True], ids=["bf16-dense", "bf16-discrete"])
def test_grouped_gemm_bf16_wrapper(discrete):
    from cudnn import grouped_gemm_wrapper_sm100

    problem = make_grouped_gemm_bf16_problem(discrete=discrete, enable_bias=True)
    expected_c, expected_d = grouped_gemm_bf16_reference(problem)
    kwargs = dict(
        a_tensor=problem["a"],
        padded_offsets=problem["offsets"],
        alpha_tensor=problem["alpha"],
        bias_tensor=problem["bias"],
        prob_tensor=problem["prob"],
        c_dtype=torch.bfloat16,
        d_dtype=torch.bfloat16,
        generate_c=True,
    )
    if discrete:
        kwargs.update(b_ptrs=problem["b_ptrs"], n=problem["n"], b_dtype=torch.bfloat16)
    else:
        kwargs.update(b_tensor=problem["b"])

    result = grouped_gemm_wrapper_sm100(**kwargs)
    assert_grouped_gemm_close(result["c_tensor"], expected_c)
    assert_grouped_gemm_close(result["d_tensor"], expected_d)


# Experts 1 and 4 get no rows; N = 1280 leaves a half-filled 512-wide N tile.
RAGGED_GROUP_M = (512, 0, 256, 768, 0, 256)
RAGGED_ALPHA = (0.75, -1.25, 0.5, 1.5, -0.25, 1.0)


def _run_tile(problem, tile, b_major="k", generate_c=False, discrete=False, dynamic=False):
    from cudnn import grouped_gemm_wrapper_sm100

    kwargs = dict(
        a_tensor=problem["a"],
        padded_offsets=problem["offsets"],
        alpha_tensor=problem["alpha"],
        bias_tensor=problem["bias"],
        prob_tensor=problem["prob"],
        c_dtype=torch.bfloat16,
        d_dtype=torch.bfloat16,
        generate_c=generate_c,
        mma_tiler_mn=tile,
        cluster_shape_mn=(2, 1),
        use_dynamic_sched=dynamic,
    )
    if discrete:
        kwargs.update(b_ptrs=problem["b_ptrs"], n=problem["n"], b_dtype=torch.bfloat16)
    elif b_major == "n":
        kwargs.update(b_tensor=problem["b_storage"].transpose(1, 2).contiguous(), b_major="n")
    else:
        kwargs.update(b_tensor=problem["b"])
    return grouped_gemm_wrapper_sm100(**kwargs)


@pytest.mark.L0
@pytest.mark.parametrize(
    "n,b_major,generate_c,enable_bias,discrete,dynamic",
    [
        (1024, "k", False, False, False, False),
        (1024, "k", False, False, False, True),
        (1280, "k", True, True, False, True),
        (1280, "n", False, False, False, False),
        (1280, "n", False, False, False, True),
        (1280, "k", True, False, True, True),
    ],
    ids=["fwd-n1024", "fwd-n1024-dyn", "fwd-n1280-c-bias-dyn", "dgrad-n1280", "dgrad-n1280-dyn", "fwd-discrete-n1280-c-dyn"],
)
def test_grouped_gemm_bf16_tile_256x512(n, b_major, generate_c, enable_bias, discrete, dynamic):
    """The 2-CTA 256x512 tile matches the reference, and the default 256x256 tile bit for bit."""
    problem = make_grouped_gemm_bf16_problem(n=n, k=512, group_m=RAGGED_GROUP_M, alpha_values=RAGGED_ALPHA, discrete=discrete, enable_bias=enable_bias)
    expected_c, expected_d = grouped_gemm_bf16_reference(problem)
    wide = _run_tile(problem, (256, 512), b_major, generate_c, discrete, dynamic)
    assert_grouped_gemm_close(wide["d_tensor"], expected_d)
    if generate_c:
        assert_grouped_gemm_close(wide["c_tensor"], expected_c)
    # Each output element still accumulates its K blocks in the same order, so the result is bitwise the 256x256 one.
    narrow = _run_tile(problem, (256, 256), b_major, generate_c, discrete, dynamic)
    again = _run_tile(problem, (256, 512), b_major, generate_c, discrete, dynamic)
    for name in ("d_tensor", "c_tensor") if generate_c else ("d_tensor",):
        assert torch.equal(wide[name].view(torch.int16), narrow[name].view(torch.int16)), name
        assert torch.equal(wide[name].view(torch.int16), again[name].view(torch.int16)), name


@pytest.mark.L0
@pytest.mark.parametrize(
    "tile,cluster",
    [((128, 512), (1, 1)), ((128, 512), (2, 1)), ((256, 512), (1, 1)), ((256, 512), (4, 1)), ((256, 512), (2, 2)), ((256, 384), (2, 1)), ((256, 1024), (2, 1))],
    ids=["1cta-c11", "1cta-c21", "2cta-c11", "c41", "c22", "n384", "n1024"],
)
def test_grouped_gemm_bf16_tile_n_above_256_rejected(tile, cluster):
    """N > 256 is only the 2-CTA (256, 512) tile in a (2, 1) cluster."""
    from cudnn import grouped_gemm_wrapper_sm100

    problem = make_grouped_gemm_bf16_problem(n=1024)
    with pytest.raises(ValueError, match="Unsupported BF16 grouped GEMM tile"):
        grouped_gemm_wrapper_sm100(
            a_tensor=problem["a"],
            padded_offsets=problem["offsets"],
            alpha_tensor=problem["alpha"],
            b_tensor=problem["b"],
            prob_tensor=problem["prob"],
            c_dtype=torch.bfloat16,
            d_dtype=torch.bfloat16,
            mma_tiler_mn=tile,
            cluster_shape_mn=cluster,
        )


@pytest.mark.L0
def test_grouped_gemm_bf16_tile_256x512_kernel_scope():
    """The SM90 kernels and the SM100 dGLU kernel keep rejecting (256, 512)."""
    import cutlass
    from cudnn.gemm.cutedsl.grouped.dglu.moe_grouped_gemm_dglu_dbias import MoEGroupedGemmDgluDbiasBf16Kernel
    from cudnn.gemm.cutedsl.grouped.glu.moe_grouped_gemm_glu_sm90 import MoEGroupedGemmGluBf16KernelSm90
    from cudnn.gemm.cutedsl.grouped.unfused.moe_grouped_gemm import MoEGroupedGemmBf16Kernel
    from cudnn.gemm.cutedsl.grouped.unfused.moe_grouped_gemm_sm90 import MoEGroupedGemmBf16KernelSm90

    bf16 = cutlass.BFloat16
    args = lambda tile: (bf16, bf16, bf16, cutlass.Float32, tile[0] == 256, tile, (2, 1), 1024, 1024, 1024, 4, "k", "k", "n", 256)
    assert MoEGroupedGemmBf16Kernel.can_implement(*args((256, 512)))
    for kernel in (MoEGroupedGemmBf16KernelSm90, MoEGroupedGemmGluBf16KernelSm90):
        assert not kernel.can_implement(*args((256, 512))), kernel.__name__
    assert MoEGroupedGemmDgluDbiasBf16Kernel.can_implement(*args((256, 256)), "dswiglu")
    assert not MoEGroupedGemmDgluDbiasBf16Kernel.can_implement(*args((256, 512)), "dswiglu")

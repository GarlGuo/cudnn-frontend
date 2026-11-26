# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The BF16 grouped GEMM kernels size their A/B stages for the compile target's shared memory (SM107: 334848 B)."""

import pytest
import torch

pytestmark = pytest.mark.L0


@pytest.fixture(autouse=True)
def require_sm10x():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100/103/107 is required")


def bf16_kernels():
    import cutlass

    from cudnn.gemm.cutedsl.grouped.dglu.moe_grouped_gemm_dglu_dbias import MoEGroupedGemmDgluDbiasBf16Kernel
    from cudnn.gemm.cutedsl.grouped.glu.moe_grouped_gemm_glu_bias import MoEGroupedGemmGluBiasBf16Kernel
    from cudnn.gemm.cutedsl.grouped.unfused.moe_grouped_gemm import MoEGroupedGemmBf16Kernel
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_grouped_gemm_wgrad import MoEGroupedGemmWgradBF16Kernel
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_grouped_gemm_wgrad_wide import MoEGroupedGemmWgradWideBF16Kernel

    common = dict(acc_dtype=cutlass.Float32, use_2cta_instrs=True, mma_tiler_mn=(256, 256), cluster_shape_mn=(2, 1))
    return {
        "plain": (MoEGroupedGemmBf16Kernel(**common, vectorized_f32=False, generate_c=False, enable_bias=False, expert_cnt=2), "num_smem_capacity"),
        "glu": (MoEGroupedGemmGluBiasBf16Kernel(**common, vectorized_f32=False, expert_cnt=2), "num_smem_capacity"),
        "dglu": (MoEGroupedGemmDgluDbiasBf16Kernel(**common, vectorized_f32=False, expert_cnt=2), "num_smem_capacity"),
        "wgrad": (MoEGroupedGemmWgradBF16Kernel(**common, expert_cnt=2), "smem_capacity"),
        "wgrad_wide": (MoEGroupedGemmWgradWideBF16Kernel(expert_cnt=2), "smem_capacity"),
    }


def test_bf16_kernels_size_shared_memory_for_the_device():
    from cudnn._cutlass_compat import get_smem_capacity_in_bytes

    major, minor = torch.cuda.get_device_capability()
    expected = get_smem_capacity_in_bytes(f"sm_{major}{minor}")
    for name, (kernel, attribute) in bf16_kernels().items():
        assert getattr(kernel, attribute) == expected, name

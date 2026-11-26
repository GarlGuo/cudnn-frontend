# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical (m, x) operands whose m x x passes 2^31 elements.

``kernel_facing_mx`` adds the unit L mode of the kernel-facing (m, x, 1) view; its stride m x x must be formed in 64
bits, as the rank-3 fakes do (a dynamic i64 stride).  The trace test checks the stride width; the L1 test runs the BF16
grouped GEMM with A (and, separately, C/D) past 2^31 elements in both layouts (about 13 GB of device memory).
"""

import pytest
import torch

LIMIT = 2**31


@pytest.mark.L0
def test_kernel_facing_mx_unit_l_stride_is_64_bit():
    import cutlass
    import cutlass.cute as cute

    from cudnn.gemm.cutedsl.grouped.canonical import kernel_facing_mx

    seen = {}

    @cute.jit
    def trace(tensor: cute.Tensor):
        stride = kernel_facing_mx(tensor).stride[2]
        seen["stride"] = (type(stride).__name__, cute.is_static(stride))

    dynamic_m = cute.runtime.make_fake_compact_tensor(cutlass.BFloat16, (cute.sym_int(divisibility=256), 7168), stride_order=(1, 0), assumed_align=16)
    cute.compile(trace, dynamic_m)
    assert seen["stride"] == ("Int64", False)
    static_m = cute.runtime.make_fake_compact_tensor(cutlass.BFloat16, (512, 7168), stride_order=(1, 0), assumed_align=16)
    cute.compile(trace, static_m)
    assert seen["stride"] == ("int", True)


def repeated_rows(rows, columns, generator, block=65536):
    pattern = (torch.randn((block, columns), generator=generator, device="cuda") * 0.125).to(torch.bfloat16)
    out = torch.empty((rows, columns), device="cuda", dtype=torch.bfloat16)
    for row in range(0, rows, block):
        out[row : row + block] = pattern[: min(block, rows - row)]
    return out


@pytest.mark.L1
@pytest.mark.parametrize("m,k,n", [(532480, 4096, 256), (532480, 256, 4096)], ids=["a_past_2g", "cd_past_2g"])
def test_bf16_grouped_gemm_canonical_past_2g_elements(m, k, n):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100 is required")
    if torch.cuda.mem_get_info()[0] < 16 * 2**30:
        pytest.skip("needs about 16 GB of free device memory")
    assert max(m * k, m * n) >= LIMIT
    import cudnn

    generator = torch.Generator(device="cuda").manual_seed(31)
    a = repeated_rows(m, k, generator)
    b = (torch.randn((2, n, k), generator=generator, device="cuda") * 0.125).to(torch.bfloat16)
    offsets = torch.tensor([m - 2048, m], dtype=torch.int32, device="cuda")
    alpha = torch.tensor([0.75, -1.25], device="cuda")
    prob = torch.linspace(0.25, 0.875, m, device="cuda")
    rows = torch.cat([torch.arange(0, 1024, device="cuda"), torch.arange(m - 4096, m, device="cuda")])
    expert = (rows >= m - 2048).long()
    expected_c = torch.empty((len(rows), n), device="cuda")
    for e in range(2):
        mask = expert == e
        expected_c[mask] = alpha[e] * (a[rows[mask]].float() @ b[e].float().t())
    expected_d = expected_c * prob[rows, None]

    outputs = {}
    for layout in ("legacy", "canonical"):
        if layout == "legacy":
            args = dict(a_tensor=a.unsqueeze(-1), b_tensor=b.permute(1, 2, 0), prob_tensor=prob.view(m, 1, 1))
        else:
            args = dict(a_tensor=a, b_tensor=b, prob_tensor=prob)
        result = cudnn.grouped_gemm_wrapper_sm100(
            padded_offsets=offsets, alpha_tensor=alpha, c_dtype=torch.bfloat16, d_dtype=torch.bfloat16, generate_c=True, **args
        )
        c, d = result["c_tensor"].reshape(m, n)[rows], result["d_tensor"].reshape(m, n)[rows]
        torch.testing.assert_close(c.float(), expected_c.bfloat16().float(), rtol=2e-2, atol=3e-2)
        torch.testing.assert_close(d.float(), expected_d.bfloat16().float(), rtol=2e-2, atol=3e-2)
        outputs[layout] = (c, d)
        del result
        torch.cuda.empty_cache()
    assert torch.equal(outputs["legacy"][0], outputs["canonical"][0])
    assert torch.equal(outputs["legacy"][1], outputs["canonical"][1])

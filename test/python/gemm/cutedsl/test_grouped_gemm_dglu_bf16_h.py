# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The h output of the BF16 dGLU: ``generate_h`` / ``sample_h`` / ``h_tensor``, the API shape of the MXFP8 h output.

h = prob * act(beta * gate) * (beta * up) in BF16, written by the BF16 kernel as its activation output (no column
scales: ``sfh_col_tensor`` is None).  h matches the torch reference, and dH and dprob do not change when h is on.
"""

import pytest
import torch

import cudnn
from gemm.cutedsl.test_grouped_gemm_dglu_activation import ACT_FUNCS, check_activation, problem, reference
from gemm.cutedsl._workspace import ws

pytestmark = pytest.mark.L0
ENDS = [256, 768, 768]  # the rows past the last offset are capacity the kernel must not touch


def wrapper_call(p, *, canonical, act_func, generate_h, deterministic):
    a, c, b, prob, dp = p["a"], p["c"], p["b"], p["prob"], p["dp"]
    if canonical:
        a, c, b, prob, dp = a[:, :, 0], c[:, :, 0], p["weights"], prob.view(-1), dp.view(-1)
    dp.zero_()
    return cudnn.grouped_gemm_dglu_wrapper_sm100(
        a,
        c,
        None,
        p["offsets"],
        p["alpha"],
        p["beta"],
        prob,
        dp,
        b_tensor=b,
        act_func=act_func,
        vector_f32=True,
        geglu_alpha=1.0,
        glu_clamp_min=-10.0,
        glu_clamp_max=10.0,
        linear_offset=0.0,
        deterministic=deterministic,
        generate_h=generate_h,
    )


@pytest.mark.parametrize("act_func", ACT_FUNCS)
@pytest.mark.parametrize("canonical", [False, True])
@pytest.mark.parametrize("deterministic", [False, True])
def test_dglu_bf16_generate_h(act_func, canonical, deterministic):
    p = problem(False)
    m, n = p["c"].shape[0], p["c"].shape[1] // 2
    base = wrapper_call(p, canonical=canonical, act_func=act_func, generate_h=False, deterministic=deterministic)
    base_d, base_dprob = base["d_row_tensor"].clone(), base["dprob_tensor"].clone()
    assert "h_tensor" not in base
    expected = reference(p, ENDS, act_func=act_func)
    previous = None
    for _ in range(2):  # the second call takes the wrapper's memo path and must allocate a fresh h
        out = wrapper_call(p, canonical=canonical, act_func=act_func, generate_h=True, deterministic=deterministic)
        h = out["h_tensor"]
        assert out["sfh_col_tensor"] is None and "activation_tensor" not in out
        assert h.dtype == torch.bfloat16 and h.shape == ((m, n) if canonical else (m, n, 1))
        assert h is not previous
        previous = h
        h3 = h.view(m, n, 1)
        check_activation(h3[: ENDS[-1]], expected[: ENDS[-1]])
        # dH unchanged with h on; dprob too (bitwise when deterministic, else up to the atomic-add order).
        assert torch.equal(out["d_row_tensor"][: ENDS[-1]], base_d[: ENDS[-1]])
        if deterministic:
            assert torch.equal(out["dprob_tensor"][: ENDS[-1]], base_dprob[: ENDS[-1]])
        else:
            torch.testing.assert_close(out["dprob_tensor"][: ENDS[-1]], base_dprob[: ENDS[-1]], rtol=2e-5, atol=2e-5)


def class_plan(p, act_func, **output):
    return cudnn.GroupedGemmDgluSm100(
        p["a"],
        p["c"],
        p["d"],
        None,
        None,
        p["offsets"],
        p["alpha"],
        p["beta"],
        p["prob"],
        p["dp"],
        **p["decl"],
        act_func=act_func,
        vector_f32=True,
        geglu_alpha=1.0,
        glu_clamp_min=-10.0,
        glu_clamp_max=10.0,
        linear_offset=0.0,
        **output,
    )


@pytest.mark.parametrize("act_func", ACT_FUNCS)
def test_dglu_bf16_sample_h_is_the_activation_output(act_func):
    """The class API: a plan declared with sample_h and run with h_tensor writes the bytes a sample_activation /
    activation_tensor plan writes, and dH is the same."""
    p = problem(False)
    h_plan = class_plan(p, act_func, sample_h=p["activation"])
    act_plan = class_plan(p, act_func, sample_activation=p["activation"])
    for plan in (h_plan, act_plan):
        assert plan.check_support()
        plan.compile()
    outputs = []
    for plan, key in ((h_plan, "h_tensor"), (act_plan, "activation_tensor")):
        h = torch.full_like(p["activation"], float("nan"))
        d = torch.full_like(p["d"], float("nan"))
        p["dp"].zero_()
        plan.execute(p["a"], p["c"], d, None, None, p["offsets"], p["alpha"], p["beta"], p["prob"], p["dp"], b_tensor=p["b"], workspace=ws(plan), **{key: h})
        outputs.append((h, d))
    assert torch.equal(outputs[0][0][: ENDS[-1]], outputs[1][0][: ENDS[-1]])
    assert torch.isnan(outputs[0][0][ENDS[-1] :]).all()
    assert torch.equal(outputs[0][1][: ENDS[-1]], outputs[1][1][: ENDS[-1]])
    check_activation(outputs[0][0][: ENDS[-1]], reference(p, ENDS, act_func=act_func)[: ENDS[-1]])
    with pytest.raises(ValueError, match="same BF16 output"):
        h_plan.execute(
            p["a"],
            p["c"],
            p["d"],
            None,
            None,
            p["offsets"],
            p["alpha"],
            p["beta"],
            p["prob"],
            p["dp"],
            b_tensor=p["b"],
            workspace=ws(h_plan),
            h_tensor=p["activation"],
            activation_tensor=p["activation"],
        )

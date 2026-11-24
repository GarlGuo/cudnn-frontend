# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MXFP8 dGLU (SM100) epilogue opt-ins against the default kernel: ``tight_sf_tmem_cols`` and ``sfd_col_lane_scales``
change where scale factors live and how the column scales are computed, never a value, so the E4M3 row and column D,
both SFD planes and the deterministic dprob slots must equal the default kernel's bytes, on the stock dynamic scheduler
and on the streaming input signal scheduler (every M tile ready).  Ragged experts (one empty) vary the per-expert column
SFD strides; NaN pre-activations exercise the NaN-propagating maxima."""

import pytest
import torch

OPTIONS = {
    "tight_sf_tmem_cols": dict(tight_sf_tmem_cols=True),
    "sfd_col_lane_scales": dict(sfd_col_lane_scales=True),
    "tight_and_lane": dict(tight_sf_tmem_cols=True, sfd_col_lane_scales=True),
}
ROWS = [256, 1280, 512, 0, 768, 2048, 256, 1024]
K, I = 1024, 1024
#: The scheduler: the stock dynamic one, or the streaming input signal scheduler with every M tile ready.
SCHEDULERS = {"dynamic": {}, "signal": dict(streaming_signal_capacity=sum(ROWS) // 256)}


@pytest.fixture(autouse=True)
def require_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("the SM100 MXFP8 dGLU kernel")
    from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old

    if cutedsl_too_old(cutedsl_state()[1]):
        pytest.skip("CuTe DSL older than the library floor")


def make_inputs(nan):
    g = torch.Generator(device="cuda").manual_seed(3)
    m, e = sum(ROWS), len(ROWS)
    fp8 = lambda *s: (torch.randn(*s, generator=g, device="cuda") * 2).to(torch.float8_e4m3fn)
    scales = lambda n: torch.randint(124, 131, (n,), generator=g, device="cuda", dtype=torch.uint8)
    c = (torch.randn(m, 2 * I, generator=g, device="cuda") * 2).to(torch.bfloat16)
    if nan:
        c.view(-1)[::99991] = float("nan")
    return dict(
        a=fp8(m, K),
        sfa=scales(m * K // 32),
        b=fp8(e, I, K),
        sfb=scales(e * I * K // 32),
        c=c,
        offsets=torch.tensor(ROWS, device="cuda").cumsum(0).to(torch.int32),
        alpha=torch.rand(e, generator=g, device="cuda") + 0.5,
        beta=torch.rand(e, generator=g, device="cuda") + 0.5,
        prob=torch.rand(m, 1, 1, generator=g, device="cuda"),
    )


def run(inputs, **options):
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.nvgpu import OperandMajorMode
    from cutlass.cute.runtime import from_dlpack
    from cutlass.utils import HardwareInfo
    from cudnn.gemm.cutedsl.grouped.dglu.moe_blockscaled_grouped_gemm_dglu_dbias import BlockScaledMoEGroupedGemmDgluDbiasKernel
    from cudnn.gemm.cutedsl.grouped.moe_utils import MoEWeightMode

    m, e = sum(ROWS), len(ROWS)
    kernel = BlockScaledMoEGroupedGemmDgluDbiasKernel(
        sf_vec_size=32,
        acc_dtype=cutlass.Float32,
        use_2cta_instrs=True,
        mma_tiler_mn=(256, 256),
        cluster_shape_mn=(2, 1),
        vectorized_f32=False,
        discrete_col_sfd=True,
        expert_cnt=e,
        weight_mode=MoEWeightMode.DENSE,
        use_dynamic_sched=True,
        act_func="dswiglu",
        deterministic=True,
        **options,
    )
    if kernel.use_streaming_signal_sched:
        from cudnn.gemm.cutedsl.grouped.moe_kernel_helpers import reset_streaming_signal_workspace

    fill = lambda n: torch.full((n,), 0xA5, device="cuda", dtype=torch.uint8)
    out = dict(d=fill(m * 2 * I), d_col=fill(m * 2 * I), sfd_row=fill(m * 2 * I // 32), sfd_col=fill(m * 2 * I // 32))
    dprob = torch.zeros(m, I // 256, device="cuda")
    workspace = torch.zeros(max(256, kernel.get_workspace_bytes()), device="cuda", dtype=torch.uint8)
    t = lambda x: from_dlpack(x, assumed_align=16)
    e8 = lambda x: t(x.view(torch.float8_e8m0fnu))
    codes = lambda x: t(x.view(m, 2 * I).view(torch.float8_e4m3fn))
    args = [
        t(inputs["a"]),
        t(inputs["b"].permute(1, 2, 0)),
        e8(inputs["sfb"]),
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int64(0),
        OperandMajorMode.K,
        from_dlpack(workspace, assumed_align=128).iterator,
        t(inputs["c"]),
        codes(out["d"]),
        codes(out["d_col"]),
        e8(inputs["sfa"]),
        e8(out["sfd_row"]),
        e8(out["sfd_col"]),
        None,
        t(torch.ones(1, device="cuda")),
        t(inputs["offsets"]),
        t(inputs["alpha"]),
        t(inputs["beta"]),
        t(inputs["prob"]),
        t(dprob.unsqueeze(-1)),
        None,
    ]
    if kernel.use_streaming_signal_sched:
        reset_streaming_signal_workspace(workspace, kernel, all_complete=True)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(kernel, *args, HardwareInfo().get_max_active_clusters(2), stream)
    compiled(*[x for i, x in enumerate(args) if i != 6], stream)
    torch.cuda.synchronize()
    out["dprob"] = dprob
    return out


@pytest.mark.L0
@pytest.mark.parametrize("nan", [False, True], ids=["finite", "nan_inputs"])
@pytest.mark.parametrize("option", sorted(OPTIONS))
@pytest.mark.parametrize("scheduler", sorted(SCHEDULERS))
def test_mxfp8_dglu_epilogue_option_matches_stock(scheduler, option, nan):
    inputs = make_inputs(nan)
    stock = run(inputs, **SCHEDULERS[scheduler])
    got = run(inputs, **SCHEDULERS[scheduler], **OPTIONS[option])
    for name in stock:
        assert torch.equal(got[name].view(torch.uint8), stock[name].view(torch.uint8)), f"{option}: {name} differs from the stock kernel"
    assert not (stock["sfd_col"] == 0xA5).all(), "the column SFD was written"

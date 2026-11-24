# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MXFP8 dGLU (SM100) ``compact_sfa_tma_warp`` against the SF warp: with compact SFA rows the TMA warp builds the SF atoms
itself instead of a dedicated SF warp, which changes who stores the atoms and when they are released, never a value. So
the E4M3 row and column D, both SFD planes and the deterministic dprob slots must equal the SF warp kernel's bytes, on the
streaming input signal scheduler (every M tile ready) and on the stock dynamic scheduler, with and without the epilogue
opt-ins.  Ragged experts (one empty); K of 2, 3 and 4 groups of 4 K tiles (the TMA warp holds two groups in flight);
NaN pre-activations."""

import pytest
import torch

ROWS = [256, 1280, 512, 0, 768, 2048, 256, 1024]
I = 1024
KS = [1024, 1536, 2048]
#: The scheduler: the streaming input signal scheduler with every M tile ready (the EP consumer), or the stock one.
SCHEDULERS = {
    "signal": dict(streaming_signal_capacity=sum(ROWS) // 256),
    "signal_multi_batch": dict(streaming_signal_capacity=sum(ROWS) // 256, streaming_signal_multi_batch_opening=True),
    "dynamic": {},
}
EPILOGUE = {"default": {}, "tight_and_lane": dict(tight_sf_tmem_cols=True, sfd_col_lane_scales=True)}


@pytest.fixture(autouse=True)
def require_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("the SM100 MXFP8 dGLU kernel")
    from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old

    if cutedsl_too_old(cutedsl_state()[1]):
        pytest.skip("CuTe DSL older than the library floor")


def make_inputs(k, nan):
    g = torch.Generator(device="cuda").manual_seed(5)
    m, e = sum(ROWS), len(ROWS)
    fp8 = lambda *s: (torch.randn(*s, generator=g, device="cuda") * 2).to(torch.float8_e4m3fn)
    scales = lambda n: torch.randint(124, 131, (n,), generator=g, device="cuda", dtype=torch.uint8)
    c = (torch.randn(m, 2 * I, generator=g, device="cuda") * 2).to(torch.bfloat16)
    if nan:
        c.view(-1)[::99991] = float("nan")
    return dict(
        k=k,
        a=fp8(m, k),
        # Compact rows [M, K / 32]: one E8M0 scale per 32 K elements of each row.
        sfa=scales(m * k // 32),
        b=fp8(e, I, k),
        sfb=scales(e * I * k // 32),
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
    from cudnn.gemm.cutedsl.grouped.moe_kernel_helpers import reset_streaming_signal_workspace
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
        compact_sfa_sf_warp=True,
        **options,
    )
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
    return out, kernel


@pytest.mark.L0
@pytest.mark.parametrize("nan", [False, True], ids=["finite", "nan_inputs"])
@pytest.mark.parametrize("k", KS, ids=[f"k{k}" for k in KS])
@pytest.mark.parametrize("epilogue", sorted(EPILOGUE))
@pytest.mark.parametrize("scheduler", sorted(SCHEDULERS))
def test_mxfp8_dglu_compact_sfa_tma_warp_matches_sf_warp(scheduler, epilogue, k, nan):
    inputs = make_inputs(k, nan)
    options = dict(SCHEDULERS[scheduler], **EPILOGUE[epilogue])
    sf_warp, sf_warp_kernel = run(inputs, **options)
    got, kernel = run(inputs, compact_sfa_tma_warp=True, **options)
    assert sf_warp_kernel.threads_per_cta == 288 and kernel.threads_per_cta == 256, "the TMA warp kernel has no SF warp"
    for name in sf_warp:
        assert torch.equal(got[name].view(torch.uint8), sf_warp[name].view(torch.uint8)), f"{name} differs from the SF warp kernel"
    assert not (sf_warp["sfd_col"] == 0xA5).all(), "the column SFD was written"


@pytest.mark.L0
def test_compact_sfa_tma_warp_requires_compact_sfa():
    import cutlass
    from cudnn.gemm.cutedsl.grouped.dglu.moe_blockscaled_grouped_gemm_dglu_dbias import BlockScaledMoEGroupedGemmDgluDbiasKernel
    from cudnn.gemm.cutedsl.grouped.moe_utils import MoEWeightMode

    with pytest.raises(ValueError, match="compact_sfa_tma_warp requires compact_sfa_sf_warp"):
        BlockScaledMoEGroupedGemmDgluDbiasKernel(
            sf_vec_size=32,
            acc_dtype=cutlass.Float32,
            use_2cta_instrs=True,
            mma_tiler_mn=(256, 256),
            cluster_shape_mn=(2, 1),
            vectorized_f32=False,
            discrete_col_sfd=True,
            expert_cnt=8,
            weight_mode=MoEWeightMode.DENSE,
            use_dynamic_sched=True,
            compact_sfa_tma_warp=True,
        )

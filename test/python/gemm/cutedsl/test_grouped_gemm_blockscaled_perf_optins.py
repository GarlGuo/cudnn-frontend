# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Perf options of the MXFP8 GLU and dGLU kernels: each one leaves the outputs it still writes byte-equal to the kernel
without it, on the stock dynamic scheduler and on the streaming input signal scheduler (every M tile ready).  On SM100
the outputs are E4M3 D with its SFD (the EP layer's FC1 and dGLU outputs), which the streaming signal suite (BF16 D) does
not cover.

- generate_sfd_col=False (SM100 GLU): the row-quantized D and row SFD only; the column pair is not written, and C is
  staged once when that buys another A/B stage.
- streaming_signal_multi_batch_opening (on by default; GLU and dGLU on the signal scheduler): one batch opening also
  publishes the other lanes' complete ready runs; off, one batch per opening.  On SM107 the Rubin GLU and dGLU build
  the same scheduler (BF16 D; their dGLU's dprob is an atomic sum, compared within FP32 reassociation).
"""

import pytest
import torch

from gemm.cutedsl import _streaming_signal_blockscaled as bs
from gemm.cutedsl._streaming_signal import signal_ready
from gemm.cutedsl._streaming_signal_blockscaled import require_sm10x  # noqa: F401  (autouse fixture)

#: E4M3 NaN codes and E8M0 NaN scales: an output the kernel should write but does not stays poisoned.
NAN_CODE, NAN_SCALE = 0x7F, 0xFF


@pytest.fixture
def require_sm100():
    if bs.is_rubin():
        pytest.skip("the SM100 kernels")


def sfd_buffers(kind, problem, kernel):
    """(C, D, D_col, SFD_row, SFD_col, norm_const, [dprob,] workspace): poisoned E4M3 codes and E8M0 scales."""
    rows, n = problem["rows"], problem["n"]
    width = n // 2 if kind == "glu" else 2 * n

    def codes():
        return torch.full((rows, width), NAN_CODE, device="cuda", dtype=torch.uint8)

    def scales():
        return torch.full((rows * width // bs.SF_VEC_SIZE,), NAN_SCALE, device="cuda", dtype=torch.uint8)

    workspace = torch.zeros(max(kernel.get_workspace_bytes(), 1), device="cuda", dtype=torch.uint8)
    out = dict(d=codes(), d_col=codes(), sfd_row=scales(), sfd_col=scales(), norm_const=torch.ones(1, device="cuda", dtype=torch.float32))
    if kind == "glu":
        out["c"] = torch.full((rows, n), float("nan"), device="cuda", dtype=torch.bfloat16)
    else:
        out["dprob"] = torch.zeros(rows, n // 256 if kernel.test_deterministic else 1, device="cuda", dtype=torch.float32)
    out["workspace"] = workspace
    return out


def sfd_args(kind, kernel, problem, buf, col):
    import cutlass
    from cutlass.cute.nvgpu import OperandMajorMode
    from cutlass.cute.runtime import from_dlpack

    def tensor(x):
        return from_dlpack(x, assumed_align=16)

    def e4m3(x):
        return tensor(x.view(torch.float8_e4m3fn))

    def e8m0(x):
        return tensor(x.view(torch.float8_e8m0fnu))

    n, k = problem["n"], problem["k"]
    head = [tensor(problem["a"]), tensor(problem["b"].permute(1, 2, 0) if kind == "dglu" else problem["b"]), e8m0(problem["sfb"])]
    head += [cutlass.Int32(n), cutlass.Int32(k), cutlass.Int64(k), OperandMajorMode.K, from_dlpack(buf["workspace"], assumed_align=128).iterator]
    offsets, alpha, prob = from_dlpack(problem["offsets"], assumed_align=4), tensor(problem["alpha"]), tensor(problem["prob"])
    sfa = e8m0(problem["a_exp"] if kernel.test_compact else problem["sfa"])
    d_col = e4m3(buf["d_col"]) if col else None
    sfd_col = e8m0(buf["sfd_col"]) if col else None
    quant = [e4m3(buf["d"]), d_col, sfa, e8m0(buf["sfd_row"]), sfd_col, None, tensor(buf["norm_const"]), offsets, alpha]
    if kind == "glu":
        # (c, d, d_col, sfa, sfd_row, sfd_col, amax, norm_const, offsets, alpha, prob, bias)
        return head + [tensor(buf["c"])] + quant + [prob, None]
    # (c, d, d_col, sfa, sfd_row, sfd_col, amax, norm_const, offsets, alpha, beta, prob, dprob, dbias)
    return head + [tensor(problem["c"])] + quant + [tensor(problem["beta"]), prob, tensor(buf["dprob"].unsqueeze(-1)), None]


COMPILED = {}


def run_sfd(kind, problem, col=True, **options):
    """One launch with E4M3 D and SFD (and the column pair unless ``col`` is False); returns the outputs by name."""
    import cutlass.cute as cute
    from cutlass.utils import HardwareInfo

    kernel = bs.make_kernel(kind, len(problem["layout"]) - 1, **({"generate_sfd": True} if kind == "glu" else {}), **options)
    buf = sfd_buffers(kind, problem, kernel)
    args = sfd_args(kind, kernel, problem, buf, col)
    clusters = HardwareInfo().get_max_active_clusters(kernel.test_cluster_size)
    key = (kernel.test_compile_key, col, problem["rows"], problem["n"])
    if key not in COMPILED:
        COMPILED[key] = cute.compile(kernel, *args, clusters, bs.cu_stream())
    if kernel.test_signal:
        signal_ready(kernel)(buf["workspace"])
    COMPILED[key](*[x for i, x in enumerate(args) if i != bs.b_major_arg(kernel)], bs.cu_stream())
    torch.cuda.synchronize()
    rows = bs.valid_rows(problem)
    names = ["d", "sfd_row"] + (["d_col", "sfd_col"] if col else []) + (["c"] if kind == "glu" else ["dprob"])
    return {name: buf[name][:rows] if name in ("c", "d", "d_col", "dprob") else buf[name] for name in names}


def assert_equal_bytes(got, want, label):
    for name in got:
        a, b = got[name], want[name]
        if a.dtype in (torch.bfloat16, torch.float32):
            a, b = a.view(torch.uint8), b.view(torch.uint8)
        assert torch.equal(a, b), f"{label}: {name} differs from the default kernel"


#: (scheduler variant, routing): the stock dynamic scheduler, and the signal scheduler with compact SFA.
VARIANTS = ("dynamic", "compact_signal")
CASES = [(variant, routing) for variant in VARIANTS for routing in sorted(bs.ROUTINGS)]
L0 = {("dynamic", "uneven"), ("compact_signal", "many_experts_64")}


def options_of(kind, problem, variant):
    return bs.variant_options(kind, problem, "compact_signal" if variant == "compact_signal" else "dynamic")


@pytest.mark.parametrize("variant, routing", bs.levels(CASES, L0, lambda variant, routing: f"{variant}-{routing}"))
def test_glu_generate_sfd_col_off_matches_row_outputs(variant, routing, require_sm100):
    problem = bs.make_problem("glu", bs.ROUTINGS[routing], seed=3)
    options = options_of("glu", problem, variant)
    want = run_sfd("glu", problem, **options)
    assert not torch.isnan(want["c"].float()).any()
    got = run_sfd("glu", problem, col=False, generate_sfd_col=False, **options)
    assert_equal_bytes(got, want, f"generate_sfd_col=False {variant} {routing}")


MB_CASES = [(kind, variant) for kind in ("glu", "dglu") for variant in ("signal", "compact_signal")]
MB_L0 = {("glu", "signal"), ("glu", "compact_signal"), ("dglu", "signal")}


def run_rubin_bf16(kind, problem, multi_batch, **options):
    """SM107: the Rubin twin's BF16 outputs.  Its constructor takes no multi-batch argument (the shared scheduler's
    default, on, applies), so off is set on the built kernel before its compile, under its own compile key."""
    kernel = bs.make_kernel(kind, len(problem["layout"]) - 1, **options)
    if not multi_batch:
        kernel.streaming_signal_multi_batch_opening = False
        kernel.test_compile_key = (*kernel.test_compile_key, "multi_batch_off")
    return bs.Launcher(kernel, problem)(problem)[0]


@pytest.mark.parametrize("kind, variant", bs.levels(MB_CASES, MB_L0, lambda kind, variant: f"{bs.KIND_IDS[kind]}-{variant}"))
def test_signal_multi_batch_opening_off_matches(kind, variant):
    """The batch opener changes how many batches one opening publishes, never the tile math: with
    streaming_signal_multi_batch_opening off and on (the default) over 64 experts (several lane rounds) the outputs are
    byte-equal.  SM100: the E4M3 D, both SFD planes and C (GLU) or the dprob slots (dGLU).  SM107 (the Rubin twins, stock
    SF atoms): the BF16 C and D (GLU) or D and the atomic dprob (dGLU)."""
    problem = bs.make_problem(kind, bs.ROUTINGS["many_experts_64"], seed=6)
    options = bs.variant_options(kind, problem, variant)
    label = f"streaming_signal_multi_batch_opening off vs on, {kind} {variant}"
    if bs.is_rubin():
        want = run_rubin_bf16(kind, problem, False, **options)
        bs.assert_same_bytes(run_rubin_bf16(kind, problem, True, **options), want, label)
        return
    want = run_sfd(kind, problem, streaming_signal_multi_batch_opening=False, **options)
    got = run_sfd(kind, problem, **options)
    assert_equal_bytes(got, want, label)

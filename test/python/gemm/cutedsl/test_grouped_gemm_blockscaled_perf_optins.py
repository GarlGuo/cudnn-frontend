# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The MXFP8 GLU kernel without its column outputs, and the input signal scheduler's batch openings with many small
experts.

- d_col and sfd_col None (SM100 GLU): the row-quantized D and row SFD only, byte-equal to the kernel that also writes
  the column pair; the column pair is not written, and C is staged once when that buys another A/B stage.  On SM100 the
  outputs are E4M3 D with its SFD (the EP layer's FC1 and dGLU outputs), which the streaming signal suite (BF16 D) does
  not cover.
- One batch opening of the input signal scheduler publishes the chosen batch and every other lane's complete ready run.
  With 64 small experts and every M tile ready, most lanes hold one, so each opening publishes many batches; every
  kernel on that scheduler must still give the stock dynamic scheduler's bytes: the MXFP8 GLU and dGLU (E4M3 D and
  SFD on SM100; on SM107 the Rubin twins with BF16 D, their dGLU's atomic dprob within FP32 reassociation), the BF16
  dGLU, and the BF16 GLU and plain grouped GEMM.
"""

import pytest
import torch

from gemm.cutedsl import _streaming_signal as bf16
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
        assert torch.equal(a, b), f"{label}: {name} differs"


#: (scheduler variant, routing): the stock dynamic scheduler, and the signal scheduler with compact SFA.
VARIANTS = ("dynamic", "compact_signal")
CASES = [(variant, routing) for variant in VARIANTS for routing in sorted(bs.ROUTINGS)]
L0 = {("dynamic", "uneven"), ("compact_signal", "many_experts_64")}


def options_of(kind, problem, variant):
    return bs.variant_options(kind, problem, "compact_signal" if variant == "compact_signal" else "dynamic")


@pytest.mark.parametrize("variant, routing", bs.levels(CASES, L0, lambda variant, routing: f"{variant}-{routing}"))
def test_glu_without_column_outputs_matches_row_outputs(variant, routing, require_sm100):
    problem = bs.make_problem("glu", bs.ROUTINGS[routing], seed=3)
    options = options_of("glu", problem, variant)
    want = run_sfd("glu", problem, **options)
    assert not torch.isnan(want["c"].float()).any()
    got = run_sfd("glu", problem, col=False, **options)
    assert_equal_bytes(got, want, f"no column outputs {variant} {routing}")


#: 64 small experts of 0 to 3 M tiles: with every M tile ready, most lanes of a batch opening hold a complete run.
SMALL_EXPERTS_64 = [(0, 1, 256, 300, 512, 600, 768)[i % 7] for i in range(64)]
#: (kind, variant): every kernel on the input signal scheduler; the MXFP8 kernels also with compact SFA.
SIGNAL_CASES = [
    ("glu", "signal"),
    ("glu", "compact_signal"),
    ("dglu", "signal"),
    ("dglu", "compact_signal"),
    ("dglu_bf16", "signal"),
    ("glu_bf16", "signal"),
    ("plain_bf16", "signal"),
]
SIGNAL_L0 = {case for case in SIGNAL_CASES if case[1] == "signal"}
#: Parameter ids of the BF16 GLU and plain grouped GEMM (the other kinds: bs.KIND_IDS).
BF16_IDS = {"glu_bf16": "fprop-bf16-glu", "plain_bf16": "fprop-bf16-plain"}


def run_bf16_grouped_gemm(glu, tokens, signal):
    """The BF16 GLU or plain grouped GEMM on the stock dynamic scheduler or the signal scheduler (every M tile ready);
    returns C and D over the experts' rows."""
    from gemm.cutedsl.test_grouped_gemm_streaming_signal import FFN, HIDDEN, run

    problem = bf16.make_problem(tokens, 2 * FFN if glu else FFN, HIDDEN, seed=6)
    options = dict(use_dynamic_sched=True)
    if signal:
        options["streaming_signal_capacity"] = problem["rows"] // bf16.ROWS_ALIGNMENT
    kernel = bf16.make_kernel(glu, len(tokens), **options)
    c, d, _, _ = run(kernel, problem, glu, prepare_workspace=signal_ready(kernel) if signal else None)
    torch.cuda.synchronize()
    rows = int(problem["offsets"][-1].item())
    return {"c": c[:rows], "d": d[:rows]}


@pytest.mark.parametrize("kind, variant", bs.levels(SIGNAL_CASES, SIGNAL_L0, lambda kind, variant: f"{BF16_IDS.get(kind) or bs.KIND_IDS[kind]}-{variant}"))
def test_signal_matches_dynamic_many_small_experts(kind, variant):
    """Batch openings that publish many batches change the tile order, never the tile math: the outputs equal the stock
    dynamic scheduler's bytes.  SM100 MXFP8: the E4M3 D, both SFD planes and C (GLU) or the dprob slots (dGLU)."""
    label = f"signal vs stock dynamic, 64 small experts, {kind} {variant}"
    if kind in BF16_IDS:
        glu = kind == "glu_bf16"
        want = run_bf16_grouped_gemm(glu, SMALL_EXPERTS_64, signal=False)
        got = run_bf16_grouped_gemm(glu, SMALL_EXPERTS_64, signal=True)
        bs.assert_same_bytes(got, want, label)
        return
    problem = bs.make_problem(kind, SMALL_EXPERTS_64, seed=6)
    stock_options, options = (bs.variant_options(kind, problem, name) for name in ("dynamic", variant))
    if kind == "dglu_bf16" or bs.is_rubin():
        experts = len(SMALL_EXPERTS_64)
        want, _ = bs.Launcher(bs.make_kernel(kind, experts, **stock_options), problem)(problem)
        got, _ = bs.Launcher(bs.make_kernel(kind, experts, **options), problem)(problem)
        torch.cuda.synchronize()
        bs.assert_same_bytes(got, want, label)
        return
    assert_equal_bytes(run_sfd(kind, problem, **options), run_sfd(kind, problem, **stock_options), label)

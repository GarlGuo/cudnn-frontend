#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reproduce the SM100 dSReLU grouped-GEMM TMEM accumulator bug in one run.

The epilogue publishes the overlapping accumulator stage as empty
(acc_pipeline.consumer_release) and only then issues the tcgen05.ld that reads it, so
the MMA producer can overwrite those TMEM columns mid-read.

This runs the kernel once and compares it against cudnn-frontend's own fp32 reference.
Run it on an UNPATCHED tree; on a patched tree it will say so and exit.

    CUDA_VISIBLE_DEVICES=0 python repro_simple.py

Needs one SM100 GPU and `pip install -e ".[cutedsl]"`.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
KERNEL = "cudnn.gemm.cutedsl.grouped.dsrelu.moe_blockscaled_grouped_gemm_dsrelu_quant"

# Big enough that each SM gets several tiles. That matters: the accumulator is only
# reused -- and so only at risk -- when a CTA moves on to a second tile.
EXPERTS, ROWS_PER_EXPERT, N, K = 16, 2048, 4096, 512

STALL_NS = 100_000

# Appended to the kernel module; see load_kernel() below.
STALL_HELPER = """

import cutlass as _c
from cutlass.cutlass_dsl import dsl_user_op as _op
from cutlass._mlir.dialects import llvm as _llvm


@_op
def _stall(*, loc=None, ip=None):
    _llvm.inline_asm(None, [_c.Int32({ns}).ir_value(loc=loc, ip=ip)],
                     "nanosleep.u32 $0;", "r", has_side_effects=True,
                     is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT)
"""


def load_kernel(stall_ns):
    """Load the dSReLU kernel with a stall inserted just after the accumulator release.

    The window between "stage published empty" and "stage actually read" is only
    nanoseconds wide, so without a stall the race almost never fires. The stall does not
    create the bug -- it only widens the interval in which the released TMEM columns can
    be overwritten before the epilogue reads them.

    CuTeDSL re-reads each kernel's source text at compile time (inspect.getsourcelines),
    so the patched text has to be registered in linecache under a filename it will look
    up -- editing the string in memory alone would have no effect.
    """
    import importlib, importlib.util, linecache

    original = importlib.import_module(KERNEL)
    source = Path(original.__file__).read_text()

    anchor = " " * 28 + "acc_consumer_state.advance()\n"
    if source.count(anchor) != 1:
        sys.exit("could not find the accumulator release site; kernel layout changed")
    source = source.replace(anchor, anchor + " " * 28 + "_stall()\n")
    source += STALL_HELPER.format(ns=stall_ns)

    fake = str(Path(original.__file__).with_name("kernel_with_stall.py"))
    linecache.cache[fake] = (len(source), None, source.splitlines(keepends=True), fake)
    module = importlib.util.module_from_spec(importlib.util.spec_from_file_location("kernel_with_stall", fake))
    module.__package__, module.__file__ = original.__package__, fake
    sys.modules["kernel_with_stall"] = module
    exec(compile(source, fake, "exec"), module.__dict__)
    return module.BlockScaledMoEGroupedGemmQuantBwdKernel


def allow_import_without_building():
    """Let `import cudnn.gemm...` work from a plain checkout, with no C++ extension built.

    Only cudnn/__init__.py and cudnn/datatypes.py need the compiled backend; the CuTeDSL
    kernels themselves do not. Skipped entirely if cudnn is properly installed.
    """
    import importlib.util, types

    if importlib.util.find_spec("cudnn") is not None:
        return
    backend = types.ModuleType("cudnn._compiled_module")
    backend.data_type = type("_dt", (), {"__getattr__": lambda self, name: name})()
    package = types.ModuleType("cudnn")
    package.__path__ = [str(REPO / "python" / "cudnn")]
    sys.modules.update({"cudnn": package, "cudnn._compiled_module": backend})
    fp4 = types.ModuleType("test_low_precision_matmul")  # eagerly imported by the test utils
    fp4._bfloat16_to_float4_e2m1fn_x2 = fp4.float4_e2m1fn_x2_to_float32 = None
    sys.modules["test_low_precision_matmul"] = fp4


def main():
    import torch

    sys.path.insert(0, str(REPO / "test" / "python"))
    allow_import_without_building()
    from cudnn.gemm.cutedsl.grouped.dsrelu.api import GroupedGemmDsreluSm100
    from cuda.bindings import driver as cuda
    from fe_api.grouped_gemm.test_grouped_gemm_dsrelu_utils import (
        allocate_grouped_gemm_dsrelu_tensors,
        allocate_grouped_gemm_input_tensors,
        run_grouped_gemm_dsrelu_ref,
    )

    if torch.cuda.get_device_capability()[0] < 10:
        sys.exit("needs an SM100+ GPU")

    # Is this tree still unpatched? The bug is the release sitting ABOVE the copy.
    import importlib

    src = Path(importlib.import_module(KERNEL).__file__).read_text()
    release = src.index("acc_pipeline.consumer_release(acc_consumer_state)")
    copy = src.index("cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)")
    if copy < release:
        sys.exit("this tree already has the fix (copy precedes release) -- nothing to reproduce")

    # ---- inputs -------------------------------------------------------------------
    # Small integers, and the block scales are powers of two, so every partial sum of
    # the k=512 reduction is an exact fp32 integer. That makes the reference EXACT:
    # summation order cannot change the result, so any mismatch below is a real error
    # and not floating-point rounding.
    fp8, e8m0 = torch.float8_e4m3fn, torch.float8_e8m0fnu
    torch.manual_seed(0)
    inp = allocate_grouped_gemm_input_tensors(
        n=N,
        k=K,
        l=EXPERTS,
        group_m_list=[ROWS_PER_EXPERT] * EXPERTS,
        ab_dtype=fp8,
        b_major="k",
        sf_dtype=e8m0,
        sf_vec_size=32,
        m_aligned=256,
    )
    inp, out = allocate_grouped_gemm_dsrelu_tensors(
        tensor_m=inp["tensor_m"],
        n=N,
        l=EXPERTS,
        ab_dtype=fp8,
        c_dtype=torch.bfloat16,
        d_dtype=fp8,
        cd_major="n",
        sf_dtype=e8m0,
        sf_vec_size=32,
        input_tensors=inp,
    )
    for ref, tensor in (("a_ref", "a_tensor"), ("b_ref", "b_tensor")):
        inp[tensor].copy_(torch.randint(-2, 3, inp[tensor].shape, device="cuda", dtype=torch.float32))
        inp[ref].copy_(inp[tensor].to(torch.float32))

    # ---- run the kernel -----------------------------------------------------------
    api = GroupedGemmDsreluSm100(
        sample_a=inp["a_tensor"],
        sample_b=inp["b_tensor"],
        sample_c=inp["c_tensor"],
        sample_d_row=out["d_row_tensor"],
        sample_d_col=out["d_col_tensor"],
        sample_d_srelu=out["d_srelu_tensor"],
        sample_sfa=inp["sfa_tensor"],
        sample_sfb=inp["sfb_tensor"],
        sample_padded_offsets=inp["padded_offsets_tensor"],
        sample_alpha=inp["alpha_tensor"],
        sample_prob=inp["prob_tensor"],
        sample_dprob=out["dprob_tensor"],
        sample_sfd_row=out["sfd_row_tensor"],
        sample_sfd_col=out["sfd_col_tensor"],
        sample_sfd_col_d_srelu=out["sfd_col_d_srelu_tensor"],
        sample_norm_const=inp["norm_const_tensor"],
        acc_dtype=torch.float32,
        mma_tiler_mn=(256, 256),
        cluster_shape_mn=(2, 1),
        sf_vec_size=32,
        vector_f32=False,
        m_aligned=256,
        discrete_col_sfd=True,
    )
    api._kernel = load_kernel(STALL_NS)
    api.compile()
    out["dprob_tensor"].zero_()
    api.execute(
        a_tensor=inp["a_tensor"],
        b_tensor=inp["b_tensor"],
        c_tensor=inp["c_tensor"],
        d_row_tensor=out["d_row_tensor"],
        d_col_tensor=out["d_col_tensor"],
        d_srelu_tensor=out["d_srelu_tensor"],
        sfa_tensor=inp["sfa_tensor"],
        sfb_tensor=inp["sfb_tensor"],
        padded_offsets=inp["padded_offsets_tensor"],
        alpha_tensor=inp["alpha_tensor"],
        prob_tensor=inp["prob_tensor"],
        dprob_tensor=out["dprob_tensor"],
        sfd_row_tensor=out["sfd_row_tensor"],
        sfd_col_tensor=out["sfd_col_tensor"],
        sfd_col_d_srelu_tensor=out["sfd_col_d_srelu_tensor"],
        norm_const_tensor=inp["norm_const_tensor"],
        amax_tensor=None,
        current_stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    torch.cuda.synchronize()

    # ---- compare against the fp32 reference ---------------------------------------
    ref = run_grouped_gemm_dsrelu_ref(
        a_ref=inp["a_ref"],
        b_ref=inp["b_ref"],
        c_ref=inp["c_ref"],
        sfa_ref=inp["sfa_ref"],
        sfb_ref=inp["sfb_ref"],
        alpha_tensor=inp["alpha_tensor"],
        prob_tensor=inp["prob_tensor"],
        aligned_group_m_list=inp["aligned_group_m_list"],
        valid_m=inp["valid_m"],
        generate_sfd=True,
        norm_const_tensor=inp["norm_const_tensor"],
        c_dtype=torch.bfloat16,
        d_dtype=fp8,
        sf_vec_size=32,
        sf_dtype=e8m0,
    )

    print(f"\nshape: {EXPERTS} experts x {ROWS_PER_EXPERT} rows, N={N}, K={K}   stall={STALL_NS} ns\n")
    print(f"  {'output':<24} {'mismatches':>12} {'of':>12}   {'max |error|':>12}")
    worst = 0
    for name, reference in (("d_row_tensor", "d_ref"), ("d_col_tensor", "d_col_ref"), ("d_srelu_tensor", "d_srelu_col_ref")):
        got = out[name].float()
        if name != "d_row_tensor":
            got = got.permute(1, 0, 2)
        want = ref[reference].float()
        bad = int((got != want).sum().item())
        note = "  <- computed without the accumulator; should stay clean" if "srelu" in name else ""
        print(f"  {name:<24} {bad:>12} {got.numel():>12}   {(got - want).abs().max().item():>12.4g}{note}")
        if "srelu" not in name:
            worst = max(worst, bad)

    if worst:
        print(f"\nBUG REPRODUCED: {worst} elements differ from the fp32 reference.")
        print("The accumulator-derived outputs are wrong; the accumulator-independent one is not.")
    else:
        print("\nNot reproduced this run. The window is narrow -- try a larger --stall or rerun.")
    return 1 if worst else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Can the TMEM accumulator race be observed WITHOUT injecting a stall?

repro_simple.py widens the invalid window with a nanosleep. This asks whether the
window can be hit on its own, by shrinking the epilogue's head start using only
legitimate configuration, and by taking far more samples.

The race needs the MMA producer to observe the accumulator-empty mbarrier, clear its
AB wait, issue tcgen05.mma and have it land -- all before the epilogue's tcgen05.ld
completes. Levers, none of which modify the kernel:

  cluster (1,1)   producer and consumer share a CTA, so the arrive does not have to
                  propagate through mapa.shared::cluster to another CTA
  small K         fewer k-tiles, so the AB pipeline is more likely already full and
                  the producer's consumer_wait after acquire is a no-op
  positive C      d = 2*relu(C)*acc, so C<0 masks corruption entirely; all-positive C
                  makes every corrupted element visible
  many runs       each execution gives (#tiles) independent handoffs

Run on an UNPATCHED tree. Reports the first hit, or an upper bound on the per-handoff
probability if none is seen.

    CUDA_VISIBLE_DEVICES=0 python zero_delay_hunt.py [--iters 200]
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
KERNEL = "cudnn.gemm.cutedsl.grouped.dsrelu.moe_blockscaled_grouped_gemm_dsrelu_quant"

# (label, mma_tiler_mn, cluster_shape_mn, K, experts, rows_per_expert, N)
CONFIGS = [
    ("baseline  cluster(2,1) K=512", (256, 256), (2, 1), 512, 16, 2048, 4096),
    ("cta-local cluster(1,1) K=512", (128, 256), (1, 1), 512, 16, 2048, 4096),
    ("cta-local cluster(1,1) K=128", (128, 256), (1, 1), 128, 16, 2048, 4096),
    ("baseline  cluster(2,1) K=128", (256, 256), (2, 1), 128, 16, 2048, 4096),
]


def allow_import_without_building():
    import importlib.util, types

    if importlib.util.find_spec("cudnn") is not None:
        return
    backend = types.ModuleType("cudnn._compiled_module")
    backend.data_type = type("_dt", (), {"__getattr__": lambda self, name: name})()
    package = types.ModuleType("cudnn")
    package.__path__ = [str(REPO / "python" / "cudnn")]
    sys.modules.update({"cudnn": package, "cudnn._compiled_module": backend})
    fp4 = types.ModuleType("test_low_precision_matmul")
    fp4._bfloat16_to_float4_e2m1fn_x2 = fp4.float4_e2m1fn_x2_to_float32 = None
    sys.modules["test_low_precision_matmul"] = fp4


def build(experts, rows, n, k, mma, cluster):
    """Inputs chosen so the fp32 reference is EXACT and corruption is never masked."""
    import torch

    from cudnn.gemm.cutedsl.grouped.dsrelu.api import GroupedGemmDsreluSm100
    from fe_api.grouped_gemm.test_grouped_gemm_dsrelu_utils import (
        allocate_grouped_gemm_dsrelu_tensors,
        allocate_grouped_gemm_input_tensors,
    )

    fp8, e8m0 = torch.float8_e4m3fn, torch.float8_e8m0fnu
    torch.manual_seed(0)
    inp = allocate_grouped_gemm_input_tensors(
        n=n, k=k, l=experts, group_m_list=[rows] * experts, ab_dtype=fp8,
        b_major="k", sf_dtype=e8m0, sf_vec_size=32, m_aligned=256,
    )  # fmt: skip
    inp, out = allocate_grouped_gemm_dsrelu_tensors(
        tensor_m=inp["tensor_m"], n=n, l=experts, ab_dtype=fp8, c_dtype=torch.bfloat16,
        d_dtype=fp8, cd_major="n", sf_dtype=e8m0, sf_vec_size=32, input_tensors=inp,
    )  # fmt: skip

    # Integer operands -> every partial sum is an exact fp32 integer -> the reference is
    # bit-exact and reduction order cannot matter.
    for ref, tensor in (("a_ref", "a_tensor"), ("b_ref", "b_tensor")):
        inp[tensor].copy_(torch.randint(-2, 3, inp[tensor].shape, device="cuda", dtype=torch.float32))
        inp[ref].copy_(inp[tensor].to(torch.float32))
    # All-positive C: d = 2*relu(C)*acc, so C<0 would zero the output and hide any
    # corruption of the accumulator behind it.
    inp["c_tensor"].copy_(torch.randint(1, 3, inp["c_tensor"].shape, device="cuda", dtype=torch.float32))
    inp["c_ref"].copy_(inp["c_tensor"].to(torch.float32))

    api = GroupedGemmDsreluSm100(
        sample_a=inp["a_tensor"], sample_b=inp["b_tensor"], sample_c=inp["c_tensor"],
        sample_d_row=out["d_row_tensor"], sample_d_col=out["d_col_tensor"], sample_d_srelu=out["d_srelu_tensor"],
        sample_sfa=inp["sfa_tensor"], sample_sfb=inp["sfb_tensor"], sample_padded_offsets=inp["padded_offsets_tensor"],
        sample_alpha=inp["alpha_tensor"], sample_prob=inp["prob_tensor"], sample_dprob=out["dprob_tensor"],
        sample_sfd_row=out["sfd_row_tensor"], sample_sfd_col=out["sfd_col_tensor"],
        sample_sfd_col_d_srelu=out["sfd_col_d_srelu_tensor"], sample_norm_const=inp["norm_const_tensor"],
        acc_dtype=torch.float32, mma_tiler_mn=mma, cluster_shape_mn=cluster,
        sf_vec_size=32, vector_f32=False, m_aligned=256, discrete_col_sfd=True,
    )  # fmt: skip
    return api, inp, out


def run_once(api, inp, out):
    import torch
    from cuda.bindings import driver as cuda

    out["dprob_tensor"].zero_()
    api.execute(
        a_tensor=inp["a_tensor"], b_tensor=inp["b_tensor"], c_tensor=inp["c_tensor"],
        d_row_tensor=out["d_row_tensor"], d_col_tensor=out["d_col_tensor"], d_srelu_tensor=out["d_srelu_tensor"],
        sfa_tensor=inp["sfa_tensor"], sfb_tensor=inp["sfb_tensor"], padded_offsets=inp["padded_offsets_tensor"],
        alpha_tensor=inp["alpha_tensor"], prob_tensor=inp["prob_tensor"], dprob_tensor=out["dprob_tensor"],
        sfd_row_tensor=out["sfd_row_tensor"], sfd_col_tensor=out["sfd_col_tensor"],
        sfd_col_d_srelu_tensor=out["sfd_col_d_srelu_tensor"], norm_const_tensor=inp["norm_const_tensor"],
        amax_tensor=None, current_stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )  # fmt: skip
    torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=200, help="executions per config")
    args = ap.parse_args()

    import torch

    sys.path.insert(0, str(REPO / "test" / "python"))
    allow_import_without_building()
    import importlib

    from fe_api.grouped_gemm.test_grouped_gemm_dsrelu_utils import run_grouped_gemm_dsrelu_ref

    src = Path(importlib.import_module(KERNEL).__file__).read_text()
    if src.index("cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)") < src.index("acc_pipeline.consumer_release(acc_consumer_state)"):
        sys.exit("this tree has the fix; run on an unpatched tree")

    total_handoffs = 0
    for label, mma, cluster, k, experts, rows, n in CONFIGS:
        try:
            api, inp, out = build(experts, rows, n, k, mma, cluster)
            if not api.check_support():
                print(f"  {label}   unsupported, skipped")
                continue
            api.compile()
        except Exception as exc:  # config the API refuses
            print(f"  {label}   unsupported ({type(exc).__name__}: {str(exc)[:60]}), skipped")
            continue

        ref = run_grouped_gemm_dsrelu_ref(
            a_ref=inp["a_ref"], b_ref=inp["b_ref"], c_ref=inp["c_ref"], sfa_ref=inp["sfa_ref"],
            sfb_ref=inp["sfb_ref"], alpha_tensor=inp["alpha_tensor"], prob_tensor=inp["prob_tensor"],
            aligned_group_m_list=inp["aligned_group_m_list"], valid_m=inp["valid_m"], generate_sfd=True,
            norm_const_tensor=inp["norm_const_tensor"], c_dtype=torch.bfloat16,
            d_dtype=torch.float8_e4m3fn, sf_vec_size=32, sf_dtype=torch.float8_e8m0fnu,
        )  # fmt: skip
        want = ref["d_ref"].float()
        tiles = experts * -(-rows // mma[0]) * -(-n // mma[1])

        hits = 0
        for i in range(args.iters):
            run_once(api, inp, out)
            if bool((out["d_row_tensor"].float() != want).any().item()):
                hits += 1
                if hits == 1:
                    bad = int((out["d_row_tensor"].float() != want).sum().item())
                    print(f"  {label}   *** HIT on iteration {i}: {bad} elements differ ***")
        handoffs = tiles * args.iters
        total_handoffs += handoffs
        rate = f"{hits}/{args.iters} runs" if hits else f"0/{args.iters} runs"
        print(f"  {label}   tiles={tiles:<5} handoffs={handoffs:>9}   {rate}")
        del api

    print(f"\n{total_handoffs} accumulator handoffs sampled with no injected stall.")
    print(f"If none fired, the per-handoff probability is below ~{3.0 / max(total_handoffs, 1):.2e} (95% CI).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

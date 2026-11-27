#!/usr/bin/env python3
"""Lock-step A/B of the BF16 grouped wgrad, stock (MoEGroupedGemmWgradBF16Kernel) vs wide (MoEGroupedGemmWgradWideBF16Kernel),
on a list of GEMMs given directly as m:n:K:E (dW is E x m x n, K rows per expert, balanced), the same list on every rank.

torchrun --nproc-per-node N rule_ls.py --gemms m:n:K:E,m:n:K:E,... --out FILE [--budget S] [--reps R] [--claim clc|counter]
         [--extra wide100,stockT] [--reverse] [--no-cases] [--no-api-check]
Env: FE_PYTHON = the cuDNN-FE python/ root; GATE_MS (idle gate, default 20).

Arms per GEMM:
  stock  : the stock kernel as the tree builds it (the API's non-wide kernel)
  wide   : the wide kernel as the tree builds it
  stock2 : the stock compiled kernel again on its own dW buffer (A/A timing control)
  extras : wide100 = wide forced to 232448 B smem; stockT = stock forced to the target's smem capacity
Checks before timing: every arm's dW bytes vs stock (balanced offsets and, unless --no-cases, zero_token / ragged /
all_empty / partial_tail_then_data); API: grouped_gemm_wgrad_wrapper_sm100 (dense, library defaults) records which kernel
it picked for this GEMM and its dW bytes vs stock.
Timing: lockstep.run (all ranks fire the same arm after a drain + barrier + GATE_MS idle, 512 MB flush, one launch timed).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lockstep  # noqa: E402

lockstep.pin_rank_gpu()
from wgrad_ls import install_cudnn  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gemms", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=float, default=600)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--claim", choices=("clc", "counter"), default="clc")
    ap.add_argument("--extra", default="")
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--no-cases", action="store_true")
    ap.add_argument("--no-api-check", action="store_true")
    args = ap.parse_args()
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    gemms = [tuple(int(x) for x in g.split(":")) for g in args.gemms.split(",") if g]
    if args.reverse:
        gemms = gemms[::-1]
    t_start = time.time()
    install_cudnn(os.environ["FE_PYTHON"])
    import torch
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as cutlass_utils
    from cutlass.cute.runtime import from_dlpack
    from cutlass.utils import HardwareInfo
    import cuda.bindings.driver as cuda
    from cudnn.gemm.cutedsl.grouped.moe_utils import MoEWeightMode, WGradInputOrder
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_grouped_gemm_wgrad import MoEGroupedGemmWgradBF16Kernel as Stock
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_grouped_gemm_wgrad_wide import MoEGroupedGemmWgradWideBF16Kernel as Wide

    out = open(f"{args.out}.r{rank}", "a")
    host = os.uname().nodename
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")

    def emit(rec):
        rec.update(claim=args.claim, rank=rank, host=host, gpu=gpu, tray=os.environ.get("SCHED_TRAY", ""),
                   entry=os.environ.get("POOL_ENTRY", ""), pid=os.getpid())
        out.write(json.dumps(rec) + "\n")
        out.flush()

    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(1)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    clusters = HardwareInfo().get_max_active_clusters(2)
    flush = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=dev)
    counter = args.claim == "counter"
    try:
        target_smem = cutlass_utils.get_smem_capacity_in_bytes()
    except Exception as ex:  # noqa: BLE001
        target_smem = repr(ex)
    props = torch.cuda.get_device_properties(0)
    emit(dict(kind="meta", clusters=clusters, sms=props.multi_processor_count, cc=list(torch.cuda.get_device_capability()),
              gpu_name=torch.cuda.get_device_name(), target_smem=target_smem, sha=os.environ.get("FE_SHA"),
              gemms=[list(x) for x in gemms]))

    def t(x, align=16):
        return from_dlpack(x, assumed_align=align)

    extras = [x for x in args.extra.split(",") if x]
    arms, outs, pairs = {}, {}, []
    for (m, n, per, el) in gemms:
        gname = f"{m}x{n}k{per}e{el}"
        rows = el * per
        tw = el * (m // 256) * (n // 512)
        a = torch.randn(rows, m, generator=g, device=dev).to(torch.bfloat16).t()  # (m, tokens), M-major
        b = torch.randn(rows, n, generator=g, device=dev).to(torch.bfloat16)  # (tokens, n), N-major
        balanced = (torch.arange(1, el + 1, device=dev, dtype=torch.int32) * per).contiguous()
        offsets = balanced.clone()
        compiled = {}
        names = ["stock", "wide", "stock2"] + extras
        for arm in names:
            o = torch.full((el, m, n), float("nan"), device=dev, dtype=torch.bfloat16)
            outs[f"{gname}_{arm}"] = o
            if arm == "stock2":
                fn, ws = compiled["stock"]
            else:
                if arm.startswith("stock"):
                    kernel = Stock(acc_dtype=cutlass.Float32, use_2cta_instrs=True, mma_tiler_mn=(256, 256), cluster_shape_mn=(2, 1),
                                   expert_cnt=el, weight_mode=MoEWeightMode.DENSE, input_order=WGradInputOrder.Tensor2D,
                                   atomic_counter_claim=counter)
                else:
                    kernel = Wide(expert_cnt=el, atomic_counter_claim=counter)
                if arm == "wide100":
                    kernel.smem_capacity = 232448
                elif arm == "stockT":
                    kernel.smem_capacity = cutlass_utils.get_smem_capacity_in_bytes()
                ws = torch.zeros(max(kernel.get_workspace_bytes(), 256), device=dev, dtype=torch.uint8)
                t0 = time.time()
                try:
                    fn = cute.compile(kernel, t(a), t(b), t(o), t(offsets, 4), t(ws, 128), clusters, stream, None)
                except Exception:  # noqa: BLE001
                    import traceback

                    emit(dict(kind="build_error", gemm=gname, arm=arm, error=traceback.format_exc()[-1500:]))
                    raise
                compiled[arm] = (fn, ws)
                emit(dict(kind="compile", gemm=gname, arm=arm, s=round(time.time() - t0, 1), smem_capacity=kernel.smem_capacity,
                          num_ab_stage=getattr(kernel, "num_ab_stage", None)))
            kargs = [t(a), t(b), t(o), t(offsets, 4), t(ws, 128)]

            def launch(fn=fn, kargs=kargs):
                fn(*kargs, stream, None)

            arms[f"{gname}_{arm}"] = (lambda: None, [launch])

        def check(gname=gname, names=names):
            for x in names:
                outs[f"{gname}_{x}"].fill_(float("nan"))
            for x in names:
                arms[f"{gname}_{x}"][1][0]()
            torch.cuda.synchronize()
            ref = outs[f"{gname}_stock"].view(torch.uint16)
            rec = dict(gemm=gname, base_nan=int(torch.isnan(outs[f"{gname}_stock"]).sum().item()))
            for x in names[1:]:
                rec[f"{x}_diff_bytes"] = int((outs[f"{gname}_{x}"].view(torch.uint16) != ref).sum().item())
            return rec

        emit(dict(kind="check", m=m, n=n, K=per, E=el, wide_tiles=tw, waves=tw / clusters, **check()))
        pairs += [(f"{gname}_{x}", f"{gname}_stock") for x in names[1:]]
        if not args.no_cases:
            ragged = [per * 2, 0, per // 2, 0, 128, per + 384, 0, 256]
            cases = {
                "zero_token": [0] * (el // 2) + [per * 2] * (el // 4) + [0] * (el - el // 2 - el // 4),
                "ragged": (ragged * (-(-el // len(ragged))))[:el],
                "all_empty": [0] * el,
                "partial_tail_then_data": [per] * (el - 1) + [per - 64],
            }
            for cname, groups in cases.items():
                if sum(groups) > rows:
                    continue
                offsets.copy_(torch.tensor(groups, dtype=torch.int32, device=dev).cumsum(0, dtype=torch.int32))
                emit(dict(kind="case", case=cname, **check()))
            offsets.copy_(balanced)
            emit(dict(kind="check_after_cases", **check()))
        if not args.no_api_check:
            try:
                import cudnn.gemm.cutedsl.grouped.wgrad.api as wapi
                from cudnn.gemm.cutedsl.grouped.wgrad import _bf16_api

                api_cls = _bf16_api.GroupedGemmWgradBf16API
                original = api_cls._wide_applies
                picked = []

                def recorded(self):
                    r = original(self)
                    picked.append(r)
                    return r

                api_cls._wide_applies = recorded
                try:
                    wapi._wgrad_wrapper_memo.clear()
                    wapi._cache_of_GroupedGemmWgradSm100Objects.clear()
                    res = wapi.grouped_gemm_wgrad_wrapper_sm100(
                        a_tensor=a, b_tensor=b, sfa_tensor=None, sfb_tensor=None, offsets_tensor=offsets, output_mode="dense",
                        acc_dtype=torch.float32, wgrad_dtype=torch.bfloat16, mma_tiler_mn=(256, 256), cluster_shape_mn=(2, 1),
                        atomic_counter_claim=counter)
                    torch.cuda.synchronize()
                    w = res["wgrad_tensor"]
                    emit(dict(kind="api", gemm=gname, picked_wide=picked[-1:],
                              diff_bytes=int((w.view(torch.uint16) != outs[f"{gname}_stock"].view(torch.uint16)).sum().item())))
                    del res, w
                finally:
                    api_cls._wide_applies = original
                    wapi._wgrad_wrapper_memo.clear()
                    wapi._cache_of_GroupedGemmWgradSm100Objects.clear()
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                import traceback

                emit(dict(kind="api_error", gemm=gname, error=traceback.format_exc()[-1500:]))

    if os.environ.get("LS_NO_TIME") == "1":
        emit(dict(kind="done", passes=0, s=round(time.time() - t_start, 1)))
        return
    lockstep.run(arms, pairs, emit, args.budget, args.reps, flush, t_start)


if __name__ == "__main__":
    main()

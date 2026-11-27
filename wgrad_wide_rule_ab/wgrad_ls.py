#!/usr/bin/env python3
"""Lock-step kernel A/B of the BF16 grouped wgrad: stock (MoEGroupedGemmWgradBF16Kernel) vs wide
(MoEGroupedGemmWgradWideBF16Kernel), both sized for the compile target's shared memory (sm_107a: 334848 B).

torchrun --nproc-per-node N wgrad_ls.py --shapes m:ep:t,m:ep:t,... --out FILE [--budget S] [--reps R] [--claim clc|counter]
         [--extra stock100,wide100] [--no-cases] [--no-api-check]
Rank r takes shape r of --shapes (one per rank). Env: FE_PYTHON = the cuDNN-FE python/ root (branch
wip/wgrad-wide-bf16-sm107); GATE_MS (lockstep idle gate, default 20).

Per rank, both GEMMs of its shape (dW1: m = 2I, n = H; dW2: m = H, n = I), each with the arms
  stock  : stock kernel (the BF16 API's default kernel)
  wide   : wide kernel (what the API runs with GroupedGemmWgradBf16API._use_wide_tiles = True)
  stock2 : stock again, its own output buffer (A/A control)
  [stock100 / wide100 with --extra: the same kernels with smem_capacity forced to sm_100's 232448 B]
Operands as in the GB200 A/B (wgrad_opts_results.md): A = dH^T / dY^T view of row-major (tokens, m) BF16 (M-major),
B = (tokens, n) row-major (N-major), dW BF16 (experts, m, n), 8192 tokens/rank, routed rows spread evenly over the
local experts, each padded to 256 (per = ceil(T * top_k * EP / E / 256) * 256).

Checks (before timing; emitted as kind=check / case / api):
  * every arm's dW bytes vs stock's on the balanced offsets;
  * cases (same compiled kernels, offsets rewritten in place, operands hold nonzero data past offs[-1]):
    zero_token, ragged, all_empty, partial_tail_then_data;
  * API: grouped_gemm_wgrad_wrapper_sm100 (dense) with _use_wide_tiles monkeypatched True, then False; records which
    kernel _wide_applies() picked and the dW bytes vs the stock kernel's.
Timing: lockstep.run: per sample every rank drains, barriers, sleeps GATE_MS, then times one launch of the same arm
(events); a 512 MB L2 flush before each sample. Per pass the arms run forward then backward from a rotating start,
R samples each way; one JSON line per pass with per-arm medians and the ratios wide/stock, stock2/stock (and extras).
"""
import argparse
import json
import os
import sys
import time
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lockstep  # noqa: E402

lockstep.pin_rank_gpu()

MODELS = {
    "deepseek_v3": dict(hidden=7168, experts=256, top_k=8, intermediate=2048),
    "qwen35_122b": dict(hidden=3072, experts=256, top_k=8, intermediate=1024),
    "qwen38_2p4t": dict(hidden=8192, experts=512, top_k=10, intermediate=2048),
    "qwen35_397b": dict(hidden=4096, experts=512, top_k=10, intermediate=1024),
    "glm53_flash": dict(hidden=4096, experts=288, top_k=8, intermediate=2048),
}


def install_cudnn(python_root):
    """Import cudnn from python_root without its compiled module (the CuTe DSL kernels do not need it)."""

    class _AnyMeta(type):
        def __getattr__(cls, name):
            if name.startswith("__"):
                raise AttributeError(name)
            return f"{cls.__name__}.{name}"

    class _Anything(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            return _AnyMeta(name, (), {})

    package = types.ModuleType("cudnn")
    package.__path__ = [os.path.join(python_root, "cudnn")]
    sys.modules["cudnn"] = package
    for name in ("_compiled_module", "_pybind_module"):
        stub = _Anything("cudnn." + name)
        sys.modules["cudnn." + name] = stub
        setattr(package, name, stub)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=float, default=600)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--claim", choices=("clc", "counter"), default="clc")
    ap.add_argument("--extra", default="")
    ap.add_argument("--no-cases", action="store_true")
    ap.add_argument("--no-api-check", action="store_true")
    args = ap.parse_args()
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    shapes = args.shapes.split(",")
    model, ep, tokens = shapes[rank % len(shapes)].split(":")
    ep, tokens = int(ep), int(tokens)
    t_start = time.time()
    install_cudnn(os.environ["FE_PYTHON"])
    import torch
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from cutlass.utils import HardwareInfo
    import cuda.bindings.driver as cuda
    from cudnn.gemm.cutedsl.grouped.moe_utils import MoEWeightMode, WGradInputOrder
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_grouped_gemm_wgrad import MoEGroupedGemmWgradBF16Kernel as Stock
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_grouped_gemm_wgrad_wide import MoEGroupedGemmWgradWideBF16Kernel as Wide

    out = open(f"{args.out}.r{rank}", "a")

    def emit(rec):
        rec.update(model=model, ep=ep, tokens=tokens, claim=args.claim, rank=rank, host=os.uname().nodename,
                   gpu=os.environ.get("CUDA_VISIBLE_DEVICES", "?"))
        out.write(json.dumps(rec) + "\n")
        out.flush()

    md = MODELS[model]
    H, E, topk, I = md["hidden"], md["experts"], md["top_k"], md["intermediate"]
    el = E // ep
    per = -(-(tokens * topk * ep // E) // 256) * 256
    rows = el * per
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(1)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    clusters = HardwareInfo().get_max_active_clusters(2)
    flush = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=dev)
    counter = args.claim == "counter"
    import cutlass.utils as cutlass_utils

    try:
        target_smem = cutlass_utils.get_smem_capacity_in_bytes()
    except Exception as ex:  # noqa: BLE001
        target_smem = repr(ex)
    emit(dict(kind="shape", rows=rows, per_expert=per, experts=el, H=H, I=I, clusters=clusters,
              cc=list(torch.cuda.get_device_capability()), gpu_name=torch.cuda.get_device_name(), target_smem=target_smem,
              sha=os.environ.get("FE_SHA")))

    def t(x, align=16):
        return from_dlpack(x, assumed_align=align)

    balanced = (torch.arange(1, el + 1, device=dev, dtype=torch.int32) * per).contiguous()
    offsets = balanced.clone()
    arms, outs, pairs = {}, {}, []
    extras = [x for x in args.extra.split(",") if x]
    arm_defs = [("stock", Stock, None), ("wide", Wide, None), ("stock2", Stock, None)]
    arm_defs += [(x, Stock if x.startswith("stock") else Wide, 232448) for x in extras]
    operands = {}

    for gname, (m, n) in {"dW1": (2 * I, H), "dW2": (H, I)}.items():
        a = torch.randn(rows, m, generator=g, device=dev).to(torch.bfloat16).t()  # (m, tokens), M-major
        b = torch.randn(rows, n, generator=g, device=dev).to(torch.bfloat16)  # (tokens, n), N-major
        operands[gname] = (a, b, m, n)
        for arm, cls, smem in arm_defs:
            if cls is Stock:
                kernel = cls(acc_dtype=cutlass.Float32, use_2cta_instrs=True, mma_tiler_mn=(256, 256), cluster_shape_mn=(2, 1),
                             expert_cnt=el, weight_mode=MoEWeightMode.DENSE, input_order=WGradInputOrder.Tensor2D,
                             atomic_counter_claim=counter)
            else:
                kernel = cls(expert_cnt=el, atomic_counter_claim=counter)
            if smem is not None:
                kernel.smem_capacity = smem
            o = torch.full((el, m, n), float("nan"), device=dev, dtype=torch.bfloat16)
            ws = torch.zeros(max(kernel.get_workspace_bytes(), 256), device=dev, dtype=torch.uint8)
            kargs = [t(a), t(b), t(o), t(offsets, 4), t(ws, 128)]
            t0 = time.time()
            try:
                fn = cute.compile(kernel, *kargs, clusters, stream, None)
            except Exception:  # noqa: BLE001
                import traceback

                emit(dict(kind="build_error", arm=f"{gname}_{arm}", error=traceback.format_exc()[-1500:]))
                raise

            def launch(fn=fn, kargs=kargs):
                fn(*kargs, stream, None)

            arms[f"{gname}_{arm}"] = (lambda: None, [launch])
            outs[f"{gname}_{arm}"] = o
            emit(dict(kind="compile", arm=f"{gname}_{arm}", s=round(time.time() - t0, 1), smem_capacity=kernel.smem_capacity,
                      num_ab_stage=getattr(kernel, "num_ab_stage", None)))
        names = [x for x, _, _ in arm_defs]

        def check(tag, gname=gname, names=names):
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

        emit(dict(kind="check", m=m, n=n, **check("balanced")))
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
                    emit(dict(kind="case_skip", gemm=gname, case=cname, why="more tokens than the operands hold"))
                    continue
                offsets.copy_(torch.tensor(groups, dtype=torch.int32, device=dev).cumsum(0, dtype=torch.int32))
                emit(dict(kind="case", case=cname, case_tokens=sum(groups), **check(cname)))
            offsets.copy_(balanced)
            emit(dict(kind="check_after_cases", m=m, n=n, **check("balanced")))

        if not args.no_api_check:
            try:
                import cudnn.gemm.cutedsl.grouped.wgrad.api as wapi
                from cudnn.gemm.cutedsl.grouped.wgrad import _bf16_api

                api_cls = _bf16_api.GroupedGemmWgradBf16API
                original_flag, original_applies = api_cls._use_wide_tiles, api_cls._wide_applies
                picked = []

                def recorded(self):
                    r = original_applies(self)
                    picked.append(r)
                    return r

                api_cls._wide_applies = recorded
                rec = dict(kind="api", gemm=gname, default_use_wide_tiles=original_flag)
                try:
                    for flag in (True, False):
                        api_cls._use_wide_tiles = flag  # test-only switch: no knob in the library
                        wapi._wgrad_wrapper_memo.clear()
                        wapi._cache_of_GroupedGemmWgradSm100Objects.clear()
                        picked.clear()
                        res = wapi.grouped_gemm_wgrad_wrapper_sm100(
                            a_tensor=a, b_tensor=b, sfa_tensor=None, sfb_tensor=None, offsets_tensor=offsets, output_mode="dense",
                            acc_dtype=torch.float32, wgrad_dtype=torch.bfloat16, mma_tiler_mn=(256, 256), cluster_shape_mn=(2, 1),
                            atomic_counter_claim=counter)
                        torch.cuda.synchronize()
                        w = res["wgrad_tensor"]
                        rec[f"use_wide_{flag}_picked_wide"] = picked[:1]
                        rec[f"use_wide_{flag}_diff_bytes"] = int((w.view(torch.uint16) != outs[f"{gname}_stock"].view(torch.uint16)).sum().item())
                        del res, w
                finally:
                    api_cls._use_wide_tiles, api_cls._wide_applies = original_flag, original_applies
                    wapi._wgrad_wrapper_memo.clear()
                    wapi._cache_of_GroupedGemmWgradSm100Objects.clear()
                    torch.cuda.empty_cache()
                emit(rec)
            except Exception:  # noqa: BLE001
                import traceback

                emit(dict(kind="api_error", gemm=gname, error=traceback.format_exc()[-1500:]))

    if os.environ.get("LS_NO_TIME") == "1":
        emit(dict(kind="done", passes=0, s=round(time.time() - t_start, 1)))
        return
    lockstep.run(arms, pairs, emit, args.budget, args.reps, flush, t_start)


if __name__ == "__main__":
    main()

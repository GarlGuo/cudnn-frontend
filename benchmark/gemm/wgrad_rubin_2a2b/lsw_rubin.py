#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lock-step kernel A/B of the MXFP8 grouped wgrad on Rubin (SM107), also runnable on SM100 for a smoke test.

torchrun --nproc-per-node <GPUs> lsw_rubin.py --shapes m:ep:t,m:ep:t,... --out F [--budget S] [--reps R]
Rank r takes shape r. All ranks fire the same arm together after a 20 ms idle gate (lockstep.py).
Env: FE_PYTHON (the cuDNN-FE python/ root of the tree under test; the compiled module is not needed),
     WAB_COUNTER=1 (atomic_counter_claim on every arm), LSW_ARMS (comma list, default below), LSW_CPU_INIT=1 (build the
     FP8 operands and E8M0 scales on the CPU, for a device where those GPU conversions are unavailable).

Arms (kernel classes constructed directly, so nothing depends on the API's selection):
  stock        the Rubin wgrad (BlockScaledMoEGroupedGemmWgradRubinKernel): what the API runs on Rubin today
  stock2       A/A control (a second instance of stock)
  dense        the SM100 global-descriptor wgrad (BlockScaledMoEGroupedGemmWgradDenseKernel), forced: SM100 K32 MMA
  stock_sm100  the SM100 stock wgrad, forced: the bit reference of dense (same MMA)
  reuse        the Rubin A-reuse wgrad (BlockScaledMoEGroupedGemmWgradReuseRubinKernel), a_stages = a_stages_for(rows, experts)
On SM100 the "stock" and "reuse" arms are the SM100 classes (smoke test of the harness only).

Byte checks per GEMM: stock2, reuse vs stock (expected 0 differing bytes); dense vs stock_sm100 (expected 0); dense vs
stock (a different MMA K, so bits may differ: reported as the differing count and the max |diff| / max |stock|).
Shape: T * top_k * ep / E rows per local expert (balanced), padded to 256; K-major FP8 A/B, random E8M0 scales, BF16 dW.
"""
import argparse, json, os, sys, time, types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lockstep

lockstep.pin_rank_gpu()

MODELS = {
    "deepseek_v3": dict(hidden=7168, experts=256, top_k=8, intermediate=2048),
    "qwen35_122b": dict(hidden=3072, experts=256, top_k=8, intermediate=1024),
    "qwen38_2p4t": dict(hidden=8192, experts=512, top_k=10, intermediate=2048),
    "qwen35_397b": dict(hidden=4096, experts=512, top_k=10, intermediate=1024),
    "glm53_flash": dict(hidden=4096, experts=288, top_k=8, intermediate=2048),
}


def install_cudnn(python_root):
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
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad import BlockScaledMoEGroupedGemmWgradKernel as StockSm100
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad_dense import BlockScaledMoEGroupedGemmWgradDenseKernel as Dense

    cap = torch.cuda.get_device_capability()
    rubin = cap == (10, 7) or os.environ.get("LSW_ASSUME_RUBIN") == "1"
    if rubin:
        from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad_rubin import BlockScaledMoEGroupedGemmWgradRubinKernel as Stock
        from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad_reuse_rubin import BlockScaledMoEGroupedGemmWgradReuseRubinKernel as Reuse
    else:
        Stock = StockSm100
        from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad_reuse import BlockScaledMoEGroupedGemmWgradReuseKernel as Reuse
    classes = dict(stock=Stock, stock2=Stock, dense=Dense, stock_sm100=StockSm100, reuse=Reuse)
    names = [x for x in os.environ.get("LSW_ARMS", "stock,dense,reuse,stock2,stock_sm100").split(",") if x]
    assert names[0] == "stock", "the first arm is the reference"

    out = open(f"{args.out}.r{rank}", "a")

    def emit(rec):
        rec.update(model=model, ep=ep, tokens=tokens, host=os.uname().nodename, gpu=os.environ.get("CUDA_VISIBLE_DEVICES", "?"))
        out.write(json.dumps(rec) + "\n")
        out.flush()

    md = MODELS[model]
    H, E, topk, I = md["hidden"], md["experts"], md["top_k"], md["intermediate"]
    el = E // ep
    per = -(-(tokens * topk * ep // E) // 256) * 256
    rows = el * per
    dev = "cuda"
    cpu_init = os.environ.get("LSW_CPU_INIT") == "1"
    gdev = "cpu" if cpu_init else dev
    g = torch.Generator(device=gdev).manual_seed(1)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    clusters = HardwareInfo().get_max_active_clusters(2)
    flush = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=dev)
    counter = os.environ.get("WAB_COUNTER", "0") == "1"
    a_stages = Reuse.a_stages_for(rows, el) if hasattr(Reuse, "a_stages_for") else None
    emit(dict(kind="shape", rows=rows, per_expert=per, experts=el, H=H, I=I, clusters=clusters, counter=counter, capability=list(cap),
              rubin=rubin, arms=names, reuse_a_stages=a_stages, sms=torch.cuda.get_device_properties(0).multi_processor_count))

    def t(x, align=16):
        return from_dlpack(x, assumed_align=align)

    def fp8(*shape):
        return (torch.randn(*shape, generator=g, device=gdev) * 2).to(torch.float8_e4m3fn).to(dev)

    def e8m0(*shape):
        return torch.randint(124, 131, shape, generator=g, device=gdev, dtype=torch.uint8).to(dev).view(torch.float8_e8m0fnu)

    offsets = (torch.arange(1, el + 1, dtype=torch.int32) * per).to(dev).contiguous()
    arms, pairs = {}, []
    for gname, (m, n) in {"dW1": (2 * I, H), "dW2": (H, I)}.items():
        a = fp8(m, rows)
        b = fp8(n, rows).t()
        cols = -(-(rows // 32) // 4) * 4
        sfa, sfb = e8m0(m, cols), e8m0(n, cols)
        epc = Reuse.can_implement(m, n, ab_bits=8, c_bits=16, has_global_scale=False, tokens=rows, experts=el)
        emit(dict(kind="rule", gemm=gname, m=m, n=n, epc_reuse=bool(epc), refined_reuse=bool(epc and n // 256 > Reuse.n_group_for(n)),
                  n_group=Reuse.n_group_for(n)))
        outs, shared = {}, None
        for arm in names:
            cls = classes[arm]
            kw = dict(a_stages=a_stages) if (cls is Reuse and a_stages is not None) else {}
            try:
                kernel = cls(sf_vec_size=32, acc_dtype=cutlass.Float32, use_2cta_instrs=True, mma_tiler_mn=(256, 256), cluster_shape_mn=(2, 1),
                             expert_cnt=el, weight_mode=MoEWeightMode.DENSE, input_order=WGradInputOrder.Tensor2D, atomic_counter_claim=counter, **kw)
                o = torch.full((el, m, n), float("nan"), device=dev, dtype=torch.bfloat16)
                ws = torch.zeros(max(kernel.get_workspace_bytes(), 256), device=dev, dtype=torch.uint8)
                kargs = [t(a), t(b), t(sfa), t(sfb), t(o), t(offsets, 4), t(ws, 128)]
                t0 = time.time()
                fn = cute.compile(kernel, *kargs, clusters, stream, None, None, None)
                fn(*kargs, stream, None, None, None)
                torch.cuda.synchronize()
            except Exception:  # noqa: BLE001
                import traceback

                emit(dict(kind="build_error", arm=f"{gname}_{arm}", error=traceback.format_exc()[-2000:]))
                raise
            emit(dict(kind="compile", arm=f"{gname}_{arm}", s=round(time.time() - t0, 1)))
            outs[arm] = o
            if shared is None:
                shared = kargs[4]
            else:
                kargs[4] = shared  # timing: every arm writes the first arm's dW buffer

            def launch(fn=fn, kargs=kargs):
                fn(*kargs, stream, None, None, None)

            arms[f"{gname}_{arm}"] = ((lambda: None), [launch])
        rec = dict(kind="check", gemm=gname, m=m, n=n, stock_nan=int(torch.isnan(outs["stock"]).sum().item()))
        ref = outs["stock"].view(torch.uint16)
        for arm, o in outs.items():
            if arm == "stock":
                continue
            rec[f"{arm}_diff_bytes"] = int((o.view(torch.uint16) != ref).sum().item())
            rec[f"{arm}_nan"] = int(torch.isnan(o).sum().item())
        if "dense" in outs and "stock_sm100" in outs:
            rec["dense_vs_stock_sm100_diff_bytes"] = int((outs["dense"].view(torch.uint16) != outs["stock_sm100"].view(torch.uint16)).sum().item())
        if "dense" in outs:
            s = outs["stock"].float()
            rec["dense_max_rel_err"] = float((outs["dense"].float() - s).abs().max().item() / max(s.abs().max().item(), 1e-30))
        emit(rec)
        del outs
        pairs += [(f"{gname}_{x}", f"{gname}_stock") for x in names if x != "stock"]
        if "reuse" in names and "dense" in names:
            pairs.append((f"{gname}_reuse", f"{gname}_dense"))
    torch.cuda.empty_cache()
    lockstep.run(arms, pairs, emit, args.budget, args.reps, flush, t_start)


if __name__ == "__main__":
    main()

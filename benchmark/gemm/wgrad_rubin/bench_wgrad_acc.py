#!/usr/bin/env python3
"""SM100 grouped GEMM wgrad: atomic_counter_claim (counter) vs CLC, CUDA Graph device time, ABBA.

  python bench_wgrad_acc.py --shape deepseek_v3|qwen35 --ep 8 --precision bf16|mxfp8 [--gemm fc1,fc2] [--out JSON]

Per EP rank: microbatch 4096 tokens, top-k routing over E experts, E/ep local experts. Uneven routing: per-expert counts
from a seeded log-normal (sigma 1), floored to the precision's group granularity (BF16 256 tokens, MXFP8 128), so
some experts are empty. FC1 wgrad: dW1 (E_local, 2F, H) = dH^T x X; FC2 wgrad: dW2 (E_local, H, F) = dY^T x act.
BF16 operands use the training layouts (A M-major, B N-major); MXFP8 operands are K-major with E8M0 scales per 32
tokens. Both arms run the wrapper (tile 256x256, cluster (2, 1)) and must agree byte for byte. Timing: a graph of
--bundle launches per arm, --warmup replays, then --repeats rounds in alternating AB/BA order; median per launch.
"""

import argparse, json, math, statistics, time

import torch
import cudnn

SHAPES = {"deepseek_v3": dict(h=7168, f=2048, experts=256, topk=8), "qwen35": dict(h=4096, f=1024, experts=512, topk=10)}


def counts(args, local, gran):
    shape = SHAPES[args.shape]
    total = args.microbatch * shape["topk"]  # tokens one rank receives: ep ranks x microbatch x topk x (1 / ep of the experts)
    g = torch.Generator().manual_seed(args.seed)
    w = torch.exp(torch.randn(local, generator=g))
    raw = (w / w.sum() * total).tolist()
    return [int(x) // gran * gran for x in raw]


def pgmig():
    for line in open("/proc/vmstat"):
        if line.startswith("pgmigrate_success"):
            return int(line.split()[1])
    return -1


def problem(args, gemm, local):
    shape = SHAPES[args.shape]
    m, n = (2 * shape["f"], shape["h"]) if gemm == "fc1" else (shape["h"], shape["f"])
    gran = 256 if args.precision == "bf16" else 128
    groups = counts(args, local, gran)
    tokens = sum(groups)
    torch.manual_seed(args.seed)
    kw = dict(
        offsets_tensor=torch.tensor(groups, dtype=torch.int32, device="cuda").cumsum(0, dtype=torch.int32),
        output_mode="dense",
        acc_dtype=torch.float32,
        wgrad_dtype=torch.bfloat16,
        mma_tiler_mn=(256, 256),
        cluster_shape_mn=(2, 1),
    )
    if args.precision == "bf16":
        kw.update(
            a_tensor=(torch.randn(tokens, m, device="cuda") * 0.125).to(torch.bfloat16).t(),
            b_tensor=(torch.randn(tokens, n, device="cuda") * 0.125).to(torch.bfloat16),
            sfa_tensor=None,
            sfb_tensor=None,
        )
    else:
        a = (torch.randn(m, tokens, device="cuda") * 0.5).to(torch.float8_e4m3fn)
        b = (torch.randn(n, tokens, device="cuda") * 0.5).to(torch.float8_e4m3fn).t()
        sfa = (
            torch.randint(-2, 1, (math.ceil(m / 128) * 128, math.ceil(tokens / 32 / 4) * 4), device="cuda").cpu().float().exp2().to(torch.float8_e8m0fnu).cuda()
        )
        sfb = (
            torch.randint(-2, 1, (math.ceil(n / 128) * 128, math.ceil(tokens / 32 / 4) * 4), device="cuda").cpu().float().exp2().to(torch.float8_e8m0fnu).cuda()
        )
        kw.update(a_tensor=a, b_tensor=b, sfa_tensor=sfa, sfb_tensor=sfb, sf_vec_size=32)
    kw["wgrad_tensor"] = torch.empty((local, m, n), dtype=torch.bfloat16, device="cuda")
    meta = dict(M=m, N=n, experts=local, tokens=tokens, empty=sum(1 for x in groups if x == 0), min=min(groups), max=max(groups))
    return kw, meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shape", required=True, choices=SHAPES)
    p.add_argument("--ep", type=int, default=8)
    p.add_argument("--precision", required=True, choices=["bf16", "mxfp8"])
    p.add_argument("--gemm", default="fc1,fc2")
    p.add_argument("--microbatch", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bundle", type=int, default=10)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    local = SHAPES[args.shape]["experts"] // args.ep
    rows = []
    for gemm in args.gemm.split(","):
        kw, meta = problem(args, gemm, local)
        arms = {"clc": False, "counter": True}
        outs, graphs = {}, {}
        for arm, flag in arms.items():
            run = lambda flag=flag: cudnn.grouped_gemm_wgrad_wrapper_sm100(atomic_counter_claim=flag, **kw)
            for _ in range(3):
                run()
            torch.cuda.synchronize()
            outs[arm] = kw["wgrad_tensor"].clone()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(args.bundle):
                    run()
            graphs[arm] = graph
        same = torch.equal(outs["clc"].view(torch.uint8), outs["counter"].view(torch.uint8))
        for _ in range(args.warmup):
            for g in graphs.values():
                g.replay()
        torch.cuda.synchronize()
        samples = {arm: [] for arm in arms}
        ev = {arm: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for arm in arms}
        mig0 = pgmig()
        for r in range(args.repeats):
            order = ("clc", "counter") if r % 2 == 0 else ("counter", "clc")
            for arm in order:
                b, e = ev[arm]
                b.record()
                graphs[arm].replay()
                e.record()
                e.synchronize()
                samples[arm].append(b.elapsed_time(e) * 1000 / args.bundle)
        mig1 = pgmig()
        med = {arm: statistics.median(v) for arm, v in samples.items()}
        ratios = sorted(c / l for c, l in zip(samples["counter"], samples["clc"]))
        row = dict(
            shape=args.shape,
            ep=args.ep,
            precision=args.precision,
            gemm=gemm,
            **meta,
            clc_us=med["clc"],
            counter_us=med["counter"],
            ratio_median=statistics.median(ratios),
            ratio_q25=ratios[len(ratios) // 4],
            ratio_q75=ratios[3 * len(ratios) // 4],
            bytes_equal=same,
            pgmigrate_delta=mig1 - mig0,
            samples=samples,
        )
        rows.append(row)
        print(
            f"PERF {args.shape} ep{args.ep} {args.precision} {gemm} M={meta['M']} N={meta['N']} E={local} T={meta['tokens']} empty={meta['empty']} "
            f"clc={med['clc']:.1f}us counter={med['counter']:.1f}us ratio={row['ratio_median']:.4f} [{row['ratio_q25']:.4f},{row['ratio_q75']:.4f}] "
            f"bytes_equal={same} pgmig={mig1 - mig0}",
            flush=True,
        )
        for g in graphs.values():
            g.reset()
    if args.out:
        json.dump(rows, open(args.out, "w"))


if __name__ == "__main__":
    main()

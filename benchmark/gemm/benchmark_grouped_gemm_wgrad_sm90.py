# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Wentao Guo, Phuong Nguyen

"""Time the SM90 BF16 grouped GEMM wgrad (grouped_gemm_wgrad_wrapper_sm100 on H100) against torch._grouped_mm.

One item is (model, op) at one EP rank's sizes: tokens per GPU x top_k routed rows spread evenly over the rank's
experts, each expert padded to 256 rows. Ops: fc1_wgrad (dW1 = dH1^T X, M=2I, N=H) and fc2_wgrad (dW2 = dY^T h,
M=H, N=I), both from row-major (tokens, features) activations and gradients. Timings interleave the backends rep by
rep, each rep behind an L2 flush. Same CLI as benchmark_grouped_gemm_sm90.py.

  python benchmark/gemm/benchmark_grouped_gemm_wgrad_sm90.py --list-items
  python benchmark/gemm/benchmark_grouped_gemm_wgrad_sm90.py --item deepseek_v3.fc2_wgrad --ep 8 --out perf.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys

import torch

#: hidden H, global experts E, top_k, expert intermediate I (FC1 N = 2I, gate + up): HF config.json dims.
MODEL_SHAPES = {
    "deepseek_v3": dict(hidden=7168, experts=256, top_k=8, intermediate=2048),
    "qwen35_122b": dict(hidden=3072, experts=256, top_k=8, intermediate=1024),
    "qwen35_397b": dict(hidden=4096, experts=512, top_k=10, intermediate=1024),
    "qwen38_2p4t": dict(hidden=8192, experts=512, top_k=10, intermediate=2048),
    "glm53_flash": dict(hidden=4096, experts=288, top_k=8, intermediate=2048),
}
OPS = ("fc1_wgrad", "fc2_wgrad")
ITEMS = [f"{model}.{op}" for model in MODEL_SHAPES for op in OPS]


def item_problem(item: str, ep: int, tokens: int):
    model, op = item.split(".")
    shape = MODEL_SHAPES[model]
    if shape["experts"] % ep:
        raise ValueError(f"{model}: {shape['experts']} experts do not divide over EP {ep}")
    experts = shape["experts"] // ep
    m, n = (2 * shape["intermediate"], shape["hidden"]) if op == "fc1_wgrad" else (shape["hidden"], shape["intermediate"])
    rows = -(-tokens * shape["top_k"] // experts)
    count = -(-rows // 256) * 256
    return dict(model=model, op=op, experts=experts, m=m, n=n, rows_per_expert=rows, count=count, k=count * experts)


def make_tensors(p, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    grad = torch.randn((p["k"], p["m"]), generator=g, device="cuda", dtype=torch.bfloat16) * 0.125
    act = torch.randn((p["k"], p["n"]), generator=g, device="cuda", dtype=torch.bfloat16) * 0.125
    offsets = torch.arange(1, p["experts"] + 1, device="cuda", dtype=torch.int32) * p["count"]
    out = torch.empty((p["experts"], p["m"], p["n"]), device="cuda", dtype=torch.bfloat16)
    return dict(a=grad.t(), b=act, offsets=offsets, out=out)


def backends(p, t, names):
    import cudnn

    def ours():
        return cudnn.grouped_gemm_wgrad_wrapper_sm100(
            a_tensor=t["a"],
            b_tensor=t["b"],
            sfa_tensor=None,
            sfb_tensor=None,
            offsets_tensor=t["offsets"],
            wgrad_tensor=t["out"],
            acc_dtype=torch.float32,
            wgrad_dtype=torch.bfloat16,
        )["wgrad_tensor"]

    runs = {"cudnn_sm90": ours}
    if "torch_grouped_mm" in names:
        # 2D x 2D grouped mm: offs splits the shared K (tokens) per expert.
        runs["torch_grouped_mm"] = lambda: torch._grouped_mm(t["a"], t["b"], offs=t["offsets"], out_dtype=torch.bfloat16)
    return runs


def time_item(runs, n_runs, warmup, reps, l2_flush):
    flush = torch.empty(int(l2_flush * torch.cuda.get_device_properties(0).L2_cache_size), dtype=torch.uint8, device="cuda")
    for fn in runs.values():
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()
    medians = {name: [] for name in runs}
    for _ in range(n_runs):
        events = {name: [] for name in runs}
        for _ in range(reps):
            for name, fn in runs.items():  # A/B interleaved within the item
                flush.zero_()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                fn()
                end.record()
                events[name].append((start, end))
        torch.cuda.synchronize()
        for name, pairs in events.items():
            medians[name].append(statistics.median(s.elapsed_time(e) * 1e3 for s, e in pairs))
    return {name: dict(us_median=statistics.median(v), us_runs=v) for name, v in medians.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list-items", action="store_true")
    parser.add_argument("--item", choices=ITEMS)
    parser.add_argument("--ep", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=8192, help="tokens per GPU")
    parser.add_argument("--baselines", default="torch_grouped_mm")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--l2-flush", type=float, default=2.0, help="flush buffer size in multiples of L2")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    if args.list_items:
        print("\n".join(ITEMS))
        return 0
    if args.item is None:
        parser.error("--item or --list-items is required")
    if torch.cuda.get_device_capability() != (9, 0):
        raise SystemExit(f"SM90 is required, found {torch.cuda.get_device_capability()}")

    p = item_problem(args.item, args.ep, args.tokens)
    t = make_tensors(p)
    runs = backends(p, t, set(filter(None, args.baselines.split(","))))
    outputs = {name: fn().float() for name, fn in runs.items()}
    torch.cuda.synchronize()
    check = {}
    if "torch_grouped_mm" in outputs:
        diff = (outputs["cudnn_sm90"] - outputs["torch_grouped_mm"]).abs().max().item()
        check = dict(max_abs_diff_vs_torch=diff, scale=outputs["torch_grouped_mm"].abs().max().item())
    timing = time_item(runs, args.runs, args.warmup, args.reps, args.l2_flush)
    flops = 2.0 * p["m"] * p["n"] * p["k"]
    for value in timing.values():
        value["tflops"] = flops / (value["us_median"] * 1e-6) / 1e12
    record = dict(
        item=args.item,
        ep=args.ep,
        tokens=args.tokens,
        problem=p,
        check=check,
        timing=timing,
        device=torch.cuda.get_device_name(),
        torch=torch.__version__,
        args=dict(runs=args.runs, warmup=args.warmup, reps=args.reps, l2_flush=args.l2_flush),
    )
    if "torch_grouped_mm" in timing:
        record["speedup_vs_torch"] = timing["torch_grouped_mm"]["us_median"] / timing["cudnn_sm90"]["us_median"]
    text = json.dumps(record, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as file:
            file.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

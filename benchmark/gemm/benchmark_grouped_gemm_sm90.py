# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Wentao Guo, Phuong Nguyen

"""Time the SM90 BF16 grouped GEMMs (the *_wrapper_sm100 APIs on H100) against torch._grouped_mm on MoE shapes.

One item is (model, op) at one EP rank's sizes: tokens per GPU x top_k routed rows spread evenly over the rank's
experts, each expert padded to 256 rows. Ops: fc1_fwd (N=2I, K=H), fc2_fwd (N=H, K=I), fc1_dgrad (N=H, K=2I),
fc2_dgrad (N=I, K=H); the dgrad ops read the forward weights N-major. fc1_glu is fc1_fwd with the SwiGLU fused
(grouped_gemm_glu_wrapper_sm100); fc2_dglu is fc2_dgrad with the dSwiGLU fused (grouped_gemm_dglu_wrapper_sm100).
Their torch baseline is torch._grouped_mm plus the activation in eager torch. Timings interleave the backends rep by
rep, each rep behind an L2 flush.

  python benchmark/gemm/benchmark_grouped_gemm_sm90.py --list-items
  python benchmark/gemm/benchmark_grouped_gemm_sm90.py --item deepseek_v3.fc2_fwd --ep 8 --out perf.json
  python benchmark/gemm/benchmark_grouped_gemm_sm90.py --item deepseek_v3.fc2_fwd --schedules cooperative,pingpong --tile 128,192
"""

from __future__ import annotations

import argparse
import functools
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
OPS = ("fc1_fwd", "fc2_fwd", "fc1_dgrad", "fc2_dgrad", "fc1_glu", "fc2_dglu")
BLOCK = 32  # interleaved_32 gate/up column blocks
ITEMS = [f"{model}.{op}" for model in MODEL_SHAPES for op in OPS]


def item_problem(item: str, ep: int, tokens: int):
    model, op = item.split(".")
    shape = MODEL_SHAPES[model]
    hidden, inter2 = shape["hidden"], 2 * shape["intermediate"]
    if shape["experts"] % ep:
        raise ValueError(f"{model}: {shape['experts']} experts do not divide over EP {ep}")
    experts = shape["experts"] // ep
    n, k = {
        "fc1_fwd": (inter2, hidden),
        "fc2_fwd": (hidden, shape["intermediate"]),
        "fc1_dgrad": (hidden, inter2),
        "fc2_dgrad": (shape["intermediate"], hidden),
        "fc1_glu": (inter2, hidden),
        "fc2_dglu": (shape["intermediate"], hidden),
    }[op]
    rows = -(-tokens * shape["top_k"] // experts)
    count = -(-rows // 256) * 256
    return dict(
        model=model,
        op=op,
        experts=experts,
        n=n,
        k=k,
        rows_per_expert=rows,
        count=count,
        m=count * experts,
        b_major="n" if op in ("fc1_dgrad", "fc2_dgrad", "fc2_dglu") else "k",
    )


def make_tensors(p, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn((p["m"], p["k"]), generator=g, device="cuda", dtype=torch.bfloat16) * 0.125
    # (E, N, K) K-major for fwd; the forward weights (E, K, N) read N-major for dgrad.
    shape = (p["experts"], p["n"], p["k"]) if p["b_major"] == "k" else (p["experts"], p["k"], p["n"])
    b = torch.randn(shape, generator=g, device="cuda", dtype=torch.bfloat16) * 0.125
    offsets = torch.arange(1, p["experts"] + 1, device="cuda", dtype=torch.int32) * p["count"]
    alpha = torch.ones(p["experts"], device="cuda", dtype=torch.float32)
    prob = torch.ones(p["m"], device="cuda", dtype=torch.float32)
    d = torch.empty((p["m"], p["n"]), device="cuda", dtype=torch.bfloat16)
    # fc2_dglu reads the forward's saved pre-activation C (gate | up, 2N wide) and accumulates dprob.
    c = torch.randn((p["m"], 2 * p["n"]), generator=g, device="cuda", dtype=torch.bfloat16) if p["op"] == "fc2_dglu" else None
    dprob = torch.zeros(p["m"], device="cuda", dtype=torch.float32)
    return dict(a=a, b=b, offsets=offsets, alpha=alpha, prob=prob, d=d, c=c, dprob=dprob)


def split_gate_up(x):
    m, two_n = x.shape
    blocks = x.view(m, two_n // (2 * BLOCK), 2, BLOCK)
    return blocks[:, :, 0].reshape(m, -1), blocks[:, :, 1].reshape(m, -1)


def torch_swiglu(acc, prob):
    gate, up = split_gate_up(acc.float())
    return (up * torch.nn.functional.silu(gate) * prob[:, None]).to(torch.bfloat16)


def torch_dswiglu(g, c, prob):
    """D (dgate | dup, interleaved) of SwiGLU from g = dY @ W^T and the saved C; alpha = beta = 1."""
    g = g.float()
    x1, x2 = (half.float() for half in split_gate_up(c))
    s = torch.sigmoid(x1)
    d1 = g * prob[:, None] * x2 * s * (1 + x1 * (1 - s))
    d2 = g * prob[:, None] * x1 * s
    m, n = d1.shape
    return torch.stack((d1.view(m, n // BLOCK, BLOCK), d2.view(m, n // BLOCK, BLOCK)), dim=2).reshape(m, 2 * n).to(torch.bfloat16)


def forced_schedule(fn, schedule):
    """``fn`` under a forced SM90 schedule (a test / benchmark hook; the library picks it by shape otherwise)."""
    import cudnn.gemm.cutedsl.grouped.unfused.moe_grouped_gemm_sm90 as sm90

    def run():
        saved, sm90._FORCE_SCHEDULE = sm90._FORCE_SCHEDULE, schedule
        try:
            return fn()
        finally:
            sm90._FORCE_SCHEDULE = saved

    return run


def backends(p, t, names, config=None, schedules=(None,)):
    """``config``: mma_tiler_mn / cluster_shape_mn for the GEMM and GLU ops; one cudnn arm per schedule (None: the heuristic's).

    The pingpong arm always runs its one configuration, a 128x192 tile on a (2, 1) cluster.
    """
    import cudnn

    config = config or {}

    def ours_glu(config):
        return cudnn.grouped_gemm_glu_wrapper_sm100(
            a_tensor=t["a"],
            sfa_tensor=None,
            padded_offsets=t["offsets"],
            alpha_tensor=t["alpha"],
            b_tensor=t["b"],
            prob_tensor=t["prob"],
            c_dtype=torch.bfloat16,
            d_dtype=torch.bfloat16,
            act_func="swiglu",
            **config,
        )["d_tensor"]

    def ours_dglu(config):
        return cudnn.grouped_gemm_dglu_wrapper_sm100(
            a_tensor=t["a"],
            c_tensor=t["c"],
            sfa_tensor=None,
            padded_offsets=t["offsets"],
            alpha_tensor=t["alpha"],
            beta_tensor=t["alpha"],
            prob_tensor=t["prob"],
            dprob_tensor=t["dprob"],
            b_tensor=t["b"],
            b_major=p["b_major"],
            d_dtype=torch.bfloat16,
            act_func="dswiglu",
        )["d_row_tensor"]

    def ours(config):
        cudnn.grouped_gemm_wrapper_sm100(
            a_tensor=t["a"],
            padded_offsets=t["offsets"],
            alpha_tensor=t["alpha"],
            b_tensor=t["b"],
            b_major=p["b_major"],
            prob_tensor=t["prob"],
            d_tensor=t["d"],
            c_dtype=torch.bfloat16,
            d_dtype=torch.bfloat16,
            **config,
        )
        return t["d"]

    # torch._grouped_mm takes (G, K, N) weights: a view of either layout.
    b_kn = t["b"].transpose(1, 2) if p["b_major"] == "k" else t["b"]
    grouped_mm = lambda: torch._grouped_mm(t["a"], b_kn, offs=t["offsets"], out_dtype=torch.bfloat16)
    if p["op"] == "fc1_glu":
        fn, reference = ours_glu, lambda: torch_swiglu(grouped_mm(), t["prob"])
    elif p["op"] == "fc2_dglu":
        fn, reference = ours_dglu, lambda: torch_dswiglu(grouped_mm(), t["c"], t["prob"])
    else:
        fn, reference = ours, grouped_mm
    pingpong_config = dict(mma_tiler_mn=(128, 192), cluster_shape_mn=(2, 1))
    arm = lambda s: functools.partial(fn, pingpong_config if s == "pingpong" else config)
    runs = {"cudnn_sm90" + ("" if s is None else f"_{s}"): forced_schedule(arm(s), s) for s in schedules}
    if "torch_grouped_mm" in names:
        runs["torch_grouped_mm"] = reference
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
    pair = lambda text: tuple(int(v) for v in text.split(","))
    parser.add_argument(
        "--schedules",
        default=None,
        help="comma list of forced SM90 schedules (cooperative, pingpong), one arm each; default: the heuristic's. "
        "The pingpong arm always runs 128x192 on a (2, 1) cluster",
    )
    parser.add_argument("--tile", type=pair, default=None, help="mma_tiler_mn as M,N (GEMM and GLU ops)")
    parser.add_argument("--cluster", type=pair, default=None, help="cluster_shape_mn as M,N (GEMM and GLU ops)")
    args = parser.parse_args(argv)
    if args.list_items:
        print("\n".join(ITEMS))
        return 0
    if args.item is None:
        parser.error("--item or --list-items is required")
    from cudnn.tensor_adapter import has_wgmma

    major, _ = torch.cuda.get_device_capability()
    if not has_wgmma(major):
        raise SystemExit(f"SM90 is required, found {torch.cuda.get_device_capability()}")

    p = item_problem(args.item, args.ep, args.tokens)
    t = make_tensors(p)
    names = {"mma_tiler_mn": args.tile, "cluster_shape_mn": args.cluster}
    config = {key: value for key, value in names.items() if value is not None}
    schedules = tuple(args.schedules.split(",")) if args.schedules else (None,)
    if (config or args.schedules) and p["op"] == "fc2_dglu":
        parser.error("--schedules / --tile / --cluster apply to the GEMM and GLU ops, not fc2_dglu")
    runs = backends(p, t, set(filter(None, args.baselines.split(","))), config, schedules)
    ours_name = next(iter(runs))
    outputs = {name: fn().float().view(p["m"], -1) for name, fn in runs.items()}
    torch.cuda.synchronize()
    check = {}
    if "torch_grouped_mm" in outputs:
        diff = max((outputs[name] - outputs["torch_grouped_mm"]).abs().max().item() for name in runs if name.startswith("cudnn_sm90"))
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
        config={key: list(value) if isinstance(value, tuple) else value for key, value in config.items()},
        schedules=list(schedules),
    )
    if "torch_grouped_mm" in timing:
        record["speedup_vs_torch"] = timing["torch_grouped_mm"]["us_median"] / timing[ours_name]["us_median"]
    text = json.dumps(record, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as file:
            file.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

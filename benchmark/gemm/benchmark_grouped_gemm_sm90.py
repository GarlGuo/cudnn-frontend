# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Time the SM90 (Hopper) BF16 grouped GEMMs against torch._grouped_mm.

One row is (model, op, arm) at one EP rank's sizes: --tokens per GPU times top_k
routed rows, spread evenly over the rank's experts, each expert padded to 256 rows.
This is a synthetic balanced-routing benchmark. Ops: fc1_fwd (N=2I, K=H), fc2_fwd
(N=H, K=I), fc1_dgrad (N=H, K=2I) and fc2_dgrad (N=I, K=H); dgrad reads the forward
weights N-major (b_major="n"). TFLOPs count the padded rows.

python benchmark/gemm/benchmark_grouped_gemm_sm90.py --model deepseek_v3 --ep 8 --arms default cooperative pingpong torch

Arms: "default" is the library's own schedule (its shape rule); "cooperative" and
"pingpong" force one through the test hook moe_grouped_gemm_sm90._SCHEDULE_OVERRIDE
(pingpong runs only its 128x192 tile on a (2, 1) cluster); "torch" is
torch._grouped_mm, which allocates its output per call. Each cudnn arm is built
through its class API (fwd and dgrad: GroupedGemmSm90) with check_support and
compile, and times the compiled plan's execute with fixed outputs and a
preallocated workspace, so no wrapper allocation or dispatch. Input generation and
JIT are outside timing; graph timing is device time for an actual captured bundle,
while eager timing includes host dispatch and synchronization. Routing and token
permutation are excluded; padding rows are zero, with a zero prob. Every cudnn
arm's output is checked against the torch arm outside timing, and its hash is a
cross-process control.
"""

import argparse
import contextlib
import functools
import hashlib
import importlib.metadata
import json
import statistics
import time
from pathlib import Path

import torch
import cudnn
from cudnn.tensor_adapter import has_wgmma

#: hidden H, global experts, top_k and expert intermediate I (FC1 N = 2I, gate + up), from the HF config.json.
MODEL_SHAPES = {
    "deepseek_v3": dict(hidden=7168, experts=256, top_k=8, intermediate=2048),
    "qwen35_122b": dict(hidden=3072, experts=256, top_k=8, intermediate=1024),
    "qwen35_397b": dict(hidden=4096, experts=512, top_k=10, intermediate=1024),
    "qwen38_2p4t": dict(hidden=8192, experts=512, top_k=10, intermediate=2048),
    "glm53_flash": dict(hidden=4096, experts=288, top_k=8, intermediate=2048),
}
OPS = ("fc1_fwd", "fc2_fwd", "fc1_dgrad", "fc2_dgrad")
ARMS = ("default", "cooperative", "pingpong", "torch")


def problem(args, model, op):
    shape = MODEL_SHAPES[model]
    hidden, inter = shape["hidden"], shape["intermediate"]
    n, k = dict(fc1_fwd=(2 * inter, hidden), fc2_fwd=(hidden, inter), fc1_dgrad=(hidden, 2 * inter), fc2_dgrad=(inter, hidden))[op]
    groups = shape["experts"] // args.ep
    valid_m = -(-args.tokens * shape["top_k"] // groups)
    m = -(-valid_m // 256) * 256
    b_major = "n" if op.endswith("dgrad") else "k"
    torch.manual_seed(args.seed)
    tensors = dict(
        a=torch.randn(groups * m, k, device="cuda", dtype=torch.bfloat16) * 0.125,
        # (E, N, K) K-major for fwd; dgrad reads the forward weights (E, K, N) N-major.
        b=torch.randn((groups, n, k) if b_major == "k" else (groups, k, n), device="cuda", dtype=torch.bfloat16) * 0.125,
        padded_offsets=torch.arange(1, groups + 1, device="cuda", dtype=torch.int32) * m,
        alpha=torch.ones(groups, device="cuda"),
        prob=torch.ones(groups * m, device="cuda"),
        c=torch.empty(groups * m, n, device="cuda", dtype=torch.bfloat16),  # unused without generate_c
    )
    metadata = dict(model=model, op=op, b_major=b_major, groups=groups, M=m, valid_M=valid_m, N=n, K=k, total_rows=groups * m)
    if valid_m != m:
        for expert in range(groups):
            tensors["a"][expert * m + valid_m : (expert + 1) * m].zero_()
            tensors["prob"][expert * m + valid_m : (expert + 1) * m].zero_()
    return tensors, metadata


def grouped_mm(t, b_major):
    b_kn = t["b"].transpose(1, 2) if b_major == "k" else t["b"]  # torch._grouped_mm takes (G, K, N)
    return functools.partial(torch._grouped_mm, t["a"], b_kn, offs=t["padded_offsets"], out_dtype=torch.bfloat16)


@contextlib.contextmanager
def forced_schedule(schedule):
    """The SM90 schedule test hook: plans built inside run ``schedule`` (None: the shape rule's pick)."""
    import cudnn.gemm.cutedsl.grouped.unfused.moe_grouped_gemm_sm90 as sm90

    saved, sm90._SCHEDULE_OVERRIDE = sm90._SCHEDULE_OVERRIDE, schedule
    try:
        yield
    finally:
        sm90._SCHEDULE_OVERRIDE = saved


def prepare(t, b_major, arm):
    """One arm's timed call, its output buffer (None for torch) and its plan's configuration."""
    if arm == "torch":
        return grouped_mm(t, b_major), None, {}
    d = torch.empty_like(t["c"])
    with forced_schedule(None if arm == "default" else arm):
        api = cudnn.GroupedGemmSm90(
            sample_a=t["a"],
            sample_c=t["c"],
            sample_d=d,
            sample_padded_offsets=t["padded_offsets"],
            sample_alpha=t["alpha"],
            sample_b=t["b"],
            sample_prob=t["prob"],
            b_major=b_major,
            use_dynamic_sched=True,
        )
        if not api.check_support():
            raise RuntimeError(f"Unsupported configuration for the {arm} arm")
        api.compile()
    # The backend plan directly, with its per-call workspace allocated once: no wrapper work in timing.
    implementation = api._implementation
    nbytes = implementation._scratch_nbytes
    workspace = torch.empty(nbytes, device="cuda", dtype=torch.uint8) if nbytes else None
    run = functools.partial(
        implementation.execute,
        a_tensor=t["a"],
        c_tensor=t["c"],
        d_tensor=d,
        padded_offsets=t["padded_offsets"],
        alpha_tensor=t["alpha"],
        b_tensor=t["b"],
        prob_tensor=t["prob"],
        _workspace=workspace,
    )
    run()
    torch.cuda.synchronize()
    schedule = "pingpong" if implementation._pingpong else "cooperative"
    return run, d, dict(schedule=schedule, mma_tiler_mn=implementation.mma_tiler_mn, cluster_shape_mn=implementation.cluster_shape_mn)


def output_sha256(value):
    """Cross-process control; copying/hashing is outside timing."""
    return hashlib.sha256(value.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def distribution(samples):
    quartiles = statistics.quantiles(samples, n=4, method="inclusive") if len(samples) > 1 else [samples[0]] * 3
    return dict(median=statistics.median(samples), q25=quartiles[0], q75=quartiles[2], samples=samples)


def measure(runs, args):
    # Every arm is compiled before any is warmed or sampled. Adjacent repeats
    # alternate AB/BA so clock drift cannot separate the arms' measurement windows.
    graphs, events = {}, {}
    samples = {arm: dict(graph_device_us=[], eager_wall_us=[]) for arm in runs}
    for arm, run in runs.items():
        for _ in range(args.warmup):
            run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(args.bundle):
                run()
        graphs[arm] = graph
        events[arm] = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    for _ in range(args.warmup):
        for graph in graphs.values():
            graph.replay()
    torch.cuda.synchronize()
    for repeat in range(args.repeats):
        order = tuple(runs) if repeat % 2 == 0 else tuple(reversed(runs))
        for arm in order:
            begin, end = events[arm]
            begin.record()
            graphs[arm].replay()
            end.record()
            end.synchronize()
            samples[arm]["graph_device_us"].append(begin.elapsed_time(end) * 1000 / args.bundle)
        for arm in order:
            start = time.perf_counter()
            for _ in range(args.bundle):
                runs[arm]()
            torch.cuda.synchronize()
            samples[arm]["eager_wall_us"].append((time.perf_counter() - start) * 1e6 / args.bundle)
    measurements = {}
    for arm, metrics in samples.items():
        measurements[arm] = {name: statistics.median(values) for name, values in metrics.items()}
        measurements[arm]["distributions"] = {name: distribution(values) for name, values in metrics.items()}
    if "torch" in runs:
        for arm in runs:
            if arm != "torch":
                measurements[arm]["paired_ratio_to_torch"] = {
                    name: distribution([a / t for a, t in zip(samples[arm][name], values)]) for name, values in samples["torch"].items()
                }
    return measurements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", nargs="+", choices=tuple(MODEL_SHAPES), default=["deepseek_v3"], metavar="MODEL", help=", ".join(MODEL_SHAPES))
    parser.add_argument("--op", nargs="+", choices=OPS, default=list(OPS), metavar="OP", help=", ".join(OPS))
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=["default", "torch"], metavar="ARM", help=", ".join(ARMS))
    parser.add_argument("--ep", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=8192, help="Tokens per GPU")
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--bundle", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.ep < 1 or any(MODEL_SHAPES[model]["experts"] % args.ep for model in args.model):
        parser.error("EP must divide every selected model's expert count")
    if min(args.tokens, args.warmup, args.bundle, args.repeats) < 1:
        parser.error("tokens and timing counts must be positive")
    capability = torch.cuda.get_device_capability()
    if not has_wgmma(capability[0]):
        raise SystemExit(f"This benchmark requires an SM90 (Hopper) GPU, found compute capability {capability}")
    runtime = dict(
        gpu=torch.cuda.get_device_name(),
        capability=capability,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        cudnn_source=cudnn.__file__,
        cudnn_frontend=cudnn.__version__,
        cutlass=importlib.metadata.version("nvidia-cutlass-dsl"),
    )
    print(json.dumps(dict(runtime=runtime, configuration=vars(args)), default=str), flush=True)
    rows = []
    for model in args.model:
        for op in args.op:
            tensors, shape = problem(args, model, op)
            reference = grouped_mm(tensors, shape["b_major"])().float()
            runs, cases = {}, {}
            for arm in args.arms:
                run, d, plan = prepare(tensors, shape["b_major"], arm)
                cases[arm] = dict(**shape, arm=arm, **plan)
                if d is not None:
                    cases[arm]["max_abs_diff_vs_torch"] = (d.float() - reference).abs().max().item()
                    cases[arm]["output_sha256"] = output_sha256(d)
                    torch.testing.assert_close(d.float(), reference, rtol=2e-2, atol=2e-2)
                runs[arm] = run
            measurements = measure(runs, args)
            flops = 2.0 * shape["total_rows"] * shape["N"] * shape["K"]
            for arm, case in cases.items():
                row = {**case, **measurements[arm], "tflops": flops / measurements[arm]["graph_device_us"] / 1e6}
                rows.append(row)
                print(json.dumps(row), flush=True)
            del tensors, reference, run, d, runs, cases
    report = dict(runtime=runtime, configuration=vars(args), measurements=rows)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()

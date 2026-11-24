#!/usr/bin/env python3
"""SM100 grouped GEMM wgrad streaming input signal: cost with every tile ready, and overlap with a delayed producer.

  python bench_wgrad_signal.py --shape deepseek_v3|qwen35 --ep 8 --precision bf16|mxfp8 [--gemm fc1,fc2] [--mode ready|overlap|off] [--out JSON]

Routing as bench_wgrad_acc.py (the atomic_counter_claim PR): per EP rank a 4096-token microbatch, top-k, log-normal
(sigma 1) expert counts floored to 256 (BF16) / 128 (MXFP8), so some experts are empty. FC1 dW1 (E, 2F, H), FC2 dW2
(E, H, F). Kernels: tile 256x256, cluster (2, 1), atomic_counter_claim; K-major operands.

--mode basehead: "off" = the base tree's kernel (--base-wgrad-dir), "on" = the head's, both without the signal, in one
                process; otherwise as ready.
--mode ready:   "off" (no signal) vs "on" (signal, every tile ready), CUDA Graph bundles of --bundle, ABBA x --repeats,
                median per launch; outputs must be byte-equal.
--mode off:     "off" only (the base tree has no signal): per-launch median, for a cross-process base vs head A/B.
--mode overlap: a one-CTA producer publishes token tile i at start + (i + 1) * D / tiles (gpu-scope release), in
                expert-major or round-robin-over-experts order, D = --window x the off-path GEMM time.
                serial: producer, then the off GEMM on the same stream (D + G).
                stream: producer on one stream, the signal GEMM (max_active_clusters - 1) on another, from the same start.
                End-to-end event time, median of --repeats; outputs byte-equal to the off path.
"""
import argparse
import json
import math
import statistics

import torch

SHAPES = {"deepseek_v3": dict(h=7168, f=2048, experts=256, topk=8), "qwen35": dict(h=4096, f=1024, experts=512, topk=10)}
TILE = 256


def counts(args, local, gran):
    shape = SHAPES[args.shape]
    total = args.microbatch * shape["topk"]
    g = torch.Generator().manual_seed(args.seed)
    w = torch.exp(torch.randn(local, generator=g))
    raw = (w / w.sum() * total).tolist()
    return [int(x) // gran * gran for x in raw]


def pgmig():
    for line in open("/proc/vmstat"):
        if line.startswith("pgmigrate_success"):
            return int(line.split()[1])
    return -1


def make_problem(args, gemm, local):
    shape = SHAPES[args.shape]
    m, n = (2 * shape["f"], shape["h"]) if gemm == "fc1" else (shape["h"], shape["f"])
    gran = 256 if args.precision == "bf16" else 128
    groups = counts(args, local, gran)
    tokens = sum(groups)
    torch.manual_seed(args.seed)
    p = {
        "kind": args.precision,
        "group_k_list": tuple(groups),
        "experts": local,
        "offsets": torch.tensor(groups, dtype=torch.int32, device="cuda").cumsum(0, dtype=torch.int32),
        "output": torch.empty((local, m, n), dtype=torch.bfloat16, device="cuda"),
    }
    if args.precision == "bf16":
        p["a"] = (torch.randn(m, tokens, device="cuda") * 0.125).to(torch.bfloat16)
        p["b"] = (torch.randn(n, tokens, device="cuda") * 0.125).to(torch.bfloat16).t()
    else:
        p["a"] = (torch.randn(m, tokens, device="cuda") * 0.5).to(torch.float8_e4m3fn)
        p["b"] = (torch.randn(n, tokens, device="cuda") * 0.5).to(torch.float8_e4m3fn).t()
        p["sfa"] = torch.randint(-2, 1, (math.ceil(m / 128) * 128, math.ceil(tokens / 32 / 4) * 4), device="cuda").float().exp2().to(torch.float8_e8m0fnu)
        p["sfb"] = torch.randint(-2, 1, (math.ceil(n / 128) * 128, math.ceil(tokens / 32 / 4) * 4), device="cuda").float().exp2().to(torch.float8_e8m0fnu)
    meta = dict(M=m, N=n, experts=local, tokens=tokens, empty=sum(1 for x in groups if x == 0), min=min(groups), max=max(groups))
    return p, meta


class Producer:
    """One thread publishes signal[order[i]] = order[i] (gpu-scope release) at start + (i + 1) * interval_ns."""

    def __init__(self):
        import cutlass
        import cutlass.cute as cute

        @cute.kernel
        def producer_kernel(signal: cute.Tensor, order: cute.Tensor, tiles: cutlass.Int32, interval_ns: cutlass.Int32):
            if cute.arch.thread_idx()[0] == 0:
                start = cute.arch.globaltimer()
                for i in cutlass.range(0, tiles, 1, unroll=1):
                    due = start + cutlass.Int64(interval_ns) * cutlass.Int64(i + 1)
                    now = cute.arch.globaltimer()
                    while now < due:
                        now = cute.arch.globaltimer()
                    tile = order[i]
                    cute.arch.atomic_exch((signal.iterator + tile).llvm_ptr, tile, sem="release", scope="gpu")

        producer_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)

        @cute.jit
        def launch(signal: cute.Tensor, order: cute.Tensor, tiles: cutlass.Int32, interval_ns: cutlass.Int32, stream):
            producer_kernel(signal, order, tiles, interval_ns).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

        self._launch = launch
        self._compiled = None

    def __call__(self, signal, order, interval_ns, stream):
        import cuda.bindings.driver as cuda
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack

        cu = cuda.CUstream(stream.cuda_stream)
        s = from_dlpack(signal, assumed_align=4).mark_layout_dynamic()
        o = from_dlpack(order, assumed_align=4).mark_layout_dynamic()
        if self._compiled is None:
            self._compiled = cute.compile(self._launch, s, o, cutlass.Int32(1), cutlass.Int32(0), cu)
        self._compiled(s, o, cutlass.Int32(order.numel()), cutlass.Int32(interval_ns), cu)


def tile_orders(problem):
    """expert_major: tiles in token order. round_robin: one tile of each expert in turn (dispatch-like)."""
    tiles = -(-sum(problem["group_k_list"]) // TILE)
    per_expert, begin = [], 0
    for k in problem["group_k_list"]:
        if k:
            per_expert.append(list(range(begin // TILE, -(-(begin + k) // TILE))))
        begin += k
    rr, seen = [], set()
    for j in range(max(len(t) for t in per_expert)):
        for t in per_expert:
            if j < len(t) and t[j] not in seen:
                rr.append(t[j])
                seen.add(t[j])
    rr += [t for t in range(tiles) if t not in seen]
    return {"expert_major": list(range(tiles)), "round_robin": rr}


def base_kernel(args, head_kernel):
    """The same kernel configuration built from the base tree's wgrad module, loaded beside the head's (the modules
    it imports are shared and unchanged), so base and head run in one process."""
    import importlib.util
    import os

    name = "moe_grouped_gemm_wgrad.py" if args.precision == "bf16" else "moe_blockscaled_grouped_gemm_wgrad.py"
    mod_name = "cudnn.gemm.cutedsl.grouped.wgrad._base_" + name[:-3]
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(args.base_wgrad_dir, name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = getattr(module, type(head_kernel).__name__)
    keep = ("acc_dtype", "use_2cta_instrs", "cluster_shape_mn", "accumulate_on_output", "expert_cnt", "weight_mode", "input_order", "atomic_counter_claim")
    kwargs = {k: getattr(head_kernel, k) for k in keep}
    kwargs["mma_tiler_mn"] = head_kernel.mma_tiler[:2]
    if args.precision != "bf16":
        kwargs["sf_vec_size"] = head_kernel.sf_vec_size
    kernel = cls(**kwargs)
    assert not hasattr(kernel, "streaming_signal_capacity") and type(kernel).__module__ == mod_name
    return kernel


def graph_time(launch, args):
    for _ in range(3):
        launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(args.bundle):
            launch()
    return graph


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shape", required=True, choices=SHAPES)
    p.add_argument("--ep", type=int, default=8)
    p.add_argument("--precision", required=True, choices=["bf16", "mxfp8"])
    p.add_argument("--gemm", default="fc1,fc2")
    p.add_argument("--mode", default="ready", choices=["ready", "overlap", "off", "basehead"])
    p.add_argument("--base-wgrad-dir", default=None, help="basehead: the base tree's python/cudnn/gemm/cutedsl/grouped/wgrad")
    p.add_argument("--microbatch", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bundle", type=int, default=10)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--windows", default="0.5,1.0")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    from cutlass.utils import HardwareInfo
    from gemm.cutedsl.test_grouped_gemm_wgrad_streaming_signal_utils import WgradLauncher, capacity_for, reset_signal, signal_view

    local = SHAPES[args.shape]["experts"] // args.ep
    torch.empty(1, device="cuda")  # create the primary context before HardwareInfo queries it
    mac = HardwareInfo().get_max_active_clusters(2)
    rows = []
    for gemm in args.gemm.split(","):
        problem, meta = make_problem(args, gemm, local)
        off = WgradLauncher(args.precision, local, max_active_clusters=mac)
        ws_off = off.workspace()
        out_off = torch.empty_like(problem["output"])
        run_off = lambda: off.run(problem, out_off, ws_off)
        if args.mode == "off":
            graph = graph_time(run_off, args)
            for _ in range(args.warmup):
                graph.replay()
            torch.cuda.synchronize()
            b, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            samples = []
            for _ in range(args.repeats):
                b.record()
                graph.replay()
                e.record()
                e.synchronize()
                samples.append(b.elapsed_time(e) * 1000 / args.bundle)
            row = dict(mode="off", shape=args.shape, ep=args.ep, precision=args.precision, gemm=gemm, **meta, off_us=statistics.median(samples), samples=samples)
            print(f"PERF off {args.shape} ep{args.ep} {args.precision} {gemm} E={local} T={meta['tokens']} off={row['off_us']:.1f}us", flush=True)
            rows.append(row)
            graph.reset()
            continue

        capacity = capacity_for(problem)
        if args.mode in ("ready", "basehead"):
            if args.mode == "ready":
                on = WgradLauncher(args.precision, local, capacity=capacity, max_active_clusters=mac)
                ws_on = on.workspace()
                reset_signal(on, ws_on, all_complete=True)
            else:
                # "off" is the base tree's kernel, "on" the head's, both without the signal: ratio = head / base.
                on = WgradLauncher(args.precision, local, max_active_clusters=mac)
                ws_on = on.workspace()
                off.kernel = base_kernel(args, off.kernel)
                off.key = off.key + ("base",)
                assert ws_off.numel() == max(off.kernel.get_workspace_bytes(), 1)
            out_on = torch.empty_like(problem["output"])
            graphs = {"off": graph_time(run_off, args), "on": graph_time(lambda: on.run(problem, out_on, ws_on), args)}
            same = torch.equal(out_off.view(torch.int16), out_on.view(torch.int16))
            for _ in range(args.warmup):
                for g in graphs.values():
                    g.replay()
            torch.cuda.synchronize()
            samples = {arm: [] for arm in graphs}
            ev = {arm: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for arm in graphs}
            mig0 = pgmig()
            for r in range(args.repeats):
                for arm in (("off", "on") if r % 2 == 0 else ("on", "off")):
                    b, e = ev[arm]
                    b.record()
                    graphs[arm].replay()
                    e.record()
                    e.synchronize()
                    samples[arm].append(b.elapsed_time(e) * 1000 / args.bundle)
            mig1 = pgmig()
            med = {arm: statistics.median(v) for arm, v in samples.items()}
            ratios = sorted(o / f for o, f in zip(samples["on"], samples["off"]))
            row = dict(mode=args.mode, shape=args.shape, ep=args.ep, precision=args.precision, gemm=gemm, **meta, off_us=med["off"], on_us=med["on"],
                       ratio_median=statistics.median(ratios), ratio_q25=ratios[len(ratios) // 4], ratio_q75=ratios[3 * len(ratios) // 4],
                       bytes_equal=same, pgmigrate_delta=mig1 - mig0, samples=samples)
            print(f"PERF {args.mode} {args.shape} ep{args.ep} {args.precision} {gemm} M={meta['M']} N={meta['N']} E={local} T={meta['tokens']} empty={meta['empty']} "
                  f"off={med['off']:.1f}us on={med['on']:.1f}us ratio={row['ratio_median']:.4f} [{row['ratio_q25']:.4f},{row['ratio_q75']:.4f}] "
                  f"bytes_equal={same} pgmig={mig1 - mig0}", flush=True)
            rows.append(row)
            for g in graphs.values():
                g.reset()
            continue

        # overlap
        producer = Producer()
        on = WgradLauncher(args.precision, local, capacity=capacity, max_active_clusters=mac - 1)
        ws_on = on.workspace()
        signal = signal_view(on, ws_on)
        dummy = torch.empty(capacity, dtype=torch.int32, device="cuda")
        out_on = torch.empty_like(problem["output"])
        # GEMM alone (off), eager median, for the window.
        for _ in range(3):
            run_off()
        torch.cuda.synchronize()
        b, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        alone = []
        for _ in range(10):
            b.record()
            run_off()
            e.record()
            e.synchronize()
            alone.append(b.elapsed_time(e) * 1000)
        g_us = statistics.median(alone)
        cur = torch.cuda.current_stream()
        ps, gs = torch.cuda.Stream(), torch.cuda.Stream()
        for order_name, order in tile_orders(problem).items():
            order_t = torch.tensor(order, dtype=torch.int32, device="cuda")
            for window in (float(w) for w in args.windows.split(",")):
                interval = max(1, int(window * g_us * 1000 / len(order)))

                def serial():
                    dummy.fill_(-1)
                    b.record()
                    producer(dummy, order_t, interval, cur)
                    run_off()
                    e.record()

                def stream():
                    reset_signal(on, ws_on, all_complete=False)
                    b.record()
                    ps.wait_stream(cur)
                    gs.wait_stream(cur)
                    producer(signal, order_t, interval, ps)
                    on.run(problem, out_on, ws_on, stream=gs)
                    cur.wait_stream(ps)
                    cur.wait_stream(gs)
                    e.record()

                for f in (serial, stream, serial, stream):
                    f()
                torch.cuda.synchronize()
                samples = {"serial": [], "stream": []}
                for r in range(args.repeats):
                    for arm in (("serial", "stream") if r % 2 == 0 else ("stream", "serial")):
                        (serial if arm == "serial" else stream)()
                        e.synchronize()
                        samples[arm].append(b.elapsed_time(e) * 1000)
                same = torch.equal(out_off.view(torch.int16), out_on.view(torch.int16))
                med = {k: statistics.median(v) for k, v in samples.items()}
                row = dict(mode="overlap", shape=args.shape, ep=args.ep, precision=args.precision, gemm=gemm, **meta, order=order_name, window=window,
                           gemm_alone_us=g_us, producer_us=interval * len(order) / 1000, serial_us=med["serial"], stream_us=med["stream"],
                           speedup=med["serial"] / med["stream"], bytes_equal=same, samples=samples)
                print(f"PERF overlap {args.shape} ep{args.ep} {args.precision} {gemm} E={local} T={meta['tokens']} order={order_name} window={window} "
                      f"G={g_us:.1f}us D={row['producer_us']:.1f}us serial={med['serial']:.1f}us stream={med['stream']:.1f}us "
                      f"speedup={row['speedup']:.3f} bytes_equal={same}", flush=True)
                rows.append(row)
    if args.out:
        json.dump(rows, open(args.out, "w"))


if __name__ == "__main__":
    main()

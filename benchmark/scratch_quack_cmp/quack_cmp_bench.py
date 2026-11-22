# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cuDNN-FE SM90 BF16 grouped GEMMs vs quack (quack-kernels, public gemm_interface only) vs torch._grouped_mm.

One item = (model, op) at one EP rank's sizes, every arm interleaved in one process on one GPU.

Routing (--routing auto): at EP8 / 8192 tokens per GPU, the per-expert real row counts of h100-coordinator's
quack_cmp_1122 f1 run (qcmp.py, bench_data routing, rank 0; qcmp_f1_routing.json next to this script), so the numbers
line up with their EPC-vs-quack cells; otherwise uniform rows (tokens x top_k / experts, every row real), as in our
benchmark/gemm/benchmark_grouped_gemm_sm90.py. Each expert is padded to 256 rows for cuDNN-FE (pad rows zero); quack
and torch run on the real rows, quack_pad on the padded layout (the same work as ours). TFLOPS count real rows only.

Timing (--protocol qcmp, default): their bwd_common.time_one_gpu. Per fire: reset, 2x-L2 float32 flush (add_),
host sync, an 80 ms GPU sleep (start gate) while the host enqueues the arm, then events around the arm alone; arms
rotated rep by rep; 5 runs x (10 warmup + 100 reps); result = median of the 5 run means; a fire whose host enqueue
outlasted the sleep is counted and flagged late. --protocol bench = our benchmarks' protocol (2x-L2 zero_ flush,
no gate, median of run medians).

Arms:
  cudnn          the *_wrapper_sm100 public APIs (SM90 kernels on cc 9.0; default tile 128x256, cluster 2x1, dynamic)
  cudnn_dyn      as cudnn with use_dynamic_sched=True (SM100 fwd / dgrad / glu / glu_c only)
  cudnn_512      as cudnn with mma_tiler_mn=(256, 512), cluster_shape_mn=(2, 1) (SM100 fwd / dgrad / glu / glu_c only)
  cudnn_512_dyn  cudnn_512 with use_dynamic_sched=True
  quack_tile512  as quack_tile pinned to 256x512, cluster 2x1, dynamic persistent (SM100)
  quack          quack public API with its defaults, tuned=True (quack autotunes on the first call), real rows
  quack_dyn      as quack with dynamic_scheduler=True (gemm / gemm_gated); for gemm_dgated (default dynamic) the
                 flipped arm is quack_static (dynamic_scheduler=False)
  quack_pad      as quack on the 256-row padded layout
  quack_default  tuned=False: quack's per-arch default config (SM90: 128x192 pingpong, cluster 2x1, static)
  quack_tile     pinned to our SM90 tile: 128x256 cooperative, cluster 2x1, dynamic persistent
  torch          torch._grouped_mm on the real rows (+ the activation in eager torch for the GLU ops)

  python quack_cmp_bench.py --list-items
  python quack_cmp_bench.py --item deepseek_v3.fc1_glu --ep 8 --out perf.json
  python quack_cmp_bench.py --summarize OUT_DIR       # perf_*.json under OUT_DIR -> OUT_DIR/summary.{md,json}
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import statistics
import sys
import time
import traceback

import torch

#: hidden H, global experts E, top_k, expert intermediate I (FC1 N = 2I, gate + up): HF config.json dims.
MODEL_SHAPES = {
    "deepseek_v3": dict(hidden=7168, experts=256, top_k=8, intermediate=2048),
    "qwen35_122b": dict(hidden=3072, experts=256, top_k=8, intermediate=1024),
    "qwen35_397b": dict(hidden=4096, experts=512, top_k=10, intermediate=1024),
    "qwen38_2p4t": dict(hidden=8192, experts=512, top_k=10, intermediate=2048),
    "glm53_flash": dict(hidden=4096, experts=288, top_k=8, intermediate=2048),
}
OPS = ("fc1_fwd", "fc2_fwd", "fc1_dgrad", "fc2_dgrad", "fc1_glu", "fc1_glu_c", "fc2_dglu", "fc1_wgrad", "fc2_wgrad")
ITEMS = [f"{model}.{op}" for model in MODEL_SHAPES for op in OPS]
QUACK_ARMS = ("quack", "quack_dyn", "quack_static", "quack_pad", "quack_default", "quack_tile")
# name -> (SM100 mma_tiler_mn or None for the default, use_dynamic_sched); explicit tiles use cluster (2, 1)
CUDNN_ARMS = {"cudnn": (None, False), "cudnn_dyn": (None, True), "cudnn_512": ((256, 512), False), "cudnn_512_dyn": ((256, 512), True)}
QUACK_ARMS = QUACK_ARMS + ("quack_tile512",)
ALL_ARMS = tuple(CUDNN_ARMS) + QUACK_ARMS + ("torch",)
DEFAULT_ARMS = ",".join(ALL_ARMS)
BLOCK = 32  # our interleaved_32 gate/up column blocks
ROUTING_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qcmp_f1_routing.json")

SEMANTICS = {
    "fwd": "ours: D = alpha_e * (A @ B_e) * prob (prob = alpha = 1; the multiply still runs); quack gemm(cu_seqlens_m): D = A @ B_e",
    "wgrad": "both: dW_e = A[:, K_e] @ B[K_e]; A m-major, B n-major, out (E, M, N) bf16; quack gemm(cu_seqlens_k)",
    "glu": "ours: D = silu(g) * u * prob, gate/up in 32-column blocks, no C; quack gemm_gated(swiglu, store_preact=False): "
    "postact = silu(g) * u, gate/up column-interleaved, no prob",
    "glu_c": "as glu plus the pre-activation stored: ours generate_c=True (C, 32-col blocks); quack store_preact=True",
    "dglu": "both: D = dSwiGLU(C, prob * dY @ B_e), dprob = rowsum(dY @ B_e * silu(g) * u); ours: dprob by fp32 atomics "
    "(zeroed in the untimed reset), no postact output; quack gemm_dgated(colvec_scale=prob, colvec_reduce=True): dprob by "
    "per-tile partials + a sum inside the call, and also writes postact = silu(g) * u * prob (M x I bf16)",
}


def op_kind(op):
    return {"fc1_glu": "glu", "fc1_glu_c": "glu_c", "fc2_dglu": "dglu", "fc1_wgrad": "wgrad", "fc2_wgrad": "wgrad"}.get(op, "fwd")


def load_routing():
    try:
        with open(ROUTING_FILE) as f:
            return json.load(f)
    except OSError:
        return None


def item_problem(item, ep, tokens, routing="auto", shape_override=None, real_counts=None):
    model, op = item.split(".")
    shape = dict(MODEL_SHAPES[model])
    if shape_override:
        shape.update(shape_override)
    if shape["experts"] % ep:
        raise ValueError(f"{model}: {shape['experts']} experts do not divide over EP {ep}")
    experts = shape["experts"] // ep
    hidden, inter = shape["hidden"], shape["intermediate"]
    data = load_routing()
    if routing == "auto":
        routing = "qcmp" if (ep == 8 and tokens == 8192 and not shape_override and data and model in data["rows_per_expert"]) else "uniform"
    if real_counts:  # smoke runs: explicit real rows per expert
        routing, real = "custom", list(real_counts)
        assert len(real) == experts, (len(real), experts)
        padded = [-(-c // 256) * 256 for c in real]
    elif routing == "qcmp":
        real = [int(c) for c in data["rows_per_expert"][model]]
        assert len(real) == experts, (model, len(real), experts)
        padded = [-(-c // 256) * 256 for c in real]
    else:  # our benchmarks: every expert gets ceil(tokens * top_k / experts) rows, padded to 256, all of them real
        rows = -(-tokens * shape["top_k"] // experts)
        padded = [-(-rows // 256) * 256] * experts
        real = list(padded)
    p = dict(model=model, op=op, kind=op_kind(op), experts=experts, routing=routing, real=real, padded=padded,
             m_real=sum(real), m_padded=sum(padded), shape=shape)
    if p["kind"] == "wgrad":  # benchmark_grouped_gemm_wgrad_sm90.py naming: (m, n) output per expert, K = routed rows
        m, n = (2 * inter, hidden) if op == "fc1_wgrad" else (hidden, inter)
        p.update(m=m, n=n, k=p["m_real"])
        p["flops"] = 2.0 * m * n * p["m_real"]
    else:
        n, k = {
            "fc1_fwd": (2 * inter, hidden), "fc2_fwd": (hidden, inter), "fc1_dgrad": (hidden, 2 * inter),
            "fc2_dgrad": (inter, hidden), "fc1_glu": (2 * inter, hidden), "fc1_glu_c": (2 * inter, hidden),
            "fc2_dglu": (inter, hidden),
        }[op]
        p.update(n=n, k=k, m=p["m_real"], b_major="n" if op in ("fc1_dgrad", "fc2_dgrad", "fc2_dglu") else "k")
        p["flops"] = 2.0 * p["m_real"] * n * k
    return p


# ---------------------------------------------------------------------------- layouts
def split_gate_up(x):
    """(m, 2n) in 32-column gate/up blocks -> (gate, up), each (m, n) in canonical column order."""
    m, two_n = x.shape
    blocks = x.view(m, two_n // (2 * BLOCK), 2, BLOCK)
    return blocks[:, :, 0].reshape(m, -1), blocks[:, :, 1].reshape(m, -1)


def join_i32(gate, up):
    m, n = gate.shape
    return torch.stack((gate.reshape(m, n // BLOCK, BLOCK), up.reshape(m, n // BLOCK, BLOCK)), dim=2).reshape(m, 2 * n)


def join_i1(gate, up):
    """quack's gated layout: gate and up column-interleaved."""
    return torch.stack((gate, up), dim=-1).reshape(gate.shape[0], -1)


def rows_i32_to_i1(w):
    """(E, 2n, K) weights with rows in 32-row gate/up blocks -> rows interleaved one by one (a copy)."""
    e, two_n, k = w.shape
    blocks = w.view(e, two_n // (2 * BLOCK), 2, BLOCK, k)
    gate, up = blocks[:, :, 0].reshape(e, -1, k), blocks[:, :, 1].reshape(e, -1, k)
    return torch.stack((gate, up), dim=2).reshape(e, two_n, k).contiguous()


def torch_swiglu(acc, prob):
    gate, up = split_gate_up(acc.float())
    return (up * torch.nn.functional.silu(gate) * prob[:, None]).to(torch.bfloat16)


def torch_dswiglu(g, c, prob):
    """D (dgate | dup, 32-col blocks) of SwiGLU from g = dY @ W^T and the saved C; alpha = beta = 1."""
    g = g.float()
    x1, x2 = (half.float() for half in split_gate_up(c))
    s = torch.sigmoid(x1)
    return join_i32(g * prob[:, None] * x2 * s * (1 + x1 * (1 - s)), g * prob[:, None] * x1 * s).to(torch.bfloat16)


def diff(x, ref, chunk=1 << 26):
    """max |x - ref| and max |ref|, in fp32 chunks (no full-size fp32 copies: the wgrad outputs reach 8 GiB)."""
    assert x.shape == ref.shape, (tuple(x.shape), tuple(ref.shape))
    x, ref = x.reshape(-1), ref.reshape(-1)
    d = s = 0.0
    for i in range(0, x.numel(), chunk):
        r = ref[i:i + chunk].float()
        d = max(d, (x[i:i + chunk].float() - r).abs().max().item())
        s = max(s, r.abs().max().item())
    return dict(max_abs_diff=d, ref_absmax=s, rel=d / s if s else float("nan"))


# ---------------------------------------------------------------------------- tensors
def make_tensors(p, seed=0):
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)
    real, padded = p["real"], p["padded"]
    cu = lambda counts: torch.tensor([0] + torch.tensor(counts).cumsum(0).tolist(), dtype=torch.int32, device=dev)
    t = dict(cu_r=cu(real), cu_p=cu(padded))
    t["offs_r"], t["offs_p"] = t["cu_r"][1:].clone(), t["cu_p"][1:].clone()
    starts = [0] + torch.tensor(padded).cumsum(0).tolist()[:-1]
    same = real == padded
    t["idx"] = None if same else torch.cat([torch.arange(s, s + c, device=dev) for s, c in zip(starts, real)])
    P = p["m_padded"]
    keep = None
    if not same:
        keep = torch.zeros(P, dtype=torch.bool, device=dev)
        keep[t["idx"]] = True

    def rows(width, scale=0.125):
        """(P, width) padded rows (pad rows zero) and its real rows (the same tensor when nothing is padding)."""
        x = torch.randn((P, width), generator=g, device=dev, dtype=torch.bfloat16) * scale
        if same:
            return x, x
        x[~keep] = 0
        return x, x[t["idx"]].contiguous()

    t["alpha"] = torch.ones(p["experts"], device=dev, dtype=torch.float32)
    if p["kind"] == "wgrad":
        grad_p, grad_r = rows(p["m"])
        act_p, act_r = rows(p["n"])
        t.update(a_p=grad_p.t(), b_p=act_p, a_r=grad_r.t(), b_r=act_r)  # A (m, K) m-major, B (K, n) n-major
        return t
    t["a_p"], t["a_r"] = rows(p["k"])
    t["b"] = torch.randn((p["experts"], p["n"], p["k"]) if p["b_major"] == "k" else (p["experts"], p["k"], p["n"]),
                         generator=g, device=dev, dtype=torch.bfloat16) * 0.125
    t["b_kn"] = t["b"].transpose(1, 2) if p["b_major"] == "k" else t["b"]  # (E, K, N) view for torch and quack
    if p["kind"] == "dglu":
        t["c_p"], t["c_r"] = rows(2 * p["n"], scale=1.0)
        t["ci1_p"] = join_i1(*split_gate_up(t["c_p"])).contiguous()
        t["ci1_r"] = t["ci1_p"] if same else t["ci1_p"][t["idx"]].contiguous()
        prob = 0.05 + 0.95 * torch.rand(P, generator=g, device=dev, dtype=torch.float32)  # as qcmp.py
        if not same:
            prob[~keep] = 0
        t["prob_p"], t["prob_r"] = prob, (prob if same else prob[t["idx"]].contiguous())
        t["dprob"] = torch.zeros(P, device=dev, dtype=torch.float32)
    else:
        t["prob_p"] = torch.ones(P, device=dev, dtype=torch.float32)
        t["prob_r"] = torch.ones(p["m_real"], device=dev, dtype=torch.float32)
    if p["kind"] in ("glu", "glu_c"):
        t["bq_kn"] = rows_i32_to_i1(t["b"]).transpose(1, 2)  # quack gated B: (E, K, 2I) K-major, column-interleaved
    return t


# ---------------------------------------------------------------------------- arms
def quack_tile_config(cap):
    from quack.gemm_config import GemmConfig

    if cap == 9:
        return GemmConfig(tile_m=128, tile_n=256, pingpong=False, cluster_m=2, cluster_n=1, is_dynamic_persistent=True, device_capacity=9)
    # Smoke runs on SM100: our SM100 default tile (256x256, 2-CTA, cluster 2x1).
    return GemmConfig(tile_m=256, tile_n=256, pingpong=False, cluster_m=2, cluster_n=1, is_dynamic_persistent=True, device_capacity=cap)


def limit_quack_tuning(cap, limit):
    """Smoke only: keep the first `limit` configs of this arch in quack's tuning spaces."""
    import quack.gemm_interface as gi
    import quack.gemm_runtime.autotune as at

    gi.gemm_tuned.configs = [c for c in gi.gemm_tuned.configs if c.kwargs["config"].device_capacity == cap][:limit]
    full = at.get_all_configs
    at.get_all_configs = lambda *a, **k: [c for c in full(*a, **k) if c.device_capacity == cap][:limit]


def quack_tuners():
    import quack.gemm_interface as gi
    import quack.gemm_runtime.autotune as at

    return [gi.gemm_tuned, *at._MOD_TUNERS.values()]


def tuner_snapshot():
    try:
        return {(id(tu), str(k)) for tu in quack_tuners() for k in getattr(tu, "cache", {})}
    except Exception:
        return set()


def tuned_since(before):
    """Configs quack's autotuners chose since `before` (the arm's first call)."""
    out = []
    for tu in quack_tuners():
        timings = getattr(tu, "configs_timings", None) or {}
        for k, cfg in getattr(tu, "cache", {}).items():
            if (id(tu), str(k)) in before:
                continue
            top = sorted(((v[0] if isinstance(v, (list, tuple)) else v), str(c)) for c, v in timings.items())[:5]
            out.append(dict(tuner=getattr(getattr(tu, "fn", None), "__name__", "?"), config=str(cfg), top5_ms=top, n_configs=len(timings)))
    return out


def build_arms(p, t, names, cap):
    """name -> (reset, launch, check); check(result) -> diffs vs the torch reference on the real rows."""
    kind, idx = p["kind"], t["idx"]
    sel = (lambda x: x) if idx is None else (lambda x: x[idx])
    noop = lambda: None
    arms, ref = {}, {}

    # torch reference / baseline, real rows (as qcmp.py)
    if kind == "wgrad":
        grouped_mm = lambda: torch._grouped_mm(t["a_r"], t["b_r"], offs=t["offs_r"], out_dtype=torch.bfloat16)
        torch_fn = grouped_mm
    else:
        grouped_mm = lambda: torch._grouped_mm(t["a_r"], t["b_kn"], offs=t["offs_r"], out_dtype=torch.bfloat16)
        if kind in ("glu", "glu_c"):
            torch_fn = lambda: torch_swiglu(grouped_mm(), t["prob_r"])
        elif kind == "dglu":
            torch_fn = lambda: torch_dswiglu(grouped_mm(), t["c_r"], t["prob_r"])
        else:
            torch_fn = grouped_mm
    acc = grouped_mm()
    ref["out"] = torch_fn()
    if kind == "glu_c":
        ref["preact"] = acc
    if kind == "dglu":
        x1, x2 = (h.float() for h in split_gate_up(t["c_r"]))
        act = torch.nn.functional.silu(x1) * x2
        ref["dprob"] = (acc.float() * act).sum(-1)
        ref["postact"] = (act * t["prob_r"][:, None]).to(torch.bfloat16)
        del x1, x2, act
    del acc
    if "torch" in names:
        arms["torch"] = (noop, torch_fn, lambda r: dict(out=diff(r, ref["out"])))

    def cudnn_arm(tile, dyn):
        import cudnn

        tile_kw = {} if tile is None else dict(mma_tiler_mn=tile, cluster_shape_mn=(2, 1))
        if dyn:
            tile_kw["use_dynamic_sched"] = True
        reset = noop
        if kind == "wgrad":
            out = torch.empty((p["experts"], p["m"], p["n"]), device="cuda", dtype=torch.bfloat16)
            fn = lambda: cudnn.grouped_gemm_wgrad_wrapper_sm100(
                a_tensor=t["a_p"], b_tensor=t["b_p"], sfa_tensor=None, sfb_tensor=None, offsets_tensor=t["offs_p"],
                wgrad_tensor=out, acc_dtype=torch.float32, wgrad_dtype=torch.bfloat16,
            )["wgrad_tensor"]
            check = lambda r: dict(out=diff(r, ref["out"]))
        elif kind in ("glu", "glu_c"):
            gen_c = kind == "glu_c"
            fn = lambda: cudnn.grouped_gemm_glu_wrapper_sm100(
                a_tensor=t["a_p"], sfa_tensor=None, padded_offsets=t["offs_p"], alpha_tensor=t["alpha"], b_tensor=t["b"],
                prob_tensor=t["prob_p"], c_dtype=torch.bfloat16, d_dtype=torch.bfloat16, act_func="swiglu", generate_c=gen_c, **tile_kw,
            )

            def check(r):
                res = dict(out=diff(sel(r["d_tensor"].view(p["m_padded"], -1)), ref["out"]))
                if gen_c:
                    res["preact"] = diff(sel(r["c_tensor"].view(p["m_padded"], -1)), ref["preact"])
                return res
        elif kind == "dglu":
            reset = lambda: t["dprob"].zero_()  # the atomic accumulator, zeroed outside the timed window (as qcmp.py)
            fn = lambda: cudnn.grouped_gemm_dglu_wrapper_sm100(
                a_tensor=t["a_p"], c_tensor=t["c_p"], sfa_tensor=None, padded_offsets=t["offs_p"], alpha_tensor=t["alpha"],
                beta_tensor=t["alpha"], prob_tensor=t["prob_p"], dprob_tensor=t["dprob"], b_tensor=t["b"], b_major=p["b_major"],
                d_dtype=torch.bfloat16, act_func="dswiglu",
            )
            check = lambda r: dict(out=diff(sel(r["d_row_tensor"].view(p["m_padded"], -1)), ref["out"]),
                                   dprob=diff(sel(t["dprob"]), ref["dprob"]))
        else:
            d = torch.empty((p["m_padded"], p["n"]), device="cuda", dtype=torch.bfloat16)

            def fn():
                cudnn.grouped_gemm_wrapper_sm100(
                    a_tensor=t["a_p"], padded_offsets=t["offs_p"], alpha_tensor=t["alpha"], b_tensor=t["b"], b_major=p["b_major"],
                    prob_tensor=t["prob_p"], d_tensor=d, c_dtype=torch.bfloat16, d_dtype=torch.bfloat16, **tile_kw,
                )
                return d

            check = lambda r: dict(out=diff(sel(r), ref["out"]))
        return reset, fn, check

    for name, (tile, dyn) in CUDNN_ARMS.items():
        if name in names and (name == "cudnn" or kind in ("fwd", "glu", "glu_c")):
            arms[name] = cudnn_arm(tile, dyn)

    qnames = [n for n in QUACK_ARMS if n in names]
    # gemm_dgated defaults to the dynamic scheduler, so its flipped arm is quack_static; the others flip to quack_dyn
    qnames = [n for n in qnames if n != ("quack_dyn" if kind == "dglu" else "quack_static")]
    if qnames:
        import quack.gemm_interface as qg

        tile = quack_tile_config(cap)
        from quack.gemm_config import GemmConfig

        tile512 = GemmConfig(tile_m=256, tile_n=512, pingpong=False, cluster_m=2, cluster_n=1, is_dynamic_persistent=True, device_capacity=cap)
        opts = {"quack": {}, "quack_pad": {}, "quack_dyn": dict(dynamic_scheduler=True), "quack_static": dict(dynamic_scheduler=False),
                "quack_default": dict(tuned=False), "quack_tile": dict(tuned=False, config=tile), "quack_tile512": dict(tuned=False, config=tile512)}
        P, M = p["m_padded"], p["m_real"]
        bufs = {}
        ident = lambda x: x
        if kind == "wgrad":
            out = torch.empty((p["experts"], p["m"], p["n"]), device="cuda", dtype=torch.bfloat16)

            def make(name):
                a, b, cu = (t["a_p"], t["b_p"], t["cu_p"]) if name == "quack_pad" else (t["a_r"], t["b_r"], t["cu_r"])
                if name == "quack_tile":  # gemm() takes no config: the tuner's undecorated function with it pinned
                    return lambda: (qg.gemm_tuned.fn(a, b, out, cu_seqlens_k=cu, config=tile), out)[1]
                return lambda: qg.gemm(a, b, out=out, cu_seqlens_k=cu, **opts[name])

            check = lambda name: (lambda r: dict(out=diff(r, ref["out"])))
        elif kind in ("glu", "glu_c"):
            store = kind == "glu_c"

            def make(name):
                pad = name == "quack_pad"
                rows, a, cu = (P, t["a_p"], t["cu_p"]) if pad else (M, t["a_r"], t["cu_r"])
                if rows not in bufs:
                    bufs[rows] = (torch.empty((rows, p["n"]), device="cuda", dtype=torch.bfloat16) if store else None,
                                  torch.empty((rows, p["n"] // 2), device="cuda", dtype=torch.bfloat16))
                pre, post = bufs[rows]
                return lambda: qg.gemm_gated(a, t["bq_kn"], activation="swiglu", preact_out=pre, postact_out=post,
                                             cu_seqlens_m=cu, store_preact=store, **opts[name])

            def check(name):
                s = sel if name == "quack_pad" else ident

                def f(r):
                    preact, postact = r
                    res = dict(out=diff(s(postact), ref["out"]))  # prob = 1: ours' reference is quack's too
                    if store:
                        pq = s(preact)
                        res["preact"] = diff(join_i32(pq[:, 0::2], pq[:, 1::2]), ref["preact"])
                    return res
                return f
        elif kind == "dglu":
            def make(name):
                pad = name == "quack_pad"
                rows = P if pad else M
                a, c, prob, cu = (t["a_p"], t["ci1_p"], t["prob_p"], t["cu_p"]) if pad else (t["a_r"], t["ci1_r"], t["prob_r"], t["cu_r"])
                if rows not in bufs:
                    bufs[rows] = (torch.empty((rows, 2 * p["n"]), device="cuda", dtype=torch.bfloat16),
                                  torch.empty((rows, p["n"]), device="cuda", dtype=torch.bfloat16))
                dx, post = bufs[rows]
                return lambda: qg.gemm_dgated(a, t["b_kn"], c, activation="swiglu", dx_out=dx, postact_out=post, colvec_scale=prob,
                                              colvec_reduce=True, cu_seqlens_m=cu, **opts[name])

            def check(name):
                s = sel if name == "quack_pad" else ident

                def f(r):
                    dx, post, dprob = r
                    dx = s(dx)
                    return dict(out=diff(join_i32(dx[:, 0::2], dx[:, 1::2]), ref["out"]), dprob=diff(s(dprob.view(-1)), ref["dprob"]),
                                postact=diff(s(post), ref["postact"]))
                return f
        else:
            def make(name):
                pad = name == "quack_pad"
                rows, a, cu = (P, t["a_p"], t["cu_p"]) if pad else (M, t["a_r"], t["cu_r"])
                if rows not in bufs:
                    bufs[rows] = torch.empty((rows, p["n"]), device="cuda", dtype=torch.bfloat16)
                d = bufs[rows]
                if name in ("quack_tile", "quack_tile512"):
                    cfg = tile if name == "quack_tile" else tile512
                    return lambda: (qg.gemm_tuned.fn(a, t["b_kn"], d, cu_seqlens_m=cu, config=cfg), d)[1]
                return lambda: qg.gemm(a, t["b_kn"], out=d, cu_seqlens_m=cu, **opts[name])

            def check(name):
                s = sel if name == "quack_pad" else ident
                return lambda r: dict(out=diff(s(r), ref["out"]))
        for name in qnames:
            arms[name] = (noop, make(name), check(name))
    return arms, ref


# ---------------------------------------------------------------------------- timing
def _sleep_cycles(sleep_ms, probe_cycles=20_000_000):
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    torch.cuda._sleep(probe_cycles)
    stop.record()
    stop.synchronize()
    return int(probe_cycles * sleep_ms / start.elapsed_time(stop))


def time_qcmp(arms, runs, warmup, reps, gate_ms):
    """h100-coordinator's bwd_common.time_one_gpu on one stream: {name: (reset, launch)} -> {name: result}."""
    stream = torch.cuda.current_stream()
    flush = torch.empty(2 * torch.cuda.get_device_properties(0).L2_cache_size // 4, dtype=torch.float32, device="cuda")
    cycles = _sleep_cycles(gate_ms)
    sleeps = []
    for _ in range(3):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        torch.cuda._sleep(cycles)
        e.record()
        e.synchronize()
        sleeps.append(s.elapsed_time(e) * 1e3)
    sleep_us = min(sleeps)
    names = list(arms)

    def fire(name):
        reset, launch = arms[name]
        reset()
        flush.add_(1.0)
        stream.synchronize()
        torch.cuda._sleep(cycles)  # the gate: the GPU sleeps while the host enqueues the arm
        t0 = time.perf_counter()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        launch()
        e1.record()
        host_us = (time.perf_counter() - t0) * 1e6
        e1.synchronize()
        return e0.elapsed_time(e1) * 1e3, host_us >= sleep_us, host_us

    for name in names:  # prime: ungated, then gated
        arms[name][0]()
        arms[name][1]()
        torch.cuda.synchronize()
    for name in names:
        fire(name)
    samples = {n: [] for n in names}
    late = {n: 0 for n in names}
    host = {n: [] for n in names}
    gc.collect()
    gc.disable()  # no CPython gen-2 collection inside the timed loop
    try:
        for _ in range(runs):
            for _ in range(warmup):
                for name in names:
                    fire(name)
            for rep in range(reps):
                first = rep % len(names)
                for name in names[first:] + names[:first]:
                    us, is_late, host_us = fire(name)
                    samples[name].append(us)
                    late[name] += is_late
                    host[name].append(host_us)
    finally:
        gc.enable()
    out = {}
    for name in names:
        run_means = [statistics.fmean(samples[name][r * reps:(r + 1) * reps]) for r in range(runs)]
        out[name] = dict(us_median=statistics.median(run_means), us_runs=run_means, late_reps=late[name],
                         host_enqueue_us_max=max(host[name]), rep_us_min=min(samples[name]), rep_us_max=max(samples[name]))
    del flush
    return out, dict(protocol="qcmp", gate_sleep_us=sleep_us, gate_ms=gate_ms)


def time_bench(arms, runs, warmup, reps, l2_flush=2.0):
    """Our benchmark_grouped_gemm_sm90.py protocol (resets are no-ops there except the dGLU dprob zero)."""
    flush = torch.empty(int(l2_flush * torch.cuda.get_device_properties(0).L2_cache_size), dtype=torch.uint8, device="cuda")
    for reset, fn in arms.values():
        for _ in range(warmup):
            reset()
            fn()
    torch.cuda.synchronize()
    medians = {name: [] for name in arms}
    gc.collect()
    gc.disable()
    try:
        for _ in range(runs):
            events = {name: [] for name in arms}
            for _ in range(reps):
                for name, (reset, fn) in arms.items():
                    reset()
                    flush.zero_()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    fn()
                    end.record()
                    events[name].append((start, end))
            torch.cuda.synchronize()
            for name, pairs in events.items():
                medians[name].append(statistics.median(s.elapsed_time(e) * 1e3 for s, e in pairs))
    finally:
        gc.enable()
    del flush
    return {n: dict(us_median=statistics.median(v), us_runs=v) for n, v in medians.items()}, dict(protocol="bench", l2_flush=l2_flush)


def env_info():
    info = dict(torch=torch.__version__, device=torch.cuda.get_device_name(), cc=list(torch.cuda.get_device_capability()),
                host=os.uname().nodename, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    for mod in ("cutlass", "quack", "cudnn", "triton", "tvm_ffi"):
        try:
            module = __import__(mod)
            info[mod] = getattr(module, "__version__", "?")
            info[mod + "_file"] = getattr(module, "__file__", None)
            if mod == "cudnn":
                try:
                    info["cudnn_backend"] = module.backend_version()
                except Exception as e:  # stubbed core in local smoke runs
                    info["cudnn_backend"] = f"n/a ({type(e).__name__})"
        except Exception as e:
            info[mod] = f"not importable: {type(e).__name__}: {e}"
    return info


def run_item(args):
    cap = torch.cuda.get_device_capability()[0]
    if cap != 9 and not args.allow_non_sm90:
        raise SystemExit(f"SM90 is required, found {torch.cuda.get_device_capability()} (smoke runs: --allow-non-sm90)")
    override = None
    if args.model_shape:
        h, e, k, i = (int(x) for x in args.model_shape.split(","))
        override = dict(hidden=h, experts=e, top_k=k, intermediate=i)
    counts = [int(x) for x in args.real_counts.split(",")] if args.real_counts else None
    p = item_problem(args.item, args.ep, args.tokens, args.routing, override, counts)
    names = [n for n in args.arms.split(",") if n]
    unknown = set(names) - set(ALL_ARMS)
    if unknown:
        raise SystemExit(f"unknown arms {sorted(unknown)}; known {ALL_ARMS}")
    errors = {}
    if any(n.startswith("quack") for n in names):
        try:
            import quack.gemm_interface  # noqa: F401

            if args.quack_tune_limit:
                limit_quack_tuning(cap, args.quack_tune_limit)
        except Exception as e:
            errors["quack_import"] = f"{type(e).__name__}: {e}"
            names = [n for n in names if not n.startswith("quack")]
    if any(n in CUDNN_ARMS for n in names):
        try:
            import cudnn  # noqa: F401
        except Exception as e:
            errors["cudnn_import"] = f"{type(e).__name__}: {e}"
            names = [n for n in names if n not in CUDNN_ARMS]
    if "torch" not in names:
        names.append("torch")

    t = make_tensors(p)
    built, ref = build_arms(p, t, set(names), cap)
    order = [n for n in ALL_ARMS if n in built]
    arms, checks, first_call_s, tuned = {}, {}, {}, {}
    for name in order:  # first calls (compile, quack autotune) and checks, outside timing; a failing arm is dropped
        reset, fn, check = built[name]
        try:
            reset()
            torch.cuda.synchronize()
            before = tuner_snapshot() if name.startswith("quack") else None
            t0 = time.perf_counter()
            result = fn()
            torch.cuda.synchronize()
            first_call_s[name] = round(time.perf_counter() - t0, 2)
            if before is not None:
                tuned[name] = tuned_since(before)
            checks[name] = check(result)
            del result
            arms[name] = (reset, fn)
        except Exception as e:
            errors[name] = f"{type(e).__name__}: {e}"[:2000] + "\n" + traceback.format_exc()[-3000:]
            print(f"[{args.item}] arm {name} failed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            try:
                torch.cuda.synchronize()
            except Exception as e2:  # sticky device error: nothing more can run in this process
                errors["sticky"] = f"{type(e2).__name__}: {e2}"
                break
    del ref
    torch.cuda.empty_cache()
    timing, proto = {}, {}
    timing_all = {}
    if arms and "sticky" not in errors:
        for protocol in args.protocol.split(","):  # e.g. "qcmp,bench": the gated and the back-to-back protocol on the same arms
            if protocol == "qcmp":
                tm, pr = time_qcmp(arms, args.runs, args.warmup, args.reps, args.gate_ms)
            else:
                tm, pr = time_bench(arms, args.runs, args.warmup, args.reps)
            for value in tm.values():
                value["tflops"] = p["flops"] / (value["us_median"] * 1e-6) / 1e12
            timing_all[protocol] = dict(timing=tm, protocol=pr)
        first = args.protocol.split(",")[0]
        timing, proto = timing_all[first]["timing"], timing_all[first]["protocol"]
    for name, value in timing.items():
        value["tflops"] = p["flops"] / (value["us_median"] * 1e-6) / 1e12
        value["first_call_s"] = first_call_s.get(name)
        value["check"] = checks.get(name)
        if name in tuned:
            value["tuned_configs"] = tuned[name]
    quack_us = {n: v["us_median"] for n, v in timing.items() if n.startswith("quack")}
    ratios = {}
    if "cudnn" in timing:
        for other in list(quack_us) + ["torch"]:
            if other in timing:
                ratios[f"cudnn_over_{other}_time"] = timing["cudnn"]["us_median"] / timing[other]["us_median"]
        if quack_us:
            best = min(quack_us, key=quack_us.get)
            ratios["best_quack"] = best
            ratios["cudnn_over_best_quack_time"] = timing["cudnn"]["us_median"] / quack_us[best]
    if "cudnn_512" in timing:
        for other in ["cudnn"] + list(quack_us):
            if other in timing:
                ratios[f"cudnn_512_over_{other}_time"] = timing["cudnn_512"]["us_median"] / timing[other]["us_median"]
    data = load_routing()
    f1 = data["f1_reference"].get(args.item) if data and p["routing"] == "qcmp" else None
    record = dict(
        item=args.item, ep=args.ep, tokens=args.tokens, problem=p, semantics=SEMANTICS[p["kind"]], timing=timing, ratios=ratios,
        timing_protocol=proto, timing_all=timing_all, quack_tile_config=str(quack_tile_config(cap)) if "quack_tile" in timing else None,
        f1_reference=f1, errors=errors, env=env_info(), peak_mem_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
        args=dict(runs=args.runs, warmup=args.warmup, reps=args.reps, arms=args.arms, protocol=args.protocol, routing=args.routing,
                  gate_ms=args.gate_ms, model_shape=args.model_shape, real_counts=args.real_counts, quack_tune_limit=args.quack_tune_limit),
    )
    text = json.dumps(record, indent=1, default=str)
    print(text)
    if args.out:
        with open(args.out, "w") as file:
            file.write(text + "\n")
    failed = [k for k in errors if k != "quack_import"]
    return 0 if timing and not failed else 3


# ---------------------------------------------------------------------------- summary
def summarize(directory):
    rows = []
    for path in sorted(glob.glob(os.path.join(directory, "**", "perf_*.json"), recursive=True)):
        try:
            with open(path) as f:
                rows.append((path, json.load(f)))
        except Exception as e:
            print(f"skip {path}: {e}", file=sys.stderr)
    rows.sort(key=lambda x: (-x[1]["ep"], x[1]["item"]))
    cols = ("cudnn", "cudnn_dyn", "cudnn_512", "cudnn_512_dyn", "quack", "quack_dyn", "quack_static", "quack_pad", "quack_default", "quack_tile", "quack_tile512", "torch")
    head = (["EP", "item", "routing", "M_real x N x K, E"] + [f"{c} us" for c in cols]
            + ["cudnn TF", "best quack TF", "cudnn/best quack", "cudnn/quack", "cudnn/quack_pad", "cudnn/torch",
               "f1 EPC us", "cudnn/f1 EPC", "max rel diff cudnn; quack", "late reps", "quack tuned config"])
    lines = ["# cuDNN-FE SM90 vs quack vs torch._grouped_mm (BF16)", "",
             "Ratios are time ratios cudnn / other (< 1: cudnn faster). us: median of 5 run means (qcmp protocol) unless noted.",
             "f1 EPC us: h100-coordinator's quack_cmp_1122 f1 'ours' (EPC SM90) arm on the same routing, for cross-checking.", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out = []
    fmt = lambda v, f="{:.1f}": f.format(v) if isinstance(v, (int, float)) else "-"
    for path, r in rows:
        tm, p, rat = r.get("timing", {}), r["problem"], r.get("ratios", {})
        q = {c: tm[c]["us_median"] for c in tm if c.startswith("quack")}
        best = min(q, key=q.get) if q else None

        def rel(c):
            ch = (tm.get(c) or {}).get("check") or {}
            return f"{max(v['rel'] for v in ch.values()):.1e}" if ch else "-"

        f1 = r.get("f1_reference") or {}
        epc = (f1.get("results_us") or {}).get("ours")
        cfgs = (tm.get("quack") or {}).get("tuned_configs") or []
        cfg = "; ".join(c["config"] for c in cfgs).replace("GemmConfig", "").replace("device_capacity=9, ", "")[:140]
        cells = [str(r["ep"]), r["item"], p["routing"], f"{p['m_real']}x{p['n']}x{p['k']}, {p['experts']}"]
        cells += [fmt(tm[c]["us_median"]) if c in tm else "-" for c in cols]
        cells += [fmt((tm.get("cudnn") or {}).get("tflops"), "{:.0f}"), (fmt(tm[best]["tflops"], "{:.0f}") + f" ({best})") if best else "-",
                  fmt(rat.get("cudnn_over_best_quack_time"), "{:.3f}"), fmt(rat.get("cudnn_over_quack_time"), "{:.3f}"),
                  fmt(rat.get("cudnn_over_quack_pad_time"), "{:.3f}"), fmt(rat.get("cudnn_over_torch_time"), "{:.3f}"),
                  fmt(epc), fmt(tm["cudnn"]["us_median"] / epc, "{:.3f}") if epc and "cudnn" in tm else "-",
                  f"{rel('cudnn')}; {rel('quack')}", str(sum(v.get("late_reps", 0) for v in tm.values())), cfg]
        lines.append("| " + " | ".join(cells) + " |")
        if r.get("errors"):
            lines.append(f"| | {r['item']} errors: {', '.join(sorted(r['errors']))} |" + " |" * (len(head) - 2))
        out.append(dict(path=os.path.relpath(path, directory), item=r["item"], ep=r["ep"], routing=p["routing"],
                        us={c: tm[c]["us_median"] for c in tm}, ratios=rat, f1_epc_us=epc,
                        checks={c: tm[c].get("check") for c in tm}, errors=sorted(r.get("errors", {}))))
    lines += ["", "Semantics per op kind:", ""] + [f"- {k}: {v}" for k, v in SEMANTICS.items()]
    with open(os.path.join(directory, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(directory, "summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\n".join(lines))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list-items", action="store_true")
    parser.add_argument("--item", choices=ITEMS)
    parser.add_argument("--ep", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=8192, help="tokens per GPU")
    parser.add_argument("--routing", choices=("auto", "qcmp", "uniform"), default="auto")
    parser.add_argument("--arms", default=DEFAULT_ARMS)
    parser.add_argument("--protocol", default="qcmp", help="qcmp, bench, or a comma list (each timed on the same arms)")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--gate-ms", type=float, default=80.0, help="start-gate GPU sleep per fire (qcmp protocol)")
    parser.add_argument("--out", default=None)
    parser.add_argument("--summarize", metavar="DIR", default=None)
    # smoke-test knobs (not used on H100)
    parser.add_argument("--allow-non-sm90", action="store_true")
    parser.add_argument("--model-shape", default=None, help="H,E,TOPK,I override of the item's model dims")
    parser.add_argument("--real-counts", default=None, help="comma list of real rows per local expert")
    parser.add_argument("--quack-tune-limit", type=int, default=0, help="keep only the first N quack tuning configs")
    args = parser.parse_args(argv)
    if args.list_items:
        print("\n".join(ITEMS))
        return 0
    if args.summarize:
        return summarize(args.summarize)
    if args.item is None:
        parser.error("--item, --list-items or --summarize is required")
    return run_item(args)


if __name__ == "__main__":
    sys.exit(main())

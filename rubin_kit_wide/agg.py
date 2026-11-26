#!/usr/bin/env python3
"""agg.py <jsonl files or dirs>...: tables of the lock-step BF16 wgrad A/B (wgrad_ls.py output).

Per (model, EP, GEMM, claim): median over passes of stock / wide / stock2 us, wide/stock and A/A (stock2/stock) as the
median of per-pass ratios with a bootstrap 95% CI, passes and GPUs, and the dW bytes differing (balanced + cases + API).
Then per GEMM the geomean of wide/stock and A/A, and per shape the dW1 + dW2 time ratio (per pass) and its geomean.
Also lists build / API errors and the smem capacity and A/B stage count each arm compiled with.
"""
import collections
import glob
import json
import math
import os
import random
import statistics
import sys

files = []
for x in sys.argv[1:]:
    files += sorted(glob.glob(os.path.join(x, "*.jsonl.r*"))) if os.path.isdir(x) else [x]
passes = collections.defaultdict(list)
diffs = collections.defaultdict(lambda: collections.Counter())
nan_info = collections.defaultdict(list)
compiles = collections.defaultdict(set)
api = collections.defaultdict(list)
errors = []
meta = set()
for f in files:
    for line in open(f):
        line = line.strip()
        if line.startswith("KP "):
            line = line[3:]
        if not line.startswith("{"):
            continue
        r = json.loads(line)
        key = (r["model"], r["ep"], r["tokens"], r.get("claim", "?"))
        k = r["kind"]
        if k == "pass":
            passes[key].append(r)
        elif k in ("check", "case", "check_after_cases"):
            for f2, v in r.items():
                if f2.endswith("_diff_bytes"):
                    diffs[(key, r["gemm"])][f2[: -len("_diff_bytes")]] += v
            if k == "case":
                nan_info[(key, r["gemm"])].append((r["case"], r["base_nan"]))
        elif k == "compile":
            compiles[(r["arm"].split("_", 1)[1], r.get("smem_capacity"), r.get("num_ab_stage"))].add(key[:2])
        elif k == "api":
            api[(key, r["gemm"])].append(r)
        elif k == "shape":
            meta.add((r.get("gpu_name"), tuple(r.get("cc") or ()), r.get("target_smem"), r.get("sha")))
        elif k.endswith("error"):
            errors.append((key, r.get("gemm") or r.get("arm"), r["error"][-400:]))


def ci(v):
    random.seed(0)
    if len(v) < 3:
        return float("nan"), float("nan")
    bs = sorted(statistics.median(random.choices(v, k=len(v))) for _ in range(1000))
    return bs[25], bs[974]


def geo(v):
    return math.exp(statistics.mean(math.log(x) for x in v)) if v else float("nan")


print("GPUs / cc / target smem / sha:", sorted(meta, key=str))
print("compiled (arm, smem_capacity, num_ab_stage):", sorted((a, s, n) for a, s, n in compiles))
seen_err = collections.Counter()
for key, where, err in errors:
    seen_err[(where, err.strip().splitlines()[-1][:300])] += 1
for (where, last), n in sorted(seen_err.items()):
    print(f"ERROR x{n} {where}: {last}")
order = ["deepseek_v3", "qwen35_122b", "qwen35_397b", "glm53_flash", "qwen38_2p4t"]
keys = sorted(passes, key=lambda k: (k[3], k[1], order.index(k[0]) if k[0] in order else 9, k[2]))
for claim in sorted({k[3] for k in keys}):
    summary = {}
    for gemm in ("dW1", "dW2"):
        print(f"\n#### BF16 {gemm}, {claim}: wide vs stock (both sized for the target's shared memory)\n")
        extra = sorted({a.split("_", 1)[1] for k in keys for p in passes[k] for a in p["us"] if a.startswith(gemm)} - {"stock", "wide", "stock2"})
        hdr = "| Model | EP | Tokens/rank | stock (us) | wide (us) | wide / stock [95% CI] | A/A [95% CI] |"
        hdr += "".join(f" {x} / stock |" for x in extra) + " passes (GPUs) | dW bytes differing (wide, A/A) | API picks wide (on, off) |"
        print(hdr)
        print("|" + " --- |" * (hdr.count("|") - 1))
        rw, ra = [], []
        for key in [k for k in keys if k[3] == claim]:
            ps = [p for p in passes[key] if f"{gemm}_stock" in p["us"]]
            if not ps:
                continue
            us = {a: statistics.median(p["us"][f"{gemm}_{a}"] for p in ps) for a in ("stock", "wide")}
            w = [p["ratios"][f"{gemm}_wide/{gemm}_stock"] for p in ps]
            aa = [p["ratios"][f"{gemm}_stock2/{gemm}_stock"] for p in ps]
            ex = [statistics.median(p["ratios"][f"{gemm}_{x}/{gemm}_stock"] for p in ps) for x in extra]
            d = diffs.get((key, gemm), {})
            ap = api.get((key, gemm), [])
            apis = ",".join(f"{r.get('use_wide_True_picked_wide')}/{r.get('use_wide_False_picked_wide')} "
                            f"diff {r.get('use_wide_True_diff_bytes')}/{r.get('use_wide_False_diff_bytes')}" for r in ap[:1]) or "-"
            gpus = len({p["host"] + str(p["gpu"]) for p in ps})
            cw, ca = ci(w), ci(aa)
            rw.append(statistics.median(w))
            ra.append(statistics.median(aa))
            print(f"| {key[0]} | {key[1]} | {key[2]} | {us['stock']:.1f} | {us['wide']:.1f} | {rw[-1]:.3f} [{cw[0]:.3f}, {cw[1]:.3f}] | "
                  f"{ra[-1]:.3f} [{ca[0]:.3f}, {ca[1]:.3f}] |" + "".join(f" {x:.3f} |" for x in ex)
                  + f" {len(ps)} ({gpus}) | {d.get('wide', '-')}, {d.get('stock2', '-')} | {apis} |")
        print(f"| **geomean** | | | | | **{geo(rw):.3f}** | {geo(ra):.3f} |" + " |" * len(extra) + " | | |")
        summary[gemm] = (geo(rw), geo(ra), min(rw, default=float("nan")), max(rw, default=float("nan")), len(rw))
    both, both_aa = [], []
    for key in [k for k in keys if k[3] == claim]:
        ps = [p for p in passes[key] if "dW1_stock" in p["us"] and "dW2_stock" in p["us"]]
        if ps:
            both.append(statistics.median((p["us"]["dW1_wide"] + p["us"]["dW2_wide"]) / (p["us"]["dW1_stock"] + p["us"]["dW2_stock"]) for p in ps))
            both_aa.append(statistics.median((p["us"]["dW1_stock2"] + p["us"]["dW2_stock2"]) / (p["us"]["dW1_stock"] + p["us"]["dW2_stock"]) for p in ps))
    print(f"\n| claim | GEMM | geomean wide/stock | min | max | geomean A/A | cells |\n| --- | --- | --- | --- | --- | --- | --- |")
    for gemm, (g, ga, lo, hi, n) in summary.items():
        print(f"| {claim} | {gemm} | {g:.3f} | {lo:.3f} | {hi:.3f} | {ga:.3f} | {n} |")
    if both:
        print(f"| {claim} | **dW1 + dW2 time** | **{geo(both):.3f}** | {min(both):.3f} | {max(both):.3f} | {geo(both_aa):.3f} | {len(both)} |")
tot = collections.Counter()
for v in diffs.values():
    tot.update(v)
print("\nbyte checks, total differing dW bytes over every shape / GEMM / case:", dict(tot), "records", len(diffs))
cases_nan = collections.Counter()
for v in nan_info.values():
    for c, nn in v:
        cases_nan[c] += nn
print("stock NaN elements per case (outputs are NaN-filled before each launch):", dict(cases_nan))

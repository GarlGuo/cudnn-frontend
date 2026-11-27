#!/usr/bin/env python3
"""agg_rule.py <files or dirs>...: tables of rule_ls.py output (lines may carry a "KP " prefix).

Per (GEMM, claim): per process, the median over passes of the per-pass ratio x/stock (x = wide, stock2, extras); then the
median over processes with the min and max, the number of processes and passes, the waves (wide tiles / clusters) and
the dW bytes that differ from stock (balanced + cases + API). Option --csv FILE writes one row per (GEMM, claim, arm).
Option --fast host:gpu,host:gpu,... (or a file with one host:gpu per line) keeps only those GPUs' processes for the
tables. Always printed first: per physical GPU (host:gpu), its stock speed index (geomean over GEMMs of its stock time /
the median stock time of all GPUs on that GEMM and claim; 1.00 = typical, larger = slower) and its wide/stock geomean.
"""
import collections
import glob
import json
import math
import os
import statistics
import sys

args = [x for x in sys.argv[1:] if not x.startswith("--")]
csv_out = None
if "--csv" in sys.argv:
    csv_out = sys.argv[sys.argv.index("--csv") + 1]
    args = [x for x in args if x != csv_out]
fast = None
if "--fast" in sys.argv:
    spec = sys.argv[sys.argv.index("--fast") + 1]
    args = [x for x in args if x != spec]
    items = open(spec).read().split() if os.path.isfile(spec) else spec.split(",")
    fast = {tuple(x.rsplit(":", 1)) for x in items if x}
gpu_stock = collections.defaultdict(dict)  # (gemm, claim) -> (host, gpu) -> [stock us medians]
gpu_ratio = collections.defaultdict(list)  # (host, gpu) -> [wide/stock per process and GEMM]
files = []
for x in args:
    files += sorted(glob.glob(os.path.join(x, "**", "*"), recursive=True)) if os.path.isdir(x) else [x]
proc_ratios = collections.defaultdict(lambda: collections.defaultdict(list))  # (gemm, claim) -> arm -> [per-proc med]
proc_us = collections.defaultdict(lambda: collections.defaultdict(list))
npass = collections.Counter()
info = {}
diffs = collections.defaultdict(collections.Counter)
api = collections.defaultdict(list)
errors = []
meta = collections.Counter()
compiles = collections.Counter()
for f in files:
    if not os.path.isfile(f):
        continue
    per_proc = collections.defaultdict(lambda: collections.defaultdict(list))
    per_proc_us = collections.defaultdict(lambda: collections.defaultdict(list))
    for line in open(f, errors="replace"):
        line = line.strip()
        if line.startswith("KP "):
            line = line[3:]
        if not line.startswith("{"):
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        k = r.get("kind")
        proc = (r.get("host"), r.get("gpu"), r.get("pid"), r.get("entry"))
        claim = r.get("claim")
        if k == "meta":
            meta[(r.get("gpu_name"), tuple(r.get("cc") or ()), r.get("clusters"), r.get("sms"), r.get("target_smem"), r.get("sha"))] += 1
        elif k == "check":
            info[r["gemm"]] = (r["m"], r["n"], r["K"], r["E"], r["wide_tiles"], r["waves"])
        if k in ("check", "case", "check_after_cases"):
            for key, v in r.items():
                if key.endswith("_diff_bytes"):
                    diffs[(r["gemm"], claim)][key[: -len("_diff_bytes")]] += v
        elif k == "api":
            api[(r["gemm"], claim)].append((tuple(r["picked_wide"]), r["diff_bytes"]))
        elif k == "compile":
            compiles[(r["arm"], r.get("smem_capacity"), r.get("num_ab_stage"))] += 1
        elif k and k.endswith("error"):
            errors.append((r.get("gemm"), r.get("arm"), r["error"].strip().splitlines()[-1][:300]))
        elif k == "pass":
            for pair, v in r["ratios"].items():
                x, _ = pair.split("/")
                gemm, arm = x.rsplit("_", 1)
                per_proc[(proc, gemm, claim)][arm].append(v)
            for name, us in r["us"].items():
                gemm, arm = name.rsplit("_", 1)
                per_proc_us[(proc, gemm, claim)][arm].append(us)
    for (proc, gemm, claim), arms in per_proc.items():
        if "wide" in arms:
            gpu_ratio[(proc[0], proc[1])].append(statistics.median(arms["wide"]))
        if fast is not None and (proc[0], proc[1]) not in fast:
            continue
        for arm, v in arms.items():
            proc_ratios[(gemm, claim)][arm].append(statistics.median(v))
        npass[(gemm, claim)] += len(next(iter(arms.values())))
    for (proc, gemm, claim), arms in per_proc_us.items():
        if "stock" in arms:
            gpu_stock[(gemm, claim)].setdefault((proc[0], proc[1]), []).append(statistics.median(arms["stock"]))
        if fast is not None and (proc[0], proc[1]) not in fast:
            continue
        for arm, v in arms.items():
            proc_us[(gemm, claim)][arm].append(statistics.median(v))

speed = collections.defaultdict(list)
for key, per_gpu in gpu_stock.items():
    med = statistics.median([statistics.median(v) for v in per_gpu.values()])
    for g, v in per_gpu.items():
        speed[g].append(statistics.median(v) / med)
print("per GPU (host:gpu): stock speed index (geomean; 1.00 = median GPU, larger = slower), wide/stock geomean, GEMMs"
      + ("; * = in --fast" if fast is not None else ""))
for g in sorted(speed, key=lambda g: math.exp(statistics.mean(math.log(x) for x in speed[g]))):
    gi = math.exp(statistics.mean(math.log(x) for x in speed[g]))
    gr = math.exp(statistics.mean(math.log(x) for x in gpu_ratio[g])) if gpu_ratio[g] else float("nan")
    print(f"  {g[0]}:{g[1]} {gi:.3f} {gr:.3f} {len(speed[g])}" + (" *" if fast is not None and g in fast else ""))
print("meta (gpu, cc, clusters, sms, target smem, sha): count")
for k, v in sorted(meta.items(), key=str):
    print("  ", k, v)
print("compiled (arm, smem, stages):", sorted(compiles))
for e, c in collections.Counter(errors).items():
    print("ERROR", c, e)


def fmt(v):
    return f"{statistics.median(v):.3f} ({min(v):.3f}-{max(v):.3f})" if v else "-"


rows = []
for claim in sorted({c for _, c in proc_ratios}):
    arms_all = sorted({a for (g, c), d in proc_ratios.items() if c == claim for a in d}, key=lambda a: (a != "wide", a))
    print(f"\n#### {claim}\n")
    print("| GEMM (m x n, K, E) | wide tiles | waves | stock us | " + " | ".join(f"{a}/stock" for a in arms_all) + " | procs | passes | diff bytes | API picked wide |")
    print("|" + " --- |" * (6 + len(arms_all) + 2))
    keys = sorted((g for g, c in proc_ratios if c == claim), key=lambda g: (info.get(g, (0,) * 6)[5], g))
    for g in keys:
        m, n, K, E, tw, w = info.get(g, (0, 0, 0, 0, 0, float("nan")))
        d = proc_ratios[(g, claim)]
        su = proc_us[(g, claim)].get("stock", [])
        picks = collections.Counter(p for p, _ in api.get((g, claim), []))
        api_diff = sum(x for _, x in api.get((g, claim), []))
        dsum = sum(diffs[(g, claim)].values()) + api_diff
        print(f"| {m}x{n}, K {K}, E {E} | {tw} | {w:.1f} | {statistics.median(su):.0f} | " + " | ".join(fmt(d.get(a, [])) for a in arms_all)
              + f" | {len(d.get('wide', []))} | {npass[(g, claim)]} | {dsum} | {dict(picks)} |")
        for a in arms_all:
            v = d.get(a, [])
            if v:
                rows.append((g, m, n, K, E, tw, round(w, 2), claim, a, statistics.median(v), min(v), max(v), len(v)))
if csv_out:
    with open(csv_out, "w") as f:
        f.write("gemm,m,n,K,E,wide_tiles,waves,claim,arm,median,min,max,procs\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")

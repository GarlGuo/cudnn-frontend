#!/usr/bin/env python3
"""Tables from lsw_rubin.py output: python agg_rubin.py <files with one JSON record per line (optionally "KP "-prefixed)>
Cell = median over passes of the per-pass ratio (95% CI: bootstrap over passes). "2.a+2.b (EPC rule)": reuse where the
rule picks it, else the fallback arm; the refined rule also needs more than one work unit per M tile (n / 256 > n_group)."""
import sys, json, math, random, statistics as st, collections

MORDER = ["deepseek_v3", "qwen35_122b", "qwen35_397b", "glm53_flash", "qwen38_2p4t"]
passes, rules, checks, per, meta, errs = collections.defaultdict(list), {}, [], {}, collections.defaultdict(set), []
for f in sys.argv[1:]:
    counter = {}
    for line in open(f):
        line = line.strip()
        if line.startswith("KP "):
            line = line[3:]
        if not line.startswith("{"):
            continue
        r = json.loads(line)
        k = (r["model"], r["ep"])
        if r["kind"] == "shape":
            counter[(r["host"], r["gpu"])] = r["counter"]
            per[k] = (r["per_expert"], r.get("reuse_a_stages"))
        c = counter.get((r["host"], r["gpu"]))
        if r["kind"] == "rule":
            rules[(*k, r["gemm"])] = r
        elif r["kind"] == "check":
            checks.append(r)
        elif r["kind"] in ("build_error",):
            errs.append(r)
        elif r["kind"] == "pass":
            passes[(c, *k)].append(r["us"])
            meta[(c, *k)].add((r["host"], r["gpu"]))


def gm(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def boot(v, n=1000):
    rnd = random.Random(0)
    ms = sorted(st.median(rnd.choices(v, k=len(v))) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n)]


def t(p, k, g, arm):
    if arm.startswith("auto"):
        rule, fb = arm.split("_")[1:]  # auto_<epc|ref>_<fallback arm>
        pick = rules[(*k, g)]["epc_reuse" if rule == "epc" else "refined_reuse"]
        return p[f"{g}_reuse"] if pick else p[f"{g}_{fb}"]
    return p[f"{g}_{arm}"]


def ratio(c, k, g, x, y):
    ps = passes[(c, *k)]
    gs = ("dW1", "dW2") if g == "layer" else (g,)
    rs = [sum(t(p, k, h, x) for h in gs) / sum(t(p, k, h, y) for h in gs) for p in ps]
    return st.median(rs), boot(rs), st.median(sum(t(p, k, h, y) for h in gs) for p in ps), len(ps)


print(f"checks: {len(checks)} records; build errors: {len(errs)}")
for r in errs:
    print("BUILD_ERROR", r["model"], r["ep"], r["arm"], r["error"][-300:])
for r in checks:
    # dense and stock_sm100 vs the Rubin stock may differ (SM100 K32 vs Rubin K64 MMA); dense vs stock_sm100 may not.
    bad = {x: v for x, v in r.items() if (x.endswith("_diff_bytes") and x not in ("dense_diff_bytes", "stock_sm100_diff_bytes")) or x.endswith("_nan")}
    flag = "OK" if not any(bad.values()) else "DIFF"
    print(f"CHECK {flag} {r['model']} EP{r['ep']} {r['gemm']} " + " ".join(f"{x}={v}" for x, v in r.items() if "diff" in x or "nan" in x or "err" in x))
shapes = sorted({(m, e) for (_, m, e) in passes}, key=lambda s: (s[1], MORDER.index(s[0])))
arms_avail = set(next(iter(passes.values()))[0]) if passes else set()
has = lambda a: f"dW1_{a}" in arms_avail
for c in sorted({c for (c, _, _) in passes}):
    print(f"\n### {'atomic counter' if c else 'CLC'}\n")
    cols = [("dense", "stock"), ("reuse", "stock"), ("reuse", "dense"), ("auto_epc_stock", "stock"), ("auto_ref_stock", "stock"),
            ("auto_epc_dense", "dense"), ("auto_ref_dense", "dense"), ("stock2", "stock"), ("stock_sm100", "stock")]
    cols = [(x, y) for x, y in cols if all(has(a.split("_")[-1] if a.startswith("auto") else a) for a in (x, y)) and (not x.startswith("auto") or has("reuse"))]
    print("| Model | EP | GEMM | rows/expert (A stages) | EPC rule | refined | stock (us) | " + " | ".join(f"{x} / {y}" for x, y in cols) + " | passes (GPUs) |")
    print("|" + " --- |" * (7 + len(cols) + 1))
    agg = collections.defaultdict(list)
    for g in ("dW1", "dW2", "layer"):
        for k in shapes:
            if g == "layer":
                ru = "/".join("R" if rules[(*k, h)]["epc_reuse"] else "-" for h in ("dW1", "dW2"))
                rf = "/".join("R" if rules[(*k, h)]["refined_reuse"] else "-" for h in ("dW1", "dW2"))
            else:
                ru = "reuse" if rules[(*k, g)]["epc_reuse"] else "-"
                rf = "reuse" if rules[(*k, g)]["refined_reuse"] else "-"
            cells = []
            for x, y in cols:
                r, ci, uy, n = ratio(c, k, g, x, y)
                agg[(g, x, y)].append(r)
                cells.append(f"{r:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
            uy = ratio(c, k, g, "stock", "stock")[2]
            print(f"| {k[0]} | {k[1]} | {'dW1 + dW2' if g == 'layer' else g} | {per[k][0]} ({per[k][1]}) | {ru} | {rf} | {uy:.1f} | " + " | ".join(cells) + f" | {n} ({len(meta[(c, *k)])}) |")
    print("\n| Geomean | " + " | ".join(f"{x} / {y}" for x, y in cols) + " |")
    print("|" + " --- |" * (len(cols) + 1))
    for g in ("dW1", "dW2", "layer"):
        print(f"| {g} | " + " | ".join(f"{gm(agg[(g, x, y)]):.3f}" for x, y in cols) + " |")
    print("| all 20 GEMMs | " + " | ".join(f"{gm(agg[('dW1', x, y)] + agg[('dW2', x, y)]):.3f}" for x, y in cols) + " |")

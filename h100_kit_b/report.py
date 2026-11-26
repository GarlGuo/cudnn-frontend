#!/usr/bin/env python3
"""report.py [DIR ...]: one text report of the kit-b packages (s59sm90regress, s59sm90hangrepro, s59sm90perf).

DIR: any directory holding job output dirs (each with a status.txt), searched recursively; default
$HOME/cudnnfe_sm90/s59sm90regress, .../s59sm90hangrepro, .../s59sm90perf (job.sh copies every job's JOB_OUT to
<W>/results/<job>_a<attempt>/).  Results of several hosts: copy their <W>/results dirs under one directory (any layout)
and pass it, or run this on every host and send every output.  Python 3 standard library only; reads, never writes.

Prints, per package: jobs expected / with a result / by status, the ids still missing, and
  regress: junit cases per test file (L0 / L1 split by the on-host collection), skips with reasons, strict determinism,
           sanitizer ids and their SUMMARY lines;
  hangrepro: outcomes and launch totals per (tree, mode), bad checksums, STALL lines, fh / replica / sanitizer results;
  perf: per cell time(644492a92) / time(228f92bab) and A/A = time(ctl) / time(base) (per-pass ratios, ABBA blocks of two
        passes averaged in log space, 95% CI over blocks), geomeans per mode;
and, for every job that is not PASS, its status, summary and first failure lines (fail_head.txt).
"""
import collections, glob, json, math, os, statistics, sys
import xml.etree.ElementTree as ET

KIT = os.path.dirname(os.path.abspath(__file__))
RUNS = ("s59sm90regress", "s59sm90hangrepro", "s59sm90perf")
home = os.path.expanduser("~")
dirs = sys.argv[1:] or [os.path.join(home, "cudnnfe_sm90", r) for r in RUNS]


def status_of(d):
    lines = open(os.path.join(d, "status.txt"), errors="replace").read().splitlines()
    kv = {}
    for ln in lines[1:]:
        if "=" in ln:
            k, v = ln.split("=", 1)
            kv.setdefault(k, v)
    kv["status"] = lines[0].strip() if lines else "?"
    kv["dir"] = d
    return kv


# every job dir; per (run, job id) the latest attempt (by end=)
found = collections.defaultdict(list)
for root in dirs:
    for st in glob.glob(os.path.join(root, "**", "status.txt"), recursive=True):
        try:
            kv = status_of(os.path.dirname(st))
        except OSError:
            continue
        if kv.get("run") in RUNS:
            found[(kv["run"], kv.get("job_id", "?"))].append(kv)
latest = {}
for k, v in found.items():
    v.sort(key=lambda x: (x.get("end", ""), x.get("attempt", "")))
    seen, uniq = set(), []
    for x in v:  # the same attempt in job_out/ and results/: count once
        key = (x.get("attempt"), x.get("end"))
        if key not in seen:
            seen.add(key); uniq.append(x)
    latest[k] = dict(uniq[-1], n_attempts=len(uniq))


def expected(run):
    p = os.path.join(KIT, run, "jobs.jsonl")
    return [json.loads(l)["id"] for l in open(p)] if os.path.exists(p) else []


def header(run, title):
    exp = expected(run)
    got = {j: latest[(run, j)] for (r, j) in latest if r == run}
    c = collections.Counter(x["status"] for x in got.values())
    print(f"\n## {run}: {title}")
    print(f"jobs: {len(exp)} expected, {len(got)} with a result: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))
    miss = [j for j in exp if j not in got]
    if miss:
        print(f"no result yet ({len(miss)}): {' '.join(miss[:60])}{' ...' if len(miss) > 60 else ''}")
    retried = [f"{j}(x{x['n_attempts']})" for j, x in sorted(got.items()) if x["n_attempts"] > 1]
    if retried:
        print(f"jobs with more than one attempt: {' '.join(retried)}")
    return got


bad_jobs = []

# ---------------------------------------------------------------------------------------------- regress (a) + (c)
got = header("s59sm90regress", "SM90 regression of 644492a92 (L0+L1 shards, strict determinism, sanitizers on the streaming signal files)")
if got:
    l0 = set()
    for x in got.values():
        p = os.path.join(x["dir"], "collected_sig.txt")
        if os.path.exists(p):
            l0 |= {l.strip() for l in open(p) if l.strip()}
    per = collections.defaultdict(lambda: collections.Counter())
    skips = collections.Counter()
    for j, x in sorted(got.items()):
        if x["status"] != "PASS":
            bad_jobs.append(("s59sm90regress", j, x))
        if ".fshard." not in j:
            continue
        p = os.path.join(x["dir"], "junit.xml")
        if not os.path.exists(p):
            continue
        try:
            for tc in ET.parse(p).getroot().iter("testcase"):
                cls, name = tc.get("classname", ""), tc.get("name", "")
                f = cls.replace(".", "/") + ".py"
                tid = f"{f}::{name}"
                lvl = "L0" if tid in l0 else "L1"
                if tc.find("skipped") is not None:
                    r = "skip"; skips[f"{os.path.basename(f)}: {(tc.find('skipped').get('message') or '').strip()[:160]}"] += 1
                elif tc.find("failure") is not None or tc.find("error") is not None:
                    r = "fail"
                else:
                    r = "pass"
                per[os.path.basename(f)][lvl] += 1
                per[os.path.basename(f)][r] += 1
        except ET.ParseError as e:
            print(f"junit parse error {j}: {e}")
    tot = collections.Counter()
    print("\nL0+L1 shards (sig.fshard.*), junit cases per file:")
    print("| file | L0 | L1 | pass | fail | skip |")
    print("|---|---:|---:|---:|---:|---:|")
    for f in sorted(per):
        c = per[f]; tot.update(c)
        print(f"| {f} | {c['L0']} | {c['L1']} | {c['pass']} | {c['fail']} | {c['skip']} |")
    print(f"| total | {tot['L0']} | {tot['L1']} | {tot['pass']} | {tot['fail']} | {tot['skip']} |")
    nfull = 0
    for x in got.values():
        p = os.path.join(x["dir"], "collected_full_sig.txt")
        if os.path.exists(p):
            nfull = sum(1 for l in open(p) if l.strip()); break
    print(f"collected on the host: {len(l0)} L0 ids, {nfull} L0+L1 ids")
    for s, n in sorted(skips.items()):
        print(f"skip x{n}: {s}")
    print("\nstrict determinism (CUDNN_DETERMINISM_GOLDENS set):")
    for j, x in sorted(got.items()):
        if ".strict." in j:
            print(f"- {j.split('.strict.')[1]}: {x['status']}: {x.get('summary', '')}")
    for tool in ("memcheck", "racecheck"):
        rows = []
        for j, x in sorted(got.items()):
            if f".{tool}." in j:
                p = os.path.join(x["dir"], "san_ids.tsv")
                if os.path.exists(p):
                    rows += [l.rstrip("\n").split("\t") for l in open(p) if l.strip()]
        if not rows and not any(f".{tool}." in j for j in got):
            continue
        sl = collections.Counter(r[5] if len(r) > 5 else "-" for r in rows)
        files = collections.Counter(os.path.basename(r[3].split("::")[0]) for r in rows if len(r) > 3)
        npass = sum(1 for r in rows if r[0] == "PASS")
        print(f"\n{tool} (L0, not warm_stress, streaming signal files): {len(rows)} ids, {npass} PASS; per file "
              + ", ".join(f"{f} {n}" for f, n in sorted(files.items())))
        for s, n in sorted(sl.items()):
            print(f"  x{n}: {s}")
        for r in rows:
            if r[0] != "PASS":
                print(f"  NOT PASS: {' | '.join(r[:5])}")

# ---------------------------------------------------------------------------------------------- hang repro (b)
got = header("s59sm90hangrepro", "58-case hang repro, cell dglu / deepseek_v3 / EP4 / T4096 (base 644492a92, old 228f92bab)")
if got:
    agg = collections.defaultdict(lambda: dict(n=0, ok=0, launches=0, bad=0, outcomes=collections.Counter()))
    for j, x in sorted(got.items()):
        if x["status"] != "PASS":
            bad_jobs.append(("s59sm90hangrepro", j, x))
        parts = j.split(".")
        if parts[0] in ("solo", "fh"):
            key = f"{parts[0]} {parts[1]} {parts[2]}.{parts[3]}" if parts[2].startswith("delay") else f"{parts[0]} {parts[1]} {parts[2]}"
            a = agg[key]; a["n"] += 1; a["ok"] += x["status"] == "PASS"
            a["launches"] += int(x.get("launches", "0") or 0); a["bad"] += int(x.get("bad", "0") or 0)
            a["outcomes"][x.get("outcome", "?")] += 1
    if agg:
        print("\n| jobs | n | PASS | outcomes | launches | bad checksums |")
        print("|---|---:|---:|---|---:|---:|")
        for k in sorted(agg):
            a = agg[k]
            print(f"| {k} | {a['n']} | {a['ok']} | {', '.join(f'{o} {n}' for o, n in sorted(a['outcomes'].items()))} | "
                  f"{a['launches'] / 1e6:.2f}M | {a['bad']} |")
    for j, x in sorted(got.items()):
        if j.startswith("replica") or j.startswith("san"):
            print(f"- {j}: {x['status']}: outcome={x.get('outcome', '')} {x.get('summary', '')[:300]}")
    stalls = [f"{j}: {v}" for j, x in sorted(got.items()) for k, v in x.items() if k.startswith("stall")]
    print(f"STALL / STALL_EXT lines: {len(stalls)}")
    for s in stalls[:10]:
        print(f"  {s[:300]}")

# ---------------------------------------------------------------------------------------------- perf (d)
got = header("s59sm90perf", "short dGLU perf replicate, 644492a92 (base) vs 228f92bab (old), ctl = base again (A/A)")
T975 = lambda n: 1.96 if n > 60 else {1: 12.7, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 10: 2.23, 20: 2.09, 30: 2.04}.get(n, 2.0 + 2.0 / max(n, 1))


def blocks(passes, a, b):
    pr = {p: math.log(r[a] / r[b]) for p, r in passes.items() if a in r and b in r and r[a] > 0 and r[b] > 0}
    out = []
    for jj in sorted({p // 2 for p in pr}):
        v = [pr[p] for p in (2 * jj, 2 * jj + 1) if p in pr]
        if len(v) == 2:
            out.append(sum(v) / 2)
    return out


def est(bl):
    if len(bl) < 2:
        return None
    m, sd = statistics.fmean(bl), statistics.stdev(bl)
    h = T975(len(bl) - 1) * sd / math.sqrt(len(bl))
    return math.exp(m), math.exp(m - h), math.exp(m + h), len(bl)


fmt = lambda e: "-" if e is None else f"{e[0]:.4f} [{e[1]:.4f}, {e[2]:.4f}]"
if got:
    rows = []
    for j, x in sorted(got.items()):
        if x["status"] != "PASS":
            bad_jobs.append(("s59sm90perf", j, x))
            continue
        samples = collections.defaultdict(lambda: collections.defaultdict(dict))
        for role in ("base", "old", "ctl"):
            f = os.path.join(x["dir"], f"{role}.log")
            if not os.path.exists(f):
                continue
            for ln in open(f, errors="replace"):
                if ln.startswith("SAMPLE "):
                    try:
                        s = json.loads(ln.split(" ", 1)[1])
                    except ValueError:
                        continue
                    if not s.get("nob"):
                        samples[s["cid"]][s["pass_"]][s["role"]] = s["med"]
        for cid, ps in samples.items():
            kind, mode, model, ep, tok = cid.split("|")
            tb = [r["base"] for r in ps.values() if "base" in r]
            rows.append(dict(mode=mode, model=model, ep=ep, tok=tok, ab=est(blocks(ps, "base", "old")), aa=est(blocks(ps, "ctl", "base")),
                             us=statistics.median(tb) if tb else float("nan"), host=x.get("host", "")))
    if rows:
        print("\nnew/old = time(644492a92) / time(228f92bab), < 1 = new faster; A/A = time(ctl) / time(base), the same tree. "
              "95% CI over ABBA blocks (narrow; the A/A spread is the real noise floor).")
        print("| mode | model | EP | tokens | base us | new/old [95% CI] | A/A [95% CI] | blocks | host |")
        print("|---|---|---|---|---:|---|---|---:|---|")
        for r in sorted(rows, key=lambda r: (r["mode"] != "delay1.0", r["model"], r["ep"], r["tok"])):
            print(f"| {r['mode']} | {r['model']} | {r['ep']} | {r['tok']} | {r['us']:.1f} | {fmt(r['ab'])} | {fmt(r['aa'])} | "
                  f"{r['ab'][3] if r['ab'] else 0} | {r['host']} |")
        for mode in ("delay1.0", "ready"):
            rs = [r for r in rows if r["mode"] == mode and r["ab"]]
            if not rs:
                continue
            g = math.exp(statistics.fmean(math.log(r["ab"][0]) for r in rs))
            aas = [r["aa"][0] for r in rs if r["aa"]]
            ga = math.exp(statistics.fmean(math.log(a) for a in aas)) if aas else float("nan")
            print(f"GEOMEAN dglu {mode}: new/old {g:.4f} over {len(rs)} cells (min {min(r['ab'][0] for r in rs):.4f}, "
                  f"max {max(r['ab'][0] for r in rs):.4f}; cells > 1.002: {sum(r['ab'][0] > 1.002 for r in rs)}, "
                  f"< 0.998: {sum(r['ab'][0] < 0.998 for r in rs)}); A/A geomean {ga:.4f}, "
                  f"range {min(aas, default=float('nan')):.4f} - {max(aas, default=float('nan')):.4f}")

# ---------------------------------------------------------------------------------------------- failures
print(f"\n## jobs not PASS: {len(bad_jobs)}")
for run, j, x in bad_jobs:
    print(f"\n### {run} {j}: {x['status']} (host {x.get('host', '?')}, attempt {x.get('attempt', '?')}, {x.get('secs', '?')} s)")
    print(f"summary: {x.get('summary', '')[:600]}")
    fh = os.path.join(x["dir"], "fail_head.txt")
    if os.path.exists(fh):
        for ln in open(fh, errors="replace").read().splitlines()[:40]:
            print(f"    {ln[:300]}")

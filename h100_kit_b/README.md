# H100 kit b: native SM90 validation of #2908 at 644492a92

Kit-only branch `wip/sm90-h100-kit-b` = the #2908 head `644492a92` (sm90-streaming-signal) + this directory. No library
change. Git delivery, no S3: every source tree is `git archive` of a commit in this clone, and results stay on the host.

Trees:
- new = `644492a923c79bcae3b9214391b79a760a326742` (the PR head; this branch's parent commit)
- old = `228f92bab65ae26f9bac192a16e0a512fcbc2029` (the previous head, validated in s57/s58; an ancestor of 644492a92).
  228f92bab..644492a92 changes only python/ and docs/, so "old" is 228f92bab's python/cudnn over the new venv's `.so`
  (`site_old`, as s54c/s57/s58 did).

The same harness as the last H100 campaign (s57sm90regress, s58sm90hangrepro, s57sm90plainperf3), in the coordinator's
dynamic-queue format (setup.sh, job.sh, jobs.jsonl, caps.json, MD5SUMS). Sized for 4 hosts / 32 GPUs (caps per host:
test 8, memcheck 3, racecheck 2, synccheck 2, perf 2; host_min_avail_gb 64). About 35 GPU-hours, so about 70-90 min on
32 GPUs.

| package | jobs | content | priority |
|---|---:|---|---:|
| s59sm90regress | 28 | (a) 16 L0+L1 shards of every SM90 test file (fwd/dgrad, GLU, dGLU, streaming signal, determinism incl. warm stress, overlap sm90 ids) + 4 strict determinism files with goldens; (c) compute-sanitizer memcheck x4 and racecheck x4 shards over the L0, non-warm-stress ids of the two SM90 streaming signal files, `--kernel-name-exclude regex=at6native` | 10 / 11 |
| s59sm90hangrepro | 58 | (b) s58's 58-case hang repro of dglu / deepseek_v3 / EP4 / T4096: 48 solo (base, old) x (ready, delay1.0, delay0.5) x 8, 4 replica, 4 fh, racecheck + synccheck | 8 |
| s59sm90perf | 15 | (d) one short replicate, the 15 dGLU shapes, cells ready (calibrates the producer) + delay1.0, roles base / old / ctl taking turns on one GPU; about 17 min of sampling per job | 5 |

## 1. Clone (each host)

```bash
git clone --branch wip/sm90-h100-kit-b --single-branch https://github.com/GarlGuo/cudnn-frontend.git ~/cudnnfe_sm90/kit-b/cudnn-frontend
git -C ~/cudnnfe_sm90/kit-b/cudnn-frontend log --oneline -2   # the kit commit, then 644492a92
cd ~/cudnnfe_sm90/kit-b/cudnn-frontend/h100_kit_b && for p in s59sm90regress s59sm90hangrepro s59sm90perf; do (cd $p && md5sum -c MD5SUMS); done
```

A host without GitHub access can get a copy of another host's clone (the whole repo dir); set `KIT_REPO=<path>` if it is
not at `~/cudnnfe_sm90/kit-b/cudnn-frontend`.

## 2. Setup (each host, once; idempotent)

```bash
K=~/cudnnfe_sm90/kit-b/cudnn-frontend/h100_kit_b
CUDA_VISIBLE_DEVICES=0 bash $K/s59sm90regress/setup.sh    # builds ~/venvs/cfe_pp_644492a9 from ~/venvs/cfe_quack (setup_base_venv.sh pins), collects the ids on the GPU
bash $K/s59sm90hangrepro/setup.sh                          # reuses that venv; builds site_old (228f92bab)
bash $K/s59sm90perf/setup.sh                               # same
```

Each copies its package into `~/cudnnfe_sm90/<pkg>/` (md5-checked), which is where the jobs' `cmd` points. A job also
runs its setup itself when the host is not ready. Expect `tree sig ready: venv OK, 288 L0 ids, 379 L0+L1 ids`.

## 3. Enqueue

`$K/<pkg>/jobs.jsonl` (+ `caps.json`) for the three packages. Every job is 1 GPU, not exclusive. `result_key` is not used
(no upload); `JOB_ID`, `ATTEMPT`, `JOB_OUT`, `CUDA_VISIBLE_DEVICES` from the dispatcher as before. Every job copies its
JOB_OUT to `~/cudnnfe_sm90/<pkg>/results/<job id>_a<attempt>/`. Exit 0 iff PASS.

## 4. Report

```bash
python3 ~/cudnnfe_sm90/kit-b/cudnn-frontend/h100_kit_b/report.py                # this host's results
python3 ~/cudnnfe_sm90/kit-b/cudnn-frontend/h100_kit_b/report.py DIR [DIR ...]  # e.g. every host's results/ dirs copied under DIR
```

It prints pass/fail counts per package, junit cases per test file (L0 / L1), skips, strict determinism, the
sanitizer SUMMARY lines, hang outcomes and launch totals, the perf ratios (per cell and geomean), and for every job that is
not PASS its summary and first failure lines. Running it on each host and sending every output is fine.

## Expected (from the 228f92bab run, s57/s58)

- regress: 28/28 PASS; 379 cases = 378 pass + 1 skip (the dGLU determinism case that needs jax); strict 4/4.
- memcheck: 24 ids, `ERROR SUMMARY: 0 errors`; racecheck: 24 ids, `RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)`.
- hangrepro: 58/58 PASS, no STALL / STALL_EXT, 0 bad checksums; the fh jobs' outcome is SEGV (the periodic-dump harness
  artifact) or OK, PASS either way.
- perf: dGLU delay1.0 new/old near 1.000 (644492a92 vs 228f92bab); A/A within about +-1.2%.

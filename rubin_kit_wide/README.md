# Wide BF16 wgrad on Rubin (VR200): tests + lock-step kernel A/B

Request from the cudnn-fe session (item "2.c": wide BF16 wgrad). Priority: normal. Run it after any
higher-priority Rubin work you already have, and after the MXFP8 wgrad kit (2.a / 2.b) from the same request.
The two kits use the same lock-step harness style.

## What is being tested

Library branch `wip/wgrad-wide-bf16-sm107` on github.com/GarlGuo/cudnn-frontend @ c43e5c663. The kit branch
`wip/wgrad-wide-rubin-kit` is that commit plus this directory (`rubin_kit_wide/`) and nothing else, so checking out
the kit branch gives both.

- `python/cudnn/gemm/cutedsl/grouped/wgrad/moe_grouped_gemm_wgrad_wide.py`: `MoEGroupedGemmWgradWideBF16Kernel`
  (2-CTA 256 x 512 tiles per CTA pair, each A k-tile shared by two 256x256 MMA halves; the same dW bits as stock).
  Its A/B stage count now follows the compile target's shared memory: 4 stages on sm_100a, 6 on sm_107a (334848 B).
  This is the SM107 change of EPC b2d5154c (EPC's sm107 subclass only sets the capacity).
- The stock BF16 wgrad (`moe_grouped_gemm_wgrad.py`) also takes the target's capacity (merged from
  wip/sm107-bf16-smem): 9 stages on sm_107a instead of 6.
- `GroupedGemmWgradBf16API._wide_applies()` now admits (10, 7). The wide kernel is still **off by default**
  (`_use_wide_tiles = False`). The only way to turn it on is test-only: the plugin `wide_on.py` here, or the
  harness's own monkeypatch. There is no env var or knob in the library.

Expected on sm_107a (checked by CPU trace here): stock smem 334848 / 9 stages, wide 334848 / 6 stages.
The harness records both in its `compile` lines. If you see 232448, the DSL is not targeting sm_107a.

## 0. Setup

```bash
git clone https://github.com/GarlGuo/cudnn-frontend.git fe && cd fe
git checkout wip/wgrad-wide-rubin-kit
git diff --stat c43e5c663 HEAD -- python test     # must be empty: the library is wip/wgrad-wide-bf16-sm107 @ c43e5c663
export FE=$PWD
# Make $FE/python importable as `cudnn`. Either build the branch the way you build cuDNN-FE on Rubin, or copy
# _compiled_module*.so from an existing cuDNN-FE build of the same backend into $FE/python/cudnn/.
export KIT=$FE/rubin_kit_wide         # wide_on.py, run_tests.sh, lockstep.py, wgrad_ls.py, run_ab.sh, agg.py
export PY=python3                      # your Rubin env: torch + a nvidia-cutlass-dsl that targets sm_107a
```

## 1. Tests (VR200, one node)

```bash
OUT=$PWD/tests_out $KIT/run_tests.sh off     # library default: stock BF16 wgrad
OUT=$PWD/tests_out $KIT/run_tests.sh on      # wide forced on (plugin); prints WIDE_PICKED counts
```

Files: test_grouped_gemm_wgrad.py, _atomic_counter.py, _pdl.py, _empty.py, test_wgrad_tensormap_acquire.py,
test_grouped_gemm_wgrad_wide.py (forces wide itself, bitwise vs stock; now runs on (10, 7)) and
test_grouped_gemm_bf16_smem_capacity.py (expects 334848 B for the four BF16 kernels and the wide one), `-m "L0 or L1"`.
In `on` mode `test_wide_tiles_are_off_by_default` is deselected (it asserts the library default).
`WIDE_PICKED` must be > 0 in `on` mode. On GB200 it is about 30 tests from the wide file plus the atomic-counter
BF16 cases. GB200 reference counts are in the cudnn-fe notes.
Add `-n 4` (or your xdist setting) as extra args if you want it parallel: `$KIT/run_tests.sh on -n 4`.

## 2. Lock-step kernel A/B

All GPUs of the node run the same arm at the same time. Before each sample every rank drains its GPU, meets a gloo
barrier and idles 20 ms (GATE_MS), then a 512 MB L2 flush runs and the arm is timed with events. Each rank takes one
shape. Both GEMMs of that shape run in one process:

- dW1: m = 2I, n = H
- dW2: m = H, n = I

Arms per GEMM:

- `stock`: the API's default kernel, sized for the target.
- `wide`: what the API runs with `_use_wide_tiles = True`.
- `stock2`: an A/A control.

Shapes: 5 models x EP4 / EP8 at 8192 tokens per rank = 10 shapes = 20 GEMMs, the same as the GB200 notes. Rows are
spread evenly over the local experts, each padded to 256. A is M-major, B is N-major, dW is BF16.

Before timing, every rank byte-checks wide and stock2 against stock on:

- the balanced offsets;
- zero_token, ragged, all_empty and partial_tail_then_data offsets (operands hold data past offs[-1]);
- the API path: `grouped_gemm_wgrad_wrapper_sm100` with `_use_wide_tiles` monkeypatched True, then False. It records
  which kernel `_wide_applies()` picked and the dW bytes vs the stock kernel.

Every pass runs all arms of both GEMMs interleaved (rotating start, forward then backward), so any slice of the
work is a complete A/B on its own. A job runs a subset of (claim, round) units and stays inside `CAP` seconds
(default 1620 = 27 min, for 30-minute single-tray jobs). Each unit gets `min(BUDGET, remaining CAP / units left - 45)`
seconds, and its compiles and checks count against that. A round is NPROC shapes, one per GPU: 10 shapes on a
4-GPU tray = rounds 0, 1, 2, with round 2 padded with shapes 0 and 1. Outputs carry a per-job tag, and `agg.py`
merges any number of job outputs. Passes are pooled per (shape, claim), so extra jobs simply add passes.

```bash
cd $WORKDIR
# One 30-min job per line (any order, any trays). Copy every job's OUT dir into one place afterwards.
FE_PYTHON=$FE/python OUT=$PWD/ab_out CLAIMS=clc     ROUNDS=0   $KIT/run_ab.sh   # ~25 min timed
FE_PYTHON=$FE/python OUT=$PWD/ab_out CLAIMS=clc     ROUNDS=1,2 $KIT/run_ab.sh   # ~12 min per round
FE_PYTHON=$FE/python OUT=$PWD/ab_out CLAIMS=counter ROUNDS=0   $KIT/run_ab.sh
FE_PYTHON=$FE/python OUT=$PWD/ab_out CLAIMS=counter ROUNDS=1,2 $KIT/run_ab.sh
# Fewer jobs: one claim per job, 3 rounds of ~8 min each (still about 100 passes per shape on GB200):
#   FE_PYTHON=$FE/python OUT=$PWD/ab_out CLAIMS=clc $KIT/run_ab.sh
# Merged tables, every job and both claims:
$PY $KIT/agg.py $PWD/ab_out            # or: $PY $KIT/agg.py dirA dirB file.jsonl.r0 ...
```

Each job prints `UNIT <claim> round <k> rc <rc> budget <s>` per unit and `UNIT_SKIPPED` for units the cap did not
leave room for. Rerun those units with `ROUNDS=`. A first uncached compile of the six kernels per rank plus the
API check can take a few minutes; `MIN_UNIT` (default 240 s) keeps a unit from starting without room for timing.

Optional extra arms: `EXTRA=stock100,wide100` adds the same kernels forced to sm_100's 232448 B. That separates
the item-14 smem effect from the wide-tile effect. Do not add them to the main run: in the GB200 runs, more arms
added about 2% of position effects.

## 3. What to report back (to session "cudnn-fe")

1. Tests: passed / failed / skipped for `off` and `on`, the WIDE_PICKED line, and the failures' first lines.
2. A/B, for CLC and then counter: the agg.py tables. For each of the 20 GEMMs: wide/stock with CI, A/A, passes and
   GPUs, and the differing bytes. Then the dW1 and dW2 geomeans, the **dW1 + dW2 time geomean** with its A/A, the
   `compiled (arm, smem_capacity, num_ab_stage)` line, and any ERROR lines.
3. GPU name, cc, DSL version, driver, node name.

GB200 reference (sm_100a; stock and wide both have 232448 B):

- old 2-GPU harness: dW1 0.998, dW2 1.019, dW1 + dW2 1.006, A/A 0.994 / 0.998;
- this kit on GB200 (4 jobs merged, entries 20261126T194547-1457209961, -194548-1457245166, -194822-1459903905,
  -194823-1459919795; 102-264 passes per cell):
  - CLC: dW1 0.962, dW2 0.965, **dW1 + dW2 0.963** (0.951-0.988), A/A 1.000;
  - counter: dW1 0.982, dW2 0.988, **dW1 + dW2 0.984** (0.975-1.008), A/A 1.000;
  - 0 differing dW bytes in every check, case and API record; the API picks wide only with the flag on.
  The lock-step method idles every GPU 20 ms before each single launch. It gives a clear gain where the old
  harness, with back-to-back launches and no gate, gave none. Report the numbers as measured: the gap between the
  methods is itself a finding.

The question is whether sm_107a gives the 6-stage wide kernel a gain over the 9-stage stock one. EPC reported -4.7%
e2e bwd on VR200, which has not been reproduced.

## Files

| File | Purpose |
| --- | --- |
| `wide_on.py` | pytest plugin: `_use_wide_tiles = True` in every pytest process, logs `_wide_applies()` decisions (WIDE_LOG_DIR) |
| `run_tests.sh` | the L0 + L1 test run, `off` / `on` |
| `lockstep.py` | the lock-step timer (the same as the MXFP8 kit's) |
| `wgrad_ls.py` | per-rank A/B: builds the arms, runs the byte checks and the API check, then calls lockstep |
| `run_ab.sh` | one torchrun per round of NPROC shapes, then agg.py |
| `agg.py` | the tables |

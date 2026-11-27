# Wide BF16 wgrad selection rule on VR200: lock-step kernel A/B + tests

## What is being tested

Library: branch `wip/wgrad-wide-bf16-v2` on github.com/GarlGuo/cudnn-frontend @ e528f6d8a. The branch
`wip/wgrad-wide-bf16-v2-ab` is that commit plus this directory (`wgrad_wide_rule_ab/`) and nothing else.

- `MoEGroupedGemmWgradWideBF16Kernel` (256 x 512 tiles per CTA pair; the same dW bits as stock) takes the target's
  shared memory: 6 A/B stages on sm_107a (334848 B), 4 on sm_100a.
- The stock BF16 wgrad (`MoEGroupedGemmWgradBF16Kernel`) is unchanged from the base (#2898 head): it is sized for
  sm_100's 232448 B on every target, i.e. **6 stages on sm_107a** (the earlier VR200 study compared against a 9-stage
  stock that is not in this branch).
- The BF16 API now picks the wide kernel by default when the dW has at least 12 (CLC) / 24 (atomic counter) wide tiles
  (experts x m/256 x n/512) per resident 2-CTA cluster (`GroupedGemmWgradBf16API._wide_tiles_pay_off`), on
  SM100/SM103/SM107 alike. On VR200 (106 clusters) that is >= 1272 / 2544 wide tiles. The thresholds come from GB200;
  this run decides whether VR200 needs others.

Expected `compile` lines on sm_107a: stock 232448 / 6 stages, wide 334848 / 6, stockT 334848 / 9, wide100 232448 / 4.
If wide shows 232448 / 4, the DSL is not targeting sm_107a.

## 0. Setup

```bash
git clone https://github.com/GarlGuo/cudnn-frontend.git fe && cd fe
git checkout wip/wgrad-wide-bf16-v2-ab
git diff --stat e528f6d8a HEAD -- python test     # must be empty
export FE=$PWD KIT=$PWD/wgrad_wide_rule_ab PY=python3   # PY: torch + a nvidia-cutlass-dsl that targets sm_107a
# Make $FE/python importable as `cudnn` as before (build, or copy _compiled_module*.so into
# $FE/python/cudnn/). The A/B harness itself stubs the compiled module; the tests need it.
```

## 1. Tests (one VR200 node, any GPUs)

```bash
OUT=$PWD/tests_out $KIT/run_tests.sh default   # library default (the rule)
OUT=$PWD/tests_out $KIT/run_tests.sh forced    # wide forced on every eligible call (plugin wide_on.py); prints WIDE_PICKED
```

Files: test_grouped_gemm_wgrad.py, _atomic_counter.py, _pdl.py, _empty.py, test_wgrad_tensormap_acquire.py,
`-m "L0 or L1"`. The wide tests (bitwise wide vs stock, the rule at its boundary on this GPU's cluster count) are in
test_grouped_gemm_wgrad.py and run on (10, 7).

## 2. Lock-step kernel A/B

Same method as the previous wide-tile A/B: all GPUs of the node fire the same arm together after a drain, a gloo barrier
and a 20 ms idle; a 512 MB L2 flush; one launch timed with events. **New:** every GPU runs the same GEMM list (not one
shape per GPU), so each process is a complete paired comparison on one physical GPU. Arms per GEMM, interleaved in
every pass (rotating start, forward then backward):

- `stock`: the branch's stock kernel (6 stages on sm_107a);
- `wide`: the branch's wide kernel (6 stages on sm_107a);
- `stock2`: the stock compiled kernel again on its own dW (A/A);
- `stockT`: stock forced to the target's 334848 B (9 stages; the earlier study's stock);
- `wide100`: wide forced to 232448 B (4 stages).

Byte checks before timing: every arm vs stock on balanced offsets and on zero_token / ragged / all_empty /
partial_tail_then_data; the API with library defaults (which kernel it picked, its bytes vs stock).

GEMMs are `m:n:K:E` (dW is E x m x n, K rows per expert, balanced; A M-major, B N-major, dW BF16). Sets (in
run_rule_ab.sh): `models_ep4`, `models_ep8` (the 20 MoE GEMMs, 8192 tokens per rank) and the boundary families `f1`
(3072 x 1024, K 2048), `f2` (2048 x 3072, K 2048), `f3` (4096 x 2048, K 1024), each at about 8, 12, 14, 17, 23, 26 and
30 wide tiles per cluster on VR200, and `f4` (other n and K). Each set runs both claims (CLC, atomic counter) in one job, about 25 minutes.

Please run on the GPUs where the stock kernel ran at its fastest in the previous study (the request lists them). Two
jobs per set, the second mirrored (`CLAIMS=counter,clc REVERSE=1`), on two different nodes if you can:

```bash
cd $WORKDIR
for SET in models_ep4 models_ep8 f1 f2 f3 f4; do
  FE_PYTHON=$FE/python OUT=$PWD/rule_out SET=$SET                              $KIT/run_rule_ab.sh   # job A, node 1
  FE_PYTHON=$FE/python OUT=$PWD/rule_out SET=$SET CLAIMS=counter,clc REVERSE=1 $KIT/run_rule_ab.sh   # job B, node 2
done
# 12 jobs of ~25 min; any order, any split over jobs. Merge everything at the end:
$PY $KIT/agg_rule.py $PWD/rule_out > all.md
$PY $KIT/agg_rule.py $PWD/rule_out --fast <host:gpu,host:gpu,...> > fast.md   # the fast GPUs
```

The `--fast` list uses `<hostname>:<GPU index>` exactly as the first table of all.md prints them (the per-GPU stock
speed index, 1.00 = median GPU). Use the GPUs whose index is within ~3% of the fastest, and say which.

## 3. What to report back

1. Tests: passed / failed / skipped for `default` and `forced`, the WIDE_PICKED lines, the first lines of any failure.
   (`run_tests.sh forced` deselects the three tests that check the library's own threshold.)
2. fast.md and all.md in full (both are short): the per-GPU speed table, the `compiled (arm, smem, stages)` line, any
   ERROR lines, and per claim the per-GEMM table (wide/stock, stock2/stock, stockT/stock, wide100/stock as the
   median over processes with min-max, processes, passes, differing bytes, API pick).
3. GPU name, cc, DSL version, driver, node names, and which jobs ran on which node.

GB200 reference (this harness, 76 clusters, stock 6 / wide 4 stages; wide/stock, median over 8 processes per GEMM,
37 boundary GEMMs + 14 refinements): the ratio follows wide tiles per cluster only. CLC: 1.01-1.18 at <= 5 per
cluster, 0.985-1.014 at 10, <= 0.999 from 11.4 on (0.96-0.98 from 20). Counter: 1.00-1.07 up to 23.6, 0.987-0.996 at
24-25, 0.976-0.994 from 26.9. 0 differing dW bytes everywhere.

## Files

| File | Purpose |
| --- | --- |
| `rule_ls.py` | per-rank A/B on a GEMM list: builds the arms, byte checks, API check, then lockstep |
| `lockstep.py` | the lock-step timer (unchanged from the previous A/B) |
| `wgrad_ls.py` | the previous per-shape A/B; only its `install_cudnn` helper is used here |
| `run_rule_ab.sh` | one torchrun per claim over a GEMM set, then agg_rule.py |
| `agg_rule.py` | per-GPU speed table and per-GEMM tables (`--fast`, `--csv`) |
| `run_tests.sh`, `wide_on.py` | the wgrad L0 + L1 tests, library default or wide forced |

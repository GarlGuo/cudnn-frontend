# cuDNN-FE MXFP8 grouped wgrad on Rubin: stock vs global descriptors (2.a) vs A reuse (2.b)

From the cudnn-fe session (same request as the wide BF16 wgrad (2.c) kit another agent sent; please run this one first).
Priority: normal, after any higher-priority Rubin work.

Update 2026-11-26 (kit v3): new arm `dense_rubin`, a Rubin-native 2.a (`BlockScaledMoEGroupedGemmWgradDenseRubinKernel`,
branch `wip/wgrad-dense-rubin`): global A/B/SF descriptors, no helper, per-warp epilogue, on the Rubin K64 MMA, 256-token
K tile and Rubin TMEM plan, without A reuse. It is ReuseRubin minus the reuse, so `reuse / dense_rubin` measures 2.b
alone at the same MMA. Expected bits: dense_rubin = stock = reuse (0 differing bytes). The kit branch now contains it
(merge of `wip/wgrad-dense-rubin` into the kit).

Update 2026-11-27 (kit v4): the library now picks the kernels on Rubin. Branch `wip/wgrad-rubin-kit` = the kit
merged with `wip/wgrad-areuse-v3` (the code under test), plus this kit commit. What is new for Rubin:
- `_select_kernel` on SM107 picks `BlockScaledMoEGroupedGemmWgradDenseRubinKernel` (Rubin 2.a) under the 2.a conditions
  (MXFP8 E8M0 vec32, fp32 accumulation, dense output, Tensor2D, no accumulate, 2-CTA 256x256 tiles in a (2, 1) cluster,
  16-bit dW, m and n multiples of 256, no global scales); otherwise the Rubin wgrad as before.
- Within those conditions it picks `BlockScaledMoEGroupedGemmWgradReuseRubinKernel` (2.b) only when the mean tokens per
  expert are at most 5 x 256 = 1280 and dW has at least 3 work units of N tiles per M tile (`prefer_over_dense`), with
  `a_stages_for` A stages. At the 20 benchmark GEMMs that is dW1 at EP4 (all 5 models) and at EP8 for qwen35_397b and
  qwen38_2p4t; every dW2 gets 2.a.
- Tests were folded: `test_grouped_gemm_wgrad_reuse.py` and `test_grouped_gemm_wgrad_reuse_rubin.py` are gone, their
  cases are in `test_grouped_gemm_wgrad.py`.

### v4 runs (in this order; total about 30 min of one node)

From the kit worktree root (`W`), with that tree's `python/` on PYTHONPATH for pytest as before:
```
cd $W/test/python
# 1. wgrad L0 + L1 (about 10 min). Expected on Rubin: the SM100-only global-descriptor / A-reuse cases skip; the Rubin
#    cases run: test_rubin_global_descriptor_wgrad_matches_rubin_stock (10), test_rubin_reuse_wgrad_matches_rubin_stock
#    (16), test_rubin_api_picks_wgrad_variants (2), the host-side selection tests (4), and the PDL file's Rubin picks.
python -m pytest -q -rs -m "L0 or L1" gemm/cutedsl/test_grouped_gemm_wgrad.py gemm/cutedsl/test_grouped_gemm_wgrad_atomic_counter.py \
  gemm/cutedsl/test_grouped_gemm_wgrad_pdl.py gemm/cutedsl/test_grouped_gemm_wgrad_empty.py gemm/cutedsl/test_wgrad_tensormap_acquire.py
# 2. API-level selection at the 20 benchmark GEMMs, CLC and counter, one GPU (84 s on GB200; peak about 20 GB of GPU memory; needs apache-tvm-ffi like the tests):
#    the public wrapper picks the expected kernel, and its dW equals the Rubin wgrad's byte for byte.
cd $W && CUDA_VISIBLE_DEVICES=0 FE_PYTHON=$W/python python rubin_kit/api_check.py --out api_check.jsonl | tee api_check.log
#    expect: "device ... (10, 7) type rubin dense BlockScaledMoEGroupedGemmWgradDenseRubinKernel reuse ...ReuseRubinKernel",
#    40 "API OK" lines for the GEMMs and 10 for the thresholds, then "SUMMARY ok 50 fail 0".
# 3. Optional, only if the per-GEMM table for the library rule is wanted: the lock-step run as in v3 (section 2), e.g.
#    LSW_ARMS=stock,dense_rubin,reuse,stock2 MODES="0 1" PASSES=a BUDGET=300; the tables now carry a "library" column and
#    auto_lib_dense_rubin columns.
```
Reply with: pytest summary line(s) and any failure, the api_check SUMMARY line and any "API FAIL" lines.

## Code

Fork https://github.com/GarlGuo/cudnn-frontend.git, branch `wip/wgrad-dense-rubin` (`wip/wgrad-areuse-v2` + the Rubin
2.a kernel), which has:
- `#2898` head 73490c74c (atomic-counter claim + use_pdl for the wgrad kernels);
- 2.a: `moe_blockscaled_grouped_gemm_wgrad_dense.py` (global A/B/SF descriptors, no helper kernel, per-warp epilogue).
  The API picks it on SM100/SM103 only (MXFP8, dense output, Tensor2D, 2-CTA 256x256, cluster (2, 1)).
- 2.b: `moe_blockscaled_grouped_gemm_wgrad_reuse.py` (A kept in shared memory across a unit's N tiles) and its Rubin
  subclass `moe_blockscaled_grouped_gemm_wgrad_reuse_rubin.py` (Rubin K64 MMA, 256-token K tile, Rubin TMEM plan).
  The API does not pick either kernel on Rubin.
- Rubin 2.a: `moe_blockscaled_grouped_gemm_wgrad_dense_rubin.py`, `BlockScaledMoEGroupedGemmWgradDenseRubinKernel(Dense,
  Rubin)`, composed like ReuseRubin. Not picked by the API; not run on hardware yet. It compiles for sm_107a (CuTe DSL
  4.8.0 cu13, CPU) at all 20 GEMMs x {CLC, counter} x {use_pdl off, on} and issues the K64 MMA
  (`kind::mxf8f6f4.block_scale`), 4 A/B stages, TMEM 544 columns (2 x 256 accumulator + 24 SF), as the Rubin wgrad.

The SM100 2.a kernel (arm `dense`) also builds for sm_107a, but it issues the SM100 K32 MMA
(`kind::mxf8f6f4.block_scale.block32`), so its dW bits differ from the Rubin wgrad's (kept for continuity).

Two ways to get the kit:
- Kit branch `wip/wgrad-rubin-kit`. It is the earlier kit (`wip/wgrad-areuse-v2` @ 59c2b32e3 plus kit-only commits
  under `rubin_kit/`) with `wip/wgrad-dense-rubin` merged in, plus kit-only commits. One worktree then gives both the
  code under test and the kit.
- The files sent verbatim by message: save them unchanged into one directory.

```
git fetch https://github.com/GarlGuo/cudnn-frontend.git wip/wgrad-rubin-kit
git worktree add ../w_wgrad3 FETCH_HEAD      # the wip/wgrad-rubin-kit head
git -C ../w_wgrad3 log --oneline -3          # expect: the kit v3 commit (rubin_kit/ only), the merge of wip/wgrad-dense-rubin, its head
```
Kit dir: `../w_wgrad3/rubin_kit` (or wherever you saved the files). The code under test: `../w_wgrad3/python`.

CPU prep (optional, while GPUs are busy): with `CUDA_VISIBLE_DEVICES=` and `CUTE_DSL_ARCH=sm_107a`, all six arm
classes compiled to sm_107a cubins here with CuTe DSL 4.8.0 (cu13 libs): Rubin wgrad, Rubin 2.a (dense_rubin), Rubin A
reuse (a_stages 3 and 5), SM100 global-descriptor and SM100 stock, with CLC, the atomic counter and use_pdl.

## 1. Tests (wgrad L0 + L1)

From `../w_wgrad3/test/python`, with that tree's `python/` on PYTHONPATH as for the earlier cuDNN-FE runs:
```
python -m pytest -q -rs -m "L0 or L1" \
  gemm/cutedsl/test_grouped_gemm_wgrad.py gemm/cutedsl/test_grouped_gemm_wgrad_atomic_counter.py \
  gemm/cutedsl/test_grouped_gemm_wgrad_pdl.py gemm/cutedsl/test_grouped_gemm_wgrad_empty.py \
  gemm/cutedsl/test_wgrad_tensormap_acquire.py \
  gemm/cutedsl/test_grouped_gemm_wgrad_reuse.py gemm/cutedsl/test_grouped_gemm_wgrad_reuse_rubin.py
```
(`test_grouped_gemm_wgrad_global_desc.py` is gone: its tests were folded into `test_grouped_gemm_wgrad.py`.)
Expected on Rubin:
- The global-descriptor tests of `test_grouped_gemm_wgrad.py` and `_reuse` skip (SM100/SM103 only).
- The PDL file's new global-descriptor cases skip.
- `test_grouped_gemm_wgrad_reuse_rubin.py` runs its 26 L1 tests, bitwise against the Rubin wgrad:
  - 16 force the Rubin A-reuse kernel: N 512/768/1024/1536, experts longer than the A ring, partial K tiles followed by
    data, empty experts, unaligned starts, CLC and atomic counter, a_stages 5 and 2;
  - 10 (`test_rubin_dense_wgrad_matches_rubin_stock`) force the Rubin 2.a kernel: the same 4 cases plus N 256 with a
    96-token expert, CLC and atomic counter.
- Everything else passes as on #2898.

Optional, if time allows (one process per GPU, under your usual memory cap):
```
compute-sanitizer --tool racecheck --kernel-name-exclude regex=at6native --launch-timeout 0 --report-api-errors no \
  --target-processes all --error-exitcode 99 python -m pytest -q -m "L0 or L1" gemm/cutedsl/test_grouped_gemm_wgrad_reuse_rubin.py
compute-sanitizer --tool memcheck --kernel-name-exclude regex=at6native --launch-timeout 0 --report-api-errors no \
  --target-processes all --error-exitcode 99 python -m pytest -q -m "L0 or L1" gemm/cutedsl/test_grouped_gemm_wgrad_reuse_rubin.py
```

## 2. Lock-step kernel A/B (this kit)

Files: `lsw_rubin.py` (per-rank harness), `lockstep.py`, `run_rubin.sh` (rounds), `agg_rubin.py` (tables).
The harness imports the tree's `python/cudnn` sources directly, so it does not need the compiled module.

Method:
- Each rank takes one shape. Every GPU of the node fires the same arm at the same time, after a host barrier and a
  20 ms idle gate.
- Each sample flushes a 512 MB L2 buffer, then times the launch with CUDA events.
- Per pass, the arms run forward then backward, R samples each. The cell is the median of the per-pass ratios.
- Shapes: 5 models x EP4/EP8 at 8192 tokens/rank. Rows per local expert are T * top_k * EP / E padded to 256.
  dW1 has m = 2I, n = H; dW2 has m = H, n = I. All 20 GEMMs are MXFP8 (E4M3 A/B, E8M0 scales), BF16 dW, (2, 1) clusters.
- `run_rubin.sh` packs the 10 shapes into rounds of <GPUs per node>, padding by wrap-around. It runs each round twice
  (rank positions forward, then reversed), for CLC and for the atomic counter. Each round is 600 s.

Arms (kernel classes constructed directly; no API selection involved):

| arm | kernel |
| --- | --- |
| stock | `BlockScaledMoEGroupedGemmWgradRubinKernel`: what the API runs on Rubin today |
| stock2 | A/A control (second instance of stock) |
| dense_rubin | 2.a on Rubin: `BlockScaledMoEGroupedGemmWgradDenseRubinKernel` (K64 MMA, as stock and reuse) |
| reuse | 2.a + 2.b: `BlockScaledMoEGroupedGemmWgradReuseRubinKernel`, a_stages = a_stages_for(rows, experts) as EPC does |
| dense | continuity: the SM100 2.a `BlockScaledMoEGroupedGemmWgradDenseKernel`, forced (SM100 K32 MMA) |
| stock_sm100 | continuity: the SM100 stock wgrad, forced: the bit reference for dense |

Derived in the tables (no extra arms):
- "auto_epc_<fallback>" (2.a + 2.b, EPC rule) = reuse where `ReuseRubin.can_implement` holds (MXFP8, more than one N
  tile per unit, mean rows/expert <= 5 x 256), else the fallback arm: dense_rubin (Rubin 2.a), stock (the library as
  is) or dense (SM100 2.a, continuity).
- "auto_ref_<fallback>" (refined rule) also requires more than one work unit per M tile (n / 256 > n_group).
- Per mode, two tables. Rubin-native: dense_rubin / stock, reuse / stock, reuse / dense_rubin, auto_{epc,ref}_dense_rubin
  over dense_rubin and over stock, stock2 / stock. Continuity: the earlier columns (dense / stock, reuse / dense,
  auto_*_stock, auto_*_dense, stock_sm100 / stock) plus dense_rubin / dense.

Run (one node, all GPUs; ~2 x 2 x ceil(10 / GPUs) x 11 min, i.e. ~66 min with 4 GPUs, ~44 min with 8; six arms per
pass now, so ~17% fewer passes per round than with five):
```
cd ../w_wgrad3/rubin_kit
FE_PYTHON=$PWD/../python OUT=$PWD/lsw_out bash run_rubin.sh
# quick check first (~5 min): BUDGET=60 REPS=3 MODES=0 PASSES=a FE_PYTHON=$PWD/../python OUT=$PWD/lsw_smoke bash run_rubin.sh
```
Splitting into shorter jobs: each job runs every arm, interleaved, for the rounds it is given.
- One MODES value and one PASSES value per job (4 jobs: MODES 0/1 x PASSES a/b). With 4 GPUs, one such job is
  3 rounds x ~11 min at BUDGET=600. For jobs of 30 min or less, use BUDGET=480, or split further with ROUNDS
  (e.g. ROUNDS="0 1" and ROUNDS="2").
- Give each job its own OUT (or the same OUT; files are named per mode, pass and round).
- Then merge all jobs in one table:
  `python agg_rubin.py <job1_out>/*.jsonl.r* <job2_out>/*.jsonl.r* ... > tables.md`
  Passes from all jobs are pooled per (mode, shape). The per-job tables.md that run_rubin.sh writes covers only that job.

If FP8 conversion or E8M0 work on the GPU fails (the earlier NVRTC issue), add `LSW_CPU_INIT=1` (operands built on the
CPU). `PY` / `TORCHRUN` override the python and torchrun binaries.

Byte checks, printed as `CHECK` lines at the top of `lsw_out/tables.md`:
- stock2, dense_rubin and reuse vs stock, and `reuse_vs_dense_rubin`: expect 0 differing bytes;
- dense vs stock_sm100: expect 0 (same MMA);
- dense (and stock_sm100) vs stock: reported as a count plus `dense_max_rel_err`. It can be nonzero (K32 vs K64 MMA), but it should be
  small (~1e-2 or less for BF16 dW).
- `*_nan` must be 0.

## Please send back (to session "cudnn-fe")

1. The pytest pass/fail/skip counts and skip reasons, plus any sanitizer SUMMARY lines.
2. `lsw_out/tables.md` in full: the CHECK lines and the per-GEMM, per-layer and geomean tables for CLC and the counter.
3. `nvidia-smi` GPU name, SM clock, power limit and GPUs per node, plus the torch / CuTe DSL / CUDA versions.
4. Any `build_error` record or compile failure (tables.md lists them; the full traceback is in `lsw_out/*.log`).

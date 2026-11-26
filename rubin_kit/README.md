# cuDNN-FE MXFP8 grouped wgrad on Rubin: stock vs global descriptors (2.a) vs A reuse (2.b)

From the cudnn-fe session (same request as the wide BF16 wgrad (2.c) kit another agent sent; please run this one first).
Priority: normal, after any higher-priority Rubin work.

## Code

Fork https://github.com/GarlGuo/cudnn-frontend.git, branch `wip/wgrad-areuse-v2` @ 59c2b32e3:
- `#2898` head 73490c74c (atomic-counter claim + use_pdl for the wgrad kernels);
- 2.a: `moe_blockscaled_grouped_gemm_wgrad_dense.py` (global A/B/SF descriptors, no helper kernel, per-warp epilogue).
  The API picks it on SM100/SM103 only (MXFP8, dense output, Tensor2D, 2-CTA 256x256, cluster (2, 1)).
- 2.b: `moe_blockscaled_grouped_gemm_wgrad_reuse.py` (A kept in shared memory across a unit's N tiles) and its Rubin
  subclass `moe_blockscaled_grouped_gemm_wgrad_reuse_rubin.py` (Rubin K64 MMA, 256-token K tile, Rubin TMEM plan).
  The Rubin subclass has never run on hardware. The API does not pick either kernel on Rubin.

There is no Rubin version of 2.a. EPC has none either: on Rubin, EPC runs the A-reuse Rubin kernel for short experts
and cuDNN's Rubin wgrad otherwise. The SM100 2.a kernel builds for sm_107a here (CuTe DSL 4.8.0 cu13, CPU compile), but
it issues the SM100 K32 MMA (`kind::mxf8f6f4.block_scale.block32`), not the Rubin K64 MMA. Its dW bits therefore
cannot match the Rubin wgrad's. The harness forces it as an arm, so we learn whether it is correct and fast on Rubin.

Two ways to get the kit:
- Kit branch `wip/wgrad-rubin-kit`. It is `wip/wgrad-areuse-v2` @ 59c2b32e3 plus kit-only commits that add only
  `rubin_kit/` (these files) at the repo root. One worktree then gives both the code under test and the kit.
- The files sent verbatim by message: save them unchanged into one directory.

```
git fetch https://github.com/GarlGuo/cudnn-frontend.git wip/wgrad-rubin-kit wip/wgrad-areuse-v2
git worktree add ../w_wgrad2 FETCH_HEAD      # the wip/wgrad-rubin-kit head; its parent is 59c2b32e3
git -C ../w_wgrad2 log --oneline -3          # expect: kit-only commits (rubin_kit/), then 59c2b32e3
```
Kit dir: `../w_wgrad2/rubin_kit` (or wherever you saved the files). The code under test: `../w_wgrad2/python`.

CPU prep (optional, while GPUs are busy): with `CUDA_VISIBLE_DEVICES=` and `CUTE_DSL_ARCH=sm_107a`, all five arm
classes compiled to sm_107a cubins here with CuTe DSL 4.8.0 (cu13 libs): Rubin wgrad, Rubin A reuse (a_stages 3 and 5),
SM100 global-descriptor and SM100 stock, with CLC, the atomic counter and use_pdl.

## 1. Tests (wgrad L0 + L1)

From `../w_wgrad2/test/python`, with that tree's `python/` on PYTHONPATH as for the earlier cuDNN-FE runs:
```
python -m pytest -q -rs -m "L0 or L1" \
  gemm/cutedsl/test_grouped_gemm_wgrad.py gemm/cutedsl/test_grouped_gemm_wgrad_atomic_counter.py \
  gemm/cutedsl/test_grouped_gemm_wgrad_pdl.py gemm/cutedsl/test_grouped_gemm_wgrad_empty.py \
  gemm/cutedsl/test_wgrad_tensormap_acquire.py gemm/cutedsl/test_grouped_gemm_wgrad_global_desc.py \
  gemm/cutedsl/test_grouped_gemm_wgrad_reuse.py gemm/cutedsl/test_grouped_gemm_wgrad_reuse_rubin.py
```
Expected on Rubin:
- `_global_desc` and `_reuse` skip (SM100/SM103 only).
- The PDL file's new global-descriptor cases skip.
- `test_grouped_gemm_wgrad_reuse_rubin.py` runs its 16 L1 tests. They force the Rubin A-reuse kernel against the
  Rubin wgrad, bitwise: N 512/768/1024/1536, experts longer than the A ring, partial K tiles followed by data, empty
  experts, unaligned starts, CLC and atomic counter, a_stages 5 and 2.
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
| dense | 2.a: the SM100 `BlockScaledMoEGroupedGemmWgradDenseKernel`, forced (SM100 K32 MMA) |
| stock_sm100 | the SM100 stock wgrad, forced: the bit reference for dense |
| reuse | 2.b: `BlockScaledMoEGroupedGemmWgradReuseRubinKernel`, a_stages = a_stages_for(rows, experts) as EPC does |

Derived in the tables (no extra arms):
- "2.a + 2.b, EPC rule" = reuse where `ReuseRubin.can_implement` holds (MXFP8, more than one N tile per unit, mean
  rows/expert <= 5 x 256), else the fallback. The fallback is stock (2.a not picked on Rubin) or dense (2.a forced).
- "Refined rule" also requires more than one work unit per M tile (n / 256 > n_group).

Run (one node, all GPUs; ~2 x 2 x ceil(10 / GPUs) x 11 min, i.e. ~66 min with 4 GPUs, ~44 min with 8):
```
cd ../w_wgrad2/rubin_kit
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
- stock2 vs stock and reuse vs stock: expect 0 differing bytes;
- dense vs stock_sm100: expect 0 (same MMA);
- dense (and stock_sm100) vs stock: reported as a count plus `dense_max_rel_err`. It can be nonzero (K32 vs K64 MMA), but it should be
  small (~1e-2 or less for BF16 dW).
- `*_nan` must be 0.

## Please send back (to session "cudnn-fe")

1. The pytest pass/fail/skip counts and skip reasons, plus any sanitizer SUMMARY lines.
2. `lsw_out/tables.md` in full: the CHECK lines and the per-GEMM, per-layer and geomean tables for CLC and the counter.
3. `nvidia-smi` GPU name, SM clock, power limit and GPUs per node, plus the torch / CuTe DSL / CUDA versions.
4. Any `build_error` record or compile failure (tables.md lists them; the full traceback is in `lsw_out/*.log`).

# Overlapping-accumulator TMEM release-before-copy race (SM100 grouped GEMMs)

Four grouped-GEMM CuTeDSL epilogues publish the overlapping accumulator stage as **empty**
before issuing the TMEM→register copy of a subtile that still lies inside the overlap
region. **This change fixes the three SM100 kernels**; the fourth is a Rubin-family kernel
left untouched here because no sm_103 hardware was available to validate it on (it is
xfailed in the guard test with `strict=True`, so fixing it will flag). The MMA producer waiting on that same mbarrier may then begin
overwriting TMEM columns the epilogue has not read yet — a write-after-read
(anti-dependency) hazard on Tensor Memory.

The ordering is fully visible in Python source; there is no compiled kernel path
involved. Eight sibling kernels in the same package do it the other way round.

## Affected

| kernel | early release | TMEM read | order |
|---|---|---|---|
| `grouped/dsrelu/moe_blockscaled_grouped_gemm_dsrelu_quant.py` | 2131 | 2135 | **release-before-copy** |
| `grouped/srelu/moe_blockscaled_grouped_gemm_srelu_quant.py` | 1912 | 1916 | **release-before-copy** |
| `grouped/quant/grouped_gemm_quant.py` | 1862 | 1866 | **release-before-copy** |
| `grouped/quant/moe_blockscaled_grouped_gemm_quant_rubin.py` | 2239 | 2243 | **release-before-copy** — *not fixed here (sm_103, unverified)* |
| `grouped/wgrad/moe_blockscaled_grouped_gemm_wgrad.py` | 1552 | 1546 | copy-before-release |
| `grouped/glu/moe_blockscaled_grouped_gemm_glu_bias.py` | 2641 | 2630 | copy-before-release |
| `grouped/glu/moe_blockscaled_grouped_gemm_glu_rubin.py` | 2955 | 2944 | copy-before-release |
| `grouped/glu_hadamard/moe_blockscaled_grouped_gemm_glu_hadamard.py` | 2068 | 2057 | copy-before-release |
| `grouped/swiglu/grouped_gemm_swiglu_quant.py` | 2352 | 2341 | copy-before-release |
| `grouped/dswiglu/grouped_gemm_dswiglu_quant.py` | 2642 | 2620 | copy-before-release |
| `grouped/dglu/moe_blockscaled_grouped_gemm_dglu_dbias.py` | 2993 | 2971 | copy-before-release |
| `grouped/dglu/moe_blockscaled_grouped_gemm_dglu_rubin.py` | 3405 | 3385 | copy-before-release |

Line numbers as of `f9e6ab5` (1.25.0.dev-117), before this change; the pytest guard in
`test/python/fe_api/grouped_gemm/test_grouped_gemm_acc_release_order.py` regenerates them.

The shipped code, `dsrelu:2127-2135`:

```python
if cutlass.const_expr(self.overlapping_accum):
    if subtile_idx == self.iter_acc_early_release_in_epilogue:
        cute.arch.fence_view_async_tmem_load()          # 2129  orders PRIOR loads only
        with cute.arch.elect_one():
            acc_pipeline.consumer_release(acc_consumer_state)  # 2131  stage published EMPTY
        acc_consumer_state.advance()

tTR_tAcc_mn = tTR_tAcc[(None, None, None, real_subtile_idx)]
cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)        # 2135  the read it had to cover
```

The control, same package, same idiom, same pipeline object — `wgrad:1544-1553`:

```python
tTR_tAcc_mn = tTR_tAcc[(None, None, None, real_subtile_idx)]
if k_tile_cnt > 0:
    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)    # copy FIRST

if cutlass.const_expr(self.overlapping_accum):
    if subtile_idx == self.iter_acc_early_release_in_epilogue:
        cute.arch.fence_view_async_tmem_load()          # then fence
        if k_tile_cnt > 0:
            acc_pipeline.consumer_release(acc_consumer_state)   # then release
            acc_consumer_state.advance()
```

`tTR_tAcc` is bound at `dsrelu:2114` from the stage index captured at `:2109`, before the
loop, so the `advance()` does **not** redirect the load to another stage — it still reads
the stage that was just handed to the producer.

## Why the fence cannot cover the later load

Both plausible lowerings of `cute.arch.fence_view_async_tmem_load()` order **prior**
operations only, so the argument does not depend on which one it is.

PTX ISA 9.7.17.8.5, `tcgen05.wait`:

> Instruction `tcgen05.wait::ld` causes the executing thread to block until all prior
> `tcgen05.ld` operations issued by the executing thread have completed.

PTX ISA 9.7.17.11.1, `tcgen05.fence`:

> The instruction `tcgen05.fence::before_thread_sync` orders all the prior asynchronous
> `tcgen05` operations with respect to the subsequent `tcgen05` and the execution
> ordering operations.

The load at `:2135` has not been issued when the fence at `:2129` executes. Nothing
orders it before the arrive at `:2131`, and PTX ISA 9.7.17.6.1 is explicit that
`tcgen05.ld` is asynchronous and "not inherently ordered with respect to other `tcgen05`
operations in the same thread".

PTX ISA 9.7.17.6.4.4, *Non-pipelined instructions, different thread* — the canonical
pattern for exactly this producer/consumer pair:

> In this pattern, the producer threads that issue the asynchronous `tcgen05`
> instructions must explicitly wait for the instructions' completion **before**
> synchronizing with the consumer threads.

```
Thread 0                            Thread 1
tcgen05.ld
tcgen05.wait::ld
tcgen05.fence::before_thread_sync
mbarrier.arrive.relaxed.cluster
                                    mbarrier.try_wait.relaxed.cluster  // loop till success
                                    tcgen05.fence::after_thread_sync
                                    tcgen05.mma
```

The epilogue emits `wait::ld → arrive → ld`. The mandated order is `ld → wait::ld →
fence → arrive`. And PTX ISA 9.7.17.6.4.5 names the hazard class directly:

> To enforce such memory orderings and avoiding anti-dependency hazards around
> `tcgen05.ld`, `tcgen05.wait::ld` must be used.

## Which columns are exposed

For every shipped dSReLU/SReLU/quant configuration (`mma_tiler_mn` is `(256, 256)` or
`(128, 256)`; N is hard-required to be 256):

- `epi_tile = (128, 32)`, `mma_inst_tile_k = 4`, `sf_atom_mn = 32`
- `num_sfa_tmem_cols = (128 / 32) × 4 = 16`, `num_sfb_tmem_cols = (256 / 32) × 4 = 32`
- `num_sf_tmem_cols = 48`, so the accumulator stage stride is `256 − 48 = 208` columns
  (`dsrelu:1508`) and `num_accumulator_tmem_cols = 2 × 256 − 48 = 464`

Stage 0 occupies TMEM columns `[0, 256)`, stage 1 occupies `[208, 464)`; they share
`[208, 256)` — **48 columns**.

`iter_acc_early_release_in_epilogue = ceil(48 / 32) − 1 = 1`, and the epilogue walks the
overlap first in both phases (descending subtiles for stage 0, ascending for stage 1):

| subtile_idx | stage 0 columns | stage 1 columns | overlap columns read |
|---|---|---|---|
| 0 | `[224, 256)` | `[208, 240)` | 32 |
| 1 | `[192, 224)` | `[240, 272)` | 16 |

So the release constant is correct — after subtile 1 all 48 overlapping columns have
been read. Only the *placement* is wrong: releasing before the subtile-1 copy leaves
**16 of the 48 shared columns read after the stage was published empty**, symmetrically
in both phases. The producer's first `tcgen05.mma` after `producer_acquire`
(`dsrelu:1781`, `:1804`) writes the full N=256 accumulator, so it overwrites those
columns immediately.

## It is not an edge case

`overlapping_accum = self.num_acc_stage == 1 and self.mma_tiler[1] == 256`
(`dsrelu:405`, `srelu:379`, `quant:356`) with `num_acc_stage = 1 if mma_tiler_mnk[1] == 256
else 2`. The API layer hard-rejects anything else — *"MMA tiler N must be 256"* at
`dsrelu/api.py:522`, `srelu/api.py:448`, `quant/api.py:460`. `overlapping_accum` is
therefore always `True` and the release-before-copy branch is the only branch.

## The fix

Move the release block below the copy in each of the three SM100 kernels, matching wgrad.
One statement per file, no semantic change to the release condition:

```diff
--- a/python/cudnn/gemm/cutedsl/grouped/dsrelu/moe_blockscaled_grouped_gemm_dsrelu_quant.py
+++ b/python/cudnn/gemm/cutedsl/grouped/dsrelu/moe_blockscaled_grouped_gemm_dsrelu_quant.py
@@ -2124,15 +2124,15 @@
                          if reverse_subtile:
                              real_subtile_idx = self.cta_tile_shape_mnk[1] // self.epi_tile_n_required - 1 - subtile_idx
 
+                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, real_subtile_idx)]
+                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)
+
                     if cutlass.const_expr(self.overlapping_accum):
                         if subtile_idx == self.iter_acc_early_release_in_epilogue:
                             cute.arch.fence_view_async_tmem_load()
                             with cute.arch.elect_one():
                                 acc_pipeline.consumer_release(acc_consumer_state)
                             acc_consumer_state.advance()
-
-                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, real_subtile_idx)]
-                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)
```

`srelu`, `quant` and `quant_rubin` take the identical change (the two `quant` files also
carry a `# C1 fix: fence + early release for overlapping_accum` comment that should move
with the release block).

`test/python/fe_api/grouped_gemm/test_grouped_gemm_acc_release_order.py` is the
regression guard: it parses every grouped kernel and asserts the TMEM read precedes the
release. It is red on the pre-fix tree (4 of 12 sites) and green after.

Note that neither arm emits a separate `tcgen05.fence::before_thread_sync` between the
wait and the `mbarrier.arrive`, which 9.7.17.6.4.4 also shows. That is a second, smaller
question about the CuTeDSL lowering and is **not** addressed by this change.

## Reproducing

Three artifacts, cheapest first. Only the first needs no GPU.

### 1. The ordering itself — static, no GPU, runs in CI

```bash
cd test/python && pytest fe_api/grouped_gemm/test_grouped_gemm_acc_release_order.py
```

Parses every kernel under `python/cudnn/gemm/cutedsl/grouped/`, finds each `for subtile_idx`
epilogue that early-releases the accumulator, and asserts the TMEM read precedes the
release. 12 sites; 4 fail before this change, 0 after. Pure source analysis — no torch, no
cutlass, no device.

### 2. The numerical flip — 1 SM100 GPU

```bash
CUDA_VISIBLE_DEVICES=0 python benchmark/gemm/tmem_acc_release_race/ab_min.py
```

Two phases, each in its own subprocess (so the CuTeDSL compile cache cannot serve a stale
cross-phase kernel). Phase A moves the release back above the TMEM read — the original
order, verified identical to the pre-fix source in git. Phase B is this tree. Both are
scored against the repo's fp32 oracle, `run_grouped_gemm_dsrelu_ref`.

```
phase A  buggy=True  stall=100000  ptx=a5deccb8  d_row 5161299/134217728 band= 64, d_col 3984005/134217728 band=256  control 0  |ref|=448
phase B  buggy=False stall=100000  ptx=7c432f8c  d_row       0/134217728 band=  0, d_col       0/134217728 band=  0  control 0  |ref|=448
phase C  buggy=True  stall=0       ptx=6e8f5ed9  d_row       0/134217728 band=  0, d_col       0/134217728 band=  0  control 0  |ref|=448
```

Job 2596074 on GB200, exit 0, 3 repeats per phase (worst reported). Phase B is **bitwise identical** to the oracle. Phase A is
wrong in 3.8% of elements. Phase C — the same reordered source with the stall removed — is
also bitwise exact, which is what separates "this is a race" from "the transform broke
something". Report mismatch counts rather than a max-ratio: the ratio saturates at 2.0 (e4m3 tops out
at 448, so `896/448`) and one bad element ties with total garbage.

`band` is the positional signature — how many of the 256 column positions mod the 256-wide
N tile contain any mismatch — and it independently confirms the column arithmetic derived
above from source. The release fires at `subtile_idx == 1`, so that subtile's 32-wide read
happens after publication, in both accumulator phases: stage 0 reads `[192, 224)` and
stage 1 reads `[240, 272) mod 256`. Two disjoint 32-column regions, **64 positions** —
exactly what `d_row` reports. (`d_col` is the column-major output, whose scale blocks run
along M, so its band saturates at 256; only `d_row`'s band is positionally informative.)

All three phases compile to distinct PTX, which the script asserts — otherwise phase C
could pass by being served phase B's artifact and would exonerate nothing. `control` is `d_srelu`, which is computed from C and
`prob` only — `compute_relu2` (`dsrelu:1176`) and `scale_srelu_from_relu2` (`:1209`) take
the accumulator fragment for its size, never its values. It stays exact in both phases,
which is what localises the fault to the accumulator lifetime rather than the epilogue at
large.

**Why phase B is exact and not merely close.** The oracle already models MXFP8 faithfully:
`a_ref`/`b_ref` *are* the fp8 values (`create_and_permute_tensor` sets
`ref = dtype_tensor.to(float32)`), the E8M0 scales are exact powers of two, and
`compute_reference_row_quant` redoes the amax → E8M0 → e4m3 quantization. The one thing it
cannot model is the tensor core's fp32 summation order, which differs from `torch.einsum`'s
and flips round-to-nearest ties on a handful of ~10^7 elements. So the repro removes the
dependence on order instead of trying to match it: operands are integers in [-2, 2], every
product is an integer in [-16, 16] after the power-of-two scales on both operands, and every
partial sum over k=512 is an integer in [-8192, 8192] — 14 bits against fp32's 24, so addition is genuinely
associative over this data and both reduction orders land on the same bits. Everything
downstream (`d = 2·relu(C)·ref`, then a divide by a power-of-two scale) is order-free.

Two preconditions the script enforces or documents:

- **A stall after the release**, injected identically into both phases. Without it neither
  fires — the window is nanoseconds wide.
- **>1 tile per cluster.** With `num_acc_stage == 1` the accumulator is only reused when a
  CTA takes a *second* tile. The shape here is 2048 tiles over ~76 clusters; the `fe_api`
  defaults (`n=512, k=512, l=4`) give 8 tiles over ~76 and are structurally immune.

### 3. The emitted instructions

Recorded from compiling the kernel with the release below vs above the copy (GB200, CuTeDSL
4.6.0). Deterministic -- no race is involved, the instruction order is the finding. The
generating script is not included here; the sequences and the dataflow below are the result.

```
  [shipped]                                 [with the copy moved up]
    ptx:1537  tcgen05.wait::ld  <-- fence     ptx:1542  tcgen05.ld
    ptx:1542  mbarrier.arrive                 ptx:1561  tcgen05.wait::ld  <-- fence
    ptx:1576  tcgen05.ld                      ptx:1566  mbarrier.arrive

  arrive targets             %r71
  which maps to local reg    %r9 = base + 88
  arrives on it, kernel-wide 1
  producer try_wait at       [992, 1005, 1155, 1192]
  first tcgen05.mma after    ptx:1087
```

Exactly one arrive targets that barrier in the whole kernel -- the epilogue's
`consumer_release` -- and it is the barrier gating `tcgen05.mma`, established by address
dataflow rather than proximity.

### Shape matters more than the delay

With `num_acc_stage == 1` the accumulator is only reused when a CTA picks up a *second*
tile, so a problem with fewer tiles than clusters can never exercise the window however long
the delay is. `tiles ~= M_total * N / 65536`; the repro prints tiles/cluster and warns at
<= 1. The `fe_api` defaults (`n=512, k=512, l=4`) give 8 tiles over ~76 clusters — 0.1 per
cluster, structurally immune.

## Status of the evidence

**Established here:** the source ordering; the TMEM column arithmetic; that the path is
always taken; the contrast with eight sibling kernels; and — verified on hardware — that
the shipped source order emits `wait::ld → mbarrier.arrive → tcgen05.ld` on the
accumulator-empty barrier, while the corrected order emits the PTX-ISA-canonical
`tcgen05.ld → wait::ld → mbarrier.arrive`.

**Not established here:**

- No vendor confirmation.
- PTX only — no SASS/CUBIN audit.
- **That it fires without an injected delay.** The sweep (job 2590531, GB200, 76 clusters)
  reproduced at every point with a post-release delay and at no point without one:

  | shape (`experts:group_m:n:k`) | tiles/cluster | 500 µs | 100 µs | 10 µs | 0 |
  |---|---|---|---|---|---|
  | `16:2048:4096:512` | 26.9 | REPRO | REPRO | REPRO | not repro |
  | `8:4096:2048:512` | 13.5 | REPRO | REPRO | — | not repro |
  | `8:2048:2048:512` | 6.7 | — | REPRO | — | not repro |

  6/6 delayed points hit, 3/3 zero-delay points clean at 8 repeats per arm. So the window
  is real and the flip is exact once there is pressure on it, but nothing here shows it
  being hit by an undisturbed workload. Treat it as a latent defect with a demonstrated
  failure mode, not as a known-active production bug.

  Note the earlier attempt (job 2590366, `n=512, k=512, l=4, m=1024`) was **8 tiles over
  ~74 clusters** — no CTA ever re-acquired the accumulator, so the window did not exist and
  that run said nothing either way.
- No claim that this ordering has caused a failure in any production workload. A prior
  investigation that suspected it of causing MoE wgrad nondeterminism **cleared it** — the
  cause there was pad-row garbage reaching columnwise MXFP8 block scales. This is a latent
  ordering defect on an always-taken path, reported on its own merits.

Verification runs (lyris `gb200`, one GB200): numerical A/B job **2591753**; PTX A/B job
**2591037**; delay/shape sweep job **2590531**.

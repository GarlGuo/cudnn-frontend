#!/usr/bin/env python3


"""Minimal A/B test for the SM100 dSReLU TMEM accumulator-release fix.

Phase A -- release before the TMEM->register copy (the original order) + a stall.
Phase B -- release after the copy (this tree) + the same stall.
Phase C -- the original order with NO stall.

The epilogue publishes the overlapping accumulator stage EMPTY, then issues the
tcgen05.ld that still needs those columns. fence_view_async_tmem_load() only orders
loads already issued, so the MMA producer can overwrite them mid-read.

The stall is injected after the release. In phase A that lands *between* the release
and the copy, widening the invalid window; in phase B the same anchor sits after the
copy and widens nothing. So B is not a control for the stall -- C is: same source as
A, no stall, and it must come back clean.

Each phase compiles in its own subprocess under its own module name and filename, with
CUTE_DSL_KEEP=ptx (which also disables the compile cache) so no phase can be served
another's artifact. The PTX of all three is hashed and required to differ.

The shape must put >1 tile on every cluster or the accumulator is never reused and
nothing can go wrong.
"""

import hashlib, json, os, re, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
KERNEL = "cudnn.gemm.cutedsl.grouped.dsrelu.moe_blockscaled_grouped_gemm_dsrelu_quant"
DELAY_NS = 100000
REPEATS = 3
E, GROUP_M, N, K = 16, 2048, 4096, 512  # 2048 tiles over ~76 clusters
PHASES = {"A": (True, DELAY_NS), "B": (False, DELAY_NS), "C": (True, 0)}

# The patched tree reads: copy, then release. Phases A and C swap them back.
RELEASE_BODY = (
    "                        if subtile_idx == self.iter_acc_early_release_in_epilogue:\n"
    "                            cute.arch.fence_view_async_tmem_load()\n"
    "                            with cute.arch.elect_one():\n"
    "                                acc_pipeline.consumer_release(acc_consumer_state)\n"
    "                            acc_consumer_state.advance()\n"
)
BLOCKS = re.compile(
    r"(?P<copy> +tTR_tAcc_mn = tTR_tAcc\[\(None, None, None, real_subtile_idx\)\]\n"
    r" +cute\.copy\(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc\)\n)"
    r"(?P<gap>(?: *(?:#[^\n]*)?\n)*?)"
    r"(?P<release> +if cutlass\.const_expr\(self\.overlapping_accum\):\n" + re.escape(RELEASE_BODY) + r")"
)

DELAY_HELPER = """

import cutlass as _c
from cutlass.cutlass_dsl import dsl_user_op as _op
from cutlass._mlir.dialects import llvm as _llvm


@_op
def _stall(*, loc=None, ip=None):
    _llvm.inline_asm(None, [_c.Int32({ns}).ir_value(loc=loc, ip=ip)],
                     "nanosleep.u32 $0;", "r", has_side_effects=True,
                     is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT)
"""

STALL_ANCHOR = "                            acc_consumer_state.advance()\n"


def fail(msg):
    raise SystemExit(f"ab_min: {msg}")  # not assert: -O would strip these


def kernel_class(phase):
    """Import the dSReLU kernel with the release moved and/or a stall injected.

    CuTeDSL re-parses each traced function with inspect.getsourcelines() at compile time,
    which reads co_filename through linecache -- not the code object. So the variant needs
    its own filename plus a seeded linecache entry, or the DSL traces the on-disk file and
    every phase compiles identically.
    """
    import importlib, importlib.util, inspect, linecache

    buggy, delay_ns = PHASES[phase]
    original = importlib.import_module(KERNEL)
    path = Path(original.__file__)
    source = path.read_text()

    if len(BLOCKS.findall(source)) != 1:
        fail("expected exactly one copy/release site in the kernel")
    if buggy:
        source = BLOCKS.sub(r"\g<release>\g<gap>\g<copy>", source)
    if delay_ns:
        # 28 spaces: unique to the release block's advance (the others are at 20).
        if source.count(STALL_ANCHOR) != 1:
            fail("stall anchor is not unique")
        source = source.replace(STALL_ANCHOR, STALL_ANCHOR + "                            _stall()\n", 1)
        source += DELAY_HELPER.format(ns=delay_ns)

    # Distinct module name AND filename per phase. Sharing either makes a stale compile
    # artifact from a previous phase a live possibility, and phase C -- which runs after
    # the clean phase and expects a clean result -- could not tell the difference.
    name, fake = f"{KERNEL}_ab_{phase}", str(path.with_name(f"{path.stem}_ab_{phase}.py"))
    linecache.cache[fake] = (len(source), None, source.splitlines(keepends=True), fake)
    module = importlib.util.module_from_spec(importlib.util.spec_from_file_location(name, fake))
    module.__package__, module.__file__ = original.__package__, fake
    sys.modules[name] = module
    exec(compile(source, fake, "exec"), module.__dict__)

    # Read the source back the way CuTeDSL will, and check both axes we varied.
    seen = inspect.getsource(module)
    if (BLOCKS.search(seen) is None) != buggy:
        fail(f"phase {phase}: the DSL would trace the wrong release/copy order")
    if ("_stall()" in seen) != bool(delay_ns):
        fail(f"phase {phase}: the DSL would trace the wrong stall state")
    return module.BlockScaledMoEGroupedGemmQuantBwdKernel


def import_from_tree():
    """Make cudnn.gemm.cutedsl importable straight from this checkout.

    cudnn/__init__.py and cudnn/datatypes.py need the compiled pybind extension; nothing on
    the grouped-GEMM path does. No-op when an editable install of this tree is present.
    Also stubs the FP4 helpers test_fe_api_utils imports eagerly (unused on the FP8 path).
    """
    import importlib.util, types

    sys.path.insert(0, str(REPO / "test" / "python"))
    sys.modules.setdefault("test_low_precision_matmul", types.ModuleType("test_low_precision_matmul"))
    for name in ("_bfloat16_to_float4_e2m1fn_x2", "float4_e2m1fn_x2_to_float32"):
        setattr(sys.modules["test_low_precision_matmul"], name, None)

    pkg_dir = (REPO / "python" / "cudnn").resolve()
    spec = importlib.util.find_spec("cudnn")
    if spec and spec.origin and Path(spec.origin).resolve().parent == pkg_dir:
        return
    stub = types.ModuleType("cudnn._compiled_module")
    stub.data_type = type("_dt", (), {"__getattr__": lambda self, n: n})()
    package = types.ModuleType("cudnn")
    package.__path__ = [str(pkg_dir)]
    sys.modules.update({"cudnn": package, "cudnn._compiled_module": stub})


def score(got, want):
    """(max |err|, max |ref|, nonzero fraction of ref, mismatch count, mod-256 band width).

    The COUNT is the informative number -- max |err| saturates at 2x the reference, since
    e4m3 tops out at 448, so one bad element ties with total garbage. The band width is the
    positional signature: the race corrupts one epilogue subtile, so mismatches should sit
    in a narrow band of N columns mod 256, not be spread uniformly.
    """
    import torch

    got, want = got.float(), want.float()
    bad = got != want  # NaN != NaN is True, so NaN counts as a mismatch (biases to INCONCLUSIVE)
    per_column = bad.squeeze(-1).sum(0)
    band = int((per_column.view(-1, 256).sum(0) > 0).sum().item()) if per_column.numel() % 256 == 0 else -1
    return {
        "max_abs": (got - want).abs().max().item(),
        "max_ref": want.abs().max().item(),
        "nonzero_frac": (want != 0).float().mean().item(),
        "mismatches": int(bad.sum().item()),
        "band": band,
        "elements": bad.numel(),
    }


def child(phase):
    import atexit, inspect, shutil, tempfile

    import torch

    # CUTE_DSL_KEEP=ptx drops every compiled artifact into the CWD. Compile from a scratch
    # directory so running this from a checkout does not litter the tree with .ptx files.
    scratch = tempfile.mkdtemp(prefix="ab_min_ptx_")
    os.chdir(scratch)
    atexit.register(lambda: (os.chdir("/"), shutil.rmtree(scratch, ignore_errors=True)))

    import_from_tree()
    from cudnn.gemm.cutedsl.grouped.dsrelu.api import GroupedGemmDsreluSm100
    from cuda.bindings import driver as cuda
    from fe_api.grouped_gemm.test_grouped_gemm_dsrelu_utils import (
        allocate_grouped_gemm_dsrelu_tensors, allocate_grouped_gemm_input_tensors,
        run_grouped_gemm_dsrelu_ref,
    )  # fmt: skip

    sf = torch.float8_e8m0fnu
    torch.manual_seed(0)
    inp = allocate_grouped_gemm_input_tensors(n=N, k=K, l=E, group_m_list=[GROUP_M] * E,
                                              ab_dtype=torch.float8_e4m3fn, b_major="k",
                                              sf_dtype=sf, sf_vec_size=32, m_aligned=256)  # fmt: skip
    inp, out = allocate_grouped_gemm_dsrelu_tensors(tensor_m=inp["tensor_m"], n=N, l=E,
                                                    ab_dtype=torch.float8_e4m3fn, c_dtype=torch.bfloat16,
                                                    d_dtype=torch.float8_e4m3fn, cd_major="n",
                                                    sf_dtype=sf, sf_vec_size=32, input_tensors=inp)  # fmt: skip

    # Make the GEMM exact so reduction ORDER cannot matter. The oracle already models MXFP8
    # (a_ref/b_ref are the fp8 values themselves, the E8M0 scales are powers of two, and the
    # reference re-does the e4m3 quantization) -- what it cannot model is the tensor core's
    # fp32 summation order. With integer operands in [-2,2] and power-of-two scales on BOTH
    # operands, every product is an integer in [-16,16] and every partial sum over k=512 is
    # an integer in [-8192,8192]: 14 bits against fp32's 24, so addition is associative here
    # and both orders land on the same bits. Phase B must then match the oracle exactly.
    for ref_key, tensor_key in (("a_ref", "a_tensor"), ("b_ref", "b_tensor")):
        whole = torch.randint(-2, 3, inp[tensor_key].shape, device=inp[tensor_key].device, dtype=torch.float32)
        inp[tensor_key].copy_(whole)
        inp[ref_key].copy_(inp[tensor_key].to(torch.float32))  # ref is exactly what the kernel sees
    for key in ("sfa_ref", "sfb_ref"):
        if not set(inp[key].unique().tolist()) <= {1.0, 2.0}:
            fail(f"{key} is not a power of two; the exactness argument breaks")

    api = GroupedGemmDsreluSm100(
        sample_a=inp["a_tensor"], sample_b=inp["b_tensor"], sample_c=inp["c_tensor"],
        sample_d_row=out["d_row_tensor"], sample_d_col=out["d_col_tensor"], sample_d_srelu=out["d_srelu_tensor"],
        sample_sfa=inp["sfa_tensor"], sample_sfb=inp["sfb_tensor"], sample_padded_offsets=inp["padded_offsets_tensor"],
        sample_alpha=inp["alpha_tensor"], sample_prob=inp["prob_tensor"], sample_dprob=out["dprob_tensor"],
        sample_sfd_row=out["sfd_row_tensor"], sample_sfd_col=out["sfd_col_tensor"],
        sample_sfd_col_d_srelu=out["sfd_col_d_srelu_tensor"], sample_norm_const=inp["norm_const_tensor"],
        acc_dtype=torch.float32, mma_tiler_mn=(256, 256), cluster_shape_mn=(2, 1),
        sf_vec_size=32, vector_f32=False, m_aligned=256, discrete_col_sfd=True,
    )  # fmt: skip
    api._kernel = kernel_class(phase)  # compile() instantiates self._kernel
    if not api.check_support():
        fail("configuration unsupported on this device")
    api.compile()

    # Proof that THIS phase's source reached codegen, and a hash the parent uses to require
    # all three phases produced distinct binaries.
    ptx = inspect.getclosurevars(api._compiled_kernel).nonlocals["_compiled_kernel"].artifacts.PTX
    if ("nanosleep" in ptx) != bool(PHASES[phase][1]):
        fail(f"phase {phase}: emitted PTX does not match the requested stall state")

    ref = run_grouped_gemm_dsrelu_ref(
        a_ref=inp["a_ref"], b_ref=inp["b_ref"], c_ref=inp["c_ref"], sfa_ref=inp["sfa_ref"],
        sfb_ref=inp["sfb_ref"], alpha_tensor=inp["alpha_tensor"], prob_tensor=inp["prob_tensor"],
        aligned_group_m_list=inp["aligned_group_m_list"], valid_m=inp["valid_m"], generate_sfd=True,
        norm_const_tensor=inp["norm_const_tensor"], c_dtype=torch.bfloat16,
        d_dtype=torch.float8_e4m3fn, sf_vec_size=32, sf_dtype=sf,
    )  # fmt: skip

    worst = {}
    for _ in range(REPEATS):  # a race is not an n=1 claim; the kernel is already compiled
        out["dprob_tensor"].zero_()
        api.execute(
            a_tensor=inp["a_tensor"], b_tensor=inp["b_tensor"], c_tensor=inp["c_tensor"],
            d_row_tensor=out["d_row_tensor"], d_col_tensor=out["d_col_tensor"], d_srelu_tensor=out["d_srelu_tensor"],
            sfa_tensor=inp["sfa_tensor"], sfb_tensor=inp["sfb_tensor"], padded_offsets=inp["padded_offsets_tensor"],
            alpha_tensor=inp["alpha_tensor"], prob_tensor=inp["prob_tensor"], dprob_tensor=out["dprob_tensor"],
            sfd_row_tensor=out["sfd_row_tensor"], sfd_col_tensor=out["sfd_col_tensor"],
            sfd_col_d_srelu_tensor=out["sfd_col_d_srelu_tensor"], norm_const_tensor=inp["norm_const_tensor"],
            amax_tensor=None, current_stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )  # fmt: skip
        torch.cuda.synchronize()
        run = {
            "d_row": score(out["d_row_tensor"], ref["d_ref"]),
            "d_col": score(out["d_col_tensor"].permute(1, 0, 2), ref["d_col_ref"]),
            # d_srelu is relu(C)^2*prob -- the accumulator's VALUES never reach it. Control.
            "d_srelu": score(out["d_srelu_tensor"].permute(1, 0, 2), ref["d_srelu_col_ref"]),
        }
        for key, value in run.items():
            if key not in worst or value["mismatches"] > worst[key]["mismatches"]:
                worst[key] = value

    print("PHASE_RESULT " + json.dumps({
        "phase": phase, "buggy": PHASES[phase][0], "delay_ns": PHASES[phase][1],
        "ptx_sha": hashlib.sha256(ptx.encode()).hexdigest()[:16], "repeats": REPEATS,
        **worst,
    }), flush=True)  # fmt: skip


def run_phase(phase):
    # CUTE_DSL_KEEP=ptx gives us artifacts.PTX and disables the compile cache outright;
    # the two explicit cache vars are belt-and-braces. PYTHONOPTIMIZE would strip nothing
    # now (fail() raises) but a stripped child is still not what we want to measure.
    env = {**os.environ, "CUTE_DSL_KEEP": "ptx", "CUTE_DSL_NO_CACHE": "1", "CUTE_DSL_DISABLE_FILE_CACHING": "1"}
    env.pop("PYTHONOPTIMIZE", None)
    out = subprocess.run([sys.executable, "-u", __file__, phase], capture_output=True, text=True, env=env, timeout=1800)
    for line in out.stdout.splitlines():
        if line.startswith("PHASE_RESULT "):
            if out.returncode:  # a child can print a result and then die on a CUDA error
                print(f"phase {phase} reported then exited {out.returncode}\n{out.stderr}", file=sys.stderr)
                return None
            return json.loads(line[len("PHASE_RESULT ") :])
    print(f"phase {phase} produced no result\n{out.stdout}\n{out.stderr}", file=sys.stderr)
    return None


def main():
    if len(sys.argv) > 1:
        if sys.argv[1] not in PHASES:
            fail(f"unknown phase {sys.argv[1]!r}; expected one of {sorted(PHASES)}")
        child(sys.argv[1])
        return 0

    results = {}
    for phase in PHASES:
        d = results[phase] = run_phase(phase)
        if d is None:
            return 2
        acc = ", ".join(f"{k} {d[k]['mismatches']:>9}/{d[k]['elements']} band={d[k]['band']:>3}" for k in ("d_row", "d_col"))
        print(f"phase {phase}  buggy={str(d['buggy']):<5} stall={d['delay_ns']:<7} ptx={d['ptx_sha']}  {acc}"
              f"  control {d['d_srelu']['mismatches']}  |ref|={d['d_row']['max_ref']:.0f}", flush=True)  # fmt: skip

    a, b, c = results["A"], results["B"], results["C"]
    acc_keys, witnesses = ("d_row", "d_col"), ("d_row", "d_col", "d_srelu")

    if len({d["ptx_sha"] for d in results.values()}) != 3:
        print("\nINCONCLUSIVE: two phases compiled to identical PTX, so they are not distinct kernels.")
        return 2
    # A degenerate reference makes every count meaningless -- a comparison against all-zero
    # trivially reports zero mismatches and would read as a clean pass.
    for phase, d in results.items():
        for key in witnesses:
            if d[key]["max_ref"] < 1.0 or d[key]["nonzero_frac"] < 0.01:
                print(f"\nINCONCLUSIVE: phase {phase} reference for {key} is degenerate "
                      f"(max={d[key]['max_ref']:.3g}, nonzero={d[key]['nonzero_frac']:.3%}).")  # fmt: skip
                return 2
    if any(d[key]["mismatches"] for d in results.values() for key in ("d_srelu",)):
        print("\nINCONCLUSIVE: accumulator-INDEPENDENT output is wrong, so the fault is not")
        print("              localised to the accumulator lifetime.")
        return 2
    if any(b[key]["mismatches"] for key in acc_keys):
        print("\nINCONCLUSIVE: the corrected order does not match the fp32 oracle, so either the")
        print("              fix is insufficient or the bit-exactness argument does not hold.")
        return 2
    if any(c[key]["mismatches"] for key in acc_keys):
        # Not "inconclusive" -- this is a STRONGER result than the one we set out to show,
        # but it also means the transform-is-broken hypothesis is no longer separable here.
        print(f"\nThe original order corrupts WITHOUT the stall ({c['d_row']['mismatches']} mismatches).")
        print("That is stronger than the claim this script makes, but it no longer isolates")
        print("timing as the cause; check the emitted PTX before reporting it.")
        return 4
    if not any(a[key]["mismatches"] for key in acc_keys):
        print("\nNOT REPRODUCED: the original order matched the oracle even with the stall.")
        return 3
    print(f"\nREPRODUCED over {REPEATS} runs per phase: the original order is wrong in "
          f"{a['d_row']['mismatches']} of {a['d_row']['elements']} d_row elements")  # fmt: skip
    print(f"            (mismatches confined to {a['d_row']['band']} of 256 column positions);")
    print("            moving the release below the TMEM read makes it bitwise exact (phase B);")
    print("            the same original order with no stall is also exact (phase C), so this")
    print("            is the timing window and not a broken source transform.")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)

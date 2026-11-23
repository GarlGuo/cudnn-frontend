# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared by the streaming input signal tests of the dGLU (dSwiGLU) kernels: the BF16 and MXFP8 SM100 kernels and, on
SM107, the MXFP8 Rubin twin; problems, buffers, launches and one compile cache.

The backward dGLU reads the dispatched output gradient dY as A (and, for MXFP8, its scales), so the producer streams A
and SFA exactly as for the forward GLU; C (the saved pre-activation), prob and B are ordinary inputs.  The SM100 kernels
run with ``deterministic=True`` (one dprob slot per N tile), so dprob is compared bit for bit.  The Rubin kernel has no
such option: it adds each tile's fp32 dprob partials with atomics in arrival order, so its dprob is not deterministic
(not even across replays of the stock scheduler) and is compared with a tolerance there; D stays bit-exact.
"""

import torch

from gemm.cutedsl import _streaming_signal as bf16_helper
from gemm.cutedsl import _streaming_signal_blockscaled as bs
from gemm.cutedsl._streaming_signal import ROWS_ALIGNMENT, frozen

RECIPES = ("bf16", "mxfp8")
HIDDEN = 512  # K: dY's width (a multiple of 512 for the compact SFA SF warp)
INTERMEDIATE = 512  # N of the dGLU GEMM (W2's rows); C and D are 2 x INTERMEDIATE wide, two 256-wide N tiles
BF16_POISON = 0xFF  # NaN in both bytes of a BF16


def is_rubin():
    return bs.is_rubin()


def make_dglu_kernel(recipe, experts, **options):
    """The dSwiGLU kernel for ``recipe``, cluster (2, 1) with the 2-CTA 256 x 256 tile, dense weights."""
    import cutlass
    from cudnn.gemm.cutedsl.grouped.moe_utils import MoEWeightMode

    rubin = recipe == "mxfp8" and is_rubin()
    common = dict(
        acc_dtype=cutlass.Float32,
        use_2cta_instrs=True,
        mma_tiler_mn=(256, 256),
        cluster_shape_mn=(2, 1),
        vectorized_f32=True,
        expert_cnt=experts,
        weight_mode=MoEWeightMode.DENSE,
        act_func="dswiglu",
    )
    common.update({name: options.pop(name) for name in list(options) if name in common})
    # Off, the SM100-only options are not passed (the Rubin kernel has none of them).
    compact = bool(options.get("compact_sfa_sf_warp"))
    for name in ("compact_sfa_sf_warp", "compact_sfa_cta_release"):
        if not options.get(name):
            options.pop(name, None)
    if not rubin:
        options.setdefault("deterministic", True)
    if recipe == "bf16":
        from cudnn.gemm.cutedsl.grouped.dglu.moe_grouped_gemm_dglu_dbias import MoEGroupedGemmDgluDbiasBf16Kernel

        kernel = MoEGroupedGemmDgluDbiasBf16Kernel(**common, **options)
        kernel.test_kind = "bf16"
    elif rubin:
        from cudnn.gemm.cutedsl.grouped.dglu.moe_blockscaled_grouped_gemm_dglu_rubin import BlockScaledMoEGroupedGemmDgluKernel

        kernel = BlockScaledMoEGroupedGemmDgluKernel(sf_vec_size=bs.SF_VEC_SIZE, discrete_col_sfd=False, **common, **options)
        kernel.test_kind = "rubin"
    else:
        from cudnn.gemm.cutedsl.grouped.dglu.moe_blockscaled_grouped_gemm_dglu_dbias import BlockScaledMoEGroupedGemmDgluDbiasKernel

        kernel = BlockScaledMoEGroupedGemmDgluDbiasKernel(sf_vec_size=bs.SF_VEC_SIZE, discrete_col_sfd=False, **common, **options)
        kernel.test_kind = "mxfp8"
    kernel.test_compact = compact
    kernel.test_dprob_slots = INTERMEDIATE // 256 if options.get("deterministic") else 1
    kernel.test_compile_key = (type(kernel).__module__, frozen({**common, **options}))
    return kernel


def dprob_is_bitwise(kernel):
    """False for the Rubin kernel: its dprob is an fp32 atomic sum in tile arrival order."""
    return kernel.test_kind != "rubin"


def exact_outputs(kernel, out):
    """The outputs the kernel promises bit for bit: D, and dprob unless its dprob is an atomic sum."""
    return out if dprob_is_bitwise(kernel) else {name: value for name, value in out.items() if name != "dprob"}


def assert_outputs_match(kernel, got, want, label=""):
    """Raw-byte equal on ``exact_outputs``; an atomic dprob within fp32 reassociation of ``want``."""
    for name in exact_outputs(kernel, want):
        assert torch.equal(got[name].view(torch.int16), want[name].view(torch.int16)), f"{label}: {name} differs from the stock scheduler"
    if not dprob_is_bitwise(kernel):
        torch.testing.assert_close(got["dprob"], want["dprob"], rtol=1e-5, atol=1e-5, msg=lambda m: f"{label}: dprob: {m}")


def make_problem(recipe, tokens_per_expert, seed=0, rows=None):
    """Expert-major rows padded to ROWS_ALIGNMENT: A (dY) [rows, HIDDEN] (BF16, or MXFP8 codes with compact and SF-atom
    scales), B (W2) [experts, INTERMEDIATE, HIDDEN], C [rows, 2 * INTERMEDIATE] BF16, alpha, beta [experts], prob."""
    if recipe == "bf16":
        problem = bf16_helper.make_problem(tokens_per_expert, INTERMEDIATE, HIDDEN, seed=seed, rows=rows)
    else:
        problem = bs.make_problem(tokens_per_expert, INTERMEDIATE, HIDDEN, seed=seed, rows=rows)
    generator = torch.Generator(device="cuda").manual_seed(seed + 1000)
    problem["c"] = (torch.randn(problem["rows"], 2 * INTERMEDIATE, generator=generator, device="cuda") * 2).to(torch.bfloat16)
    problem["beta"] = torch.rand(len(tokens_per_expert), generator=generator, device="cuda") + 0.5
    problem["recipe"] = recipe
    return problem


def new_buffers(kernel, problem):
    """NaN-filled BF16 D [rows, 2 * INTERMEDIATE], zeroed dprob [rows, slots] and workspace."""
    rows = problem["rows"]
    d = torch.full((rows, 2 * INTERMEDIATE), float("nan"), device="cuda", dtype=torch.bfloat16)
    dprob = torch.zeros(rows, kernel.test_dprob_slots, device="cuda", dtype=torch.float32)
    workspace = torch.zeros(max(kernel.get_workspace_bytes(), 1), device="cuda", dtype=torch.uint8)
    return d, dprob, workspace


def kernel_args(kernel, problem, d, dprob, workspace):
    """The kernel's arguments before max_active_clusters; index ``b_major_index(kernel)`` is a compile-time constant."""
    import cutlass
    from cutlass.cute.nvgpu import OperandMajorMode
    from cutlass.cute.runtime import from_dlpack

    def tensor(x):
        return from_dlpack(x, assumed_align=16)

    def e8m0(x):
        return tensor(x.view(torch.float8_e8m0fnu))

    k = problem["k"]
    sizes = [cutlass.Int32(INTERMEDIATE), cutlass.Int32(k), cutlass.Int64(k), OperandMajorMode.K, from_dlpack(workspace, assumed_align=128).iterator]
    a, b, c, d_cute = tensor(problem["a"]), tensor(problem["b"].permute(1, 2, 0)), tensor(problem["c"]), tensor(d)
    offsets = from_dlpack(problem["offsets"], assumed_align=4)
    alpha, beta, prob, dprob_cute = tensor(problem["alpha"]), tensor(problem["beta"]), tensor(problem["prob"]), tensor(dprob.unsqueeze(-1))
    if kernel.test_kind == "bf16":
        # (c, d, padded_offsets, alpha, beta, prob, dprob, linear_offset, dbias)
        return [a, b, *sizes, c, d_cute, offsets, alpha, beta, prob, dprob_cute, cutlass.Float32(0.0), None]
    sfa = e8m0(problem["a_exp"] if kernel.test_compact else problem["sfa"])
    # (c, d, d_col, sfa, sfd_row, sfd_col, amax, norm_const, padded_offsets, alpha, beta, prob, dprob, [linear_offset,] dbias)
    tail = [c, d_cute, d_cute, sfa, None, None, None, None, offsets, alpha, beta, prob, dprob_cute]
    tail += [cutlass.Float32(0.0), None] if kernel.test_kind == "rubin" else [None]
    return [a, b, e8m0(problem["sfb"]), *sizes, *tail]


def b_major_index(kernel):
    return 5 if kernel.test_kind == "bf16" else 6


COMPILED = {}


def compile_once(kernel, problem, buffers, max_active_clusters=None):
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    from cutlass.utils import HardwareInfo

    if max_active_clusters is None:
        max_active_clusters = HardwareInfo().get_max_active_clusters(2)
    names = ("a", "b", "c", "offsets", "alpha", "beta", "prob") + (("a_exp", "sfa", "sfb") if kernel.test_kind != "bf16" else ())
    layouts = [problem[name] for name in names] + list(buffers)
    key = (kernel.test_compile_key, max_active_clusters, tuple((tuple(x.shape), tuple(x.stride()), str(x.dtype)) for x in layouts))
    compiled = COMPILED.get(key)
    if compiled is None:
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled = COMPILED[key] = cute.compile(kernel, *kernel_args(kernel, problem, *buffers), max_active_clusters, stream)
    return compiled


class Launcher:
    """One compile of ``kernel`` for a problem layout, launched on fresh NaN-filled D, zeroed dprob and workspace."""

    def __init__(self, kernel, problem, max_active_clusters=None):
        self.kernel = kernel
        self._compiled = compile_once(kernel, problem, self.buffers(problem), max_active_clusters)

    def buffers(self, problem):
        return new_buffers(self.kernel, problem)

    def launch(self, problem, buffers, stream=None):
        import cuda.bindings.driver as cuda

        stream = stream or torch.cuda.current_stream()
        skip = b_major_index(self.kernel)
        args = [x for i, x in enumerate(kernel_args(self.kernel, problem, *buffers)) if i != skip]
        self._compiled(*args, cuda.CUstream(stream.cuda_stream))

    def __call__(self, problem, prepare_workspace=None):
        """Fresh buffers, ``prepare_workspace(workspace)``, one launch; returns ({"d", "dprob"} over the experts' rows,
        workspace)."""
        buffers = self.buffers(problem)
        if prepare_workspace is not None:
            prepare_workspace(buffers[-1])
        self.launch(problem, buffers)
        return outputs(problem, buffers), buffers[-1]


def outputs(problem, buffers):
    rows = bs.valid_rows(problem)
    return {"d": buffers[0][:rows], "dprob": buffers[1][:rows]}


def poison_inputs(problem, late):
    """NaN in the operands ``late`` names: "a" (dY), and for MXFP8 "sfa" or "both"; returns their final bytes."""
    if problem["recipe"] == "mxfp8":
        return bs.poison_inputs(problem, late)
    assert late == "a", "a BF16 A has no scales"
    final = {"a": problem["a"].view(torch.uint8).clone()}
    problem["a"].view(torch.uint8).fill_(BF16_POISON)
    return final


def produce_tiles(problem, final, late, signal, order):
    """The fake producer on the current stream: per M tile in ``order``, write the late operands' rows (and SFA bytes),
    then ``signal[t] = t``."""
    if problem["recipe"] == "mxfp8":
        return bs.produce_tiles(problem, final, late, signal, order)
    a_bytes = problem["a"].view(torch.uint8)
    for tile in order:
        rows = slice(tile * ROWS_ALIGNMENT, (tile + 1) * ROWS_ALIGNMENT)
        a_bytes[rows].copy_(final["a"][rows])
        signal[tile].fill_(tile)


def run_with_producer(kernel, problem, late, orders, device_layout=False):
    """``kernel`` (a signal kernel) launched last on its stream with every M tile not ready and ``late`` poisoned; a
    producer stream writes them tile by tile in each order of ``orders``.  Returns {order: outputs}."""
    from cudnn.gemm.cutedsl.grouped.moe_kernel_helpers import reset_streaming_signal_workspace

    capacity = kernel.streaming_signal_capacity
    # A few clusters only, so the producer finds free SMs while the GEMM waits.
    launcher = Launcher(kernel, problem, max_active_clusters=4)
    launcher(problem, bs.signal_ready(kernel))  # loads the module: under lazy loading a first launch waits for running work
    torch.cuda.synchronize()
    final_layout = problem["layout"].clone()
    gemm_stream, producer_stream = torch.cuda.Stream(), torch.cuda.Stream()
    results = {}
    for order in orders:
        buffers = launcher.buffers(problem)
        final = poison_inputs(problem, late)
        if device_layout:
            problem["layout"].fill_(-1)
        signal = bs.signal_view(kernel, buffers[-1])
        gemm_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(gemm_stream):
            reset_streaming_signal_workspace(buffers[-1], kernel)
        producer_stream.wait_stream(gemm_stream)
        with torch.cuda.stream(producer_stream):
            torch.cuda._sleep(20_000_000)
            if device_layout:
                problem["layout"].copy_(final_layout)
            produce_tiles(problem, final, late, signal, bs.release_order(order, capacity))
        launcher.launch(problem, buffers, gemm_stream)
        torch.cuda.synchronize()
        results[order] = {name: value.clone() for name, value in outputs(problem, buffers).items()}
    return results

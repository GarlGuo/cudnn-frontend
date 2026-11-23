# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared by the streaming signal tests of the MXFP8 block-scaled MoE forward kernels (quant / FC2 and GLU / FC1): MXFP8
problems with swizzled scale-factor atoms, kernel construction (SM100 or the Rubin twin, by device), one compile cache,
a consumer kernel that copies each N chunk of D as soon as the GEMM publishes it, and a fake input producer."""

import torch

from gemm.cutedsl._streaming_signal import ROWS_ALIGNMENT, frozen, padded

SF_VEC_SIZE = 32
#: Scale-factor atom: 128 rows x 4 scales, 512 bytes (tcgen05 block-scaled MMA layout).
SF_ATOM_ROWS, SF_ATOM_COLS = 128, 4


def is_rubin():
    return tuple(torch.cuda.get_device_capability()) == (10, 7)


def make_quant_kernel(experts, one_cta=False, rubin=None, **options):
    """The MXFP8 quant kernel with BF16 D, cluster (2, 1) with the 2-CTA MMA or (1, 1) with the 1-CTA MMA; the Rubin
    twin on SM107 (``rubin`` overrides the device)."""
    import cutlass
    from cudnn.gemm.cutedsl.grouped.moe_utils import MoEWeightMode

    rubin = is_rubin() if rubin is None else rubin
    common = dict(
        sf_vec_size=SF_VEC_SIZE,
        acc_dtype=cutlass.Float32,
        use_2cta_instrs=not one_cta,
        mma_tiler_mn=(128, 256) if one_cta else (256, 256),
        cluster_shape_mn=(1, 1) if one_cta else (2, 1),
        vectorized_f32=True,
        generate_sfd=False,
        discrete_col_sfd=False,
        enable_bias=False,
        expert_cnt=experts,
        weight_mode=MoEWeightMode.DENSE,
    )
    # An option naming a construction argument overrides it (the rejection tests).
    common.update({name: options.pop(name) for name in list(options) if name in common})
    if rubin:
        from cudnn.gemm.cutedsl.grouped.quant.moe_blockscaled_grouped_gemm_quant_rubin import BlockScaledMoEGroupedGemmQuantKernel

        kernel = BlockScaledMoEGroupedGemmQuantKernel(generate_c=options.pop("generate_c", False), **common, **options)
    else:
        from cudnn.gemm.cutedsl.grouped.quant.grouped_gemm_quant import BlockScaledMoEGroupedGemmQuantKernel

        kernel = BlockScaledMoEGroupedGemmQuantKernel(**common, **options)
    kernel.test_rubin, kernel.test_glu = rubin, False
    kernel.test_compile_key = (type(kernel).__module__, frozen({**common, **options}))
    return kernel


def make_glu_kernel(experts, rubin=None, **options):
    """The MXFP8 SwiGLU kernel (FC1) with BF16 C and D, cluster (2, 1) with the 2-CTA 256 x 256 tile; the Rubin twin on
    SM107."""
    import cutlass
    from cudnn.gemm.cutedsl.grouped.moe_utils import MoEWeightMode

    rubin = is_rubin() if rubin is None else rubin
    common = dict(
        sf_vec_size=SF_VEC_SIZE,
        acc_dtype=cutlass.Float32,
        use_2cta_instrs=True,
        mma_tiler_mn=(256, 256),
        cluster_shape_mn=(2, 1),
        vectorized_f32=True,
        generate_sfd=False,
        discrete_col_sfd=False,
        expert_cnt=experts,
        weight_mode=MoEWeightMode.DENSE,
        act_func="swiglu",
    )
    common.update({name: options.pop(name) for name in list(options) if name in common})
    # compact_sfa_sf_warp is the SM100 kernel's alone: off, it is not passed (the Rubin twin has no such argument).
    test_compact = bool(options.get("compact_sfa_sf_warp"))
    if not test_compact:
        options.pop("compact_sfa_sf_warp", None)
    if rubin:
        from cudnn.gemm.cutedsl.grouped.glu.moe_blockscaled_grouped_gemm_glu_rubin import BlockScaledMoEGroupedGemmGluKernel

        kernel = BlockScaledMoEGroupedGemmGluKernel(generate_c=True, **common, **options)
    else:
        from cudnn.gemm.cutedsl.grouped.glu.moe_blockscaled_grouped_gemm_glu_bias import BlockScaledMoEGroupedGemmGluBiasKernel

        kernel = BlockScaledMoEGroupedGemmGluBiasKernel(**common, **options)
    kernel.test_rubin, kernel.test_glu = rubin, True
    # compact_sfa_sf_warp: SFA is the compact E8M0 rows [rows, K / 32] (problem["a_exp"]), not the SF atoms.
    kernel.test_compact = test_compact
    # generate_sfd: E4M3 D with its row SFD (and, unless generate_sfd_col=False, the column-quantized D and SFD).
    kernel.test_sfd = common["generate_sfd"]
    kernel.test_sfd_col = kernel.test_sfd and options.get("generate_sfd_col", True)
    kernel.test_compile_key = (type(kernel).__module__, frozen({**common, **options}))
    return kernel


def swizzle_sf(scales):
    """Row-major scales [rows, cols] (rows % 128 == 0, cols % 4 == 0) -> the flat SF-atom bytes the kernel reads:
    128-row blocks outermost, then 4-column atoms, each atom 32 x (4 x 4) bytes."""
    rows, cols = scales.shape
    return scales.view(rows // 128, 4, 32, cols // 4, 4).permute(0, 3, 2, 1, 4).contiguous().view(-1)


def sf_cols(k):
    """Scale columns per row, padded to whole atoms."""
    return -(-k // (SF_VEC_SIZE * SF_ATOM_COLS)) * SF_ATOM_COLS


def mxfp8_codes(rows, k, generator):
    """E4M3 codes [rows, k] and E8M0 exponents [rows, sf_cols(k)] (2^-3 .. 2^3; padding columns 2^0)."""
    codes = (torch.randn(rows, k, generator=generator, device="cuda") * 2).to(torch.float8_e4m3fn)
    exponents = torch.full((rows, sf_cols(k)), 127, dtype=torch.uint8, device="cuda")
    exponents[:, : k // SF_VEC_SIZE] = torch.randint(124, 131, (rows, k // SF_VEC_SIZE), generator=generator, device="cuda", dtype=torch.uint8)
    return codes, exponents


def dequantize(codes, exponents):
    k = codes.shape[-1]
    # A lookup of exact powers of two: torch.exp2 JITs through NVRTC, which rejects sm_107.
    powers = torch.tensor([2.0**e for e in range(-127, 129)], device="cuda")
    scale = powers[exponents[..., : k // SF_VEC_SIZE].long()].repeat_interleave(SF_VEC_SIZE, dim=-1)
    return codes.float() * scale


def assemble_problem(a, a_exp, b, b_exp, layout, alpha, prob):
    """A problem from E4M3 codes and E8M0 exponents: A [rows, k] with ``a_exp`` [rows, sf_cols(k)], B [experts, n, k] with
    ``b_exp`` [experts * n, sf_cols(k)], the inclusive ``layout[1:]`` row ends, alpha [experts], prob [rows, 1, 1]."""
    experts, n, k = b.shape
    return {
        "a": a,
        "a_exp": a_exp,
        "sfa": swizzle_sf(a_exp),
        "b": b,
        "b_exp": b_exp,
        "sfb": torch.cat([swizzle_sf(b_exp[e * n : (e + 1) * n]) for e in range(experts)]),
        "layout": layout,
        "offsets": layout[1:],
        "alpha": alpha,
        "prob": prob,
        "rows": a.shape[0],
        "n": n,
        "k": k,
    }


def make_problem(tokens_per_expert, n, k, seed=0, rows=None):
    """Expert-major MXFP8 rows padded to ROWS_ALIGNMENT (padding rows hold data too), B [experts, n, k] with per-expert SF
    atoms, alpha, prob.  ``rows`` can exceed the experts' rows."""
    experts = len(tokens_per_expert)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    blocks = [padded(tokens) for tokens in tokens_per_expert]
    rows = rows or max(sum(blocks), ROWS_ALIGNMENT)
    a, a_exp = mxfp8_codes(rows, k, generator)
    b, b_exp = mxfp8_codes(experts * n, k, generator)
    layout = torch.zeros(experts + 1, device="cuda", dtype=torch.int32)
    layout[1:] = torch.tensor(blocks, device="cuda", dtype=torch.int64).cumsum(0).to(torch.int32)
    alpha = torch.rand(experts, generator=generator, device="cuda") + 0.5
    prob = torch.rand(rows, 1, 1, generator=generator, device="cuda")
    return assemble_problem(a, a_exp, b.view(experts, n, k), b_exp, layout, alpha, prob)


def reference(problem):
    """float32 D = alpha[e] * prob * (SFA * A) @ (SFB * B[e])^T over the experts' rows."""
    a = dequantize(problem["a"], problem["a_exp"])
    n = problem["n"]
    out = torch.zeros(problem["rows"], n, device="cuda")
    begin = 0
    for expert, end in enumerate(problem["offsets"].tolist()):
        b = dequantize(problem["b"][expert], problem["b_exp"][expert * n : (expert + 1) * n])
        out[begin:end] = problem["alpha"][expert] * (a[begin:end] @ b.T)
        begin = end
    return out * problem["prob"][:, 0, :]


def new_buffers(kernel, problem):
    """NaN-filled BF16 outputs and a zeroed workspace: (D, workspace), or (C, D, workspace) for the GLU kernel."""
    rows, n = problem["rows"], problem["n"]
    workspace = torch.zeros(max(kernel.get_workspace_bytes(), 1), device="cuda", dtype=torch.uint8)
    if kernel.test_glu:
        c = torch.full((rows, n), float("nan"), device="cuda", dtype=torch.bfloat16)
        if getattr(kernel, "test_sfd", False):
            # (C, D, D_col, SFD_row, SFD_col, norm_const, workspace): E4M3 NaN codes and 0xFF (NaN) scales; the column
            # pair is empty without generate_sfd_col.
            cols = n // 2 if kernel.test_sfd_col else 0

            def codes(width):
                return torch.full((rows, width), 0x7F, device="cuda", dtype=torch.uint8).view(torch.float8_e4m3fn)

            def scales(width):
                return torch.full((rows * width // SF_VEC_SIZE,), 0xFF, device="cuda", dtype=torch.uint8)

            norm_const = torch.ones(1, device="cuda", dtype=torch.float32)
            return c, codes(n // 2), codes(cols), scales(n // 2), scales(cols), norm_const, workspace
        d = torch.full((rows, n // 2), float("nan"), device="cuda", dtype=torch.bfloat16)
        return c, d, workspace
    d = torch.full((rows, n), float("nan"), device="cuda", dtype=torch.bfloat16)
    return d, workspace


#: Index of the B major mode, a compile-time constant that is not passed at launch.
B_MAJOR_ARG = 6


def kernel_args(kernel, problem, *buffers):
    import cutlass
    from cutlass.cute.nvgpu import OperandMajorMode
    from cutlass.cute.runtime import from_dlpack

    def tensor(x):
        return from_dlpack(x, assumed_align=16)

    workspace = buffers[-1]
    n, k = problem["n"], problem["k"]
    head = [
        tensor(problem["a"]),
        tensor(problem["b"]),
        tensor(problem["sfb"].view(torch.float8_e8m0fnu)),
        cutlass.Int32(n),
        cutlass.Int32(k),
        cutlass.Int64(k),
        OperandMajorMode.K,
        from_dlpack(workspace, assumed_align=128).iterator,
    ]
    sfa_bytes = problem["a_exp"] if getattr(kernel, "test_compact", False) else problem["sfa"]
    sfa, offsets = tensor(sfa_bytes.view(torch.float8_e8m0fnu)), from_dlpack(problem["offsets"], assumed_align=4)
    alpha, prob = tensor(problem["alpha"]), tensor(problem["prob"])
    if kernel.test_glu:
        # (c, d, d_col, sfa, sfd_row, sfd_col, amax, norm_const, offsets, alpha, prob, bias), SM100 and Rubin alike.
        c_cute, d_cute = tensor(buffers[0]), tensor(buffers[1])
        if getattr(kernel, "test_sfd", False):
            d_col, sfd_row, sfd_col, norm_const = buffers[2:6]
            d_col = tensor(d_col) if kernel.test_sfd_col else None
            sfd_col = tensor(sfd_col.view(torch.float8_e8m0fnu)) if kernel.test_sfd_col else None
            sfd_row = tensor(sfd_row.view(torch.float8_e8m0fnu))
            return head + [c_cute, d_cute, d_col, sfa, sfd_row, sfd_col, None, tensor(norm_const), offsets, alpha, prob, None]
        return head + [c_cute, d_cute, d_cute, sfa, None, None, None, None, offsets, alpha, prob, None]
    d_cute = tensor(buffers[0])
    if kernel.test_rubin:
        # (c, d, d_col, sfa, sfd_row, sfd_col, amax, norm_const, offsets, alpha, bias, prob); C is never stored.
        return head + [d_cute, d_cute, d_cute, sfa, None, None, None, None, offsets, alpha, None, prob]
    # (d, d_col, sfa, sfd_row, sfd_col, amax, norm_const, offsets, alpha, row_scale, bias, prob)
    return head + [d_cute, d_cute, sfa, None, None, None, None, offsets, alpha, None, None, prob]


def launch_args(args):
    return [x for i, x in enumerate(args) if i != B_MAJOR_ARG]


COMPILED = {}


def compile_once(kernel, problem, buffers, max_active_clusters=None):
    """The compiled kernel for this construction and argument layout (tensors pass static layouts)."""
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    from cutlass.utils import HardwareInfo

    if max_active_clusters is None:
        max_active_clusters = HardwareInfo().get_max_active_clusters(2 if kernel.use_2cta_instrs else 1)
    layouts = [problem[name] for name in ("a", "a_exp", "b", "sfa", "sfb", "offsets", "alpha", "prob")] + list(buffers)
    key = (kernel.test_compile_key, max_active_clusters, tuple((tuple(x.shape), tuple(x.stride()), str(x.dtype)) for x in layouts))
    compiled = COMPILED.get(key)
    if compiled is None:
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled = COMPILED[key] = cute.compile(kernel, *kernel_args(kernel, problem, *buffers), max_active_clusters, stream)
    return compiled


class Launcher:
    """One compile of ``kernel`` for a problem layout, launched on fresh NaN-filled D and zeroed workspace buffers."""

    def __init__(self, kernel, problem, max_active_clusters=None):
        self.kernel = kernel
        self._compiled = compile_once(kernel, problem, self.buffers(problem), max_active_clusters)

    def buffers(self, problem):
        return new_buffers(self.kernel, problem)

    def launch(self, problem, buffers, stream=None):
        import cuda.bindings.driver as cuda

        stream = stream or torch.cuda.current_stream()
        self._compiled(*launch_args(kernel_args(self.kernel, problem, *buffers)), cuda.CUstream(stream.cuda_stream))

    def __call__(self, problem, prepare_workspace=None):
        """Fresh buffers, ``prepare_workspace(workspace)``, one launch; returns (outputs over the experts' rows,
        workspace): {"d"}, or {"c", "d"} for the GLU kernel."""
        buffers = self.buffers(problem)
        if prepare_workspace is not None:
            prepare_workspace(buffers[-1])
        self.launch(problem, buffers)
        return outputs(self.kernel, problem, buffers), buffers[-1]


def outputs(kernel, problem, buffers):
    rows = valid_rows(problem)
    if kernel.test_glu:
        if getattr(kernel, "test_sfd", False):
            # Every row is a padded expert row here (the SFD atoms cover whole 128-row blocks).
            named = {"c": buffers[0], "d": buffers[1].view(torch.uint8), "sfd_row": buffers[3]}
            if kernel.test_sfd_col:
                named.update(d_col=buffers[2].view(torch.uint8), sfd_col=buffers[4])
            return {name: x[:rows] if name in ("c", "d", "d_col") else x for name, x in named.items()}
        return {"c": buffers[0][:rows], "d": buffers[1][:rows]}
    return {"d": buffers[0][:rows]}


def valid_rows(problem):
    return int(problem["offsets"][-1].item())


def notify_config(n, chunk, clusters_beside_consumer=1, caller_reset=False):
    from cudnn.gemm.cutedsl.grouped.moe_streaming_output_chunk_tile_scheduler import (
        StreamingOutputNotifyConfig,
        StreamingOutputNotifyMode,
        streaming_output_notify_capacity,
    )

    capacity = streaming_output_notify_capacity(chunk, n)
    return StreamingOutputNotifyConfig(
        StreamingOutputNotifyMode.N_CHUNK, chunk, capacity, caller_reset=caller_reset, clusters_beside_consumer=clusters_beside_consumer
    )


def chunk_signal(kernel, workspace):
    """The kernel's ``signal[]``, one int32 per N chunk, as a view of ``workspace``."""
    from cudnn.gemm.cutedsl.grouped.moe_streaming_output_chunk_tile_scheduler import streaming_output_notify_offsets

    offsets = streaming_output_notify_offsets(kernel)
    return workspace[offsets["signal"] : offsets["signal"] + 4 * offsets["capacity"]].view(torch.int32)


def chunk_words(rows, n, chunk):
    """The int32 words of a row-major BF16 D [rows, n] in each N chunk: (word indices, CSR begin per chunk).  Each chunk
    lists its rows in the order its last tiles finish (even chunks walk M forward, so last row first; odd chunks first
    row first), so a consumer reads the most recently stored rows right after the publication."""
    words_per_row = n // 2
    indices, begins = [], [0]
    for u, first_col in enumerate(range(0, n, chunk)):
        cols = torch.arange(first_col // 2, min(first_col + chunk, n) // 2, device="cuda", dtype=torch.int64)
        row_order = torch.arange(rows, device="cuda", dtype=torch.int64)
        if u % 2 == 0:
            row_order = row_order.flip(0)
        chunk_idx = (row_order[:, None] * words_per_row + cols[None, :]).reshape(-1)
        indices.append(chunk_idx)
        begins.append(begins[-1] + chunk_idx.numel())
    return torch.cat(indices).to(torch.int32), torch.tensor(begins, device="cuda", dtype=torch.int32)


CONSUMER_THREADS = 256


def _consumer_module():
    """The chunk consumer, built once: block u acquire-waits (thread 0) for ``signal[u] == u``, then copies chunk u's
    words of ``src`` into ``dst`` with gpu-scope loads, so every chunk is read the moment it is published."""
    import cutlass
    import cutlass.cute as cute
    from cutlass.cutlass_dsl import Int32

    @cute.kernel
    def consume(signal: cute.Tensor, src: cute.Tensor, dst: cute.Tensor, words: cute.Tensor, begins: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        u, _, _ = cute.arch.block_idx()
        if tidx == 0:
            seen = Int32(-1)
            while seen != u:
                seen = cute.arch.load((signal.iterator + u).llvm_ptr, Int32, sem="acquire", scope="gpu")
        cute.arch.sync_threads()
        for i in cutlass.range(begins[u] + tidx, begins[u + 1], CONSUMER_THREADS, unroll=1):
            word = words[i]
            dst[word] = cute.arch.load((src.iterator + word).llvm_ptr, Int32, sem="relaxed", scope="gpu")

    @cute.jit
    def launch(
        signal: cute.Tensor,
        src: cute.Tensor,
        dst: cute.Tensor,
        words: cute.Tensor,
        begins: cute.Tensor,
        chunks: cutlass.Constexpr,
        stream,
        use_pdl: cutlass.Constexpr,
    ):
        consume(signal, src, dst, words, begins).launch(grid=(chunks, 1, 1), block=(CONSUMER_THREADS, 1, 1), stream=stream, use_pdl=use_pdl)

    return launch


_CONSUMERS = {}


def chunk_consumer(signal, d, words, begins, use_pdl=False):
    """A launcher of the chunk consumer over ``d``'s words; returns (launch(stream), snapshot): ``snapshot`` is the
    int32 copy each chunk's words land in, -1 bytes where nothing was copied."""
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    src = d.view(torch.int32).view(-1)
    snapshot = torch.full_like(src, -1)
    chunks = begins.numel() - 1
    args = [from_dlpack(x, assumed_align=4) for x in (signal, src, snapshot, words, begins)]
    key = (use_pdl, tuple(tuple(x.shape) for x in (signal, src, words, begins)))
    if key not in _CONSUMERS:
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        _CONSUMERS[key] = cute.compile(_consumer_module(), *args, chunks, stream, use_pdl)

    def launch(stream):
        _CONSUMERS[key](*args, cuda.CUstream(stream.cuda_stream))

    return launch, snapshot


# ---- Input direction (signal[] producer -> GLU) ----

#: E4M3 0x7F and E8M0 0xFF are NaN: a tile read before its producer wrote it carries NaN into C and D.
A_POISON, SF_POISON = 0x7F, 0xFF


def sfa_tile_bytes(k):
    """SFA bytes of one 256-row M tile: two whole 128-row SF-atom rows (the contract's ``256 * Kc``)."""
    return ROWS_ALIGNMENT * sf_cols(k)


def signal_ready(kernel):
    from cudnn.gemm.cutedsl.grouped.moe_kernel_helpers import reset_streaming_signal_workspace

    return lambda workspace: reset_streaming_signal_workspace(workspace, kernel, all_complete=True)


def signal_view(kernel, workspace):
    from cudnn.gemm.cutedsl.grouped.moe_kernel_helpers import streaming_signal_workspace_offsets

    offsets = streaming_signal_workspace_offsets(kernel)
    word = offsets["signal"] // 4
    return workspace.view(torch.int32)[word : word + offsets["streaming_signal_capacity"]]


def poison_inputs(problem, late):
    """Overwrite the operands ``late`` names ("a", "sfa" or "both") with NaN codes, both SFA forms (atoms and compact
    rows); returns their final bytes."""
    final = {"a": problem["a"].view(torch.uint8).clone(), "sfa": problem["sfa"].clone(), "a_exp": problem["a_exp"].clone()}
    if late in ("a", "both"):
        problem["a"].view(torch.uint8).fill_(A_POISON)
    if late in ("sfa", "both"):
        problem["sfa"].fill_(SF_POISON)
        problem["a_exp"].fill_(SF_POISON)
    return final


def produce_tiles(problem, final, late, signal, order):
    """The fake producer, enqueued on the current stream: per M tile in ``order``, write the late operands' rows and
    SFA bytes (the atom bytes and the compact rows alike), then ``signal[t] = t`` (stream-ordered: the GEMM acquires the
    flag)."""
    k = problem["k"]
    a_bytes, sf_bytes = problem["a"].view(torch.uint8), sfa_tile_bytes(k)
    for tile in order:
        rows = slice(tile * ROWS_ALIGNMENT, (tile + 1) * ROWS_ALIGNMENT)
        if late in ("a", "both"):
            a_bytes[rows].copy_(final["a"][rows])
        if late in ("sfa", "both"):
            sf = slice(tile * sf_bytes, (tile + 1) * sf_bytes)
            problem["sfa"][sf].copy_(final["sfa"][sf])
            problem["a_exp"][rows].copy_(final["a_exp"][rows])
        signal[tile].fill_(tile)


def release_order(order, capacity):
    if order == "in_order":
        return list(range(capacity))
    if order == "reverse":
        return list(reversed(range(capacity)))
    return torch.randperm(capacity, generator=torch.Generator().manual_seed(capacity)).tolist()


# ---- Batch invariance (shared by the determinism suites) ----

# Batch invariance: one target expert keeps its rows, codes, scales, B, alpha and prob while its neighbors change.
# Four expert slots each (empty slots pad the count), so one compiled kernel serves every packing.
BANK_TOKENS = {"target": 700, "n300": 300, "n513": 513, "r1": 1, "r255": 255, "r257": 257}
PACKINGS = {
    "alone": ["target", None, None, None],
    "neighbors": ["n300", "target", "n513", None],
    "empty_neighbors": [None, "target", None, None],
    "ragged": ["r1", "r255", "target", "r257"],
    "reordered": ["n513", "n300", None, "target"],
    "duplicated": ["target", "n300", "target", None],
}
SLOTS = 4
BATCH_ROWS = max(sum(padded(BANK_TOKENS[key]) for key in packing if key) for packing in PACKINGS.values())


def expert_bank(n, k):
    """Each expert quantized once: its A rows (codes and exponents), B, alpha and prob."""
    generator = torch.Generator(device="cuda").manual_seed(1234)
    bank = {}
    for key, tokens in BANK_TOKENS.items():
        rows = padded(tokens)
        a, a_exp = mxfp8_codes(rows, k, generator)
        b, b_exp = mxfp8_codes(n, k, generator)
        alpha = torch.rand(1, generator=generator, device="cuda") + 0.5
        bank[key] = dict(a=a, a_exp=a_exp, b=b, b_exp=b_exp, alpha=alpha, prob=torch.rand(rows, 1, 1, generator=generator, device="cuda"))
    return bank


def pack(bank, packing, n, k):
    """A problem of SLOTS experts in ``packing`` order over BATCH_ROWS rows; returns it and the target's row ranges."""
    a = torch.zeros(BATCH_ROWS, k, device="cuda", dtype=torch.float8_e4m3fn)
    a_exp = torch.full((BATCH_ROWS, sf_cols(k)), 127, device="cuda", dtype=torch.uint8)
    b = torch.zeros(SLOTS, n, k, device="cuda", dtype=torch.float8_e4m3fn)
    b_exp = torch.full((SLOTS * n, sf_cols(k)), 127, device="cuda", dtype=torch.uint8)
    alpha = torch.ones(SLOTS, device="cuda")
    prob = torch.ones(BATCH_ROWS, 1, 1, device="cuda")
    layout = torch.zeros(SLOTS + 1, device="cuda", dtype=torch.int32)
    begin, target_rows = 0, []
    for slot, key in enumerate(packing):
        if key is not None:
            expert = bank[key]
            end = begin + expert["a"].shape[0]
            a[begin:end], a_exp[begin:end], prob[begin:end] = expert["a"], expert["a_exp"], expert["prob"]
            b[slot], b_exp[slot * n : (slot + 1) * n], alpha[slot] = expert["b"], expert["b_exp"], expert["alpha"][0]
            if key == "target":
                target_rows.append(slice(begin, end))
            begin = end
        layout[slot + 1] = begin
    return assemble_problem(a, a_exp, b, b_exp, layout, alpha, prob), target_rows

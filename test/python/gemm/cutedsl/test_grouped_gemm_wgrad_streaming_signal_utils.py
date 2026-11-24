# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the wgrad streaming input signal tests: problems, kernel launchers and a delayed producer.

The kernels are compiled directly (the signal is a kernel option, as for the forward signal), with the token extent
dynamic so one compile serves every token count of a configuration.
"""

import random

import torch

from gemm.cutedsl.test_grouped_gemm_wgrad_bf16_utils import make_grouped_gemm_wgrad_bf16_problem
from gemm.cutedsl.test_grouped_gemm_wgrad_utils import (
    _wgrad_assemble_scales_2d2d,
    _wgrad_create_scale_tensor,
    _wgrad_independent_reference,
)

TOKEN_TILE = 256
# Per-expert token alignment of each precision's contract: BF16 256, MXFP8 128 (one K tile).
ALIGN = {"bf16": 256, "mxfp8": 128}
SF_VEC = 32


def is_rubin():
    return torch.cuda.get_device_capability() == (10, 7)


def split_tokens(total, experts, empty, seed):
    """``total`` routed tokens over ``experts``, none to the experts in ``empty``; deterministic."""
    rng = random.Random(seed)
    live = [e for e in range(experts) if e not in empty]
    counts = [0] * experts
    for _ in range(total):
        counts[rng.choice(live)] += 1
    return counts


def pad_groups(counts, kind):
    align = ALIGN[kind]
    return tuple(-(-c // align) * align for c in counts)


def make_mxfp8_problem(groups, m, n, seed, *, initial_value=None, logical=None):
    """MXFP8 E4M3/E8M0 operands with exact values ({-1, 0, 1} data, {1, 2} scales); ``logical`` = (a, b, sfa, sfb)
    per-expert pieces to reuse (batch invariance)."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    tokens = sum(groups)
    if logical is None:
        logical_a = torch.randint(-1, 2, (m, tokens), device="cuda", generator=generator).to(torch.bfloat16)
        logical_b = torch.randint(-1, 2, (tokens, n), device="cuda", generator=generator).to(torch.bfloat16)
        torch.manual_seed(seed)
        raw_sfa = [_wgrad_create_scale_tensor((m, -(-k // SF_VEC)), torch.float8_e8m0fnu) for k in groups]
        raw_sfb = [_wgrad_create_scale_tensor((n, -(-k // SF_VEC)), torch.float8_e8m0fnu) for k in groups]
    else:
        logical_a, logical_b, raw_sfa, raw_sfb = logical
    reference = _wgrad_independent_reference(logical_a.float(), logical_b.float(), raw_sfa, raw_sfb, list(groups), SF_VEC, torch.bfloat16)
    if initial_value is not None:
        reference = (reference.float() + initial_value).to(torch.bfloat16)
    return {
        "kind": "mxfp8",
        "a": logical_a.to(torch.float8_e4m3fn).contiguous(),
        "b": logical_b.to(torch.float8_e4m3fn).t().contiguous().t(),
        "logical_a": logical_a,
        "logical_b": logical_b,
        "raw_sfa": raw_sfa,
        "raw_sfb": raw_sfb,
        "sfa": _wgrad_assemble_scales_2d2d(raw_sfa, m),
        "sfb": _wgrad_assemble_scales_2d2d(raw_sfb, n),
        "offsets": torch.tensor(groups, dtype=torch.int32, device="cuda").cumsum(0, dtype=torch.int32),
        "output": torch.empty((len(groups), m, n), dtype=torch.bfloat16, device="cuda"),
        "reference": reference,
        "group_k_list": tuple(groups),
        "m": m,
        "n": n,
        "experts": len(groups),
    }


def make_problem(kind, groups, m, n, seed=0, *, input_order="tensor2d", discrete=False, initial_value=None):
    if kind == "bf16":
        problem = make_grouped_gemm_wgrad_bf16_problem(
            group_k_list=tuple(groups), m=m, n=n, input_order=input_order, discrete=discrete, initial_value=initial_value
        )
        problem["kind"] = "bf16"
        return problem
    assert input_order == "tensor2d"
    return make_mxfp8_problem(groups, m, n, seed, initial_value=initial_value)


def token_rows(problem):
    return sum(problem["group_k_list"])


def capacity_for(problem):
    return max(1, -(-token_rows(problem) // TOKEN_TILE))


# ---------------------------------------------------------------------------
# Kernel launchers
# ---------------------------------------------------------------------------

_COMPILED = {}


class WgradLauncher:
    """One kernel configuration; ``run`` compiles it on first use (token extent dynamic) and launches it."""

    def __init__(
        self,
        kind,
        experts,
        *,
        tile=(256, 256),
        cluster=(2, 1),
        capacity=None,
        counter=True,
        accumulate=False,
        discrete=False,
        input_order="tensor2d",
        max_active_clusters=None,
        dynamic_tokens=True,
    ):
        import cutlass
        from cudnn.gemm.cutedsl.grouped.moe_utils import MoEWeightMode, WGradInputOrder

        self.kind = kind
        self.discrete = discrete
        self.dynamic_tokens = dynamic_tokens
        self.max_active_clusters = max_active_clusters
        options = dict(
            acc_dtype=cutlass.Float32,
            use_2cta_instrs=tile[0] == 256,
            mma_tiler_mn=tile,
            cluster_shape_mn=cluster,
            accumulate_on_output=accumulate,
            expert_cnt=experts,
            weight_mode=MoEWeightMode.DISCRETE if discrete else MoEWeightMode.DENSE,
            input_order=WGradInputOrder.TensorRagged if input_order == "tensor_ragged" else WGradInputOrder.Tensor2D,
            atomic_counter_claim=counter,
        )
        if capacity is not None:
            options["streaming_signal_capacity"] = capacity
        self.key = (kind, tuple(sorted((k, str(v)) for k, v in options.items())), max_active_clusters, dynamic_tokens)
        if kind == "bf16":
            from cudnn.gemm.cutedsl.grouped.wgrad.moe_grouped_gemm_wgrad import MoEGroupedGemmWgradBF16Kernel

            self.kernel = MoEGroupedGemmWgradBF16Kernel(**options)
        else:
            from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad import BlockScaledMoEGroupedGemmWgradKernel

            cls = BlockScaledMoEGroupedGemmWgradKernel
            if is_rubin():
                from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad_rubin import BlockScaledMoEGroupedGemmWgradRubinKernel

                cls = BlockScaledMoEGroupedGemmWgradRubinKernel
            self.kernel = cls(sf_vec_size=SF_VEC, **options)
        self.cluster = cluster

    def workspace(self):
        return torch.zeros(max(self.kernel.get_workspace_bytes(), 1), dtype=torch.uint8, device="cuda")

    def _args(self, problem, output, workspace):
        from cutlass.cute.runtime import from_dlpack

        def tokens_dynamic(x, mode, divisibility):
            t = from_dlpack(x, assumed_align=16)
            if self.dynamic_tokens:
                t = t.mark_compact_shape_dynamic(mode=mode, stride_order=x.dim_order(), divisibility=divisibility)
            return t

        token_div = 256 if self.kind == "bf16" else 16
        a = tokens_dynamic(problem["a"], 1, token_div)
        b = tokens_dynamic(problem["b"], 0, token_div)
        if self.discrete:
            ptrs = torch.tensor([output[e].data_ptr() for e in range(output.shape[0])], dtype=torch.int64, device="cuda")
            out = from_dlpack(ptrs, assumed_align=8).iterator
            single = from_dlpack(output[0], assumed_align=16)
            keep = [ptrs]
        else:
            out = from_dlpack(output, assumed_align=16)
            single = None
            keep = []
        offs = from_dlpack(problem["offsets"], assumed_align=4)
        ws = from_dlpack(workspace, assumed_align=128)
        if self.kind == "bf16":
            return (a, b, out, offs, ws), (single,), keep
        sfa = tokens_dynamic(problem["sfa"], 1, 4)
        sfb = tokens_dynamic(problem["sfb"], 1, 4)
        return (a, b, sfa, sfb, out, offs, ws), (None, None, single), keep

    def run(self, problem, output, workspace, stream=None):
        """Launch on ``stream`` (default: current); the caller prepared ``workspace``."""
        import cuda.bindings.driver as cuda
        import cutlass.cute as cute
        from cutlass.utils import HardwareInfo

        stream = stream or torch.cuda.current_stream()
        cu_stream = cuda.CUstream(stream.cuda_stream)
        head, tail, keep = self._args(problem, output, workspace)
        key = self.key + (() if self.dynamic_tokens else (tuple(problem["a"].shape), tuple(problem["b"].shape)),)
        compiled = _COMPILED.get(key)
        if compiled is None:
            mac = self.max_active_clusters or HardwareInfo().get_max_active_clusters(self.cluster[0] * self.cluster[1])
            compiled = cute.compile(self.kernel, *head, mac, cu_stream, *tail)
            _COMPILED[key] = compiled
        with torch.cuda.stream(stream):
            compiled(*head, cu_stream, *tail)
            # The pointer array must outlive the launch.
            self._keep = keep


def reset_signal(launcher, workspace, *, all_complete, stream=None):
    from cudnn.gemm.cutedsl.grouped.wgrad.streaming_signal import reset_streaming_signal_workspace

    with torch.cuda.stream(stream or torch.cuda.current_stream()):
        reset_streaming_signal_workspace(workspace, launcher.kernel, all_complete=all_complete)


def signal_view(launcher, workspace):
    from cudnn.gemm.cutedsl.grouped.wgrad.streaming_signal import streaming_signal_workspace_offsets

    offsets = streaming_signal_workspace_offsets(launcher.kernel)
    word = offsets["signal"] // 4
    return workspace.view(torch.int32)[word : word + offsets["streaming_signal_capacity"]]


def fresh_output(problem, initial_value=None):
    if initial_value is None:
        return torch.full_like(problem["output"], float("nan"))
    return torch.full_like(problem["output"], initial_value)


def run_ready(launcher, problem, *, initial_value=None, all_complete=True):
    """One launch with every signal[] entry ready (or none, with ``all_complete=False``); returns the output."""
    output = fresh_output(problem, initial_value)
    workspace = launcher.workspace()
    if getattr(launcher.kernel, "streaming_signal_capacity", None) is not None:
        reset_signal(launcher, workspace, all_complete=all_complete)
    launcher.run(problem, output, workspace)
    torch.cuda.synchronize()
    return output


# ---------------------------------------------------------------------------
# Delayed producer: token tiles land out of order, each published late by a kernel
# ---------------------------------------------------------------------------


class _Publisher:
    """One thread waits ``delay_ns`` (global timer), then stores ``signal[tile] = tile`` with a gpu-scope release."""

    def __init__(self):
        import cutlass
        import cutlass.cute as cute

        @cute.kernel
        def publish_kernel(signal: cute.Tensor, tile: cutlass.Int32, delay_ns: cutlass.Int32):
            if cute.arch.thread_idx()[0] == 0:
                start = cute.arch.globaltimer()
                now = start
                while now - start < cutlass.Int64(delay_ns):
                    now = cute.arch.globaltimer()
                cute.arch.atomic_exch((signal.iterator + tile).llvm_ptr, tile, sem="release", scope="gpu")

        publish_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)

        @cute.jit
        def publish(signal: cute.Tensor, tile: cutlass.Int32, delay_ns: cutlass.Int32, stream):
            publish_kernel(signal, tile, delay_ns).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

        self._publish = publish
        self._compiled = None

    def __call__(self, signal, tile, delay_ns, stream):
        import cuda.bindings.driver as cuda
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack

        cu_stream = cuda.CUstream(stream.cuda_stream)
        signal_arg = from_dlpack(signal, assumed_align=4).mark_layout_dynamic()
        if self._compiled is None:
            self._compiled = cute.compile(self._publish, signal_arg, cutlass.Int32(0), cutlass.Int32(0), cu_stream)
        self._compiled(signal_arg, cutlass.Int32(tile), cutlass.Int32(delay_ns), cu_stream)


_PUBLISHER = []


def publisher():
    if not _PUBLISHER:
        _PUBLISHER.append(_Publisher())
    return _PUBLISHER[0]


def tile_order(tiles, order, seed=0):
    if order == "reversed":
        return list(reversed(range(tiles)))
    if order == "shuffled":
        result = list(range(tiles))
        random.Random(seed).shuffle(result)
        return result
    assert order == "in_order"
    return list(range(tiles))


def _token_slices(problem, tile):
    rows = slice(tile * TOKEN_TILE, min((tile + 1) * TOKEN_TILE, token_rows(problem)))
    return (slice(None), rows), (rows, slice(None))


def poison_operands(problem):
    """NaN in every token row of A and B (E4M3 NaN is 0x7F); returns the final values to land later."""
    final = (problem["a"].clone(), problem["b"].clone())
    for x in (problem["a"], problem["b"]):
        if x.dtype == torch.float8_e4m3fn:
            x.view(torch.uint8).fill_(0x7F)
        else:
            x.fill_(float("nan"))
    return final


def run_with_delayed_producer(launcher, problem, final, *, order, seed=0, delay_ns=20_000, initial_value=None):
    """The GEMM starts with every signal[] entry -1 and NaN operands. A producer stream then writes token tile after
    token tile in ``order`` and, per tile, a kernel publishes it ``delay_ns`` later. Reading any tile early carries NaN
    into dW. Returns the output."""
    tiles = -(-token_rows(problem) // TOKEN_TILE)
    output = fresh_output(problem, initial_value)
    workspace = launcher.workspace()
    # One launch with every tile ready loads the module: under lazy loading a first launch waits for running work.
    reset_signal(launcher, workspace, all_complete=True)
    launcher.run(problem, torch.empty_like(output), workspace)
    publish = publisher()
    torch.cuda.synchronize()
    gemm_stream, producer_stream = torch.cuda.Stream(), torch.cuda.Stream()
    reset_signal(launcher, workspace, all_complete=False)
    signal = signal_view(launcher, workspace)
    gemm_stream.wait_stream(torch.cuda.current_stream())
    producer_stream.wait_stream(torch.cuda.current_stream())
    # Producer work is enqueued before the GEMM, so no host launch waits for the spinning GEMM; the sleep makes the GEMM
    # start waiting before the first row lands.
    with torch.cuda.stream(producer_stream):
        torch.cuda._sleep(20_000_000)
        for tile in tile_order(tiles, order, seed):
            a_rows, b_rows = _token_slices(problem, tile)
            problem["a"][a_rows].copy_(final[0][a_rows])
            problem["b"][b_rows].copy_(final[1][b_rows])
            publish(signal, tile, delay_ns, producer_stream)
    launcher.run(problem, output, workspace, stream=gemm_stream)
    torch.cuda.synchronize()
    expected = torch.full((signal.numel(),), -1, dtype=torch.int32)
    expected[:tiles] = torch.arange(tiles, dtype=torch.int32)
    assert torch.equal(signal.cpu(), expected), "the producer did not publish every token tile"
    return output

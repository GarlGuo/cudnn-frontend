# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming input signal for the MoE grouped GEMM wgrad kernels (2Dx2D, K = tokens).

A producer kernel, such as an expert-parallel dispatch, writes token rows of an operand while the wgrad GEMM runs.

``signal[]``: int32, one entry per token tile of ``STREAMING_SIGNAL_TOKEN_TILE`` (256) rows of the concatenated token
axis, the K offset space of ``offsets``.
- Entry ``t`` covers token rows ``[256 t, 256 (t + 1))`` of every operand the producer writes (A^T columns, B rows,
  and their scale factors when block scaled), padding rows included.
- ``t`` is in ``[0, ceil(offsets[E - 1] / 256))``, at most ``streaming_signal_capacity``.
- Reset to -1 before launch (never 0: any value >= 0 reads as ready).
- The producer stores ``signal[t] = t`` with a gpu-scope release once token tile ``t`` is in memory.

This is the forward contract's M tile of A (docs/operations/moe/streaming_signal_contract.md), so a producer that
publishes that contract feeds wgrad unchanged.

The TMA warp gates each K block: before loading it, it waits for every token tile the block reads, up to the last
valid token row ``offsets[E - 1]``, then acquires them with one fence.  Per work tile it keeps the last token tile it
knows ready and the first K block that reads past it, so a K block costs one compare unless it reaches a new tile.
The claim order (``atomic_counter_claim``) does not change, so the output bytes equal those of the kernel without the
signal.

Workspace: ``[TMA descriptors][claim counter, 4 B][signal[capacity]]``.
"""

from typing import Dict

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Int32, dsl_user_op
from cutlass.cute.typing import AddressSpace

#: Token rows per ``signal[]`` entry: the forward contract's cluster M tile.
STREAMING_SIGNAL_TOKEN_TILE = 256

#: Bytes from the claim counter to ``signal[]``, as in the forward contract.
STREAMING_SIGNAL_OFFSET = 4


def check_streaming_signal_options(streaming_signal_capacity, atomic_counter_claim: bool) -> None:
    """Build-time checks of the kernel option; ``None`` is off."""
    if streaming_signal_capacity is None:
        return
    if isinstance(streaming_signal_capacity, bool) or not isinstance(streaming_signal_capacity, int) or streaming_signal_capacity < 1:
        raise ValueError(f"streaming_signal_capacity must be a positive int, got {streaming_signal_capacity!r}")
    if not atomic_counter_claim:
        raise ValueError("streaming_signal_capacity requires atomic_counter_claim=True")
    if streaming_signal_capacity > (2**31 - 1) // 4:
        raise ValueError(f"streaming_signal_capacity {streaming_signal_capacity} does not fit an int32 byte offset")


def streaming_signal_capacity_for_tokens(token_rows: int) -> int:
    """``signal[]`` entries that cover ``token_rows`` allocated token rows."""
    return max(1, -(-token_rows // STREAMING_SIGNAL_TOKEN_TILE))


def check_streaming_signal_capacity(streaming_signal_capacity, token_rows) -> None:
    """Trace-time check that ``signal[]`` covers the allocated token rows; a dynamic extent cannot be checked."""
    if streaming_signal_capacity is None or not isinstance(token_rows, int):
        return
    needed = streaming_signal_capacity_for_tokens(token_rows)
    if streaming_signal_capacity < needed:
        raise ValueError(f"streaming_signal_capacity {streaming_signal_capacity} < {needed} token tiles of {token_rows} rows")


def streaming_signal_workspace_bytes(streaming_signal_capacity) -> int:
    """Workspace bytes of ``signal[]`` after the claim counter (0 when off)."""
    return 0 if streaming_signal_capacity is None else 4 * streaming_signal_capacity


@cute.jit
def streaming_signal_ptr(counter_ptr):
    """Base of ``signal[]`` as an ``Int32`` pointer, derived from the claim counter's pointer."""
    return cute.make_ptr(Int32, counter_ptr.toint() + STREAMING_SIGNAL_OFFSET, AddressSpace.gmem, assumed_align=4)


@dsl_user_op
def fence_acquire_gpu(*, loc=None, ip=None) -> None:
    """``fence.acquire.gpu`` (PTX ISA 8.6, sm_90+): after a strong read of a location that observed a gpu-scope
    release, it completes an acquire pattern on that location.  SASS: ``CCTL.IVALL`` alone, without the ``MEMBAR`` of
    ``fence.acq_rel.gpu``."""
    llvm.inline_asm(None, [], "fence.acquire.gpu;", "~{memory}", has_side_effects=True, asm_dialect=0, loc=loc, ip=ip)


@cute.jit
def wait_k_block_token_tiles(signal_ptr, ready_through: Int32, token_offset: Int32, k_block: Int32, valid_rows: Int32, block_k: cutlass.Constexpr):
    """Wait, without bound, for the token tiles K block ``k_block`` of the expert at ``token_offset`` reads past
    ``ready_through``; return the new ``ready_through`` and the first K block that reads past it (INT32_MAX if no valid
    row does).

    The block reads token rows up to ``min(token_offset + (k_block + 1) * block_k, valid_rows) - 1``. The caller calls
    this only for a block that reads past ``ready_through``.  Relaxed gpu-scope polls (no L1 invalidation per poll); once
    every tile reads ready, ``fence.acquire.gpu`` makes each poll that saw a tile ready an acquire of the producer's
    release, and ``fence.proxy.async.global`` orders the producer's generic-proxy stores before this thread's later TMA
    loads."""
    needed = (cutlass.min(token_offset + (k_block + 1) * block_k, valid_rows) - 1) // STREAMING_SIGNAL_TOKEN_TILE
    while ready_through < needed:
        ready_through = ready_through + 1
        # One load point in the loop: a load ahead of it would make the compiler duplicate the loop.
        value = Int32(-1)
        while value < Int32(0):
            value = cute.arch.load((signal_ptr + ready_through).llvm_ptr, Int32, sem="relaxed", scope="gpu")
    fence_acquire_gpu()
    cute.arch.fence_proxy("async.global")
    next_k_block = Int32(2**31 - 1)
    ready_end = (ready_through + 1) * STREAMING_SIGNAL_TOKEN_TILE
    if ready_end < valid_rows:
        # ready_end > token_offset: the first block whose rows reach past ready_end.
        next_k_block = (ready_end - token_offset) // block_k
    return ready_through, next_k_block


# ---- Host side: the workspace byte map and its reset ----


def streaming_signal_workspace_offsets(kernel) -> Dict[str, int]:
    """Byte offsets ``counter``/``signal`` plus ``streaming_signal_capacity`` and ``total_bytes`` for one wgrad kernel
    INSTANCE built with ``streaming_signal_capacity``."""
    capacity = kernel.streaming_signal_capacity
    assert capacity is not None, "the kernel was built without streaming_signal_capacity"
    counter = kernel.get_desc_workspace_bytes()
    assert kernel.get_workspace_bytes() == counter + STREAMING_SIGNAL_OFFSET + 4 * capacity
    return {
        "counter": counter,
        "signal": counter + STREAMING_SIGNAL_OFFSET,
        "streaming_signal_capacity": capacity,
        "total_bytes": kernel.get_workspace_bytes(),
    }


def reset_streaming_signal_workspace(workspace, kernel, all_complete: bool = False) -> None:
    """Make ``workspace`` ready for one launch: ``signal[]`` -1, or ``signal[t] = t`` with ``all_complete`` (every token
    tile ready, as with no producer).  The kernel's helper resets the claim counter itself."""
    import torch

    offsets = streaming_signal_workspace_offsets(kernel)
    words = workspace.view(torch.int32)
    signal_word = offsets["signal"] // 4
    signal = words[signal_word : signal_word + offsets["streaming_signal_capacity"]]
    if all_complete:
        signal.copy_(torch.arange(signal.numel(), dtype=torch.int32, device=workspace.device))
    else:
        # Never zero-fill signal[]: any value >= 0 reads as complete.
        signal.fill_(-1)

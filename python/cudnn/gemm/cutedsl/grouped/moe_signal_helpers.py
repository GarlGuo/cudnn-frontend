# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for the streaming signal MoE schedulers.  Device side: cluster fences and mbarrier ops.
Host side: the workspace byte map and its reset."""

from typing import Dict

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Boolean, Int32, dsl_user_op
from cutlass.cute.typing import AddressSpace
from cutlass._mlir.dialects import llvm, nvvm

#: Bytes from the claim counter to ``signal[]``.  ``streaming_signal_workspace_offsets`` is the host's
#: view of the same map.
STREAMING_SIGNAL_OFFSET = 4


@dsl_user_op
def streaming_signal_ptr(counter_ptr, *, loc=None, ip=None):
    """Base of ``signal[]`` as an ``Int32`` pointer (``+ t`` advances ``4 * t`` bytes), derived
    from the claim counter's pointer so the scheduler serializes no second one."""
    return cute.make_ptr(
        Int32,
        counter_ptr.toint() + STREAMING_SIGNAL_OFFSET,
        AddressSpace.gmem,
        assumed_align=4,
    )


@dsl_user_op
def mbarrier_arrive_expect_tx_on_peer_relaxed_cluster(mbar_ptr, tx_count: Int32, cta_rank_in_cluster: cutlass.Constexpr, *, loc=None, ip=None) -> None:
    """Arrive + expect_tx on a peer CTA's mbarrier, ``.relaxed.cluster`` (the default would be ``.release.cta``)."""
    cute.arch.mbarrier_arrive_and_expect_tx(
        mbar_ptr, tx_count, peer_cta_rank_in_cluster=cta_rank_in_cluster, relaxed=True, scope=nvvm.MemScopeKind.CLUSTER, loc=loc, ip=ip
    )


@dsl_user_op
def fence_release_cluster(*, loc=None, ip=None) -> None:
    """``fence.release.cluster``: orders this CTA's earlier stores before a peer observes the next arrive."""
    llvm.fence(llvm.AtomicOrdering.release, syncscope="cluster", loc=loc, ip=ip)


@dsl_user_op
def mbarrier_try_wait_parity_acquire_cluster(mbar_ptr, phase: Int32, *, loc=None, ip=None) -> Boolean:
    """``mbarrier.try_wait.parity.acquire.cluster``: the default ``.acquire.cta`` does not synchronize with a peer's arrive."""
    return Boolean(
        nvvm.mbarrier_wait_parity(
            mbar_ptr.to_llvm_ptr(loc=loc, ip=ip),
            Int32(phase).ir_value(loc=loc, ip=ip),
            nvvm.MBarrierWaitKind.TRY,
            scope=nvvm.MemScopeKind.CLUSTER,
            order=nvvm.MemOrderKind.ACQUIRE,
            loc=loc,
            ip=ip,
        )
    )


# ---- The signal workspace byte map: the producer's and this consumer's contract ----


def streaming_scheduler_state_offset(streaming_signal_capacity: int) -> int:
    """Bytes from the claim counter to the scheduler state: past ``signal[]``, rounded up to 128 so the producer-written
    flags and the scheduler's own words share no 128-byte line."""
    return -(-(STREAMING_SIGNAL_OFFSET + 4 * streaming_signal_capacity) // 128) * 128


def streaming_signal_workspace_offsets(kernel) -> Dict[str, int]:
    """Byte offsets ``counter``/``signal``/``state`` plus ``streaming_signal_capacity`` and ``total_bytes``,
    for one signal-scheduler kernel INSTANCE."""
    desc_workspace_bytes = kernel.get_desc_workspace_bytes()
    capacity = kernel.streaming_signal_capacity
    # The state holds 2 words per expert, assigned_claim_count and a 3-word record per M-tile: a kernel that sized
    # it otherwise overruns the workspace, and the scheduler hangs or faults.
    state_bytes = kernel.get_workspace_bytes() - (desc_workspace_bytes + streaming_scheduler_state_offset(capacity))
    assert state_bytes >= 4 * (
        2 * kernel.expert_cnt + 1 + 3 * capacity
    ), f"state of {state_bytes} bytes: {kernel.expert_cnt} experts, capacity {capacity} do not fit"
    return {
        # The claim counter is the stock dynamic scheduler's own counter word.
        "counter": desc_workspace_bytes,
        # One ready flag per tile: signal[t] is t once tile t is complete.  streaming_signal_ptr()
        # names this same address on the device.
        "signal": desc_workspace_bytes + STREAMING_SIGNAL_OFFSET,
        "state": desc_workspace_bytes + streaming_scheduler_state_offset(capacity),
        "streaming_signal_capacity": capacity,
        "total_bytes": kernel.get_workspace_bytes(),
    }


def reset_streaming_signal_workspace(workspace, kernel, all_complete: bool = False) -> None:
    """Make ``workspace`` ready for one launch: claim counter and scheduler state 0, ``signal[]``
    -1, or ``signal[t] = t`` with ``all_complete`` (every tile ready, as with no producer)."""
    import torch

    offsets = streaming_signal_workspace_offsets(kernel)
    # A view of the byte buffer: the offsets are 4-byte aligned because get_desc_workspace_bytes() is.
    int32_words = workspace.view(torch.int32)
    counter_word, signal_word = offsets["counter"] // 4, offsets["signal"] // 4
    int32_words[counter_word : counter_word + 1].fill_(0)
    int32_words[offsets["state"] // 4 : offsets["total_bytes"] // 4].fill_(0)
    signal = int32_words[signal_word : signal_word + offsets["streaming_signal_capacity"]]
    # Never zero-fill signal[]: any value >= 0 reads as complete.
    if all_complete:
        signal.copy_(torch.arange(signal.numel(), dtype=torch.int32, device=workspace.device))
    else:
        signal.fill_(-1)

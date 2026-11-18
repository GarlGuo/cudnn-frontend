# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for the dispatch-signal MoE scheduler.  Device side: inline-PTX fences and cluster
mbarrier ops.  Host side: the workspace byte map and its reset."""

from typing import Dict

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Boolean, Int32, T, const_expr, dsl_user_op
from cutlass.cute.typing import AddressSpace
from cutlass._mlir.dialects import llvm

#: Bytes from the claim counter to ``signal[]``.  ``signal_workspace_offsets`` is the host's
#: view of the same map.
SIGNAL_OFFSET = 4


@dsl_user_op
def signal_ptr(counter_ptr, *, loc=None, ip=None):
    """Base of ``signal[]`` as an ``Int32`` pointer (``+ t`` advances ``4 * t`` bytes), derived
    from the claim counter's pointer so the scheduler serializes no second one."""
    return cute.make_ptr(
        Int32,
        counter_ptr.toint() + SIGNAL_OFFSET,
        AddressSpace.gmem,
        assumed_align=4,
    )


@dsl_user_op
def mbarrier_arrive_expect_tx_on_peer_relaxed_cluster(
    mbar_ptr,
    tx_count: Int32,
    cta_rank_in_cluster: cutlass.Constexpr,
    *,
    loc=None,
    ip=None,
) -> None:
    """Arrive+expect_tx on a peer CTA mbarrier, ``.relaxed.cluster``: without it the opcode is
    ``.release.cta``, since ``.shared::cluster`` names the state space, not the scope."""
    mbar_addr = llvm.ptrtoint(T.i32(), mbar_ptr.llvm_ptr)
    llvm.inline_asm(
        res=None,
        operands_=[
            mbar_addr,
            Int32(cta_rank_in_cluster).ir_value(),
            tx_count.ir_value(),
        ],
        asm_string="""{{
            .reg .u32 remote_mbar;
            mapa.shared::cluster.u32 remote_mbar, $0, $1;
            mbarrier.arrive.expect_tx.relaxed.cluster.shared::cluster.b64 _, [remote_mbar], $2;
        }}""",
        constraints="r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def fence_release_cluster(*, loc=None, ip=None) -> None:
    """``fence.release.cluster`` via inline PTX; ``cute.arch`` exposes only ``acq_rel``, whose
    acquire half is a dead ``CCTL.IVALL``.  Assembles at PTX ISA 8.6+; ISA Table 20 omits it."""
    llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string="fence.release.cluster;",
        constraints="",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mbarrier_try_wait_parity_acquire_cluster(mbar_ptr, phase: Int32, *, loc=None, ip=None) -> Boolean:
    """``mbarrier.try_wait.parity.acquire.cluster`` via inline PTX: the DSL has no sem/scope knob
    and its default ``.acquire.cta`` is not morally strong with a peer's arrive, fence or not."""
    mbar_addr = llvm.ptrtoint(T.i32(), mbar_ptr.llvm_ptr)
    wait_result = llvm.inline_asm(
        res=T.i32(),
        operands_=[mbar_addr, Int32(phase).ir_value()],
        asm_string="""{{
            .reg .pred P_ht;
            mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P_ht, [$1], $2;
            selp.b32 $0, 1, 0, P_ht;
        }}""",
        constraints="=r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Int32(wait_result) != Int32(0)


# ---- The signal workspace byte map: nccl-ep's producer and this consumer's contract ----


def _ceil_div(x: int, y: int) -> int:
    return -(-x // y)


def scheduler_state_offset(signal_capacity):
    """Bytes from the claim counter to the scheduler state: past ``signal[]``, rounded up to 128 so the producer-written
    flags and the scheduler's own words share no 128-byte line.  An int capacity folds; an Int32 is the launch's."""
    if isinstance(signal_capacity, int):
        return const_expr(_ceil_div(SIGNAL_OFFSET + 4 * signal_capacity, 128) * 128)
    return (SIGNAL_OFFSET + 4 * signal_capacity + 127) & -128


def signal_workspace_offsets(kernel) -> Dict[str, int]:
    """Byte offsets ``counter``/``signal``/``state`` plus ``signal_capacity`` and ``total_bytes``,
    for one signal-scheduler kernel INSTANCE."""
    desc_workspace_bytes = kernel.get_desc_workspace_bytes()
    capacity = const_expr(kernel.signal_capacity)
    # The state must hold a word per expert, assigned_claim_count and a 4-word record per M-tile: a kernel that sized
    # it otherwise overruns the workspace, and the scheduler hangs or faults.
    state_bytes = kernel.get_workspace_bytes() - (desc_workspace_bytes + scheduler_state_offset(capacity))
    assert state_bytes >= 4 * (
        kernel.expert_cnt + 1 + 4 * capacity
    ), f"state of {state_bytes} bytes: {kernel.expert_cnt} experts, capacity {capacity} do not fit"
    return {
        # The claim counter is the stock dynamic scheduler's own counter word.
        "counter": desc_workspace_bytes,
        # One ready flag per tile: signal[t] is t once tile t is complete.  signal_ptr()
        # names this same address on the device.
        "signal": desc_workspace_bytes + SIGNAL_OFFSET,
        "state": desc_workspace_bytes + scheduler_state_offset(capacity),
        "signal_capacity": capacity,
        "total_bytes": kernel.get_workspace_bytes(),
    }


def signal_launch_kwargs(kernel) -> Dict[str, Int32]:
    """``kernel``'s keyword arguments at compile and at every launch: ``signal_capacity`` for a signal consumer that
    takes it at launch (SIGNAL_CAPACITY_AT_LAUNCH), else none."""
    if getattr(kernel, "SIGNAL_CAPACITY_AT_LAUNCH", False) and kernel.use_signal_sched:
        return {"signal_capacity": Int32(kernel.signal_capacity)}
    return {}


def signal_workspace_words(workspace, kernel) -> Dict:
    """``workspace``'s int32 words a reset writes: the claim ``counter``, the scheduler ``state`` and
    ``signal[]`` (``signal_workspace_offsets``)."""
    import torch

    offsets = signal_workspace_offsets(kernel)
    # view() on the byte buffer keeps a single allocation and a single pointer; the
    # offsets are 4-byte aligned because get_desc_workspace_bytes() is.
    int32_words = workspace.view(torch.int32)
    capacity = const_expr(offsets["signal_capacity"])
    return {
        "counter": int32_words[offsets["counter"] // 4 : offsets["counter"] // 4 + 1],
        "state": int32_words[offsets["state"] // 4 : offsets["total_bytes"] // 4],
        "signal": int32_words[offsets["signal"] // 4 : offsets["signal"] // 4 + capacity],
    }


def reset_signal_workspace(workspace, kernel, all_complete: bool = False) -> None:
    """Make ``workspace`` ready for one launch: claim counter and scheduler state 0, ``signal[]``
    -1, or ``signal[t] = t`` with ``all_complete`` (every tile ready, as with no producer)."""
    import torch

    words = signal_workspace_words(workspace, kernel)
    words["counter"].fill_(0)
    words["state"].fill_(0)
    signal = words["signal"]
    # Never zero-fill signal[]: any value >= 0 reads as complete.
    if all_complete:
        signal.copy_(torch.arange(signal.numel(), dtype=torch.int32, device=workspace.device))
    else:
        signal.fill_(-1)

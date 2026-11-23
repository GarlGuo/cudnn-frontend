# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DIAGNOSTIC ONLY (wip/sigsched-rubin-bisect): trace-time knobs and a per-tile event log for bisecting the streaming
signal scheduler's Rubin overhead.  Every knob is read at trace time; unset, nothing is emitted (PTX unchanged).

  SIGDIAG_STOCK_SCHED=1   a signal-scheduler build that runs the stock dynamic scheduler (kernel-side signal code kept)
  SIGDIAG_HYBRID_CLAIM=1  the signal scheduler's claim without batches: claim index -> tile directly (all tiles ready)
  SIGDIAG_HYBRID_BCAST=1  the stock scheduler's cluster broadcast and slot release in the signal scheduler
  TILELOG_ADDR=<device address of int32[1 << 22]>: per CTA and tile, the scheduler's publish, the TMA warp's first
      load, the MMA warp's tile start and the epilogue's tile end (%globaltimer_lo), plus the tile and %smid.
      Layout: counters int32[CTA * 4 + role] in the first 65536 words, then records int32[16] at
      65536 + (CTA * 256 + seq) * 16: [0:3] expert, tile m, tile n, [3] smid, [4 + role] time, [8] 1; the TMA warp
      (role 1) adds [9:16] its coordinates: expert, MMA tile m, MMA tile n, the A tensor's base coordinates (2), the B tensor's
      (1), CTA rank in the cluster.  TILELOG_ROLES=<subset of 0123> logs only those roles (0 scheduler, 1 TMA, 2 MMA,
      3 epilogue).
  SIGDIAG_EPI_SLEEP=<ns> / SIGDIAG_EPI_ATOMIC=1 / SIGDIAG_EPI_STORE=1 / SIGDIAG_EPI_WAITALL=1: at the epilogue's tile end (the
      TILELOG role-3 point), epilogue warp 0: a nanosleep / a returning gmem atomic whose result is used (address:
      TILELOG_SCRATCH) / a plain gmem store (TILELOG_SCRATCH) / cp.async.bulk.wait_group 0 (every D/C store complete,
      not only read).  Which part of the role-3 log removes the Rubin gap.
  SIGDIAG_L2_A / SIGDIAG_L2_B / SIGDIAG_L2_CD = first | last | normal: an L2 eviction-priority cache policy on the
      A + SFA loads / B + SFB loads / C + D + D_col stores (CUTLASS's CacheHintSm90 constants).  Unset: no policy.
  SIGDIAG_NO_FINALIZE=1   a signal-scheduler build without the final cluster barrier (internal_finalize)
  SIGDIAG_STOCK_FINALIZE=1  with SIGDIAG_STOCK_SCHED: add that cluster barrier to the stock scheduler
"""

import os

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import AddressSpace, Int32


def knob(name: str) -> bool:
    return os.environ.get(name, "0") == "1"


def tilelog(role: int, tile=None, extra=None):
    """One thread per CTA and role: append this role's event for the CTA's next tile.  No-op without TILELOG_ADDR."""
    addr = int(os.environ.get("TILELOG_ADDR", "0"))
    if addr == 0:
        return
    from .moe_persistent_scheduler import atomic_add_i32

    bx, by, bz = cute.arch.block_idx()
    gx, gy, _ = cute.arch.grid_dim()
    cta = bx + gx * (by + gy * bz)
    base = cute.make_ptr(Int32, addr, AddressSpace.gmem, assumed_align=16)
    seq = atomic_add_i32((base + cta * 4 + role).llvm_ptr, Int32(1))
    rec = cute.make_tensor(base + 65536 + (cta * 256 + seq) * 16, cute.make_layout((16,)))
    rec[4 + role] = cute.arch.globaltimer_lo()
    if tile is not None:
        rec[0] = tile[0]
        rec[1] = tile[1]
        rec[2] = tile[2]
        rec[3] = cute.arch.smid()
        rec[8] = Int32(1)
    if extra is not None:
        for idx, value in enumerate(extra):
            rec[9 + idx] = value


def tilelog_on(role=None) -> bool:
    """Logging enabled (TILELOG_ADDR) and, with ``role``, that role selected by TILELOG_ROLES (default "0123")."""
    if int(os.environ.get("TILELOG_ADDR", "0")) == 0:
        return False
    return role is None or str(role) in os.environ.get("TILELOG_ROLES", "0123")


def coord_words(iterator, count: int):
    """The first ``count`` integer components of a TMA tensor's coordinate iterator (zeros past its rank)."""
    flat = []

    def walk(x):
        if isinstance(x, (tuple, list)):
            for y in x:
                walk(y)
        else:
            flat.append(x)

    walk(iterator)
    words = [Int32(x) for x in flat[:count]]
    return words + [Int32(0)] * (count - len(words))


def epi_tile_end_knobs_on() -> bool:
    return any(os.environ.get(k, "0") not in ("", "0") for k in ("SIGDIAG_EPI_SLEEP", "SIGDIAG_EPI_ATOMIC", "SIGDIAG_EPI_STORE", "SIGDIAG_EPI_WAITALL"))


def epi_tile_end():
    """One thread of epilogue warp 0 at the tile end (see the module docstring)."""
    from cutlass._mlir.dialects import nvvm

    ns = int(os.environ.get("SIGDIAG_EPI_SLEEP", "0"))
    if ns > 0:
        nvvm.nanosleep(Int32(ns).ir_value())
    scratch = int(os.environ.get("TILELOG_SCRATCH", "0"))
    if knob("SIGDIAG_EPI_ATOMIC") and scratch:
        from .moe_persistent_scheduler import atomic_add_i32

        base = cute.make_ptr(Int32, scratch, AddressSpace.gmem, assumed_align=16)
        old = atomic_add_i32(base.llvm_ptr, Int32(1))
        log = cute.make_tensor(base + 64, cute.make_layout((1 << 16,)))
        log[old & 0xFFFF] = Int32(1)  # uses the atomic's result, as the role-3 log does
    if knob("SIGDIAG_EPI_STORE") and scratch:
        log = cute.make_tensor(cute.make_ptr(Int32, scratch, AddressSpace.gmem, assumed_align=16) + 128, cute.make_layout((1,)))
        log[0] = cute.arch.globaltimer_lo()
    if knob("SIGDIAG_EPI_WAITALL"):
        cute.arch.cp_async_bulk_wait_group(0, read=False)


# CUTLASS cute/arch/copy_sm90_tma.hpp CacheHintSm90 (createpolicy encodings, fraction 1.0).
_L2_POLICY = {"normal": 0x1000000000000000, "first": 0x12F0000000000000, "last": 0x14F0000000000000}


def l2_policy(operand: str):
    """The cache_policy for ``operand`` ("A", "B" or "CD") from SIGDIAG_L2_<operand>, or None (no hint argument)."""
    name = os.environ.get(f"SIGDIAG_L2_{operand}", "")
    if not name:
        return None
    return cutlass.Int64(_L2_POLICY[name])

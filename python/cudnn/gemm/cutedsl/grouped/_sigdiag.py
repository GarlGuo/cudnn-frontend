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
      65536 + (CTA * 256 + seq) * 16: [0:3] expert, tile m, tile n, [3] smid, [4 + role] time, [8] 1.
"""

import os

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import AddressSpace, Int32


def knob(name: str) -> bool:
    return os.environ.get(name, "0") == "1"


def tilelog(role: int, tile=None):
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


def tilelog_on() -> bool:
    return int(os.environ.get("TILELOG_ADDR", "0")) != 0

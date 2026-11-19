# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Expert tail tiles (< 256 rows) for the 2-CTA BF16 grouped GEMMs, run with A and B swapped (weight rows = MMA M,
tile rows = MMA N), so the accumulator is the output tile transposed: TMEM lane = column, TMEM column = row."""

import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05
from cutlass.cutlass_dsl import Int32

from .moe_utils import compute_expert_token_range

#: Rows of a full cluster tile (2-CTA MMA M).
CLUSTER_TILE_M = 256
#: Rows of the TMEM-to-register subtile of a tail accumulator.
TAIL_SUBTILE_ROWS = 32
#: Tails of at most this many rows following a full M tile of their expert fuse with it into one unit that loads each
#: weight K block once for both MMAs; 0 disables fusion.  Only 64: a swapped 128-row tail saves nothing fused.
FUSED_TAIL_MAX_ROWS = 64
#: A tail of up to this many FUSED_TAIL_MAX_ROWS-row chunks, after at least as many full M tiles, rides chunk by
#: chunk in the units of the expert's last full tiles (a 128-row tail: 64 rows in each of the last two).
FUSED_TAIL_CHUNKS = 2
#: ... the second chunk only in experts of fewer rows than this: with more full tiles a lone tail tile's weight K
#: blocks are L2 hits behind them, and two longer units at the end of each N tile's run cost more than they save.
FUSED_TAIL_CHUNKED_MAX_EXPERT_ROWS = 9 * CLUSTER_TILE_M
#: The FC1 signal scheduler fuses a tail only when its rows are ready as its batch opens, and marks each unit that
#: runs a tail chunk with this bit, in its broadcast linear tile index and its work tile's k_tile_cnt.
FUSED_UNIT = 1 << 30


def fused_tail_heights(alignment, max_rows=FUSED_TAIL_MAX_ROWS):
    """Tail heights that run fused with their expert's last full M tile."""
    return tuple(rows for rows in tail_tile_heights(alignment) if rows <= max_rows)


@cute.jit
def fused_tail_chunks(expert_i_row_count: Int32, max_rows: cutlass.Constexpr) -> Int32:
    """Chunks of ``max_rows`` rows in which an expert of ``expert_i_row_count`` rows fuses its tail into its last full M
    tiles (one chunk per tile), else 0."""
    full_tiles = expert_i_row_count // CLUSTER_TILE_M
    tail_rows = expert_i_row_count - full_tiles * CLUSTER_TILE_M
    chunks = (tail_rows + (max_rows - 1)) // max_rows
    fused = Int32(0)
    if tail_rows > 0 and chunks <= FUSED_TAIL_CHUNKS and full_tiles >= chunks:
        if chunks == 1 or expert_i_row_count < FUSED_TAIL_CHUNKED_MAX_EXPERT_ROWS:
            fused = chunks
    return fused


@cute.jit
def fused_tail_rows(expert_i_row_count: Int32, mma_tile_m: Int32, max_rows: cutlass.Constexpr) -> Int32:
    """Rows of the tail chunk fused into full M tile ``mma_tile_m`` of an expert of ``expert_i_row_count`` rows, else 0: the
    tail's last chunk (to the tail's end) rides in the last full tile, the one before it in the tile before."""
    full_tiles = expert_i_row_count // CLUSTER_TILE_M
    chunks = fused_tail_chunks(expert_i_row_count, max_rows)
    fused = Int32(0)
    if chunks > 0 and mma_tile_m >= full_tiles - chunks and mma_tile_m < full_tiles:
        fused = cutlass.min(expert_i_row_count - fused_tail_first_row(expert_i_row_count, mma_tile_m, max_rows), Int32(max_rows))
    return fused


@cute.jit
def fused_tail_first_row(expert_i_row_count: Int32, mma_tile_m: Int32, max_rows: cutlass.Constexpr) -> Int32:
    """First row (in the expert) of the tail chunk fused into full M tile ``mma_tile_m`` (see fused_tail_rows)."""
    full_tiles = expert_i_row_count // CLUSTER_TILE_M
    return full_tiles * CLUSTER_TILE_M + (mma_tile_m - (full_tiles - fused_tail_chunks(expert_i_row_count, max_rows))) * max_rows


@cute.jit
def expert_m_units(expert_i_row_count: Int32, max_rows: cutlass.Constexpr) -> Int32:
    """M work units of an expert: its M tiles, less one when its tail is fused into the last full tile."""
    units = (expert_i_row_count + (CLUSTER_TILE_M - 1)) // CLUSTER_TILE_M
    if cutlass.const_expr(max_rows > 0):
        if fused_tail_chunks(expert_i_row_count, max_rows) > 0:
            units = units - 1
    return units


def tail_tile_heights(alignment):
    """Row counts below 256 an expert's last M tile can have when experts are padded to ``alignment``."""
    if alignment is None or alignment >= CLUSTER_TILE_M:
        return ()
    assert (
        CLUSTER_TILE_M % alignment == 0 and alignment % TAIL_SUBTILE_ROWS == 0
    ), f"expert row alignment {alignment} must divide {CLUSTER_TILE_M} and be a multiple of {TAIL_SUBTILE_ROWS}"
    return tuple(range(alignment, CLUSTER_TILE_M, alignment))


@cute.jit
def m_tile_rows(padded_offsets: cute.Tensor, expert_idx: Int32, mma_tile_m: Int32) -> Int32:
    """Rows of M tile ``mma_tile_m`` of ``expert_idx``: 256, or fewer for its last tile."""
    _, expert_i_row_count = compute_expert_token_range(padded_offsets, expert_idx)
    return cutlass.min(expert_i_row_count - mma_tile_m * CLUSTER_TILE_M, CLUSTER_TILE_M)


def make_tail_mmas(ab_dtype, acc_dtype, heights, weight_major=OperandMajorMode.K):
    """{rows: 2-CTA tiled MMA with M = 256 weight rows and N = rows}; the rows K-major, the weights
    ``weight_major`` (the kernel's B major mode)."""
    return {
        rows: sm100_utils.make_trivial_tiled_mma(ab_dtype, ab_dtype, weight_major, OperandMajorMode.K, acc_dtype, tcgen05.CtaGroup.TWO, (CLUSTER_TILE_M, rows))
        for rows in heights
    }


def tail_row_operand(smem_rows, a_smem_layout_staged, rows):
    """The A stages viewed as the swapped MMA's B operand: each CTA's first ``rows // 2`` rows of every stage."""
    outer = a_smem_layout_staged.outer
    (cta_rows, k_atom), mode_m, mode_k, stages = outer.shape
    assert cta_rows == CLUSTER_TILE_M // 2, f"A stage holds {cta_rows} rows, not {CLUSTER_TILE_M // 2}"
    layout = cute.make_layout(((rows // 2, k_atom), mode_m, mode_k, stages), stride=outer.stride)
    return smem_rows.get_tensor(layout, swizzle=a_smem_layout_staged.inner)


def tail_tmem_load(tidx, acc_tail, acc_dtype, subtile_rows=TAIL_SUBTILE_ROWS):
    """(TMEM load, its per-thread source by row subtile, register fragment) for a tail accumulator.
    Thread ``tidx`` reads TMEM lane ``tidx``: one output column, ``subtile_rows`` rows per subtile."""
    atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(getattr(tcgen05.Repetition, f"x{subtile_rows}")), acc_dtype)
    acc_subtiles = cute.flat_divide(acc_tail[((None, None), 0, 0)], (CLUSTER_TILE_M // 2, subtile_rows))
    tiled_copy = tcgen05.make_tmem_copy(atom, acc_subtiles[(None, None, 0, 0)])
    thr_copy = tiled_copy.get_slice(tidx)
    source = thr_copy.partition_S(acc_subtiles)
    fragment_shape = thr_copy.partition_D(cute.make_identity_tensor((CLUSTER_TILE_M // 2, subtile_rows))).shape
    return tiled_copy, source, cute.make_rmem_tensor(fragment_shape, acc_dtype)


@cute.jit
def tail_mma_rows(real_prob: cute.Tensor, first_row: Int32, rows: cutlass.Constexpr) -> Int32:
    """Rows of a ``rows``-row tail tile from ``first_row`` that the MMA computes: up to its last nonzero-prob row,
    rounded up to ``TAIL_SUBTILE_ROWS`` (later rows have prob 0, so output 0).  Warp-collective."""
    lane_idx = cute.arch.lane_idx()
    last_row_end = Int32(0)
    for subtile in cutlass.range_constexpr(rows // TAIL_SUBTILE_ROWS):
        row = subtile * TAIL_SUBTILE_ROWS + lane_idx
        if real_prob[first_row + row, 0, 0] != cutlass.Float32(0.0):
            last_row_end = row + 1
    last_row_end = cute.arch.warp_redux_sync(last_row_end, "max")
    mma_rows = (last_row_end + TAIL_SUBTILE_ROWS - 1) // TAIL_SUBTILE_ROWS * TAIL_SUBTILE_ROWS
    return cutlass.max(mma_rows, Int32(TAIL_SUBTILE_ROWS))


@cute.jit
def mma_mainloop(tiled_mma, acc, frag_a, frag_b, ab_pipeline, ab_consumer_state, peek_ab_full_status, k_tile_cnt, is_leader_cta):
    """One tile's K loop on the leader CTA: acc = frag_a * frag_b over ``k_tile_cnt`` stages."""
    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
        if is_leader_cta:
            ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)
            num_kblocks = cute.size(frag_a, mode=[2])
            ab_consumer_state_next = ab_consumer_state.clone()
            ab_consumer_state_next.advance()
            if ab_consumer_state_next.count < k_tile_cnt:
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state_next)
            for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                kblock_coord = (None, None, kblock_idx, ab_consumer_state.index)
                cute.gemm(tiled_mma, acc, frag_a[kblock_coord], frag_b[kblock_coord], acc)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            ab_pipeline.consumer_release(ab_consumer_state)
            ab_consumer_state = ab_consumer_state_next
    return ab_consumer_state


@cute.jit
def cluster_m_tile_count(
    padded_offsets: cute.Tensor,
    expert_cnt: cutlass.Constexpr,
    cluster_tile_m: cutlass.Constexpr,
    aligned: cutlass.Constexpr,
    fused_tail_max_rows: cutlass.Constexpr = 0,
) -> Int32:
    """Cluster M tiles over all experts (sum of ceil(rows / cluster_tile_m)); ``aligned``: rows are tile multiples,
    ``fused_tail_max_rows``: count M units instead (expert_m_units).  Warp-collective."""
    total = Int32(0)
    if cutlass.const_expr(aligned):
        total = padded_offsets[expert_cnt - 1] // cluster_tile_m
    else:
        lane_idx = cute.arch.lane_idx()
        for first_expert in cutlass.range_constexpr(0, expert_cnt, cute.arch.WARP_SIZE):
            expert = first_expert + lane_idx
            if expert < expert_cnt:
                _, expert_i_row_count = compute_expert_token_range(padded_offsets, expert)
                if cutlass.const_expr(fused_tail_max_rows > 0):
                    total = total + expert_m_units(expert_i_row_count, fused_tail_max_rows)
                else:
                    total = total + (expert_i_row_count + (cluster_tile_m - 1)) // cluster_tile_m
        total = cute.arch.warp_redux_sync(total, "add")
    return total


def tail_rows_staged_view(smem_rows, staged_layout, rows):
    """A staged buffer of the fused tail's row operand (the swapped MMA's B, ``rows_max // 2`` rows per CTA per stage)
    viewed as the operand of a ``rows``-row tail: each CTA's first ``rows // 2`` rows of every stage."""
    outer = staged_layout.outer
    (_, k_atom), mode_n, mode_k, stages = outer.shape
    layout = cute.make_layout(((rows // 2, k_atom), mode_n, mode_k, stages), stride=outer.stride)
    return smem_rows.get_tensor(layout, swizzle=staged_layout.inner)


@cute.jit
def fused_mainloop(
    full_mma,
    acc,
    frag_a,
    frag_b,
    tail_mma,
    tail_acc,
    tail_frag_a,
    tail_frag_b,
    ab_pipeline,
    ab_consumer_state,
    k_tile_cnt,
    is_leader_cta,
    head_pipeline,
    head_state,
    tail_pipeline,
    tail_state,
    head_k_tiles: cutlass.Constexpr = 2,
    trail_k_tiles: cutlass.Constexpr = 3,
):
    """Fused unit K loop on the leader CTA: full MMA of the first ``head_k_tiles`` K blocks goes first (while the
    previous epilogue drains tail_acc's columns), the tail MMAs of the last ``trail_k_tiles`` before their full MMAs."""
    full_mma.set(tcgen05.Field.ACCUMULATE, False)
    tail_mma.set(tcgen05.Field.ACCUMULATE, False)
    if is_leader_cta:
        num_kblocks = cute.size(frag_a, mode=[2])
        head_tile_count = cutlass.min(Int32(head_k_tiles), k_tile_cnt)
        trail_tile_count = cutlass.min(Int32(trail_k_tiles), k_tile_cnt - head_tile_count)
        middle_end = k_tile_cnt - trail_tile_count
        # Head: the full tile's first K blocks, then (columns free) the tail's.
        ahead = ab_consumer_state.clone()
        for k_tile in cutlass.range(0, head_tile_count, 1, unroll=1):
            ab_pipeline.consumer_wait(ahead)
            for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                kblock_coord = (None, None, kblock_idx, ahead.index)
                cute.gemm(full_mma, acc, frag_a[kblock_coord], frag_b[kblock_coord], acc)
                full_mma.set(tcgen05.Field.ACCUMULATE, True)
            ahead.advance()
        head_pipeline.producer_acquire(head_state)
        for k_tile in cutlass.range(0, head_tile_count, 1, unroll=1):
            for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                kblock_coord = (None, None, kblock_idx, ab_consumer_state.index)
                cute.gemm(tail_mma, tail_acc, tail_frag_a[kblock_coord], tail_frag_b[kblock_coord], tail_acc)
                tail_mma.set(tcgen05.Field.ACCUMULATE, True)
            ab_pipeline.consumer_release(ab_consumer_state)
            ab_consumer_state.advance()
        # Middle: per K block, the full MMAs then the tail's.
        for k_tile in cutlass.range(head_tile_count, middle_end, 1, unroll=1):
            ab_pipeline.consumer_wait(ab_consumer_state)
            for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                kblock_coord = (None, None, kblock_idx, ab_consumer_state.index)
                cute.gemm(full_mma, acc, frag_a[kblock_coord], frag_b[kblock_coord], acc)
                full_mma.set(tcgen05.Field.ACCUMULATE, True)
            for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                kblock_coord = (None, None, kblock_idx, ab_consumer_state.index)
                cute.gemm(tail_mma, tail_acc, tail_frag_a[kblock_coord], tail_frag_b[kblock_coord], tail_acc)
                tail_mma.set(tcgen05.Field.ACCUMULATE, True)
            ab_pipeline.consumer_release(ab_consumer_state)
            ab_consumer_state.advance()
        # Trail: the tail's last K blocks, its commit, then the full tile's last K blocks.
        ahead = ab_consumer_state.clone()
        for k_tile in cutlass.range(middle_end, k_tile_cnt, 1, unroll=1):
            ab_pipeline.consumer_wait(ahead)
            for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                kblock_coord = (None, None, kblock_idx, ahead.index)
                cute.gemm(tail_mma, tail_acc, tail_frag_a[kblock_coord], tail_frag_b[kblock_coord], tail_acc)
                tail_mma.set(tcgen05.Field.ACCUMULATE, True)
            ahead.advance()
        tail_pipeline.producer_commit(tail_state)
        for k_tile in cutlass.range(middle_end, k_tile_cnt, 1, unroll=1):
            for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                kblock_coord = (None, None, kblock_idx, ab_consumer_state.index)
                cute.gemm(full_mma, acc, frag_a[kblock_coord], frag_b[kblock_coord], acc)
                full_mma.set(tcgen05.Field.ACCUMULATE, True)
            ab_pipeline.consumer_release(ab_consumer_state)
            ab_consumer_state.advance()
    return ab_consumer_state


@cute.jit
def fused_tail_rows_of(padded_offsets: cute.Tensor, expert_idx: Int32, mma_tile_m: Int32, max_rows: cutlass.Constexpr) -> Int32:
    """Rows of the tail fused into M tile ``mma_tile_m`` of ``expert_idx``, else 0."""
    _, expert_i_row_count = compute_expert_token_range(padded_offsets, expert_idx)
    return fused_tail_rows(expert_i_row_count, mma_tile_m, max_rows)

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
) -> Int32:
    """Cluster M tiles over all experts (sum of ceil(rows / cluster_tile_m)); ``aligned``: rows are tile multiples.
    Warp-collective."""
    total = Int32(0)
    if cutlass.const_expr(aligned):
        total = padded_offsets[expert_cnt - 1] // cluster_tile_m
    else:
        lane_idx = cute.arch.lane_idx()
        for first_expert in cutlass.range_constexpr(0, expert_cnt, cute.arch.WARP_SIZE):
            expert = first_expert + lane_idx
            if expert < expert_cnt:
                _, expert_i_row_count = compute_expert_token_range(padded_offsets, expert)
                total = total + (expert_i_row_count + (cluster_tile_m - 1)) // cluster_tile_m
        total = cute.arch.warp_redux_sync(total, "add")
    return total

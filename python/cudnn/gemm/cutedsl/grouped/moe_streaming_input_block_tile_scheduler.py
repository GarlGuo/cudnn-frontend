# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MoE 2Dx3D grouped GEMM scheduler fed by nccl-ep dispatch (``signal[t]`` = -1 until row slot ``t`` is complete):
ready M-tiles run in expert-consecutive batches; ``device_layout`` first waits (gpu-scope acquire) for the expert offsets."""

from typing import List, Tuple
import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import (
    Boolean,
    Int32,
    extract_mlir_values,
    new_from_mlir_values,
    const_expr,
)
from cutlass.cute.typing import AddressSpace
from cutlass._mlir import ir
import cutlass.pipeline as pipeline
from cutlass.pipeline import Agent, CooperativeGroup, PipelineUserType, make_pipeline_state
from .moe_signal_helpers import (
    streaming_scheduler_state_offset,
    signal_ptr,
    mbarrier_arrive_expect_tx_on_peer_relaxed_cluster,
    fence_release_cluster,
    mbarrier_try_wait_parity_acquire_cluster,
)
from .moe_utils import ceil_div
from .moe_tail_tiles import FUSED_UNIT, cluster_m_tile_count, fused_tail_chunks, fused_tail_first_row, fused_tail_rows
from .moe_persistent_scheduler import (
    atomic_add_i32,
    MoEWorkTileInfo,
    _DynamicSchedState,
)

#: Most M-tiles in one expert-consecutive-M batch: its A rows must stay in L2 beside the N-tile
#: of weights they multiply.
EXPERT_BATCH_MAX_M_TILES = 16


def _state_words(streaming_signal_capacity, expert_cnt: int) -> dict:
    """Int32 word offset of each piece of the scheduler state, and the total (an Int32 capacity: the launch's).  The
    reset is all zero; a batch record needs none, being written before ``assigned_claim_count`` exposes it."""
    capacity = streaming_signal_capacity
    # Indexed by expert, and a capacity can be below expert_cnt: fewer routed rows than experts.
    expert_table_words = max(capacity, expert_cnt) if isinstance(capacity, int) else cutlass.max(capacity, Int32(expert_cnt))
    # A batch record is 4 words per M-tile; then, per expert, 1 + the first claim number of its last batch (0: none).
    expert_last_batch = expert_table_words + 1 + 4 * capacity
    return {
        "expert_assigned_m_tiles": 0,
        "assigned_claim_count": expert_table_words,
        "expert_consecutive_m_batch_record": expert_table_words + 1,
        "expert_last_batch": expert_last_batch,
        "total": expert_last_batch + expert_cnt,
    }


def check_work_slot_range(intermediate, cluster_tile_n: int, streaming_signal_capacity, fused_tail_max_rows: int) -> None:
    """Build-time bounds on a work slot (slot * Q_N + N tile): N > 0, and every slot, END included, below the flag bit
    ``FUSED_UNIT`` with fused tails, else 2^31."""
    assert isinstance(intermediate, int), f"the signal scheduler needs a build-time N (dense B), not {intermediate}"
    assert intermediate > 0, f"the signal scheduler needs N > 0 (a work slot // Q_N), got N = {intermediate}"
    assert isinstance(streaming_signal_capacity, int) or not fused_tail_max_rows, "fused tails need a build-time streaming_signal_capacity"
    if isinstance(streaming_signal_capacity, int):
        n_tiles_per_m_tile = -(-intermediate // cluster_tile_n)  # host-side: moe_utils.ceil_div is @cute.jit
        work_slot_limit = FUSED_UNIT if fused_tail_max_rows else 2**31
        assert streaming_signal_capacity * n_tiles_per_m_tile < work_slot_limit, (
            f"streaming_signal_capacity {streaming_signal_capacity} x Q_N {n_tiles_per_m_tile} work slots reach {work_slot_limit:#x}"
            f"{' (FUSED_UNIT, fused tails on)' if fused_tail_max_rows else ''}"
        )


def signal_workspace_bytes_after_counter(streaming_signal_capacity: int, expert_cnt: int) -> int:
    """Workspace bytes past the 4-byte claim counter: ``signal[]``, the padding to a 128-byte
    line, then the scheduler state (see ``_state_words``)."""
    return const_expr(streaming_scheduler_state_offset(streaming_signal_capacity) - 4 + 4 * _state_words(streaming_signal_capacity, expert_cnt)["total"])


@cute.jit
def _spin_until_at_least(ptr, floor: Int32) -> Int32:
    """Block, without bound, until ``*ptr >= floor`` and return it, acquiring the writer's earlier stores.
    One load point in the loop: a load ahead of it would make the compiler duplicate the loop."""
    value = Int32(floor - 1)
    while value < floor:
        value = cute.arch.load(ptr.llvm_ptr, Int32, sem="acquire", scope="gpu")
    return value


@cute.jit
def _poll_until_at_least(ptr, floor: cutlass.Constexpr) -> None:
    """Block, without bound, until ``*ptr >= floor``.  Relaxed gpu-scope loads (no L1 invalidation per poll): the
    caller reads nothing the writer published, so it needs no acquire."""
    value = Int32(floor - 1)
    while value < floor:
        value = cute.arch.load(ptr.llvm_ptr, Int32, sem="relaxed", scope="gpu")


@cute.jit
def wait_slots_ready(signal, num_slots: Int32) -> None:
    """One warp waits until ``signal[0:num_slots]`` are all complete (>= 0) before its TMA loads (lane l: slots l,
    l + 32, ..., clamped); the warp barrier shares the acquires, the proxy fence orders them before the TMA."""
    lane_idx = cute.arch.lane_idx()
    for j in cutlass.range(0, ceil_div(num_slots, cute.arch.WARP_SIZE), 1, unroll=1):
        _spin_until_at_least(signal + cutlass.min(lane_idx + j * cute.arch.WARP_SIZE, num_slots - 1), 0)
    cute.arch.sync_warp()
    cute.arch.fence_proxy("async.global")


@cute.jit
def wait_count_reached(count, target: Int32) -> None:
    """From one warp, before its TMA loads: wait until ``*count >= target`` (a producer's release adds), then order
    the acquire before the warp's async-proxy loads, as wait_slots_ready."""
    _spin_until_at_least(count, target)
    cute.arch.sync_warp()
    cute.arch.fence_proxy("async.global")


#: A claim number that carries no work unit (a fused batch's last ones); the claimer takes another.
NO_WORK_SLOT = -1


class MoEInputBlockSignalSchedulerParams:
    """Parameters for MoE tile scheduler, used on host (grid shape) and on device: expert_shape is
    (expert_cnt, intermediate, hidden), a GEMM (M=tokens_i, N=intermediate, K=hidden) per expert."""

    def __init__(
        self,
        expert_shape: Tuple[int | Int32, int | Int32, int | Int32],
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
    ):
        # Whatever the caller passed, keep as-is: a Python int stays trace-time and is
        # never serialized, so expressions over it fold instead of costing a register.
        self.expert_cnt, self.intermediate, self.hidden = expert_shape
        self.cta_tile_shape_mnk = cta_tile_shape_mnk
        self.cluster_shape_mn = cluster_shape_mn

    @property
    def cluster_tile_m(self) -> int:
        """Cluster tile size along M = cta_tile_m * cluster_m."""
        return const_expr(self.cta_tile_shape_mnk[0] * self.cluster_shape_mn[0])

    @property
    def cluster_tile_n(self) -> int:
        """Cluster tile size along N = cta_tile_n * cluster_n."""
        return const_expr(self.cta_tile_shape_mnk[1] * self.cluster_shape_mn[1])

    def __extract_mlir_values__(self) -> List[ir.Value]:
        """Only serialize runtime values, not codegen-time constants."""
        values = []
        for field in (self.expert_cnt, self.intermediate, self.hidden):
            values.extend(extract_mlir_values(field))
        return values

    def __new_from_mlir_values__(self, values: List[ir.Value]) -> "MoEInputBlockSignalSchedulerParams":
        # A static field consumes no values and rebuilds as itself.
        value_iterator = iter(values)

        def take(field):
            value_count = len(extract_mlir_values(field))
            return new_from_mlir_values(field, [next(value_iterator) for _ in range(value_count)]) if value_count else field

        rebuilt = tuple(take(field) for field in (self.expert_cnt, self.intermediate, self.hidden))
        return MoEInputBlockSignalSchedulerParams(
            expert_shape=rebuilt,
            cta_tile_shape_mnk=self.cta_tile_shape_mnk,
            cluster_shape_mn=self.cluster_shape_mn,
        )


class MoEStreamingInputBlockSignalTileScheduler:
    """Persistent tile scheduler for the MoE grouped GEMM: tile iteration only (claim a work slot,
    broadcast it to the cluster).  MoESchedExtension converts domains and picks TMA descriptors."""

    def internal_init(self):
        """Create the internal cluster broadcast pipeline (dynamic mode only)."""
        cluster_size = const_expr(self.params.cluster_shape_mn[0] * self.params.cluster_shape_mn[1])
        self.cluster_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=self._sched_storage.cluster_mbar.data_ptr(),
            num_stages=1,
            producer_group=CooperativeGroup(Agent.Thread, 1),
            consumer_group=CooperativeGroup(Agent.Thread, 32 * cluster_size),
            defer_sync=True,
        )
        cute.arch.mbarrier_init_fence()

    def internal_finalize(self):
        """Cluster-wide barrier that must run before any CTA of the cluster exits.  Unconditional:
        ``__init__`` asserts a cluster of at least two CTAs."""
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()

    def __init__(
        self,
        params: MoEInputBlockSignalSchedulerParams,
        expert_freq_offsets: cute.Tensor,
        cta_id_in_cluster: cute.Coord,
        dynamic_state: _DynamicSchedState,
        sched_storage,
        cluster_pipeline,
        streaming_signal_capacity: int | Int32,
        device_layout: bool,
        slot_rows: int = None,
        fused_tail_max_rows: int = 0,
        hoist_slot_flags: bool = True,
    ):
        self.params = params
        self.expert_freq_offsets = expert_freq_offsets
        self.cta_id_in_cluster = cta_id_in_cluster
        self._dynamic_state = dynamic_state
        self._sched_storage = sched_storage
        self.cluster_pipeline = cluster_pipeline
        # Never serialized: it only places the scheduler state, whose pointer is derived from it.  An int folds into
        # the build; an Int32 is the launch's (a kernel parameter, so it needs no register of its own).
        self.streaming_signal_capacity = streaming_signal_capacity
        self.device_layout = device_layout
        # Rows per ``signal[]`` slot, codegen-time; an M-tile spans ``slots_per_m_tile`` slots.
        self.slot_rows = slot_rows or params.cluster_tile_m
        assert params.cluster_tile_m % self.slot_rows == 0, f"slot of {self.slot_rows} rows must divide the cluster tile M {params.cluster_tile_m}"
        self.slots_per_m_tile = params.cluster_tile_m // self.slot_rows
        # Codegen-time: read every candidate slot's flag in one round of loads (True), or each M-tile's only once the
        # ones before it were ready (False: a dependent load per M-tile, far fewer registers for a small setmaxnreg).
        self.hoist_slot_flags = hoist_slot_flags
        # Codegen-time: tails of at most this many rows may ride in their expert's last full M-tile(s), a chunk of at
        # most ``fused_tail_slots`` slots each (0: no fused tails).
        self.fused_tail_max_rows = fused_tail_max_rows
        self.fused_tail_slots = ceil_div(fused_tail_max_rows, self.slot_rows)
        assert (
            not fused_tail_max_rows or self.slot_rows <= fused_tail_max_rows
        ), f"fused tails of {fused_tail_max_rows} rows need slots of at most as many rows, not {self.slot_rows}"
        check_work_slot_range(params.intermediate, params.cluster_tile_n, streaming_signal_capacity, fused_tail_max_rows)
        # mapa.shared::cluster at one CTA is an illegal peer store.  This assert is the only
        # guard: no kernel checks its cluster shape for the signal scheduler.
        assert const_expr(
            params.cluster_shape_mn[0] * params.cluster_shape_mn[1] > 1
        ), "the signal scheduler broadcasts with mapa.shared::cluster, which needs >= 2 CTAs"

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values = []
        values.extend(extract_mlir_values(self.params))
        values.extend(extract_mlir_values(self.expert_freq_offsets))
        values.extend(extract_mlir_values(self.cta_id_in_cluster))
        values.extend(extract_mlir_values(self._dynamic_state))
        return values

    def __new_from_mlir_values__(self, values: List[ir.Value]) -> "MoEStreamingInputBlockSignalTileScheduler":

        def take(field):
            """Consume this field's own share off the front; `values` shrinks as it goes."""
            nonlocal values
            value_count = len(extract_mlir_values(field))
            field_values, values = (values[:value_count], values[value_count:])
            return new_from_mlir_values(field, field_values)

        return MoEStreamingInputBlockSignalTileScheduler(
            params=take(self.params),
            expert_freq_offsets=take(self.expert_freq_offsets),
            cta_id_in_cluster=take(self.cta_id_in_cluster),
            dynamic_state=take(self._dynamic_state),
            sched_storage=self._sched_storage,
            cluster_pipeline=self.cluster_pipeline,
            streaming_signal_capacity=self.streaming_signal_capacity,
            device_layout=const_expr(self.device_layout),
            slot_rows=const_expr(self.slot_rows),
            fused_tail_max_rows=const_expr(self.fused_tail_max_rows),
            hoist_slot_flags=const_expr(self.hoist_slot_flags),
        )

    @staticmethod
    def create(
        params: MoEInputBlockSignalSchedulerParams,
        expert_freq_offsets: cute.Tensor,
        counter_ptr: cute.Pointer,
        sched_storage,
        *,
        streaming_signal_capacity: int | Int32,
        device_layout: bool = False,
        slot_rows: int = None,
        fused_tail_max_rows: int = 0,
        hoist_slot_flags: bool = True,
    ) -> "MoEStreamingInputBlockSignalTileScheduler":
        """Create the scheduler from the claim counter ``counter_ptr``; ``signal[]`` and the scheduler state follow it.
        ``device_layout``: wait for the dispatch-written layout before the first claim (see the module docstring)."""
        # %cluster_ctarank, hinted warp-uniform so it lands in a uniform register -- the
        # same form the four kernels use.  CTAs are ordered x-fastest within the cluster.
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        cta_id_in_cluster = (
            cta_rank_in_cluster % const_expr(params.cluster_shape_mn[0]),
            cta_rank_in_cluster // const_expr(params.cluster_shape_mn[0]),
            0,
        )
        # Rank 0, not "even rank": this gates the atomicAdd that claims a slot, and a
        # cluster with two claimers would take two slots for one cluster's work.
        is_leader_cta = cta_rank_in_cluster == 0
        broadcast_ptr = sched_storage.cluster_broadcast.data_ptr()
        dynamic_state = _DynamicSchedState(
            counter_ptr=counter_ptr,
            broadcast_ptr=broadcast_ptr,
            is_leader_cta=is_leader_cta,
            producer_state=make_pipeline_state(PipelineUserType.Producer, 1),
            consumer_state=make_pipeline_state(PipelineUserType.Consumer, 1),
        )
        return MoEStreamingInputBlockSignalTileScheduler(
            params=params,
            expert_freq_offsets=expert_freq_offsets,
            cta_id_in_cluster=cta_id_in_cluster,
            dynamic_state=dynamic_state,
            sched_storage=sched_storage,
            cluster_pipeline=None,
            streaming_signal_capacity=streaming_signal_capacity,
            device_layout=device_layout,
            slot_rows=slot_rows,
            fused_tail_max_rows=fused_tail_max_rows,
            hoist_slot_flags=hoist_slot_flags,
        )

    @cute.jit
    def advance_to_next_work(self) -> MoEWorkTileInfo:
        """Claim the cluster's next work slot and return its work tile, invalid past the last."""
        work_tile_info = self._get_work_tile_for_linear_idx(self._claim_slot_via_cluster_broadcast())
        # The broadcast value is consumed: a full warp per CTA arrives on the leader's empty barrier.
        self.cluster_pipeline.sync_object_empty.arrive(self._dynamic_state.consumer_state.index, 0)
        self._dynamic_state.consumer_state.advance()
        return work_tile_info

    @cute.jit
    def initial_work_tile_info(self) -> MoEWorkTileInfo:
        """The first work tile, claimed like every later one (``device_layout``: once the dispatch has written the
        layout)."""
        if const_expr(self.device_layout):
            self._wait_for_dispatch_layout()
        return self.advance_to_next_work()

    @cute.jit
    def wait_all_slots_ready(self) -> None:
        """From the trigger warp: wait until the dispatch has delivered every row (``device_layout``: layout first).
        Slots are polled, not acquired (the warp reads no row; each claim acquires its own), lanes as above."""
        if const_expr(self.device_layout):
            self._wait_for_dispatch_layout()
        total_slots = self.expert_freq_offsets[const_expr(self.params.expert_cnt - 1)] // const_expr(self.slot_rows)
        signal = signal_ptr(self._dynamic_state.counter_ptr)
        lane_idx = cute.arch.lane_idx()
        for j in cutlass.range(0, ceil_div(total_slots, cute.arch.WARP_SIZE), 1, unroll=1):
            _poll_until_at_least(signal + cutlass.min(lane_idx + j * cute.arch.WARP_SIZE, total_slots - 1), 0)
        cute.arch.sync_warp()

    @cute.jit
    def _wait_for_dispatch_layout(self) -> None:
        """Each lane waits for its layout_words[0:E+1] (word w on lane w % 32) to leave the reset value -1, in every CTA
        (each SM drops stale L1); extra lanes re-wait the last word (a branch makes ptxas emulate warp collectives)."""
        lane_idx = cute.arch.lane_idx()
        word_count = const_expr(self.params.expert_cnt + 1)
        # layout_words[0] (expert 0's offset) sits one word before expert_freq_offsets = layout_words[1:E+1].
        layout_words = self.expert_freq_offsets.iterator + (-1)
        for j in cutlass.range_constexpr(ceil_div(word_count, cute.arch.WARP_SIZE)):
            word = cutlass.min(lane_idx + j * cute.arch.WARP_SIZE, word_count - 1)
            _spin_until_at_least(layout_words + word, 0)
        cute.arch.sync_warp()

    def _state_ptr(self, piece: cutlass.Constexpr):
        """Where ``piece`` of the scheduler state starts (see ``_state_words``): bytes past the claim counter, a
        trace-time number with an int capacity."""
        return cute.make_ptr(
            Int32,
            self._dynamic_state.counter_ptr.toint()
            + (
                streaming_scheduler_state_offset(self.streaming_signal_capacity)
                + 4 * _state_words(self.streaming_signal_capacity, self.params.expert_cnt)[piece]
            ),
            AddressSpace.gmem,
            assumed_align=4,
        )

    def _expert_consecutive_m_batch_record_ptr(self, m_tile_order: Int32):
        """Batch record of the k-th M-tile put in a batch (read by claims [k * Q_N, (k + 1) * Q_N)): first claim number,
        first M-tile slot, M-tile count (| ``FUSED_UNIT``), the expert's end slot (fused tails: the expert)."""
        return self._state_ptr("expert_consecutive_m_batch_record") + 4 * m_tile_order

    def _set_expert_consecutive_m_batch_record(self, m_tile_order: Int32, record: Tuple[Int32, Int32, Int32, Int32]) -> None:
        """Store ``record`` (first claim number, first M-tile's slot, M-tile count, expert end slot)."""
        record_ptr = self._expert_consecutive_m_batch_record_ptr(m_tile_order)
        # Scheduler state is read and written at gpu scope (relaxed: L1 bypassed, so other SMs' writes are seen);
        # the release on ``assigned_claim_count`` publishes these stores.
        for word, value in enumerate(record):
            cute.arch.store((record_ptr + word).llvm_ptr, value, sem="relaxed", scope="gpu")

    def _get_expert_consecutive_m_batch_record(self, m_tile_order: Int32) -> Tuple[Int32, Int32, Int32, Int32]:
        """The record (first claim number, first M-tile's slot, M-tile count, expert end slot) at ``m_tile_order``."""
        record_ptr = self._expert_consecutive_m_batch_record_ptr(m_tile_order)
        return tuple(cute.arch.load((record_ptr + word).llvm_ptr, Int32, sem="relaxed", scope="gpu") for word in range(4))

    @cute.jit
    def _expert_first_slot(self, expert: Int32) -> Int32:
        """The first slot of ``expert``.  The offsets are padded to the slot and each one is
        where its expert ENDS, so an expert starts where the one before it ends."""
        first_slot = Int32(0)
        if expert > 0:
            first_slot = self.expert_freq_offsets[expert - 1] // const_expr(self.slot_rows)
        return first_slot

    @cute.jit
    def _batch_fuses_tail(self, expert_first_slot: Int32, expert_end_slot: Int32, first_m_tile: Int32, m_tile_count: Int32) -> Boolean:
        """Whether the batch of ``m_tile_count`` M-tiles from M-tile ``first_m_tile`` runs its expert's tail fused:
        the expert has a fusable tail, and the batch takes both the M-tiles it rides in and the tail M-tile."""
        tokens_i = (expert_end_slot - expert_first_slot) * const_expr(self.slot_rows)
        full_m_tiles = tokens_i // const_expr(self.params.cluster_tile_m)
        chunks = fused_tail_chunks(tokens_i, const_expr(self.fused_tail_max_rows))
        fuses = Boolean(False)
        if chunks > 0 and first_m_tile <= full_m_tiles - chunks and first_m_tile + m_tile_count == full_m_tiles + 1:
            fuses = Boolean(True)
        return fuses

    @cute.jit
    def _fused_tail_first_slot(self, expert_first_slot: Int32, expert_end_slot: Int32, m_tile_slot: Int32) -> Int32:
        """The first slot of the tail chunk fused into the M unit at ``m_tile_slot``, or ``expert_end_slot`` when it
        has none; the chunk's slots are this one and the next ones up to ``fused_tail_slots`` before the expert end."""
        max_rows = const_expr(self.fused_tail_max_rows)
        tokens_i = (expert_end_slot - expert_first_slot) * const_expr(self.slot_rows)
        m_tile = (m_tile_slot - expert_first_slot) // const_expr(self.slots_per_m_tile)
        tail_first_slot = expert_end_slot
        if fused_tail_rows(tokens_i, m_tile, max_rows) > 0:
            tail_first_slot = expert_first_slot + fused_tail_first_row(tokens_i, m_tile, max_rows) // const_expr(self.slot_rows)
        return tail_first_slot

    @cute.jit
    def _expert_ready_m_tiles(self, expert: Int32) -> Int32:
        """How many M-tiles of ``expert`` are READY in a row from its first unassigned M-tile
        on, at most ``EXPERT_BATCH_MAX_M_TILES``.  An ``expert`` past the last expert has none."""
        ready_m_tile_count = Int32(0)
        if expert < const_expr(self.params.expert_cnt):
            slots_per_m_tile = const_expr(self.slots_per_m_tile)
            expert_end_slot = self.expert_freq_offsets[expert] // const_expr(self.slot_rows)
            expert_first_slot = self._expert_first_slot(expert)
            assigned_m_tiles_ptr = self._state_ptr("expert_assigned_m_tiles") + expert
            assigned_m_tiles = cute.arch.load(assigned_m_tiles_ptr.llvm_ptr, Int32, sem="relaxed", scope="gpu")
            first_unassigned_slot = expert_first_slot + slots_per_m_tile * assigned_m_tiles
            signal = signal_ptr(self._dynamic_state.counter_ptr)
            # Every candidate slot's flag in one round of independent loads, off the batch opener's critical path; a
            # slot past the expert's end reads its last slot instead, in bounds and ignored.
            last_slot = cutlass.max(expert_end_slot - 1, 0)
            flags = None
            if const_expr(self.hoist_slot_flags):
                flags = [
                    [
                        cute.arch.load(
                            (signal + cutlass.min(first_unassigned_slot + j * slots_per_m_tile + k, last_slot)).llvm_ptr, Int32, sem="relaxed", scope="gpu"
                        )
                        for k in range(slots_per_m_tile)
                    ]
                    for j in range(EXPERT_BATCH_MAX_M_TILES)
                ]
            for j in cutlass.range_constexpr(EXPERT_BATCH_MAX_M_TILES):
                # The j-th unassigned M-tile counts if the j before it did and it is still this expert's.
                m_tile_slot = first_unassigned_slot + j * slots_per_m_tile
                if ready_m_tile_count == j and m_tile_slot < expert_end_slot:
                    m_tile_ready = Boolean(True)
                    for k in cutlass.range_constexpr(slots_per_m_tile):
                        # Slot 0 is the expert's, by the test above; a tail M-tile has fewer slots.
                        if const_expr(k == 0) or m_tile_slot + k < expert_end_slot:
                            if const_expr(self.hoist_slot_flags):
                                flag = flags[j][k]
                            else:
                                flag = cute.arch.load((signal + (m_tile_slot + k)).llvm_ptr, Int32, sem="relaxed", scope="gpu")
                            if flag < 0:
                                m_tile_ready = Boolean(False)
                    if m_tile_ready:
                        ready_m_tile_count = ready_m_tile_count + 1
        return ready_m_tile_count

    @cute.jit
    def _open_expert_consecutive_m_batch(self, claim_number: Int32, n_tiles_per_m_tile: Int32, lane_idx: Int32) -> None:
        """Open and publish the batch at ``claim_number`` (only the ``assigned_claim_count`` warp: no atomics): prefer
        the expert opened last whose ready run ends it or fills the cap (weights in L2), else the most ready M-tiles."""
        expert_cnt = const_expr(self.params.expert_cnt)
        assigned_claim_count_ptr = self._state_ptr("assigned_claim_count")
        expert_last_batch_ptr = self._state_ptr("expert_last_batch")

        m_tile_count = Int32(0)
        expert = Int32(0)
        # No ready M-tile: look again.  ``claim_number`` exists, so one is still unassigned, and
        # the producer does not wait on this consumer, so it does get published.
        while m_tile_count == 0:
            # Score: 0 with no ready M-tile; an expert that had a batch scores past any ready count, by how recently
            # that batch opened; any other its ready count.
            lane_score = Int32(0)
            lane_ready_m_tile_count = Int32(0)
            lane_expert = Int32(0)
            for round_first_expert in cutlass.range_constexpr(0, expert_cnt, cute.arch.WARP_SIZE):
                expert_of_lane = round_first_expert + lane_idx
                ready_m_tile_count = self._expert_ready_m_tiles(expert_of_lane)
                score = ready_m_tile_count
                if ready_m_tile_count > 0:
                    last_batch = cute.arch.load((expert_last_batch_ptr + expert_of_lane).llvm_ptr, Int32, sem="relaxed", scope="gpu")
                    # Only a run that ends the expert or fills the batch cap: a shorter one (its rows still arriving)
                    # would open a small batch and cannot spare the expert another pass over its weights.
                    expert_first_slot = self._expert_first_slot(expert_of_lane)
                    assigned_m_tiles_ptr = self._state_ptr("expert_assigned_m_tiles") + expert_of_lane
                    assigned_m_tiles = cute.arch.load(assigned_m_tiles_ptr.llvm_ptr, Int32, sem="relaxed", scope="gpu")
                    run_end_slot = expert_first_slot + const_expr(self.slots_per_m_tile) * (assigned_m_tiles + ready_m_tile_count)
                    run_is_whole = ready_m_tile_count == const_expr(EXPERT_BATCH_MAX_M_TILES) or (
                        run_end_slot >= self.expert_freq_offsets[expert_of_lane] // const_expr(self.slot_rows)
                    )
                    if last_batch != 0 and run_is_whole:
                        score = const_expr(EXPERT_BATCH_MAX_M_TILES) + last_batch
                if score > lane_score:
                    lane_score = score
                    lane_ready_m_tile_count = ready_m_tile_count
                    lane_expert = expert_of_lane
            best_score = cute.arch.warp_redux_sync(lane_score, "max")
            # Only a lane that holds the best score names its expert; the others
            # answer with the expert count, which is past all of them.
            vote = Int32(expert_cnt)
            if lane_score == best_score:
                vote = lane_expert
            expert = cute.arch.warp_redux_sync(vote, "min")
            # And the lane that named it, its ready count (0 when no expert has one: look again).
            count_vote = Int32(0)
            if lane_expert == expert:
                count_vote = lane_ready_m_tile_count
            m_tile_count = cute.arch.warp_redux_sync(count_vote, "max")

        with cute.arch.elect_one():
            expert_assigned_m_tiles_ptr = self._state_ptr("expert_assigned_m_tiles") + expert
            already_assigned_m_tiles = cute.arch.load(expert_assigned_m_tiles_ptr.llvm_ptr, Int32, sem="relaxed", scope="gpu")
            first_slot = self._expert_first_slot(expert) + already_assigned_m_tiles * const_expr(self.slots_per_m_tile)
            expert_end_slot = self.expert_freq_offsets[expert] // const_expr(self.slot_rows)
            cute.arch.store(expert_assigned_m_tiles_ptr.llvm_ptr, already_assigned_m_tiles + m_tile_count, sem="relaxed", scope="gpu")
            record_m_tile_count, record_last = m_tile_count, expert_end_slot
            cute.arch.store((expert_last_batch_ptr + expert).llvm_ptr, claim_number + 1, sem="relaxed", scope="gpu")
            if const_expr(self.fused_tail_max_rows > 0):
                # Fused only when the batch's M-tiles reach the expert's end: its tail's rows are ready now.
                if self._batch_fuses_tail(self._expert_first_slot(expert), expert_end_slot, already_assigned_m_tiles, m_tile_count):
                    record_m_tile_count = m_tile_count | FUSED_UNIT
                record_last = expert
            # The batch takes the next ``m_tile_count`` M-tile orders: the same record in each.
            first_m_tile_order = claim_number // n_tiles_per_m_tile
            for j in cutlass.range_constexpr(EXPERT_BATCH_MAX_M_TILES):
                if j < m_tile_count:
                    self._set_expert_consecutive_m_batch_record(first_m_tile_order + j, (claim_number, first_slot, record_m_tile_count, record_last))
            # Published LAST, and that order is load-bearing: the release is what hands the
            # records to the claimers and ``expert_assigned_m_tiles[]`` to the next opener.
            cute.arch.store(assigned_claim_count_ptr.llvm_ptr, claim_number + m_tile_count * n_tiles_per_m_tile, sem="release", scope="gpu")
        # ``assigned_claim_count`` is past ``claim_number`` now, so this returns at once.  It is
        # here for its acquire: one lane wrote the records, and every lane is about to read them.
        _spin_until_at_least(assigned_claim_count_ptr, claim_number + 1)

    @cute.jit
    def _claim_work_slot(self, n_tiles_per_m_tile: Int32, lane_idx: Int32) -> Int32:
        """Claim a number (one atomic add), wait for every earlier one to have a batch, open one if it has none, and
        return its work slot: END past the last, ``NO_WORK_SLOT`` for fused spares, ``FUSED_UNIT`` on tail units."""
        ds = self._dynamic_state
        claim_number = 0
        if lane_idx == 0:
            claim_number = atomic_add_i32(ds.counter_ptr.llvm_ptr, Int32(1))
        # The other lanes hold 0, so the OR hands every lane lane 0's number.
        claim_number = cute.arch.warp_redux_sync(claim_number, "or")
        # Past the last slot, which is what an END claim broadcasts.  The offsets are
        # inclusive ends padded to the slot, so the last one is the slot total.
        total_slots = self.expert_freq_offsets[const_expr(self.params.expert_cnt - 1)] // const_expr(self.slot_rows)
        work_slot = total_slots * n_tiles_per_m_tile
        total_claims = (
            cluster_m_tile_count(
                self.expert_freq_offsets, const_expr(self.params.expert_cnt), const_expr(self.params.cluster_tile_m), const_expr(self.slots_per_m_tile == 1)
            )
            * n_tiles_per_m_tile
        )
        if claim_number < total_claims:
            # Every lane lands on the same side of this test: once ``assigned_claim_count``
            # is ``claim_number`` it stays there until this very warp moves it.
            if _spin_until_at_least(self._state_ptr("assigned_claim_count"), claim_number) == claim_number:
                self._open_expert_consecutive_m_batch(claim_number, n_tiles_per_m_tile, lane_idx)
            # ``assigned_claim_count`` is past ``claim_number`` and every lane has acquired
            # it, so the record at this claim number's M-tile order is written and visible.
            first_claim_number, first_slot, m_tile_count, expert_end_slot = self._get_expert_consecutive_m_batch_record(claim_number // n_tiles_per_m_tile)
            if const_expr(self.fused_tail_max_rows > 0):
                # The record's last word is the expert; its offsets, as every offset here, are read after the layout
                # wait.  A fused batch has one M unit fewer per N-tile: its tail M-tile rides in the units before it.
                fused_batch = m_tile_count & FUSED_UNIT
                m_tile_count = m_tile_count - fused_batch - fused_batch // FUSED_UNIT
                expert = expert_end_slot
                expert_first_slot = self._expert_first_slot(expert)
                expert_end_slot = self.expert_freq_offsets[expert] // const_expr(self.slot_rows)
            claim_in_batch = claim_number - first_claim_number
            n_tile = claim_in_batch // m_tile_count
            m_tile_slot = first_slot + claim_in_batch % m_tile_count * const_expr(self.slots_per_m_tile)
            # A batch holds only ready M-tiles, so these return at once.  They are here for their
            # acquire, which pairs the producer's release and makes the M-tile's rows visible.
            for k in cutlass.range_constexpr(self.slots_per_m_tile):
                if const_expr(k == 0) or m_tile_slot + k < expert_end_slot:
                    _spin_until_at_least(signal_ptr(ds.counter_ptr) + (m_tile_slot + k), 0)
            work_slot = m_tile_slot * n_tiles_per_m_tile + n_tile
            if const_expr(self.fused_tail_max_rows > 0):
                if n_tile >= n_tiles_per_m_tile:
                    # The claim numbers the fused batch's tail M-tile would have taken.
                    work_slot = Int32(NO_WORK_SLOT)
                else:
                    if fused_batch != 0:
                        tail_first_slot = self._fused_tail_first_slot(expert_first_slot, expert_end_slot, m_tile_slot)
                        if tail_first_slot < expert_end_slot:
                            # Ready when the batch opened: these acquire the tail chunk's rows.
                            for k in cutlass.range_constexpr(self.fused_tail_slots):
                                if tail_first_slot + k < expert_end_slot:
                                    _spin_until_at_least(signal_ptr(ds.counter_ptr) + (tail_first_slot + k), 0)
                            work_slot = work_slot | FUSED_UNIT
        return work_slot

    @cute.jit
    def _claim_slot_via_cluster_broadcast(self) -> Int32:
        """Claim a number with one atomic add, wait until every earlier one has a batch, open
        one if it has none, and broadcast its work slot (END past the last) to the cluster."""
        ds = self._dynamic_state
        broadcast_tensor = cute.make_tensor(ds.broadcast_ptr, cute.make_layout((1,)))
        if ds.is_leader_cta:
            self.cluster_pipeline.producer_acquire(ds.producer_state)
            full_barrier_ptr = self.cluster_pipeline.sync_object_full.get_barrier(ds.producer_state.index)
            lane_idx = cute.arch.lane_idx()
            cluster_size = const_expr(self.params.cluster_shape_mn[0] * self.params.cluster_shape_mn[1])
            # Q_N: every M-tile is this many N-tiles, and so this many claim numbers.
            n_tiles_per_m_tile = ceil_div(self.params.intermediate, const_expr(self.params.cluster_tile_n))
            work_slot = self._claim_work_slot(n_tiles_per_m_tile, lane_idx)
            if const_expr(self.fused_tail_max_rows > 0):
                while work_slot == NO_WORK_SLOT:
                    work_slot = self._claim_work_slot(n_tiles_per_m_tile, lane_idx)
            if lane_idx < cluster_size:
                cute.arch.store_async_dsmem(ds.broadcast_ptr, work_slot, full_barrier_ptr, lane_idx)
            # Fan out one arrive per CTA to the full barrier.
            with cute.arch.elect_one():
                fence_release_cluster()
                for peer in cutlass.range_constexpr(cluster_size):
                    mbarrier_arrive_expect_tx_on_peer_relaxed_cluster(full_barrier_ptr, Int32(4), peer)
        ds.producer_state.advance()
        consumer_full_barrier_ptr = self.cluster_pipeline.sync_object_full.get_barrier(ds.consumer_state.index)
        broadcast_arrived = Boolean(False)
        while not broadcast_arrived:
            broadcast_arrived = mbarrier_try_wait_parity_acquire_cluster(consumer_full_barrier_ptr, ds.consumer_state.phase)
        return broadcast_tensor[0]

    @cute.jit
    def _get_work_tile_for_linear_idx(self, work_slot: Int32) -> MoEWorkTileInfo:
        """The work tile of ``work_slot = m_tile_slot * Q_N + n_tile`` (``m_tile_slot``: the M-tile's first slot);
        END, past every expert's last slot, gives an invalid tile."""
        total_expert_cnt = const_expr(self.params.expert_cnt)
        n_tiles_per_m_tile = ceil_div(self.params.intermediate, const_expr(self.params.cluster_tile_n))
        if const_expr(self.fused_tail_max_rows > 0):
            # A fused unit keeps the mark in its k_tile_cnt.
            fused_unit = work_slot & FUSED_UNIT
            work_slot = work_slot - fused_unit
        m_tile_slot = work_slot // n_tiles_per_m_tile
        n_tile = work_slot % n_tiles_per_m_tile
        # The warp finds the first expert whose end slot is past the M-tile in one round of loads (lane l reads experts
        # l, l + 32, ...) and a min-reduction; a walk from expert 0 costs one dependent L2 load per expert per claim.
        lane_idx = cute.arch.lane_idx()
        expert = Int32(total_expert_cnt)
        for first_expert in cutlass.range_constexpr(0, total_expert_cnt, cute.arch.WARP_SIZE):
            expert_of_lane = first_expert + lane_idx
            end_slot = self.expert_freq_offsets[cutlass.min(expert_of_lane, total_expert_cnt - 1)] // const_expr(self.slot_rows)
            vote = Int32(total_expert_cnt)
            if expert_of_lane < total_expert_cnt and m_tile_slot < end_slot:
                vote = expert_of_lane
            expert = cutlass.min(expert, cute.arch.warp_redux_sync(vote, "min"))
        work_tile_info = MoEWorkTileInfo(expert_idx=Int32(-1), tile_m_idx=Int32(0), tile_n_idx=Int32(0), k_tile_cnt=Int32(0))
        if expert < total_expert_cnt:
            m_tile_in_expert = (m_tile_slot - self._expert_first_slot(expert)) // const_expr(self.slots_per_m_tile)
            cta_tile_m_idx = m_tile_in_expert * const_expr(self.params.cluster_shape_mn[0]) + self.cta_id_in_cluster[0]
            cta_tile_n_idx = n_tile * const_expr(self.params.cluster_shape_mn[1]) + self.cta_id_in_cluster[1]
            k_tile_cnt = self._compute_k_tile_cnt()
            if const_expr(self.fused_tail_max_rows > 0):
                k_tile_cnt = k_tile_cnt + fused_unit
            work_tile_info = MoEWorkTileInfo(
                expert_idx=expert,
                tile_m_idx=cta_tile_m_idx,
                tile_n_idx=cta_tile_n_idx,
                k_tile_cnt=k_tile_cnt,
            )
        return work_tile_info

    @cute.jit
    def _compute_k_tile_cnt(self) -> Int32:
        """K tiles per work tile: ``ceil(hidden / cta_tile_k)``, folded when ``hidden`` is static.
        The Int32 keeps the work tile's field type the same on both arms of the caller's branch."""
        return Int32(ceil_div(self.params.hidden, const_expr(self.params.cta_tile_shape_mnk[2])))

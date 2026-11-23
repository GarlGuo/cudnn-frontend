# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""N_CHUNK output notify of the SM100 MXFP8 block-scaled MoE quant (FC2) forward kernel.

The notifier changes the order in which tiles run and adds the chunk counts, never the tile math, so D must be raw-byte
equal to the stock dynamic scheduler's.  A concurrent consumer copies each chunk the moment ``signal[u] == u``: a chunk
published before its rows are in global memory would carry the NaN prefill into that copy.

L0 keeps one case per behaviour; the remaining chunk shapes, routings and cluster shapes are L1.
"""

import pytest
import torch

from gemm.cutedsl._streaming_signal import BIG_EXPERT, MANY_EXPERTS_64
from gemm.cutedsl._streaming_signal_blockscaled import (  # noqa: F401  (require_sm100, require_sm10x: autouse fixture)
    AMAX_ARG,
    Launcher,
    chunk_consumer,
    chunk_signal,
    chunk_words,
    cluster_id,
    kernel_args,
    make_problem,
    make_quant_kernel,
    new_buffers,
    notify_kernel,
    require_sm100,
    require_sm10x,
    valid_rows,
)
from gemm.cutedsl.test_grouped_gemm_streaming_signal import levels

HIDDEN = 512
ROUTINGS = {"uneven": [700, 0, 256, 1], "many_experts_64": MANY_EXPERTS_64, "big_expert": BIG_EXPERT}
# (D columns N, N chunk C): whole chunks; N < C (one chunk); N == C; a partial last chunk (columns 512..767).
OUTPUT_CHUNKS = {"n1024_c256": (1024, 256), "n768_c1024": (768, 1024), "n768_c768": (768, 768), "n768_c512": (768, 512)}
OUTPUT_CASES = [(chunk, "uneven") for chunk in sorted(OUTPUT_CHUNKS)] + [
    (chunk, routing) for chunk in ("n1024_c256", "n768_c512") for routing in ("many_experts_64", "big_expert")
]


def stock_kernel(experts, one_cta=False):
    return make_quant_kernel(experts, one_cta, use_dynamic_sched=True)


NOTIFY_CASES = [(beside, one_cta, chunk, routing) for beside in (1, 1000) for one_cta in (False, True) for chunk, routing in OUTPUT_CASES]
# A partial last chunk at cluster (2, 1), whole chunks at (1, 1), and N < C (one chunk) with non-stopping extras.
NOTIFY_L0 = {(1, False, "n768_c512", "many_experts_64"), (1, True, "n1024_c256", "uneven"), (1000, False, "n768_c1024", "uneven")}


@pytest.mark.parametrize(
    "clusters_beside_consumer, one_cta, n_and_chunk, routing",
    levels(NOTIFY_CASES, NOTIFY_L0, lambda beside, one_cta, chunk, routing: f"{beside}-{cluster_id(one_cta)}-{chunk}-{routing}"),
)
def test_mxfp8_output_notify_publishes_every_chunk(clusters_beside_consumer, one_cta, n_and_chunk, routing):
    """Every chunk is published (``signal[u] == u``) and D equals the stock dynamic scheduler's bytes."""
    n, chunk = OUTPUT_CHUNKS[n_and_chunk]
    tokens = ROUTINGS[routing]
    problem = make_problem(tokens, n, HIDDEN, seed=3)
    stock, _ = Launcher(stock_kernel(len(tokens), one_cta), problem)(problem)
    kernel = notify_kernel(len(tokens), n, chunk, one_cta, clusters_beside_consumer)
    out, workspace = Launcher(kernel, problem)(problem)
    torch.cuda.synchronize()
    assert torch.equal(out["d"].view(torch.int16), stock["d"].view(torch.int16))
    assert chunk_signal(kernel, workspace).tolist() == list(range(len(range(0, n, chunk))))


# The consumer is the GEMM's PDL dependent on its stream (the contract's topology: a consumer on another stream could take
# an SM a not-yet-resident GEMM cluster needs).  full_grid: the GEMM takes every cluster, so the consumer overlaps only
# once extras stop early; spare_sms: the GEMM leaves SMs free, so the consumer overlaps either way.  The GEMM is the PDL
# dependent of its own reset (waits for it at entry), except with caller_reset, where the caller resets signal[].
CONSUMER_CASES = [(topology, beside, one_cta) for topology in ("full_grid", "spare_sms", "caller_reset") for beside in (1, 1000) for one_cta in (False, True)]
CONSUMER_L0 = {("full_grid", 1, False), ("spare_sms", 1000, False), ("caller_reset", 1, True)}
SPARE_CLUSTERS = 4


@pytest.mark.parametrize(
    "topology, clusters_beside_consumer, one_cta",
    levels(CONSUMER_CASES, CONSUMER_L0, lambda topology, beside, one_cta: f"{topology}-{beside}-{cluster_id(one_cta)}"),
)
def test_mxfp8_output_notify_consumer_reads_each_published_chunk(topology, clusters_beside_consumer, one_cta):
    """A consumer kernel copies chunk u's D words as soon as it acquires ``signal[u] == u``, while the GEMM still runs.
    Each copy equals the final D, so no chunk was published before every row of it was in global memory."""
    from cutlass.utils import HardwareInfo

    n, chunk = 1024, 256
    tokens = ROUTINGS["many_experts_64"]
    problem = make_problem(tokens, n, HIDDEN, seed=5)
    stock, _ = Launcher(stock_kernel(len(tokens), one_cta), problem)(problem)
    caller_reset = topology == "caller_reset"
    kernel = notify_kernel(len(tokens), n, chunk, one_cta, clusters_beside_consumer, caller_reset=caller_reset, use_pdl=not caller_reset)
    clusters = HardwareInfo().get_max_active_clusters(1 if one_cta else 2)
    launcher = Launcher(kernel, problem, max_active_clusters=clusters if topology == "full_grid" else clusters - SPARE_CLUSTERS)
    rows = valid_rows(problem)
    words, begins = chunk_words(rows, n, chunk)
    stream = torch.cuda.Stream()
    for attempt in range(2):
        d, workspace = launcher.buffers(problem)
        signal = chunk_signal(kernel, workspace)
        consume, copied = chunk_consumer(signal, d, words, begins, use_pdl=True)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            if caller_reset:
                signal.fill_(-1)
            launcher.launch(problem, (d, workspace), stream)
            consume(stream)
        torch.cuda.synchronize()
        assert torch.equal(d[:rows].view(torch.int16), stock["d"].view(torch.int16)), f"attempt {attempt}: D differs from stock"
        final = d.view(torch.int32).view(-1)
        early = torch.count_nonzero(copied[words.long()] != final[words.long()]).item()
        assert early == 0, f"attempt {attempt}: {early} of {words.numel()} D words read at publication differ from the final D"
        assert signal.tolist() == list(range(begins.numel() - 1))


@pytest.mark.parametrize("one_cta", levels([(False,), (True,)], {(False,)}, lambda one_cta: cluster_id(one_cta)))
def test_mxfp8_output_notify_with_no_rows_exits(one_cta):
    """A GEMM with no rows exits before publishing anything; signal[] keeps its reset value."""
    problem = make_problem([0, 0, 0, 0], 1024, HIDDEN)
    kernel = notify_kernel(4, 1024, 256, one_cta)
    _, workspace = Launcher(kernel, problem)(problem)
    torch.cuda.synchronize()
    assert chunk_signal(kernel, workspace).tolist() == [-1] * 4


@pytest.mark.L0
def test_mxfp8_output_notify_rejects_invalid():
    """N_CHUNK counts BF16 D's bulk stores only: SFD, a static scheduler, discrete weights and other cluster shapes are
    rejected at construction; amax and an FP8 D at trace time."""
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    from cudnn.gemm.cutedsl.grouped.moe_utils import MoEWeightMode
    from cutlass.cute.runtime import from_dlpack

    with pytest.raises(ValueError, match="use_dynamic_sched"):
        notify_kernel(4, 1024, 256, use_dynamic_sched=False)
    with pytest.raises(ValueError, match="BF16 D"):
        notify_kernel(4, 1024, 256, generate_sfd=True)
    with pytest.raises(ValueError, match="dense weights"):
        notify_kernel(4, 1024, 256, weight_mode=MoEWeightMode.DISCRETE)
    with pytest.raises(ValueError, match="cluster"):
        notify_kernel(4, 1024, 256, cluster_shape_mn=(4, 1))
    problem = make_problem([256, 256, 0, 0], 1024, HIDDEN)
    kernel = notify_kernel(4, 1024, 256)
    d, workspace = new_buffers(kernel, problem)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    args = kernel_args(problem, d, workspace)
    amax = from_dlpack(torch.zeros(4, 1, device="cuda"), assumed_align=16)
    with pytest.raises(Exception, match="no scheduler_counter, amax or SFD"):
        cute.compile(kernel, *args[:AMAX_ARG], amax, *args[AMAX_ARG + 1 :], 4, stream)
    d_fp8 = torch.zeros(problem["rows"], problem["n"], device="cuda", dtype=torch.uint8).view(torch.float8_e4m3fn)
    with pytest.raises(Exception, match="requires BF16 D"):
        cute.compile(kernel, *kernel_args(problem, d_fp8, workspace), 4, stream)

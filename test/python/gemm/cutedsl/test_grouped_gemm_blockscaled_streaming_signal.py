# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming signal options of the SM100 block-scaled grouped GEMM kernels: N_CHUNK output notify on the MXFP8 quant
kernel (FC2).

The notifier reorders tiles and counts them; it never changes the tile math, so D must be raw-byte equal to the stock
dynamic scheduler's.  A concurrent consumer copies each N chunk the moment ``signal[u] == u``; a chunk published before
its rows are in global memory carries the NaN prefill into that copy.

Replays, chunk sizes, batch invariance and claim-slot warm stress are in the determinism file.  L0 keeps one case per
behaviour; the remaining shapes, routings and cluster shapes are L1.
"""

import pytest
import torch

from gemm.cutedsl import _streaming_signal_blockscaled as bs
from gemm.cutedsl._streaming_signal_blockscaled import require_sm10x  # noqa: F401  (autouse fixture)

# ---- N_CHUNK output notify (FC2) ----

# (D columns N, N chunk C): whole chunks; N < C (one chunk); N == C; a partial last chunk (columns 512..767).
OUTPUT_CHUNKS = {"n1024_c256": (1024, 256), "n768_c1024": (768, 1024), "n768_c768": (768, 768), "n768_c512": (768, 512)}
# (clusters beside the consumer, one CTA, N and chunk, routing): every chunk shape at both cluster shapes and both
# beside counts, every routing at both cluster shapes.  1 cluster beside the consumer: the extras stop early; 1000: never.
NOTIFY_CASES = [
    (1, False, "n768_c512", "many_experts_64"),
    (1, True, "n1024_c256", "uneven"),
    (1000, False, "n768_c1024", "uneven"),
    (1000, True, "n768_c768", "uneven"),
    (1, False, "n768_c768", "big_expert"),
    (1000, True, "n768_c512", "big_expert"),
    (1, True, "n768_c1024", "many_experts_64"),
    (1000, False, "n1024_c256", "big_expert"),
]


@pytest.mark.parametrize(
    "clusters_beside_consumer, one_cta, n_and_chunk, routing",
    bs.levels(NOTIFY_CASES, NOTIFY_CASES[:3], lambda beside, one_cta, chunk, routing: f"{beside}-{bs.cluster_id(one_cta)}-{chunk}-{routing}"),
)
def test_mxfp8_output_notify_publishes_every_chunk(clusters_beside_consumer, one_cta, n_and_chunk, routing):
    """Every chunk is published (``signal[u] == u``) and D equals the stock dynamic scheduler's bytes."""
    n, chunk = OUTPUT_CHUNKS[n_and_chunk]
    tokens = bs.ROUTINGS[routing]
    problem = bs.make_problem("quant", tokens, n, seed=3)
    stock, _ = bs.Launcher(bs.stock_kernel("quant", len(tokens), one_cta), problem)(problem)
    kernel = bs.notify_kernel(len(tokens), n, chunk, one_cta, clusters_beside_consumer)
    out, workspace = bs.Launcher(kernel, problem)(problem)
    torch.cuda.synchronize()
    bs.assert_same_bytes(out, stock, "N_CHUNK")
    assert bs.chunk_signal(kernel, workspace).tolist() == list(range(len(range(0, n, chunk))))


# The consumer is the GEMM's PDL dependent on its stream (the contract's topology: a consumer on another stream could take
# an SM a not-yet-resident GEMM cluster needs).  full_grid: the GEMM takes every cluster, so the consumer overlaps only
# once extras stop early; spare_sms: the GEMM leaves SMs free, so the consumer overlaps either way.  The GEMM is the PDL
# dependent of its own reset (waits for it at entry), except with caller_reset, where the caller resets signal[].
# Every topology with both beside counts and both cluster shapes; L0 keeps one cluster beside the consumer, where a chunk
# published early is read soonest.
CONSUMER_CASES = [
    ("full_grid", 1, False),
    ("spare_sms", 1, True),
    ("caller_reset", 1, True),
    ("full_grid", 1000, True),
    ("spare_sms", 1000, False),
    ("caller_reset", 1000, False),
]
SPARE_CLUSTERS = 4


@pytest.mark.parametrize(
    "topology, clusters_beside_consumer, one_cta",
    bs.levels(CONSUMER_CASES, CONSUMER_CASES[:3], lambda topology, beside, one_cta: f"{topology}-{beside}-{bs.cluster_id(one_cta)}"),
)
def test_mxfp8_output_notify_consumer_reads_each_published_chunk(topology, clusters_beside_consumer, one_cta):
    """A consumer kernel copies chunk u's D words as soon as it acquires ``signal[u] == u``, while the GEMM still runs.
    Each copy equals the final D, so no chunk was published before every row of it was in global memory."""
    from cutlass.utils import HardwareInfo

    n, chunk = 1024, 256
    tokens = bs.ROUTINGS["many_experts_64"]
    problem = bs.make_problem("quant", tokens, n, seed=5)
    stock, _ = bs.Launcher(bs.stock_kernel("quant", len(tokens), one_cta), problem)(problem)
    caller_reset = topology == "caller_reset"
    kernel = bs.notify_kernel(len(tokens), n, chunk, one_cta, clusters_beside_consumer, caller_reset=caller_reset, use_pdl=not caller_reset)
    clusters = HardwareInfo().get_max_active_clusters(1 if one_cta else 2)
    launcher = bs.Launcher(kernel, problem, max_active_clusters=clusters if topology == "full_grid" else clusters - SPARE_CLUSTERS)
    rows = bs.valid_rows(problem)
    words, begins = bs.chunk_words(rows, n, chunk)
    stream = torch.cuda.Stream()
    for attempt in range(2):
        d, workspace = launcher.buffers(problem)
        signal = bs.chunk_signal(kernel, workspace)
        consume, copied = bs.chunk_consumer(signal, d, words, begins, use_pdl=True)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            if caller_reset:
                signal.fill_(-1)
            launcher.launch(problem, (d, workspace), stream)
            consume(stream)
        torch.cuda.synchronize()
        bs.assert_same_bytes({"d": d[:rows]}, stock, f"attempt {attempt}")
        final = d.view(torch.int32).view(-1)
        early = torch.count_nonzero(copied[words.long()] != final[words.long()]).item()
        assert early == 0, f"attempt {attempt}: {early} of {words.numel()} D words read at publication differ from the final D"
        assert signal.tolist() == list(range(begins.numel() - 1))


@pytest.mark.L0
def test_mxfp8_output_notify_rejects_invalid():
    """N_CHUNK counts BF16 D's bulk stores only: SFD, a static scheduler, discrete weights and other cluster shapes are
    rejected at construction; amax and an FP8 D at trace time."""
    import cutlass.cute as cute
    from cudnn.gemm.cutedsl.grouped.moe_utils import MoEWeightMode
    from cutlass.cute.runtime import from_dlpack

    with pytest.raises(ValueError, match="use_dynamic_sched"):
        bs.notify_kernel(4, 1024, 256, use_dynamic_sched=False)
    with pytest.raises(ValueError, match="BF16 D"):
        bs.notify_kernel(4, 1024, 256, generate_sfd=True)
    with pytest.raises(ValueError, match="dense weights"):
        bs.notify_kernel(4, 1024, 256, weight_mode=MoEWeightMode.DISCRETE)
    with pytest.raises(ValueError, match="cluster"):
        bs.notify_kernel(4, 1024, 256, cluster_shape_mn=(4, 1))
    problem = bs.make_problem("quant", [256, 256, 0, 0])
    kernel = bs.notify_kernel(4, 1024, 256)
    d, workspace = bs.new_buffers(kernel, problem)
    args = bs.kernel_args(kernel, problem, d, workspace)
    amax = from_dlpack(torch.zeros(4, 1, device="cuda"), assumed_align=16)
    with pytest.raises(Exception, match="no scheduler_counter, amax or SFD"):
        cute.compile(kernel, *args[: bs.AMAX_ARG], amax, *args[bs.AMAX_ARG + 1 :], 4, bs.cu_stream())
    d_fp8 = torch.zeros(problem["rows"], problem["n"], device="cuda", dtype=torch.uint8).view(torch.float8_e4m3fn)
    with pytest.raises(Exception, match="requires BF16 D"):
        cute.compile(kernel, *bs.kernel_args(kernel, problem, d_fp8, workspace), 4, bs.cu_stream())


@pytest.mark.parametrize("one_cta", bs.levels([(False,), (True,)], [(False,)], bs.cluster_id))
def test_mxfp8_output_notify_with_no_rows_exits(one_cta):
    """A GEMM with no rows exits before publishing anything; signal[] keeps its reset value."""
    problem = bs.make_problem("quant", [0, 0, 0, 0])
    kernel = bs.notify_kernel(4, 1024, 256, one_cta)
    _, workspace = bs.Launcher(kernel, problem)(problem)
    torch.cuda.synchronize()
    assert bs.chunk_signal(kernel, workspace).tolist() == [-1] * 4

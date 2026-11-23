# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Wentao Guo, Phuong Nguyen

"""Bitwise replay, static/dynamic equality, batch invariance, warm stress and goldens for the SM100 BF16 grouped GEMMs.

Forward, dgrad, GLU and dGLU (D only: dprob is an atomic sum across N tiles) under the static and the dynamic scheduler.
The dynamic claim order changes which CTA runs a tile, never the tile's reduction, so both schedulers give equal bytes.
"""

from __future__ import annotations

import pytest
import torch

from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot
from gemm.cutedsl import test_grouped_gemm_bf16_wrapper_utils as bf16

pytestmark = pytest.mark.L0

OPS = ["fwd", "dgrad", "glu", "dglu"]
SCHEDS = ["static", "dynamic"]
TILES = {
    "default": (None, None),
    "128x256-c11": ((128, 256), (1, 1)),
    "128x256-c12": ((128, 256), (1, 2)),
    "128x256-c21": ((128, 256), (2, 1)),
}


@pytest.fixture(autouse=True)
def require_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100/103 is required")


def make(op, counts, n, k, seed):
    if op in ("fwd", "dgrad"):
        return bf16.make_gemm_problem(counts, n=n, k=k, direction=op, seed=seed)
    return bf16.make_glu_problem(counts, n=n, k=k, seed=seed)


def run_kwargs(sched=None, tile="default"):
    """The wrapper's own tile and cluster for "default", otherwise the named ones."""
    mma_tiler_mn, cluster_shape_mn = TILES[tile]
    kwargs = {} if mma_tiler_mn is None else dict(mma_tiler_mn=mma_tiler_mn, cluster_shape_mn=cluster_shape_mn)
    if sched is not None:
        kwargs["use_dynamic_sched"] = sched == "dynamic"
    return kwargs


def outputs(op, problem, **kwargs):
    if op in ("fwd", "dgrad"):
        result = bf16.run_gemm(problem, **kwargs)
    elif op == "glu":
        result = bf16.run_glu(problem, **kwargs)
    else:
        return {"d": bf16.run_dglu(problem, **kwargs)["d_row_tensor"]}
    return {"c": result["c_tensor"], "d": result["d_tensor"]}


def check(op, problem, **kwargs):
    if op in ("fwd", "dgrad"):
        bf16.check_gemm(problem, bf16.run_gemm(problem, **kwargs))
    elif op == "glu":
        bf16.check_glu(problem, bf16.run_glu(problem, **kwargs))
    else:
        bf16.check_dglu(problem, bf16.run_dglu(problem, **kwargs))


def segment_problem(op, segments):
    return bf16.gemm_segment_problem(op, segments) if op in ("fwd", "dgrad") else bf16.glu_segment_problem(segments)


def replay_problem(op):
    # Ragged experts with an empty one, enough tiles to cross many dynamic claims.
    return make(op, (1024, 0, 768, 256, 2048), n=2304, k=1536, seed=7)


@pytest.mark.parametrize("tile", list(TILES))
@pytest.mark.parametrize("sched", SCHEDS)
@pytest.mark.parametrize("op", OPS)
def test_sm100_replay_fresh_buffers_with_concurrent_work(op, sched, tile):
    problem = replay_problem(op)
    kwargs = run_kwargs(sched, tile)
    check(op, problem, **kwargs)
    inputs = lambda: snapshot({name: problem[name] for name in ("a", "weights", "prob", "c") if name in problem})
    original = inputs()
    # assert_replays' second-stream work shifts the dynamic claim order between replays.
    assert_replays(lambda: outputs(op, problem, **kwargs), repeats=8)
    assert_bitwise(inputs(), original, "input bytes")


@pytest.mark.parametrize("tile", list(TILES))
@pytest.mark.parametrize("op", OPS)
def test_sm100_static_matches_dynamic(op, tile):
    problem = replay_problem(op)
    static = snapshot(outputs(op, problem, **run_kwargs("static", tile)))
    dynamic = snapshot(outputs(op, problem, **run_kwargs("dynamic", tile)))
    assert_bitwise(dynamic, static, f"{op} {tile} dynamic vs static")


@pytest.mark.parametrize("sched", SCHEDS)
@pytest.mark.parametrize("op", OPS)
def test_sm100_batch_invariance(op, sched):
    kwargs = run_kwargs(sched)
    target = (512, 11)
    alone = [target]
    packed = [(768, 3), target, (256, 5), (1280, 9)]
    reordered = [(1280, 9), (256, 5), target, (768, 3)]
    duplicated = [target, (256, 5), target]
    expected = snapshot(outputs(op, segment_problem(op, alone), **kwargs))
    for name, segments in (("packed", packed), ("reordered", reordered), ("duplicated", duplicated)):
        result = snapshot(outputs(op, segment_problem(op, segments), **kwargs))
        for index, segment in enumerate(segments):
            if segment == target:
                assert_bitwise(bf16.rows_of(result, segments, index), expected, f"{name} segment {index}")
    neighbors = [(768, 4), target, (256, 6)]
    changed = snapshot(outputs(op, segment_problem(op, neighbors), **kwargs))
    assert_bitwise(bf16.rows_of(changed, neighbors, 1), expected, "changed neighbors")


@pytest.mark.parametrize("sched", SCHEDS)
@pytest.mark.parametrize("op", OPS)
def test_sm100_warm_stress(op, sched):
    import cutlass
    from warm_stress import StressCase, assert_warm_stress

    # K=4096 refills every A/B stage many times within a tile; 8 experts span several persistent tiles per CTA.
    problem = make(op, (512,) * 8, n=2048, k=4096, seed=13)
    kwargs = run_kwargs(sched)
    check(op, problem, **kwargs)
    torch.cuda.synchronize()
    dprob = torch.zeros_like(problem["prob"])
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            captured = outputs(op, problem, **kwargs) if op != "dglu" else {"d": bf16.run_dglu(problem, dprob=dprob, **kwargs)["d_row_tensor"]}
        clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(2)
        report = assert_warm_stress(
            StressCase(
                f"SM100 BF16 grouped GEMM {op} {sched} M=8x512 N=2048 K=4096",
                launch=graph.replay,
                outputs=captured,
                thread_blocks=2 * clusters,
                refills_per_launch=4096 // 64,
            )
        )
        print(report.text())
    finally:
        graph.reset()


@pytest.mark.parametrize("op", OPS)
def test_sm100_golden(op):
    problem = make(op, (256, 0, 512), n=384, k=320, seed=21)
    result = assert_replays(lambda: outputs(op, problem), repeats=2)
    check(op, problem)
    inputs = {name: problem[name] for name in ("a", "weights", "offsets", "alpha", "prob")}
    if op == "dglu":
        inputs.update(c=problem["c"], beta=problem["beta"])
    assert_golden(f"grouped_gemm_sm100_bf16_{op}", inputs, result)


def test_sm100_wrappers_default_to_dynamic(monkeypatch):
    from cudnn.gemm.cutedsl.grouped.dglu._bf16_api import GroupedGemmDgluBf16API
    from cudnn.gemm.cutedsl.grouped.glu._bf16_api import GroupedGemmGluBf16API
    from cudnn.gemm.cutedsl.grouped.unfused._bf16_api import GroupedGemmBf16API

    seen = []
    for cls in (GroupedGemmBf16API, GroupedGemmGluBf16API, GroupedGemmDgluBf16API):
        original = cls.execute
        monkeypatch.setattr(cls, "execute", lambda self, *a, _original=original, **k: (seen.append(self.use_dynamic_sched), _original(self, *a, **k))[1])
    for op in OPS:
        outputs(op, make(op, (256, 512), n=512, k=256, seed=3))
    for op in OPS:
        outputs(op, make(op, (256, 512), n=512, k=256, seed=3), use_dynamic_sched=False)
    assert seen == [True] * len(OPS) + [False] * len(OPS)

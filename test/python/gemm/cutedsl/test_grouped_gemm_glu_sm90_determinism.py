# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Wentao Guo, Phuong Nguyen

"""Bitwise replay, batch invariance, warm stress and goldens for the SM90 BF16 grouped GEMM GLU and dGLU.

dGLU's dprob is an atomic sum across N tiles, so only its D (dgate | dup) is held bitwise.
"""

from __future__ import annotations

import pytest
import torch

from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot
from gemm.cutedsl.test_grouped_gemm_glu_sm90 import check_dglu, check_glu, make_problem, run_dglu, run_glu

pytestmark = pytest.mark.L0

OPS = ["glu", "dglu"]


@pytest.fixture(autouse=True)
def require_sm90():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 is required")


def outputs(op, problem, **kwargs):
    if op == "glu":
        result = run_glu(problem, **kwargs)
        return {"c": result["c_tensor"], "d": result["d_tensor"]}
    return {"d": run_dglu(problem, **kwargs)["d_row_tensor"]}


def check(op, problem, **kwargs):
    if op == "glu":
        check_glu(problem, run_glu(problem, **kwargs))
    else:
        check_dglu(problem, run_dglu(problem, **kwargs))


def segment_problem(segments, n=768, k=640):
    """Pack the given (count, seed) segments; each segment's rows, weights, alpha, beta and prob come from its seed alone."""
    parts = [make_problem((count,), n=n, k=k, seed=seed) for count, seed in segments]
    return dict(
        parts[0],
        a=torch.cat([p["a"][: p["counts"][0]] for p in parts]),
        c=torch.cat([p["c"][: p["counts"][0]] for p in parts]),
        weights=torch.cat([p["weights"] for p in parts]),
        offsets=torch.tensor([c for c, _ in segments], dtype=torch.int64).cumsum(0).to(torch.int32).cuda(),
        alpha=torch.tensor([float(1 + seed % 5) * 0.25 for _, seed in segments], device="cuda"),
        beta=torch.tensor([float(1 + seed % 3) * 0.5 for _, seed in segments], device="cuda"),
        prob=torch.cat([p["prob"][: p["counts"][0]] for p in parts]),
        counts=tuple(c for c, _ in segments),
    )


def rows_of(result, segments, index):
    begin = sum(count for count, _ in segments[:index])
    return {name: value[begin : begin + segments[index][0]] for name, value in result.items()}


@pytest.mark.parametrize("op", OPS)
def test_sm90_glu_replay_fresh_buffers_with_concurrent_work(op):
    # Enough tiles to cross many dynamic claims; assert_replays' second-stream noise shifts the claim order.
    problem = make_problem((1024, 0, 768, 256, 2048), n=2304, k=1536, seed=7)
    check(op, problem)
    inputs = lambda: snapshot({"a": problem["a"], "weights": problem["weights"], "prob": problem["prob"], "c": problem["c"]})
    original = inputs()
    assert_replays(lambda: outputs(op, problem), repeats=8)
    assert_bitwise(inputs(), original, "input bytes")


@pytest.mark.parametrize("op", OPS)
def test_sm90_glu_batch_invariance(op):
    target = (512, 11)
    alone = [target]
    packed = [(768, 3), target, (256, 5), (1280, 9)]
    reordered = [(1280, 9), (256, 5), target, (768, 3)]
    duplicated = [target, (256, 5), target]
    expected = snapshot(outputs(op, segment_problem(alone)))
    for name, segments in (("packed", packed), ("reordered", reordered), ("duplicated", duplicated)):
        result = snapshot(outputs(op, segment_problem(segments)))
        for index, segment in enumerate(segments):
            if segment == target:
                assert_bitwise(rows_of(result, segments, index), expected, f"{name} segment {index}")
    changed = snapshot(outputs(op, segment_problem([(768, 4), target, (256, 6)])))
    assert_bitwise(rows_of(changed, [(768, 4), target, (256, 6)], 1), expected, "changed neighbors")


@pytest.mark.parametrize("op", OPS)
def test_sm90_glu_warm_stress(op):
    import cutlass
    from warm_stress import StressCase, assert_warm_stress

    # K=4096 refills every A/B stage many times within a tile; 8 experts span several persistent tiles per CTA.
    problem = make_problem((512,) * 8, n=2048, k=4096, seed=13)
    check(op, problem)
    torch.cuda.synchronize()
    dprob = torch.zeros_like(problem["prob"])
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            captured = outputs(op, problem) if op == "glu" else {"d": run_dglu(problem, dprob=dprob)["d_row_tensor"]}
        clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(2)
        report = assert_warm_stress(
            StressCase(
                f"SM90 grouped GEMM {op} M=8x512 N=2048 K=4096",
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
def test_sm90_glu_golden(op):
    problem = make_problem((256, 0, 512), n=384, k=320, seed=21)
    result = assert_replays(lambda: outputs(op, problem), repeats=2)
    check(op, problem)
    inputs = {name: problem[name] for name in ("a", "weights", "offsets", "alpha", "prob")}
    if op == "dglu":
        inputs.update(c=problem["c"], beta=problem["beta"])
    assert_golden(f"grouped_gemm_{op}_sm90_bf16", inputs, result)

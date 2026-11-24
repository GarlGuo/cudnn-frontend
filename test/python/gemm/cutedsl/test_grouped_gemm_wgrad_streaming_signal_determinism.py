# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Determinism and batch invariance of the grouped GEMM wgrad kernels with the streaming input signal.

dW[e] reduces over expert e's tokens only, one CTA (pair) per output tile in a fixed K order, whatever the claim order
and whenever the producer publishes a token tile. The suite pins that as raw bytes:
- replay of fixed inputs under concurrent work, with every token tile ready and with a delayed out-of-order producer;
- one expert's dW alone, packed with neighbors, moved to another position (another token offset, so another split of
  its rows across 256-row signal tiles), duplicated, and next to changed neighbor data.
An independent reference is checked separately. BF16 and MXFP8 (E4M3 data, E8M0 scales).
"""

import pytest
import torch

from determinism_utils import assert_bitwise, assert_replays, snapshot
from gemm.cutedsl.test_grouped_gemm_wgrad_bf16_utils import assert_grouped_gemm_wgrad_close
from gemm.cutedsl.test_grouped_gemm_wgrad_utils import (
    _wgrad_assemble_scales_2d2d,
    _wgrad_create_scale_tensor,
    _wgrad_independent_reference,
    check_ref_grouped_gemm_wgrad,
)
from gemm.cutedsl.test_grouped_gemm_wgrad_streaming_signal_utils import (
    SF_VEC,
    WgradLauncher,
    capacity_for,
    poison_operands,
    run_ready,
    run_with_delayed_producer,
)

pytestmark = pytest.mark.L0


@pytest.fixture(autouse=True)
def require_sm100():
    from cudnn.tensor_adapter import has_tcgen05

    if not torch.cuda.is_available() or not has_tcgen05(torch.cuda.get_device_capability()[0]):
        pytest.skip("Requires SM100+ for grouped GEMM wgrad")


M, N = 768, 1024
EXPERTS = 6
# Target expert tokens: BF16 groups are multiples of 256, MXFP8 of 128 (an odd number of 128-row K blocks, so the
# expert's split across 256-row signal tiles depends on its offset).
TARGET_TOKENS = {"bf16": 768, "mxfp8": 640}
NEIGHBOR_TOKENS = {"bf16": (256, 1024, 512, 2048), "mxfp8": (384, 128, 1152, 256)}
CONFIGS = {
    "bf16": [((256, 256), (2, 1)), ((128, 256), (1, 1))],
    "mxfp8": [((256, 256), (2, 1)), ((128, 128), (1, 1))],
}
CASES = [
    pytest.param(kind, config, id=f"{kind}-tile{config[0][0]}x{config[0][1]}-cluster{config[1][0]}x{config[1][1]}")
    for kind in CONFIGS
    for config in CONFIGS[kind]
]


def _expert_data(kind, tokens, seed):
    """One expert's logical operands (and raw MXFP8 scales), fixed by ``seed``."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    if kind == "bf16":
        a = torch.randn((M, tokens), device="cuda", dtype=torch.bfloat16, generator=generator) * 0.125
        b = torch.randn((tokens, N), device="cuda", dtype=torch.bfloat16, generator=generator) * 0.125
        return {"tokens": tokens, "a": a, "b": b}
    a = torch.randint(-1, 2, (M, tokens), device="cuda", generator=generator).to(torch.bfloat16)
    b = torch.randint(-1, 2, (tokens, N), device="cuda", generator=generator).to(torch.bfloat16)
    torch.manual_seed(seed)
    sfa = _wgrad_create_scale_tensor((M, -(-tokens // SF_VEC)), torch.float8_e8m0fnu)
    sfb = _wgrad_create_scale_tensor((N, -(-tokens // SF_VEC)), torch.float8_e8m0fnu)
    return {"tokens": tokens, "a": a, "b": b, "sfa": sfa, "sfb": sfb}


def _empty(kind):
    data = {"tokens": 0, "a": torch.empty((M, 0), device="cuda", dtype=torch.bfloat16), "b": torch.empty((0, N), device="cuda", dtype=torch.bfloat16)}
    if kind == "mxfp8":
        data["sfa"] = torch.empty((M, 0), device="cuda", dtype=torch.float8_e8m0fnu)
        data["sfb"] = torch.empty((N, 0), device="cuda", dtype=torch.float8_e8m0fnu)
    return data


def _pack(kind, experts):
    """Concatenate per-expert token ranges in the kernel-facing layouts (K-major A and B)."""
    groups = tuple(e["tokens"] for e in experts)
    logical_a = torch.cat([e["a"] for e in experts], dim=1)
    logical_b = torch.cat([e["b"] for e in experts], dim=0)
    problem = {
        "kind": kind,
        "group_k_list": groups,
        "experts": len(groups),
        "offsets": torch.tensor(groups, dtype=torch.int32, device="cuda").cumsum(0, dtype=torch.int32),
        "output": torch.empty((len(groups), M, N), dtype=torch.bfloat16, device="cuda"),
        "logical_a": logical_a,
        "logical_b": logical_b,
    }
    if kind == "bf16":
        problem["a"] = logical_a.contiguous()
        problem["b"] = logical_b.t().contiguous().t()
        return problem
    problem["a"] = logical_a.to(torch.float8_e4m3fn).contiguous()
    problem["b"] = logical_b.to(torch.float8_e4m3fn).t().contiguous().t()
    problem["raw_sfa"] = [e["sfa"] for e in experts]
    problem["raw_sfb"] = [e["sfb"] for e in experts]
    problem["sfa"] = _wgrad_assemble_scales_2d2d(problem["raw_sfa"], M)
    problem["sfb"] = _wgrad_assemble_scales_2d2d(problem["raw_sfb"], N)
    return problem


def _reference(problem):
    if problem["kind"] == "bf16":
        out, begin = [], 0
        for tokens in problem["group_k_list"]:
            out.append((problem["logical_a"][:, begin : begin + tokens].float() @ problem["logical_b"][begin : begin + tokens].float()).to(torch.bfloat16))
            begin += tokens
        return torch.stack(out)
    return _wgrad_independent_reference(
        problem["logical_a"].float(),
        problem["logical_b"].float(),
        problem["raw_sfa"],
        problem["raw_sfb"],
        list(problem["group_k_list"]),
        SF_VEC,
        torch.bfloat16,
    )


def _launcher(kind, problem, config, **options):
    tile, cluster = config
    # Capacity of the largest scenario, so one compiled kernel serves every packing.
    return WgradLauncher(kind, EXPERTS, tile=tile, cluster=cluster, capacity=max(capacity_for(problem), 40), **options)


def _replay_problem(kind):
    target = _expert_data(kind, TARGET_TOKENS[kind], seed=7)
    neighbors = [_expert_data(kind, t, seed=100 + i) for i, t in enumerate(NEIGHBOR_TOKENS[kind])]
    return _pack(kind, [neighbors[0], target, _empty(kind), neighbors[1], neighbors[2], neighbors[3]])


def _check_reference(problem, output):
    if problem["kind"] == "bf16":
        assert_grouped_gemm_wgrad_close(output, _reference(problem))
    else:
        check_ref_grouped_gemm_wgrad(output, _reference(problem))


@pytest.mark.parametrize("kind, config", CASES)
def test_signal_replays_bitwise(kind, config):
    """Every token tile ready: replays under concurrent work give the same bytes."""
    problem = _replay_problem(kind)
    launcher = _launcher(kind, problem, config)
    expected = assert_replays(lambda: {"wgrad": run_ready(launcher, problem)})
    _check_reference(problem, expected["wgrad"])


@pytest.mark.parametrize("kind, config", CASES)
def test_signal_replays_with_delayed_producer(kind, config):
    """The producer's timing and tile order must not change a byte."""
    problem = _replay_problem(kind)
    expected = snapshot({"wgrad": run_ready(_launcher(kind, problem, config), problem)})
    delayed = _launcher(kind, problem, config, max_active_clusters=4)
    final = poison_operands(problem)
    for seed, order in ((1, "shuffled"), (2, "shuffled"), (0, "reversed")):
        actual = run_with_delayed_producer(delayed, problem, final, order=order, seed=seed, delay_ns=5_000)
        assert_bitwise({"wgrad": actual}, expected, f"delayed {order} producer, seed {seed}")
        poison_operands(problem)


def _scenarios(kind):
    """(experts, slots holding the target expert) per packing."""
    target = _expert_data(kind, TARGET_TOKENS[kind], seed=7)
    neighbors = [_expert_data(kind, t, seed=100 + i) for i, t in enumerate(NEIGHBOR_TOKENS[kind])]
    changed = [_expert_data(kind, t, seed=200 + i) for i, t in enumerate(NEIGHBOR_TOKENS[kind])]
    empty = _empty(kind)
    n0, n1, n2, n3 = neighbors
    return {
        "alone": ([target] + [empty] * 5, [0]),
        "packed": ([n0, n1, target, n2, empty, n3], [2]),
        "reordered": ([n3, empty, n2, n0, target, n1], [4]),
        "duplicated": ([target, n0, empty, target, n1, n2], [0, 3]),
        "changed_neighbors": ([changed[0], changed[1], target, changed[2], empty, changed[3]], [2]),
    }


@pytest.mark.parametrize("kind, config", CASES)
def test_signal_batch_invariance(kind, config):
    scenarios = _scenarios(kind)
    probe = _pack(kind, scenarios["packed"][0])
    launcher = _launcher(kind, probe, config)
    results = {}
    for name, (experts, slots) in scenarios.items():
        problem = _pack(kind, experts)
        output = run_ready(launcher, problem)
        results[name] = output
        for slot in slots:
            got = snapshot({"wgrad": output[slot]})
            if name == "alone":
                alone = got
                _check_reference(problem, output)
            else:
                assert_bitwise(got, alone, f"{name} slot {slot} vs alone")
    # Neighbors moved between packings keep their bytes too.
    assert_bitwise({"wgrad": results["reordered"][3]}, snapshot({"wgrad": results["packed"][0]}), "neighbor 0 moved")
    assert_bitwise({"wgrad": results["reordered"][5]}, snapshot({"wgrad": results["packed"][1]}), "neighbor 1 moved")


@pytest.mark.parametrize("kind, config", CASES)
def test_signal_batch_invariance_with_delayed_producer(kind, config):
    """The target expert, packed and moved, with a delayed out-of-order producer: the same bytes as alone and ready."""
    scenarios = _scenarios(kind)
    probe = _pack(kind, scenarios["packed"][0])
    alone = _pack(kind, scenarios["alone"][0])
    expected = snapshot({"wgrad": run_ready(_launcher(kind, probe, config), alone)[0]})
    delayed = _launcher(kind, probe, config, max_active_clusters=4)
    for seed, name in enumerate(("packed", "reordered")):
        experts, slots = scenarios[name]
        problem = _pack(kind, experts)
        final = poison_operands(problem)
        output = run_with_delayed_producer(delayed, problem, final, order="shuffled", seed=9 + seed, delay_ns=5_000)
        assert_bitwise({"wgrad": output[slots[0]]}, expected, f"{name}, delayed producer vs alone")

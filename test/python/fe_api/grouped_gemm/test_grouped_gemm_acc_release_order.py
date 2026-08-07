# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ordering guard for the overlapping-accumulator early release (SM100 grouped GEMMs).

An epilogue that early-releases the overlapping accumulator stage must issue the
TMEM->register ``cute.copy`` of a subtile *before* publishing that stage as empty.
``cute.arch.fence_view_async_tmem_load()`` lowers to ``tcgen05.wait::ld``, which per
PTX ISA 9.7.17.8.5 blocks only until *prior* ``tcgen05.ld`` operations complete -- it
cannot cover a load that has not been issued yet. Releasing first therefore hands the
overlapping TMEM columns to the MMA producer while the epilogue is still reading them
(PTX ISA 9.7.17.6.4.4 mandates ld -> wait::ld -> fence -> arrive).

Pure source analysis: no GPU, no ``[cutedsl]`` install, no kernel launch.
"""

import ast
from pathlib import Path

import pytest

KERNEL_ROOT = Path(__file__).resolve().parents[4] / "python" / "cudnn" / "gemm" / "cutedsl" / "grouped"

# ``cute.copy`` atoms that move Tensor Memory into registers.
T2R_COPY_ATOMS = {"tiled_copy_t2r", "copy_atom_t2r"}


def _calls(node):
    return [sub for sub in ast.walk(node) if isinstance(sub, ast.Call)]


def _has_acc_consumer_release(node):
    return any(
        isinstance(c.func, ast.Attribute) and c.func.attr == "consumer_release" and isinstance(c.func.value, ast.Name) and c.func.value.id == "acc_pipeline"
        for c in _calls(node)
    )


def _first_t2r_copy(node):
    for c in _calls(node):
        fn = c.func
        if isinstance(fn, ast.Attribute) and fn.attr == "copy" and c.args:
            first = c.args[0]
            if isinstance(first, ast.Name) and first.id in T2R_COPY_ATOMS:
                return c
    return None


def _references_early_release_threshold(node):
    return any(isinstance(sub, ast.Attribute) and sub.attr == "iter_acc_early_release_in_epilogue" for sub in ast.walk(node))


# Sites carrying the same ordering defect that this change deliberately does not touch.
# strict=True, so if one is fixed the test XPASSes and fails, prompting removal from here.
# Note the other Rubin-family kernels (dglu_rubin, glu_rubin) already order this correctly --
# this is specific to quant_rubin, not to Rubin kernels in general.
KNOWN_UNFIXED = {
    "moe_blockscaled_grouped_gemm_quant_rubin.py": "same defect; not fixed here, no sm_103 hardware to validate on",
}


def _collect_sites():
    """Every ``for subtile_idx`` epilogue loop that early-releases the accumulator.

    The tail ``consumer_release`` under ``if not self.overlapping_accum`` lives outside
    the subtile loop, so restricting to the loop body correctly excludes it.
    """
    sites = []
    for path in sorted(KERNEL_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.For) or not isinstance(node.target, ast.Name) or node.target.id != "subtile_idx":
                continue
            release_idx = copy_idx = None
            release_line = copy_line = None
            for i, stmt in enumerate(node.body):
                if release_idx is None and _references_early_release_threshold(stmt) and _has_acc_consumer_release(stmt):
                    release_idx = i
                    release_line = min(
                        c.lineno
                        for c in _calls(stmt)
                        if isinstance(c.func, ast.Attribute) and c.func.attr == "consumer_release" and isinstance(c.func.value, ast.Name)
                    )
                if copy_idx is None:
                    call = _first_t2r_copy(stmt)
                    if call is not None:
                        copy_idx, copy_line = i, call.lineno
            if release_idx is not None and copy_idx is not None:
                sites.append(
                    pytest.param(
                        path.relative_to(KERNEL_ROOT),
                        release_idx,
                        copy_idx,
                        release_line,
                        copy_line,
                        id=str(path.relative_to(KERNEL_ROOT).with_suffix("")).replace("/", "-"),
                        marks=([pytest.mark.xfail(reason=KNOWN_UNFIXED[path.name], strict=True)] if path.name in KNOWN_UNFIXED else []),
                    )
                )
    return sites


ACC_RELEASE_SITES = _collect_sites()


@pytest.mark.L0
def test_early_release_sites_are_discoverable():
    """Guard the guard: if the epilogue is restructured, this file must be revisited."""
    assert ACC_RELEASE_SITES, f"no overlapping-accumulator early-release sites found under {KERNEL_ROOT}"


@pytest.mark.L0
@pytest.mark.parametrize("path, release_idx, copy_idx, release_line, copy_line", ACC_RELEASE_SITES)
def test_tmem_read_precedes_accumulator_release(path, release_idx, copy_idx, release_line, copy_line):
    assert copy_idx < release_idx, (
        f"{path}: acc_pipeline.consumer_release at line {release_line} precedes the "
        f"TMEM->register cute.copy at line {copy_line}. fence_view_async_tmem_load() "
        f"(tcgen05.wait::ld) only covers loads already issued, so the accumulator stage "
        f"is published empty while the overlapping TMEM columns are still unread. Move the "
        f"copy above the fence/release, as in grouped/wgrad/moe_blockscaled_grouped_gemm_wgrad.py."
    )

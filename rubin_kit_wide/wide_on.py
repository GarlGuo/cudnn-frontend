"""pytest plugin (test-only): force the wide BF16 wgrad kernel on, i.e. GroupedGemmWgradBf16API._use_wide_tiles = True.

Load with `-p wide_on` (this file's directory on PYTHONPATH) or PYTEST_PLUGINS=wide_on; the xdist / isolation workers
inherit PYTEST_PLUGINS. WIDE_LOG_DIR=<dir> (optional): every _wide_applies() decision is appended to
<dir>/wide.<pid>.tsv as "<test nodeid>\t<True|False>", so a run can show which tests actually ran the wide kernel.
The library has no knob for this; the attribute is private and stays False by default.
"""

import os

import pytest

_CURRENT = {"id": None}


def _force_wide():
    from cudnn.gemm.cutedsl.grouped.wgrad import _bf16_api

    cls = _bf16_api.GroupedGemmWgradBf16API
    cls._use_wide_tiles = True
    if getattr(cls._wide_applies, "_wide_on_wrapped", False):
        return
    original = cls._wide_applies
    log_dir = os.environ.get("WIDE_LOG_DIR")

    def recorded(self):
        applies = original(self)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, f"wide.{os.getpid()}.tsv"), "a") as f:
                f.write(f"{_CURRENT['id']}\t{applies}\n")
        return applies

    recorded._wide_on_wrapped = True
    cls._wide_applies = recorded


def pytest_configure(config):
    _force_wide()


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    _CURRENT["id"] = item.nodeid
    _force_wide()


def pytest_collection_modifyitems(config, items):
    # This test asserts the library default (off); it cannot pass with the plugin on.
    keep = [i for i in items if "test_wide_tiles_are_off_by_default" not in i.nodeid]
    if len(keep) != len(items):
        config.hook.pytest_deselected(items=[i for i in items if i not in keep])
        items[:] = keep

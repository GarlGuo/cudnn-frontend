"""pytest plugin (test-only): run the wide BF16 wgrad on every call where it is eligible, i.e. set
GroupedGemmWgradBf16API._WIDE_MIN_TILES_PER_CLUSTER to 0 for both claim modes (the library picks it only from a
number of wide tiles per resident cluster). Load with PYTEST_PLUGINS=wide_on (this file's directory on PYTHONPATH).
WIDE_LOG_DIR=<dir> (optional): every _wide_applies() decision is appended to <dir>/wide.<pid>.tsv as
"<test nodeid>\\t<True|False>", so a run shows which tests ran the wide kernel. Tests that monkeypatch the threshold
themselves keep their own value. WIDE_FORCE=0: log the library's own decisions without forcing anything.
"""

import os

import pytest

_CURRENT = {"id": None}
_FORCE = os.environ.get("WIDE_FORCE", "1") == "1"


def _force_wide():
    from cudnn.gemm.cutedsl.grouped.wgrad import _bf16_api

    cls = _bf16_api.GroupedGemmWgradBf16API
    if not hasattr(cls, "_wide_applies"):
        return  # a tree without the wide kernel
    if _FORCE:
        cls._WIDE_MIN_TILES_PER_CLUSTER = {False: 0, True: 0}
    if getattr(cls._wide_applies, "_wide_on_wrapped", False):
        return
    original = cls._wide_applies
    log_dir = os.environ.get("WIDE_LOG_DIR")
    summary_dir = os.environ.get("OPTIN_LOG_DIR")  # GB200 pool runner: one "WIDE <pick> <test>" line per decision

    def recorded(self):
        applies = original(self)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, f"wide.{os.getpid()}.tsv"), "a") as f:
                f.write(f"{_CURRENT['id']}\t{applies}\n")
        elif summary_dir:
            test = str(_CURRENT["id"]).split("::")[-1].split("[")[0]
            with open(os.path.join(summary_dir, f"applied.wide.{os.getpid()}"), "a") as f:
                f.write(f"WIDE {applies} {test}\n")
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
    if not _FORCE:
        return
    # These tests check the library's own threshold; with it forced to 0 they cannot pass.
    drop = ("test_bf16_wgrad_default_picks_wide_at_rule_boundary", "test_bf16_wgrad_wide_rule", "test_bf16_wgrad_wide_selection")
    keep = [i for i in items if not any(d in i.nodeid for d in drop)]
    if len(keep) != len(items):
        config.hook.pytest_deselected(items=[i for i in items if i not in keep])
        items[:] = keep

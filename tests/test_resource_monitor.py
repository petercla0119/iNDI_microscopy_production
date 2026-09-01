"""Tests for scripts/resource_monitor.py and its StageRecorder integration.

Run:
    python -m pytest tests/test_resource_monitor.py -v
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).parent.parent / "scripts"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # so run_utils can `import resource_monitor`
    spec.loader.exec_module(mod)
    return mod


rm = _load("resource_monitor", "resource_monitor.py")
ru = _load("run_utils", "run_utils.py")

psutil_required = pytest.mark.skipif(rm.psutil is None, reason="psutil not installed")


# --- core monitor -----------------------------------------------------------

@psutil_required
def test_monitor_measures_tree(monkeypatch):
    """Monitoring a busy child subprocess records CPU, memory, and wall time."""
    monkeypatch.delenv("HSARRAY_MONITOR", raising=False)
    # Child allocates ~80 MB and burns CPU for ~1.5 s.
    code = textwrap.dedent("""
        import time
        blob = bytearray(80 * 1024 * 1024)
        end = time.time() + 1.5
        x = 0
        while time.time() < end:
            x += 1
    """)
    proc = subprocess.Popen([sys.executable, "-c", code])
    mon = rm.ResourceMonitor("busy", pid=proc.pid, interval=0.1,
                             write_on_stop=False).start()
    proc.wait()
    m = mon.stop().metrics()

    assert m["wall_s"] > 0
    assert m["max_rss_mb"] > 1           # child held tens of MB
    assert m["max_cpu_pct"] > 0          # child burned CPU
    assert m["n_samples"] >= 1
    assert m["peak_nproc"] >= 1


def test_disabled_via_env(monkeypatch):
    """HSARRAY_MONITOR=0 disables monitoring; start() no-ops, metrics() empty."""
    monkeypatch.setenv("HSARRAY_MONITOR", "0")
    assert rm.enabled() is False
    mon = rm.ResourceMonitor("noop").start()
    assert mon._thread is None
    assert mon.stop().metrics() == {}


# --- StageRecorder integration ----------------------------------------------

@psutil_required
def test_stagerecorder_folds_resources(tmp_path, monkeypatch):
    """A stage record in run_metadata.json carries a `resources` block."""
    monkeypatch.delenv("HSARRAY_MONITOR", raising=False)
    monkeypatch.setenv("HSARRAY_MONITOR_INTERVAL", "0.05")

    run_id = "20260831_000000_test"
    run_dir = ru.create_run_dir(tmp_path, run_id)
    rec = ru.StageRecorder(run_dir, stage="demo", run_id=run_id)
    rec.finish(summary={"ok": True})

    doc = json.loads((run_dir / "run_metadata.json").read_text())
    resources = doc["stages"][-1]["resources"]
    for key in ("wall_s", "max_rss_mb", "max_cpu_pct", "peak_nproc", "n_samples"):
        assert key in resources

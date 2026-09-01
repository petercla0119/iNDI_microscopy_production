"""Low-overhead per-step resource monitor for the iNDI pipeline.

Samples the current process tree (parent + Dask/ProcessPoolExecutor workers) on
a background thread and records peak RSS and peak CPU% for a labeled step.

The pipeline has no orchestrator, so the natural hook is ``run_utils.StageRecorder``:
it already brackets every step and, when this module is importable, folds the
peak metrics into each stage's ``run_metadata.json`` record. This file can also
be used standalone:

1. Context manager (writes one TSV row per step)::

       from resource_monitor import monitor_step
       with monitor_step("Segment nuclei"):
           dask.compute(*tasks)

2. Standalone, wrapping any command (e.g. a sbatch submit or convert_to_zarr)::

       python -m resource_monitor --label "Convert" -- python convert_to_zarr.py data/<uuid>

3. Standalone, attaching to an already-running PID::

       python -m resource_monitor --label foo --pid 12345

Output location for the TSV (first match wins):
    --out FILE  >  $HSARRAY_BENCHMARK_DIR/benchmarks.tsv  >  ./benchmarks/benchmarks.tsv

Sampling is cheap: one ``memory_info().rss`` + ``cpu_percent()`` per process
per tick (default 2 s), guarded so a monitoring error can never crash the step.
Set HSARRAY_MONITOR=0 to disable, HSARRAY_MONITOR_INTERVAL to change the tick.
"""

import os
import sys
import threading
import time
from pathlib import Path

try:
    import psutil
except ImportError:  # monitoring must never be a hard dependency of a step
    psutil = None

# ponytail: RSS-only (memory_info), not USS/PSS — USS/PSS reads /proc smaps
# every tick and defeats "low resource". Add memory_full_info() if per-step
# shared-memory accounting is ever actually needed.

COLUMNS = [
    "iso_time", "stage", "step", "wall_s", "max_rss_mb", "max_vms_mb",
    "max_cpu_pct", "mean_cpu_pct", "cpu_time_s", "peak_nproc", "n_samples",
]

_DEFAULT_INTERVAL = float(os.environ.get("HSARRAY_MONITOR_INTERVAL", "2.0"))


def enabled() -> bool:
    """True unless psutil is missing or HSARRAY_MONITOR=0 disables monitoring."""
    return psutil is not None and os.environ.get("HSARRAY_MONITOR") != "0"


def _resolve_out(out):
    if out:
        return Path(out)
    env_dir = os.environ.get("HSARRAY_BENCHMARK_DIR")
    base = Path(env_dir) if env_dir else Path.cwd() / "benchmarks"
    return base / "benchmarks.tsv"


def _iso(t):
    # local time, no timezone math needed for a run log
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t))


class ResourceMonitor:
    """Sample a process tree's peak RSS/CPU until stopped.

    stop() finalizes metrics; metrics() returns them as a dict. When
    write_on_stop is True (standalone/context-manager use) stop() also appends
    a TSV row. StageRecorder sets write_on_stop=False and folds metrics() into
    run_metadata.json instead.
    """

    def __init__(self, step, pid=None, out=None, interval=None, stage=None,
                 write_on_stop=True):
        self.step = step
        self.pid = pid or os.getpid()
        self.out = _resolve_out(out)
        self.interval = interval or _DEFAULT_INTERVAL
        self.stage = stage if stage is not None else os.environ.get("HSARRAY_STAGE", "")
        self.write_on_stop = write_on_stop
        self._stop = threading.Event()
        self._thread = None
        self._pcache = {}  # pid -> psutil.Process, reused so cpu_percent() deltas work
        self.max_rss = 0
        self.max_vms = 0
        self.max_cpu = 0.0
        self._cpu_sum = 0.0
        self._samples = 0
        self.peak_nproc = 0
        self.cpu_time = 0.0
        self.t0 = None

    # -- context manager -----------------------------------------------------
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False  # never suppress the step's own exceptions

    # -- lifecycle -----------------------------------------------------------
    def start(self):
        if not enabled():
            if psutil is None:
                print(f"  [monitor] psutil unavailable — '{self.step}' not measured",
                      file=sys.stderr)
            return self
        self.t0 = time.time()
        self._thread = threading.Thread(target=self._run, name="resmon", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        if self._thread is None:  # never started (disabled / no psutil)
            return self
        self._stop.set()
        self._thread.join(timeout=self.interval + 5)
        self._finalize_cpu_time()
        if self.write_on_stop:
            self._write()
        return self

    # -- internals -----------------------------------------------------------
    def _live_procs(self):
        """Current process tree, reusing cached Process objects per pid.

        cpu_percent(None) reports usage since the *same object's* last call,
        so objects must persist across ticks. New processes are primed (their
        first reading is 0) and dead ones are dropped.
        """
        try:
            root = self._pcache.get(self.pid) or psutil.Process(self.pid)
        except psutil.Error:
            return []
        self._pcache[self.pid] = root
        tree = [root]
        try:
            tree += root.children(recursive=True)
        except psutil.Error:
            pass
        result, seen = [], set()
        for p in tree:
            seen.add(p.pid)
            cached = self._pcache.get(p.pid)
            if cached is None:
                self._pcache[p.pid] = p
                try:
                    p.cpu_percent(None)  # prime; delta counts from here
                except psutil.Error:
                    pass
                result.append(p)
            else:
                result.append(cached)
        for pid in [k for k in self._pcache if k not in seen and k != self.pid]:
            del self._pcache[pid]
        return result

    def _run(self):
        # prime the root so the first real reading spans a full interval
        self._live_procs()
        while not self._stop.wait(self.interval):
            self._sample()
        # one final sample after stop so short steps still record something
        self._sample()

    def _sample(self):
        rss = vms = cpu = 0.0
        n = 0
        for p in self._live_procs():
            try:
                mi = p.memory_info()
                rss += mi.rss
                vms += mi.vms
                cpu += p.cpu_percent(None)
                n += 1
            except psutil.Error:
                continue  # process vanished mid-sample; skip it
        if n == 0:
            return
        self.max_rss = max(self.max_rss, rss)
        self.max_vms = max(self.max_vms, vms)
        self.max_cpu = max(self.max_cpu, cpu)
        self.peak_nproc = max(self.peak_nproc, n)
        self._cpu_sum += cpu
        self._samples += 1

    def _finalize_cpu_time(self):
        # best-effort: parent's cumulative CPU time (children that already
        # exited are not fully counted — that's the documented ceiling)
        try:
            ct = psutil.Process(self.pid).cpu_times()
            self.cpu_time = round(ct.user + ct.system, 1)
        except psutil.Error:
            self.cpu_time = 0.0

    def metrics(self) -> dict:
        """Peak resource metrics for this step (empty if monitoring didn't run)."""
        if self.t0 is None:
            return {}
        wall = time.time() - self.t0
        mean_cpu = self._cpu_sum / self._samples if self._samples else 0.0
        return {
            "wall_s": round(wall, 1),
            "max_rss_mb": round(self.max_rss / 1e6, 1),
            "max_vms_mb": round(self.max_vms / 1e6, 1),
            "max_cpu_pct": round(self.max_cpu, 1),
            "mean_cpu_pct": round(mean_cpu, 1),
            "cpu_time_s": self.cpu_time,
            "peak_nproc": self.peak_nproc,
            "n_samples": self._samples,
        }

    def _write(self):
        m = self.metrics()
        if not m:
            return
        row = {"iso_time": _iso(self.t0), "stage": self.stage, "step": self.step, **m}
        try:
            self.out.parent.mkdir(parents=True, exist_ok=True)
            new = not self.out.exists()
            with open(self.out, "a") as f:
                if new:
                    f.write("\t".join(COLUMNS) + "\n")
                f.write("\t".join(str(row[c]) for c in COLUMNS) + "\n")
        except OSError as e:
            print(f"  [monitor] could not write benchmark row: {e}", file=sys.stderr)
        print(f"  [monitor] {self.step}: peak_rss={row['max_rss_mb']}MB "
              f"peak_cpu={row['max_cpu_pct']}% wall={row['wall_s']}s")


def monitor_step(step, **kw):
    """Convenience factory mirroring the class, for `with monitor_step(...):`."""
    return ResourceMonitor(step, **kw)


def set_benchmark_context(stage, benchmark_dir):
    """Point per-step benchmark TSVs at benchmark_dir and tag rows with stage.

    Honors a pre-set HSARRAY_BENCHMARK_DIR env var so the location can still be
    overridden (e.g. in an sbatch wrapper).
    """
    os.environ["HSARRAY_STAGE"] = stage
    os.environ.setdefault("HSARRAY_BENCHMARK_DIR", str(Path(benchmark_dir)))


# ---------------------------------------------------------------------------
# Standalone CLI: wrap a command or attach to a PID
# ---------------------------------------------------------------------------

def _main(argv):
    import argparse
    import subprocess

    ap = argparse.ArgumentParser(
        description="Monitor peak RSS/CPU of a process tree for one step.")
    ap.add_argument("--label", required=True, help="Step name recorded in the TSV")
    ap.add_argument("--stage", default=None, help="Optional stage tag column")
    ap.add_argument("--out", default=None, help="Output TSV (default: env/cwd)")
    ap.add_argument("--interval", type=float, default=None, help="Sample seconds")
    ap.add_argument("--pid", type=int, default=None,
                    help="Attach to an existing PID and exit when it exits")
    ap.add_argument("cmd", nargs=argparse.REMAINDER,
                    help="-- COMMAND ARGS to launch and monitor")
    args = ap.parse_args(argv)

    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd

    if args.pid and cmd:
        ap.error("give either --pid or a command, not both")

    if cmd:
        proc = subprocess.Popen(cmd)
        mon = ResourceMonitor(args.label, pid=proc.pid, out=args.out,
                              interval=args.interval, stage=args.stage)
        mon.start()
        rc = proc.wait()
        mon.stop()
        return rc

    if args.pid:
        if psutil is None:
            print("psutil unavailable", file=sys.stderr)
            return 1
        mon = ResourceMonitor(args.label, pid=args.pid, out=args.out,
                              interval=args.interval, stage=args.stage)
        mon.start()
        try:
            while psutil.pid_exists(args.pid):
                time.sleep(mon.interval)
        except KeyboardInterrupt:
            pass
        mon.stop()
        return 0

    ap.error("provide --pid or a -- COMMAND to monitor")


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))

#!/usr/bin/env python3
"""Continuously label the pre-generated 100k ADV dataset with the full VLM quick sim.

Sibling of ``overlap_search/label_forever.py`` with the *same operational form*
(configurable cores, time-based flush + terminal progress report, resumable,
appends to JSONL), but instead of computing the CadQuery overlap label it runs
the prewritten AeroForge quick sim (``generate_quick_report`` internals:
``collect_aero_metrics`` + ``maybe_add_asb_vlm_metrics``) and stores the curated
quick-report JSON per design (plus flattened L/D and CD0 headline scalars).

Run (laptop, e.g. 16 cores):

    conda run -n bachelor-thesis python -u \
      chapter7_aeroforge/quicksim/label_quicksim_forever.py --16

Cloud (80 cores):

    conda run -n bachelor-thesis python -u \
      chapter7_aeroforge/quicksim/label_quicksim_forever.py --80

Multicore note (the decisive knob here):
  The overlap labeler had to pin ``AEROFORGE_OCCT_THREADS=1`` so OCCT would not
  oversubscribe. The quick sim does NOT use OCCT; its analogous trap is BLAS/OMP
  thread oversubscription from numpy/casadi. We therefore pin
  ``OMP/MKL/OPENBLAS/NUMEXPR/VECLIB_NUM_THREADS=1`` at module import (before numpy
  loads) and again in the worker initializer, so W workers cleanly occupy ~W
  cores instead of W*cores threads. Each worker also warms up one VLM solve at
  init (the first AeroSandbox/casadi solve compiles/caches; warming once per
  worker keeps per-task latency steady at ~3.5 s).

Requires ``aerosandbox`` in the active env (a declared core AeroForge dep).
"""

from __future__ import annotations

from chapter7_aeroforge.release_paths import ensure_code_importable

ensure_code_importable()

import argparse
import io
import json
import math
import os
import signal
import sys
import time
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Pin numeric thread pools to 1 BEFORE numpy / aerosandbox import anywhere in
# this process. ``spawn`` re-executes this module top in every worker, so this
# guarantees workers do not each fan out BLAS threads across all cores.
_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
for _v in _THREAD_VARS:
    os.environ.setdefault(_v, "1")

from chapter7_aeroforge.release_paths import AEROFORGE_ROOT

from chapter7_aeroforge.overlap_search.core import ADV_KEYS  # noqa: E402

from chapter7_aeroforge.release_paths import DATA_ROOT, LABELS_SIM_DIR

DEFAULT_DATASET_DIR = DATA_ROOT
DEFAULT_OUT_DIR = LABELS_SIM_DIR
DEFAULT_DATASET_SIZE = 100_000
_STOP_REQUESTED = False

# Cached, per-worker handles to the AeroForge quick-sim entry points.
_SIM: dict[str, Any] = {}


def _mp():
    import multiprocessing as mp

    return mp.get_context("spawn")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _request_stop(signum: int, _frame: Any) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(f"\nReceived signal {signum}; stopping after flushing completed labels...", flush=True)


def install_signal_handlers() -> None:
    for sig_name in ("SIGINT", "SIGTERM", "SIGHUP"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            signal.signal(sig, _request_stop)


# ---------------------------------------------------------------------------
# Worker side: quick-sim evaluation
# ---------------------------------------------------------------------------


def _load_sim() -> dict[str, Any]:
    """Import AeroForge quick-sim functions once per worker (after chdir)."""
    if _SIM:
        return _SIM
    for _v in _THREAD_VARS:
        os.environ.setdefault(_v, "1")
    sys.path.insert(0, str(AEROFORGE_ROOT))
    os.chdir(AEROFORGE_ROOT)
    # RAM saver: sim_report_core's only cadquery/OCP dependency is the *optional*
    # CAD-geometry hook ``sim.sim_geom_metrics`` (guarded by try/except). The
    # quick sim never uses it (VLM builds its model from params via AeroSandbox),
    # so we block that submodule to avoid loading OCP — roughly halving each
    # worker's resident memory (~1.8 GB -> well under 1 GB) at 80-way fan-out.
    sys.modules.setdefault("sim.sim_geom_metrics", None)
    from sim.sim_report_core import (  # noqa: PLC0415
        build_essential_performance_json,
        collect_aero_metrics,
        maybe_add_asb_vlm_metrics,
    )
    from sim_params import (  # noqa: PLC0415
        FLIGHT_DEFAULTS,
        QUICK_REPORT_ALPHA_TRIM_TARGET_DEG,
    )
    from uav_params import uav_params as DEFAULT_PARAMS  # noqa: PLC0415

    _SIM.update(
        collect_aero_metrics=collect_aero_metrics,
        maybe_add_asb_vlm_metrics=maybe_add_asb_vlm_metrics,
        build_essential_performance_json=build_essential_performance_json,
        DEFAULT_PARAMS=DEFAULT_PARAMS,
        speed=float(FLIGHT_DEFAULTS.speed_mps),
        alt=float(FLIGHT_DEFAULTS.altitude_m),
        alpha=float(QUICK_REPORT_ALPHA_TRIM_TARGET_DEG),
    )
    return _SIM


def _run_quick_sim(params: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Run the full quick sim (analytic aero + VLM) and return (curated_json, vlm_error)."""
    sim = _load_sim()
    metrics = sim["collect_aero_metrics"](params, {}, speed_mps=sim["speed"], altitude_m=sim["alt"])
    vlm_error: str | None = None
    try:
        metrics = sim["maybe_add_asb_vlm_metrics"](
            metrics,
            params,
            sim["speed"],
            sim["alt"],
            assembly=None,
            use_weight_trim=False,
            alpha_target_deg=sim["alpha"],
        )
    except Exception as exc:  # noqa: BLE001
        # VLM can legitimately fail on a degenerate planform; keep the analytic
        # metrics and record why VLM was unavailable for this design.
        vlm_error = f"{type(exc).__name__}: {exc}"
    quick = sim["build_essential_performance_json"](
        metrics,
        params=params,
        include_mass_block=False,
        include_energy_performance=False,
        include_takeoff_performance=False,
        include_stability=False,
    )
    return quick, vlm_error


def _headline(quick: dict[str, Any]) -> tuple[float | None, float | None, str | None]:
    """Extract (L_over_D, CD0, vlm_status) from the curated quick-report JSON."""
    perf = quick.get("uav_performance", {}) if isinstance(quick, dict) else {}
    aero_simple = perf.get("aero_simple", {}) or {}
    asb = perf.get("asb_vlm", {}) or {}
    summary = perf.get("summary", {}) or {}
    vlm_status = asb.get("status")
    ld: float | None = None
    for cand in (
        summary.get("theoretical_uav_L_over_D_max"),
        asb.get("L_over_D_from_vlm"),
        aero_simple.get("theoretical_L_over_D_max"),
    ):
        if isinstance(cand, (int, float)) and math.isfinite(float(cand)):
            ld = float(cand)
            break
    cd0_raw = aero_simple.get("CD0")
    cd0 = float(cd0_raw) if isinstance(cd0_raw, (int, float)) and math.isfinite(float(cd0_raw)) else None
    return ld, cd0, vlm_status


def worker_init() -> None:
    """Pin threads, import the quick sim, and warm up one VLM solve per worker."""
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            sim = _load_sim()
            _run_quick_sim(dict(sim["DEFAULT_PARAMS"]))
    except Exception:  # noqa: BLE001
        # A failed warmup must not kill the worker; real tasks will report errors.
        pass


def dataset_json_path(dataset_dir: Path, sample_idx: int) -> Path:
    return dataset_dir / "advs" / f"{sample_idx:06d}.json"


def make_task(sample_idx: int, dataset_dir: Path) -> dict[str, Any]:
    path = dataset_json_path(dataset_dir, sample_idx)
    if not path.exists():
        raise FileNotFoundError(f"missing dataset ADV JSON for sample_idx={sample_idx}: {path}")
    row = json.loads(path.read_text(encoding="utf-8"))
    adv = row.get("adv", row)
    family = row.get("family", "?")
    seed = row.get("seed")
    missing = [key for key in ADV_KEYS if key not in adv]
    if missing:
        raise RuntimeError(f"sample {sample_idx} missing ADV keys: {missing}")
    return {
        "sample_idx": sample_idx,
        "seed": seed,
        "family": family,
        "adv": adv,
        "source_json": str(path),
    }


def label_task(task: dict[str, Any]) -> dict[str, Any]:
    """Run the full VLM quick sim on one ADV (merged onto the default uav_params)."""
    sim = _load_sim()
    adv = task["adv"]
    params = {**sim["DEFAULT_PARAMS"], **adv}
    out: dict[str, Any] = {
        "sample_idx": task["sample_idx"],
        "seed": task["seed"],
        "family": task["family"],
        "adv": adv,
        "ok": False,
        "error": None,
    }
    t0 = time.perf_counter()
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            quick, vlm_error = _run_quick_sim(params)
        ld, cd0, vlm_status = _headline(quick)
        out["sim_s"] = time.perf_counter() - t0
        out["total_s"] = out["sim_s"]
        out["vlm_status"] = vlm_status
        out["vlm_ok"] = vlm_status == "ok"
        out["vlm_error"] = vlm_error
        out["L_over_D"] = ld
        out["CD0"] = cd0
        out["quick_report"] = quick
        out["ok"] = True
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["total_s"] = time.perf_counter() - t0
    return out


# ---------------------------------------------------------------------------
# Stats / IO (mirrors label_forever.py, adapted to quick-sim semantics)
# ---------------------------------------------------------------------------


def load_or_create_config(out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    config_path = out_dir / "run_config.json"
    if config_path.exists():
        return json.loads(config_path.read_text(encoding="utf-8"))
    config = {
        "created_at_utc": _utc_now(),
        "dataset_dir": str(args.dataset_dir),
        "dataset_size": args.dataset_size,
        "dataset_manifest": str(args.dataset_dir / "manifest.json"),
        "generator": "pre-generated sample_adv_dataset_with_family JSON files",
        "label_metric": "generate_quick_report (collect_aero_metrics + maybe_add_asb_vlm_metrics)",
        "sim_definition": "full VLM quick sim; mass/energy/takeoff/stability sections disabled",
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config


def _accumulate(stats: Counter, family_counts: Counter, row: dict[str, Any]) -> None:
    stats["total"] += 1
    family_counts[row.get("family", "?")] += 1
    if row.get("ok"):
        stats["ok"] += 1
        stats["total_s_sum"] += float(row.get("total_s", 0.0))
        if row.get("vlm_ok"):
            stats["vlm_ok"] += 1
        else:
            stats["vlm_degraded"] += 1
        ld = row.get("L_over_D")
        if isinstance(ld, (int, float)) and math.isfinite(float(ld)):
            stats["ld_sum"] += float(ld)
            stats["ld_count"] += 1
    else:
        stats["failed"] += 1


def scan_existing_labels(
    path: Path,
    rolling_window: int,
) -> tuple[int, Counter, Counter, deque[dict[str, Any]], set[int]]:
    """Count valid rows, rebuild cumulative stats, and truncate a trailing partial line."""
    stats: Counter = Counter()
    family_counts: Counter = Counter()
    rolling_rows: deque[dict[str, Any]] = deque(maxlen=rolling_window)
    labeled_sample_idxs: set[int] = set()
    if not path.exists():
        return 0, stats, family_counts, rolling_rows, labeled_sample_idxs

    count = 0
    last_good = 0
    with path.open("rb") as handle:
        while True:
            pos = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                last_good = handle.tell()
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"Truncating partial/corrupt JSONL row at byte {pos}.", flush=True)
                break
            sample_idx = row.get("sample_idx")
            if isinstance(sample_idx, int) and row.get("ok"):
                labeled_sample_idxs.add(sample_idx)
            count += 1
            _accumulate(stats, family_counts, row)
            rolling_rows.append(row)
            last_good = handle.tell()

    if last_good != path.stat().st_size:
        with path.open("ab") as handle:
            handle.truncate(last_good)
            handle.flush()
            os.fsync(handle.fileno())
    return count, stats, family_counts, rolling_rows, labeled_sample_idxs


def write_summary(
    summary_path: Path,
    *,
    args: argparse.Namespace,
    stats: Counter,
    family_counts: Counter,
    started_at: float,
    next_idx: int,
) -> None:
    wall_s = time.perf_counter() - started_at
    ok = int(stats["ok"])
    mean_ld = (stats["ld_sum"] / stats["ld_count"]) if stats["ld_count"] else None
    summary = {
        "updated_at_utc": _utc_now(),
        "dataset_dir": str(args.dataset_dir),
        "out_dir": str(args.out_dir),
        "dataset_size": args.dataset_size,
        "workers": args.workers,
        "flush_seconds": args.flush_seconds,
        "rolling_window": args.rolling_window,
        "max_in_flight": args.max_in_flight,
        "next_sample_idx": next_idx,
        "labels_flushed_cumulative": int(stats["total"]),
        "ok": ok,
        "failed": int(stats["failed"]),
        "vlm_ok": int(stats["vlm_ok"]),
        "vlm_degraded": int(stats["vlm_degraded"]),
        "vlm_ok_rate": (stats["vlm_ok"] / ok) if ok else 0.0,
        "mean_L_over_D": mean_ld,
        "avg_sim_latency_s_per_core": (stats["total_s_sum"] / ok) if ok else None,
        "wall_s_this_run": wall_s,
        "amortized_s_per_flushed_label_this_run": (wall_s / stats["new_total"]) if stats["new_total"] else None,
        "family_counts_cumulative": dict(sorted(family_counts.items())),
    }
    tmp_path = summary_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(summary_path)


def flush_rows(
    rows: list[dict[str, Any]],
    labels_path: Path,
    summary_path: Path,
    *,
    args: argparse.Namespace,
    stats: Counter,
    family_counts: Counter,
    rolling_rows: deque[dict[str, Any]],
    started_at: float,
    next_idx: int,
) -> None:
    if not rows:
        return
    with labels_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

    for row in rows:
        stats["new_total"] += 1
        if row.get("ok"):
            stats["new_ok"] += 1
        _accumulate(stats, family_counts, row)
        rolling_rows.append(row)

    write_summary(
        summary_path,
        args=args,
        stats=stats,
        family_counts=family_counts,
        started_at=started_at,
        next_idx=next_idx,
    )
    rows.clear()


def format_hhmm(seconds: float | None) -> str:
    if seconds is None or seconds <= 0:
        return "?:??"
    minutes = int(round(seconds / 60))
    hours, mins = divmod(minutes, 60)
    return f"{hours:02d}:{mins:02d}"


def _rolling_metrics(rolling_rows: deque[dict[str, Any]]) -> dict[str, float]:
    r_ok = r_vlm_ok = r_deg = r_failed = r_ld_n = 0
    r_total_s = r_ld_sum = 0.0
    for row in rolling_rows:
        if row.get("ok"):
            r_ok += 1
            r_total_s += float(row.get("total_s", 0.0))
            if row.get("vlm_ok"):
                r_vlm_ok += 1
            else:
                r_deg += 1
            ld = row.get("L_over_D")
            if isinstance(ld, (int, float)) and math.isfinite(float(ld)):
                r_ld_sum += float(ld)
                r_ld_n += 1
        else:
            r_failed += 1
    return {
        "rows": len(rolling_rows),
        "ok": r_ok,
        "vlm_ok": r_vlm_ok,
        "degraded": r_deg,
        "failed": r_failed,
        "latency": (r_total_s / r_ok) if r_ok else 0.0,
        "mean_ld": (r_ld_sum / r_ld_n) if r_ld_n else float("nan"),
    }


def print_progress_report(
    *,
    stats: Counter,
    rolling_rows: deque[dict[str, Any]],
    started_at: float,
    workers: int,
    next_idx: int,
    dataset_size: int,
    final: bool = False,
) -> None:
    now = time.perf_counter()
    ok = int(stats["ok"])
    failed = int(stats["failed"])
    vlm_ok = int(stats["vlm_ok"])
    vlm_deg = int(stats["vlm_degraded"])
    delta_ok = ok - int(stats["last_report_ok"])
    delta_failed = failed - int(stats["last_report_failed"])
    if delta_ok <= 0 and delta_failed <= 0 and not final:
        return
    delta_vlm_ok = vlm_ok - int(stats["last_report_vlm_ok"])
    delta_vlm_deg = vlm_deg - int(stats["last_report_vlm_degraded"])
    delta_total_s = float(stats["total_s_sum"]) - float(stats["last_report_total_s_sum"])
    delta_ld_sum = float(stats["ld_sum"]) - float(stats["last_report_ld_sum"])
    delta_ld_n = int(stats["ld_count"]) - int(stats["last_report_ld_count"])
    wall_s = now - started_at
    delta_wall_s = wall_s - float(stats["last_report_wall_s"])

    avg_latency = (stats["total_s_sum"] / ok) if ok else 0.0
    delta_latency = (delta_total_s / delta_ok) if delta_ok else 0.0
    session_throughput = (wall_s / stats["new_total"]) if stats["new_total"] else 0.0
    delta_throughput = (delta_wall_s / (delta_ok + delta_failed)) if (delta_ok + delta_failed) else 0.0

    roll = _rolling_metrics(rolling_rows)
    rolling_throughput = (roll["latency"] / workers) if (workers and roll["ok"]) else 0.0
    eta_throughput = rolling_throughput or session_throughput or delta_throughput
    remaining_ok = max(0, dataset_size - ok)
    eta_hhmm = format_hhmm(remaining_ok * eta_throughput if eta_throughput else None)

    mean_ld = (stats["ld_sum"] / stats["ld_count"]) if stats["ld_count"] else float("nan")
    delta_mean_ld = (delta_ld_sum / delta_ld_n) if delta_ld_n else float("nan")
    heading = "final progress" if final else "progress"
    print(
        "\n"
        f"==== {heading}: {delta_ok} sims since last report, {ok}/{dataset_size} total ok "
        f"({100 * ok / dataset_size:5.2f}%, ~{eta_hhmm} till {dataset_size // 1000}k) ====\n"
        f"  saved rows:        {int(stats['total'])} cumulative ({int(stats['new_total'])} this session)\n"
        f"  this report:       vlm_ok {delta_vlm_ok} ({100 * delta_vlm_ok / delta_ok if delta_ok else 0.0:5.1f}%), "
        f"vlm_degraded {delta_vlm_deg}, failed {delta_failed}, mean L/D {delta_mean_ld:6.2f}\n"
        f"  rolling last {roll['rows']:4d}: vlm_ok {roll['vlm_ok']} "
        f"({100 * roll['vlm_ok'] / roll['ok'] if roll['ok'] else 0.0:5.1f}%), "
        f"degraded {roll['degraded']}, failed {roll['failed']}, mean L/D {roll['mean_ld']:6.2f}\n"
        f"  cumulative:        vlm_ok {vlm_ok} ({100 * vlm_ok / ok if ok else 0.0:5.1f}%), "
        f"degraded {vlm_deg}, failed {failed}, mean L/D {mean_ld:6.2f}\n"
        f"  avg sim/core:      {delta_latency:5.2f}s this report | {avg_latency:5.2f}s cumulative\n"
        f"  avg throughput @{workers} cores: {delta_throughput:5.2f}s/label this report | "
        f"{rolling_throughput:5.2f}s/label rolling-est | {session_throughput:5.2f}s/label this session\n"
        f"  next sample_idx:   {next_idx}\n",
        flush=True,
    )
    stats["last_report_ok"] = ok
    stats["last_report_failed"] = failed
    stats["last_report_vlm_ok"] = vlm_ok
    stats["last_report_vlm_degraded"] = vlm_deg
    stats["last_report_total_s_sum"] = float(stats["total_s_sum"])
    stats["last_report_ld_sum"] = float(stats["ld_sum"])
    stats["last_report_ld_count"] = int(stats["ld_count"])
    stats["last_report_wall_s"] = wall_s


def parse_args() -> argparse.Namespace:
    normalized_argv: list[str] = []
    for arg in sys.argv[1:]:
        if arg.startswith("--") and arg[2:].isdigit():
            normalized_argv.extend(["--workers", arg[2:]])
        else:
            normalized_argv.append(arg)

    ap = argparse.ArgumentParser(description="Label pre-generated ADV dataset with the full VLM quick sim until interrupted.")
    ap.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--cores", type=int, dest="workers", help="Alias for --workers.")
    ap.add_argument("--dataset-size", type=int, default=DEFAULT_DATASET_SIZE)
    ap.add_argument("--flush-seconds", type=float, default=120.0)
    ap.add_argument("--rolling-window", type=int, default=1000)
    ap.add_argument("--max-in-flight", type=int, default=None)
    ap.add_argument("--max-labels", type=int, default=None, help="Optional finite additional labels for tests.")
    ap.add_argument("--no-warmup", action="store_true", help="Skip the parent-side cache-warmup quick sim.")
    return ap.parse_args(normalized_argv)


def main() -> None:
    install_signal_handlers()
    args = parse_args()
    args.dataset_dir = args.dataset_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    if args.dataset_size <= 0:
        raise SystemExit("--dataset-size must be positive")
    if args.workers <= 0:
        raise SystemExit("Worker/core count must be positive")
    if args.rolling_window <= 0:
        raise SystemExit("--rolling-window must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.max_in_flight = args.max_in_flight or max(args.workers * 2, args.workers)

    labels_path = args.out_dir / "labels.jsonl"
    summary_path = args.out_dir / "summary.json"
    load_or_create_config(args.out_dir, args)
    (
        existing_rows,
        stats,
        family_counts,
        rolling_rows,
        labeled_sample_idxs,
    ) = scan_existing_labels(labels_path, args.rolling_window)
    if len(labeled_sample_idxs) > args.dataset_size:
        raise SystemExit(
            f"Unique labeled sample indices ({len(labeled_sample_idxs)}) exceed --dataset-size ({args.dataset_size})."
        )
    for key in (
        "last_report_ok",
        "last_report_failed",
        "last_report_vlm_ok",
        "last_report_vlm_degraded",
        "last_report_total_s_sum",
        "last_report_ld_sum",
        "last_report_ld_count",
    ):
        base = key.replace("last_report_", "")
        mapping = {
            "ok": "ok",
            "failed": "failed",
            "vlm_ok": "vlm_ok",
            "vlm_degraded": "vlm_degraded",
            "total_s_sum": "total_s_sum",
            "ld_sum": "ld_sum",
            "ld_count": "ld_count",
        }
        stats[key] = stats[mapping[base]]
    stats["last_report_wall_s"] = 0.0

    next_idx = 0
    submitted_new = 0
    end_idx = args.dataset_size

    print(
        f"Quick-sim label-forever run: workers={args.workers}, "
        f"threads_pinned={os.environ.get('OMP_NUM_THREADS')}, "
        f"flush_seconds={args.flush_seconds:g} (time-only), report=each flush, "
        f"max_in_flight={args.max_in_flight}",
        flush=True,
    )
    print(f"Dataset: {args.dataset_dir / 'advs'} ({args.dataset_size} JSON files expected)", flush=True)
    print(f"Output: {labels_path}", flush=True)
    while next_idx < end_idx and next_idx in labeled_sample_idxs:
        next_idx += 1

    print(
        f"Existing rows={existing_rows}, unique sample_idx={len(labeled_sample_idxs)}; "
        f"next unlabeled sample_idx={next_idx}; stopping at sample_idx={end_idx}",
        flush=True,
    )
    if len(labeled_sample_idxs) >= args.dataset_size:
        print("Dataset already fully quick-sim labeled. Nothing to do.", flush=True)
        write_summary(
            summary_path,
            args=args,
            stats=stats,
            family_counts=family_counts,
            started_at=time.perf_counter(),
            next_idx=next_idx,
        )
        return

    # Build the one-time AeroSandbox/casadi on-disk cache once in the parent so
    # that 80 fresh workers do not all hit a cold cache simultaneously (which
    # would race / serialize the first batch). Workers still warm up in-process.
    if not args.no_warmup:
        t_warm = time.perf_counter()
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                sim = _load_sim()
                _run_quick_sim(dict(sim["DEFAULT_PARAMS"]))
            print(f"Parent warmup quick sim: {time.perf_counter() - t_warm:.1f}s (disk caches built).", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"Parent warmup failed (continuing): {type(exc).__name__}: {exc}", flush=True)

    started_at = time.perf_counter()
    last_flush = started_at
    buffer: list[dict[str, Any]] = []
    future_to_task: dict[Any, dict[str, Any]] = {}
    executor = ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=worker_init,
        mp_context=_mp(),
    )

    try:
        while not _STOP_REQUESTED:
            while (
                len(future_to_task) < args.max_in_flight
                and not _STOP_REQUESTED
                and next_idx < end_idx
                and (args.max_labels is None or submitted_new < args.max_labels)
            ):
                while next_idx < end_idx and next_idx in labeled_sample_idxs:
                    next_idx += 1
                if next_idx >= end_idx:
                    break
                task = make_task(next_idx, args.dataset_dir)
                future = executor.submit(label_task, task)
                future_to_task[future] = task
                next_idx += 1
                submitted_new += 1

            if not future_to_task:
                break

            done, _pending = wait(future_to_task, timeout=5.0, return_when=FIRST_COMPLETED)
            for future in done:
                task = future_to_task.pop(future)
                try:
                    row = future.result()
                except Exception as exc:  # noqa: BLE001
                    row = {
                        "sample_idx": task["sample_idx"],
                        "seed": task["seed"],
                        "family": task["family"],
                        "adv": task["adv"],
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                buffer.append(row)
                sample_idx = row.get("sample_idx")
                if isinstance(sample_idx, int) and row.get("ok"):
                    labeled_sample_idxs.add(sample_idx)

            due_to_time = buffer and (time.perf_counter() - last_flush) >= args.flush_seconds
            if due_to_time:
                flush_rows(
                    buffer,
                    labels_path,
                    summary_path,
                    args=args,
                    stats=stats,
                    family_counts=family_counts,
                    rolling_rows=rolling_rows,
                    started_at=started_at,
                    next_idx=next_idx,
                )
                last_flush = time.perf_counter()
                print_progress_report(
                    stats=stats,
                    rolling_rows=rolling_rows,
                    started_at=started_at,
                    workers=args.workers,
                    next_idx=next_idx,
                    dataset_size=args.dataset_size,
                )

            if (next_idx >= end_idx or (args.max_labels is not None and submitted_new >= args.max_labels)) and not future_to_task:
                break
    finally:
        if future_to_task:
            print(f"Cancelling {len(future_to_task)} in-flight/pending tasks...", flush=True)
            for future in future_to_task:
                future.cancel()
        flush_rows(
            buffer,
            labels_path,
            summary_path,
            args=args,
            stats=stats,
            family_counts=family_counts,
            rolling_rows=rolling_rows,
            started_at=started_at,
            next_idx=next_idx,
        )
        executor.shutdown(wait=False, cancel_futures=True)
        if stats["ok"] > stats["last_report_ok"] or stats["failed"] > stats["last_report_failed"]:
            print_progress_report(
                stats=stats,
                rolling_rows=rolling_rows,
                started_at=started_at,
                workers=args.workers,
                next_idx=next_idx,
                dataset_size=args.dataset_size,
                final=True,
            )
        print("Stopped. Flushed completed labels and wrote summary.json.", flush=True)


if __name__ == "__main__":
    main()

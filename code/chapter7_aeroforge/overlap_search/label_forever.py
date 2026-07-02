#!/usr/bin/env python3
"""Continuously label the pre-generated 100k ADV dataset until interrupted.

This is the overnight data-labeling path:

  AEROFORGE_OCCT_THREADS=1 conda run -n bachelor-thesis python -u \
    chapter7_aeroforge/overlap_search/label_forever.py --18

Results are appended to labels.jsonl in batches. If the process is interrupted,
the current in-memory batch is flushed before shutdown when possible; a hard kill
can lose at most the unflushed batch. Re-run the same command later and labeling
continues at the next unlabeled dataset index.
"""

from __future__ import annotations

from chapter7_aeroforge.release_paths import ensure_code_importable

ensure_code_importable()

import argparse
import io
import json
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


from chapter7_aeroforge.overlap_search.core import ADV_KEYS  # noqa: E402
from chapter7_aeroforge.overlap_search.worker import worker_init  # noqa: E402

from chapter7_aeroforge.release_paths import DATA_ROOT, LABELS_CLOUD_DIR

DEFAULT_DATASET_DIR = DATA_ROOT
DEFAULT_OUT_DIR = LABELS_CLOUD_DIR
DEFAULT_SEED_BASE = 20260620
DEFAULT_DATASET_SIZE = 100_000
_STOP_REQUESTED = False


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
    """Build one ADV and compute the fast cut-each extrinsic overlap label."""
    from chapter7_aeroforge.overlap_search.core import (  # noqa: PLC0415
        adv_subset,
        build_airframe,
        cadquery_cut_each_overlap_fast,
    )

    out = {
        "sample_idx": task["sample_idx"],
        "seed": task["seed"],
        "family": task["family"],
        "adv": task["adv"],
        "ok": False,
        "error": None,
    }
    t0 = time.perf_counter()
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            uav, timings = build_airframe(task["adv"])
            t_build = time.perf_counter()
            metrics = cadquery_cut_each_overlap_fast(
                uav.components["wings"],
                uav.components["tail"],
                uav.components["fuselage_outer"],
            )
            t_label = time.perf_counter()

        out["build_s"] = t_build - t0
        out["label_s"] = t_label - t_build
        out["total_s"] = t_label - t0
        out["timings"] = {**timings, "label_s": out["label_s"], "total_s": out["total_s"]}
        out["overlap_mm3_cut_each"] = metrics["overlap_mm3_cut_each"]
        out["overlap_mm3_raw"] = metrics["overlap_mm3_raw"]
        out["raw_exit"] = metrics.get("raw_exit", False)
        out["is_overlap"] = metrics["overlap_mm3_cut_each"] > 1.0
        out["adv"] = adv_subset(task["adv"])
        out["ok"] = True
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["total_s"] = time.perf_counter() - t0
    return out


def load_or_create_config(out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    config_path = out_dir / "run_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        return config

    config = {
        "created_at_utc": _utc_now(),
        "dataset_dir": str(args.dataset_dir),
        "dataset_size": args.dataset_size,
        "dataset_manifest": str(args.dataset_dir / "manifest.json"),
        "generator": "pre-generated sample_adv_dataset_with_family JSON files",
        "label_metric": "cadquery_cut_each_overlap_fast",
        "overlap_definition": "(wings - fuselage) intersect (tail - fuselage) > 1 mm^3",
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config


def update_rolling_stats(
    rolling_rows: deque[dict[str, Any]],
    row: dict[str, Any],
    stats: Counter,
) -> None:
    if rolling_rows.maxlen and len(rolling_rows) == rolling_rows.maxlen:
        old = rolling_rows[0]
        stats["rolling_total"] -= 1
        if old.get("ok"):
            stats["rolling_ok"] -= 1
            if old.get("is_overlap"):
                stats["rolling_overlap"] -= 1
            else:
                stats["rolling_no_overlap"] -= 1
            stats["rolling_total_s_sum"] -= float(old.get("total_s", 0.0))
        else:
            stats["rolling_failed"] -= 1

    rolling_rows.append(row)
    stats["rolling_total"] += 1
    if row.get("ok"):
        stats["rolling_ok"] += 1
        if row.get("is_overlap"):
            stats["rolling_overlap"] += 1
        else:
            stats["rolling_no_overlap"] += 1
        stats["rolling_total_s_sum"] += float(row.get("total_s", 0.0))
    else:
        stats["rolling_failed"] += 1


def scan_existing_labels(
    path: Path,
    rolling_window: int,
) -> tuple[int, Counter, Counter, deque[dict[str, Any]], set[int]]:
    """Count valid rows, collect cumulative stats, and truncate a trailing partial line."""
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
            stats["total"] += 1
            family_counts[row.get("family", "?")] += 1
            if row.get("ok"):
                stats["ok"] += 1
                if row.get("is_overlap"):
                    stats["overlap"] += 1
                else:
                    stats["no_overlap"] += 1
                if row.get("raw_exit"):
                    stats["raw_early_exit"] += 1
                stats["total_s_sum"] += float(row.get("total_s", 0.0))
            else:
                stats["failed"] += 1
            update_rolling_stats(rolling_rows, row, stats)
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
    config: dict[str, Any],
    stats: Counter,
    family_counts: Counter,
    started_at: float,
    next_idx: int,
) -> None:
    wall_s = time.perf_counter() - started_at
    total = int(stats["total"])
    ok = int(stats["ok"])
    summary = {
        "updated_at_utc": _utc_now(),
        "dataset_dir": str(args.dataset_dir),
        "out_dir": str(args.out_dir),
        "dataset_size": args.dataset_size,
        "workers": args.workers,
        "flush_every": args.flush_every,
        "report_every_successful": args.report_every_successful,
        "rolling_window": args.rolling_window,
        "max_in_flight": args.max_in_flight,
        "next_sample_idx": next_idx,
        "labels_flushed_cumulative": total,
        "ok": ok,
        "failed": int(stats["failed"]),
        "overlap": int(stats["overlap"]),
        "no_overlap": int(stats["no_overlap"]),
        "raw_early_exit": int(stats["raw_early_exit"]),
        "overlap_rate_ok": (stats["overlap"] / ok) if ok else 0.0,
        "no_overlap_rate_ok": (stats["no_overlap"] / ok) if ok else 0.0,
        "avg_label_latency_s_per_core": (stats["total_s_sum"] / ok) if ok else None,
        "rolling": {
            "window": args.rolling_window,
            "rows": int(stats["rolling_total"]),
            "ok": int(stats["rolling_ok"]),
            "failed": int(stats["rolling_failed"]),
            "overlap": int(stats["rolling_overlap"]),
            "no_overlap": int(stats["rolling_no_overlap"]),
            "overlap_rate_ok": (stats["rolling_overlap"] / stats["rolling_ok"])
            if stats["rolling_ok"]
            else 0.0,
            "avg_label_latency_s_per_core": (stats["rolling_total_s_sum"] / stats["rolling_ok"])
            if stats["rolling_ok"]
            else None,
            "estimated_throughput_s_per_label": (
                (stats["rolling_total_s_sum"] / stats["rolling_ok"]) / args.workers
            )
            if stats["rolling_ok"] and args.workers
            else None,
        },
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
    config: dict[str, Any],
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
        stats["total"] += 1
        stats["new_total"] += 1
        family_counts[row.get("family", "?")] += 1
        if row.get("ok"):
            stats["ok"] += 1
            stats["new_ok"] += 1
            if row.get("is_overlap"):
                stats["overlap"] += 1
                stats["new_overlap"] += 1
            else:
                stats["no_overlap"] += 1
                stats["new_no_overlap"] += 1
            if row.get("raw_exit"):
                stats["raw_early_exit"] += 1
            stats["total_s_sum"] += float(row.get("total_s", 0.0))
        else:
            stats["failed"] += 1
            stats["new_failed"] += 1
        update_rolling_stats(rolling_rows, row, stats)

    write_summary(
        summary_path,
        args=args,
        config=config,
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


def print_progress_report(
    *,
    stats: Counter,
    started_at: float,
    workers: int,
    next_idx: int,
    dataset_size: int,
    final: bool = False,
) -> None:
    now = time.perf_counter()
    ok = int(stats["ok"])
    overlap = int(stats["overlap"])
    no_overlap = int(stats["no_overlap"])
    delta_ok = ok - int(stats["last_report_ok"])
    delta_overlap = overlap - int(stats["last_report_overlap"])
    delta_no_overlap = no_overlap - int(stats["last_report_no_overlap"])
    delta_failed = int(stats["failed"]) - int(stats["last_report_failed"])
    delta_total_s = float(stats["total_s_sum"]) - float(stats["last_report_total_s_sum"])
    wall_s = now - started_at
    delta_wall_s = wall_s - float(stats["last_report_wall_s"])
    if delta_ok <= 0 and delta_failed <= 0 and not final:
        return
    avg_latency = (stats["total_s_sum"] / ok) if ok else 0.0
    throughput = (wall_s / stats["new_total"]) if stats["new_total"] else 0.0
    delta_latency = (delta_total_s / delta_ok) if delta_ok else 0.0
    delta_throughput = (delta_wall_s / (delta_ok + delta_failed)) if (delta_ok + delta_failed) else 0.0
    rolling_ok = int(stats["rolling_ok"])
    rolling_overlap = int(stats["rolling_overlap"])
    rolling_no_overlap = int(stats["rolling_no_overlap"])
    rolling_failed = int(stats["rolling_failed"])
    rolling_latency = (stats["rolling_total_s_sum"] / rolling_ok) if rolling_ok else 0.0
    rolling_throughput = (rolling_latency / workers) if workers and rolling_ok else 0.0
    eta_throughput = rolling_throughput or throughput or delta_throughput
    remaining_ok = max(0, dataset_size - ok)
    eta_hhmm = format_hhmm(remaining_ok * eta_throughput if eta_throughput else None)
    heading = "final progress" if final else "progress"
    print(
        "\n"
        f"==== {heading}: {delta_ok} successful labels since last report, {ok}/{dataset_size} total ok "
        f"({100 * ok / dataset_size:5.2f}%, ~{eta_hhmm} till 100k) ====\n"
        f"  saved rows:        {int(stats['total'])} cumulative ({int(stats['new_total'])} this session)\n"
        f"  this report:       overlap {delta_overlap} ({100 * delta_overlap / delta_ok if delta_ok else 0.0:5.1f}%), "
        f"without overlap {delta_no_overlap} ({100 * delta_no_overlap / delta_ok if delta_ok else 0.0:5.1f}%), "
        f"failed {delta_failed}\n"
        f"  rolling last {int(stats['rolling_total']):4d}: overlap {rolling_overlap} "
        f"({100 * rolling_overlap / rolling_ok if rolling_ok else 0.0:5.1f}%), "
        f"without overlap {rolling_no_overlap} ({100 * rolling_no_overlap / rolling_ok if rolling_ok else 0.0:5.1f}%), "
        f"failed {rolling_failed}\n"
        f"  cumulative:        overlap {overlap} ({100 * overlap / ok if ok else 0.0:5.1f}%), "
        f"without overlap {no_overlap} ({100 * no_overlap / ok if ok else 0.0:5.1f}%), "
        f"failed {int(stats['failed'])}\n"
        f"  avg label/core:    {delta_latency:5.2f}s this report | {avg_latency:5.2f}s cumulative\n"
        f"  avg throughput @{workers} cores: {delta_throughput:5.2f}s/label this report | "
        f"{rolling_throughput:5.2f}s/label rolling-est | {throughput:5.2f}s/label this session\n"
        f"  next sample_idx:   {next_idx}\n",
        flush=True,
    )
    stats["last_report_ok"] = ok
    stats["last_report_overlap"] = overlap
    stats["last_report_no_overlap"] = no_overlap
    stats["last_report_failed"] = int(stats["failed"])
    stats["last_report_total_s_sum"] = float(stats["total_s_sum"])
    stats["last_report_wall_s"] = wall_s


def parse_args() -> argparse.Namespace:
    normalized_argv: list[str] = []
    for arg in sys.argv[1:]:
        if arg.startswith("--") and arg[2:].isdigit():
            normalized_argv.extend(["--workers", arg[2:]])
        else:
            normalized_argv.append(arg)

    ap = argparse.ArgumentParser(description="Label pre-generated ADV dataset samples until interrupted.")
    ap.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--cores", type=int, dest="workers", help="Alias for --workers.")
    ap.add_argument("--dataset-size", type=int, default=DEFAULT_DATASET_SIZE)
    ap.add_argument("--flush-every", type=int, default=100)
    ap.add_argument("--flush-seconds", type=float, default=60.0)
    ap.add_argument("--report-every-successful", type=int, default=100)
    ap.add_argument("--rolling-window", type=int, default=1000)
    ap.add_argument("--max-in-flight", type=int, default=None)
    ap.add_argument("--max-labels", type=int, default=None, help="Optional finite additional labels for tests.")
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

    os.environ.setdefault("AEROFORGE_OCCT_THREADS", "1")
    if os.environ.get("AEROFORGE_OCCT_THREADS") != "1":
        print(
            "WARNING: AEROFORGE_OCCT_THREADS is not 1; throughput may collapse from OCCT oversubscription.",
            flush=True,
        )

    labels_path = args.out_dir / "labels.jsonl"
    summary_path = args.out_dir / "summary.json"
    config = load_or_create_config(args.out_dir, args)
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
    stats["last_report_ok"] = stats["ok"]
    stats["last_report_overlap"] = stats["overlap"]
    stats["last_report_no_overlap"] = stats["no_overlap"]
    stats["last_report_failed"] = stats["failed"]
    stats["last_report_total_s_sum"] = stats["total_s_sum"]
    stats["last_report_wall_s"] = 0.0
    next_idx = 0
    submitted_new = 0
    end_idx = args.dataset_size
    next_report_ok = stats["ok"] + args.report_every_successful

    print(
        f"Label-forever run: workers={args.workers}, pinned_occt={os.environ.get('AEROFORGE_OCCT_THREADS')}, "
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
        print("Dataset already fully labeled. Nothing to do.", flush=True)
        write_summary(
            summary_path,
            args=args,
            config=config,
            stats=stats,
            family_counts=family_counts,
            rolling_rows=rolling_rows,
            started_at=time.perf_counter(),
            next_idx=next_idx,
        )
        return

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

            done, _pending = wait(
                future_to_task,
                timeout=5.0,
                return_when=FIRST_COMPLETED,
            )
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
                    config=config,
                    stats=stats,
                    family_counts=family_counts,
                    rolling_rows=rolling_rows,
                    started_at=started_at,
                    next_idx=next_idx,
                )
                last_flush = time.perf_counter()
                print_progress_report(
                    stats=stats,
                    started_at=started_at,
                    workers=args.workers,
                    next_idx=next_idx,
                    dataset_size=args.dataset_size,
                )
                while stats["ok"] >= next_report_ok:
                    next_report_ok += args.report_every_successful

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
            config=config,
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
                started_at=started_at,
                workers=args.workers,
                next_idx=next_idx,
                dataset_size=args.dataset_size,
                final=True,
            )
        print("Stopped. Flushed completed labels and wrote summary.json.", flush=True)


if __name__ == "__main__":
    main()

"""Aggregate Chapter 5 optimizer run statistics into comparison tables."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from shared.paths import COMPARISONS_DIR, RUNS_DIR, TASK26_ROOT


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def collect_rows(
    run_ids: list[str] | None = None,
    *,
    include_constrained: bool = False,
    runs_dir: Path = RUNS_DIR,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not runs_dir.exists():
        return rows
    run_dirs = sorted(p for p in runs_dir.iterdir() if p.is_dir())
    if run_ids:
        keep = set(run_ids)
        run_dirs = [p for p in run_dirs if p.name in keep]

    for run_dir in run_dirs:
        manifest_path = run_dir / "manifest.json"
        stats_path = run_dir / "statistics.json"
        if not manifest_path.exists() or not stats_path.exists():
            continue
        manifest = load_json(manifest_path)
        if manifest.get("constraint_id") and not include_constrained and not run_ids:
            continue
        stats = load_json(stats_path)
        dist = stats.get("distances", {})
        sparse = stats.get("sparse_metrics", {})
        rows.append({
            "run_id": run_dir.name,
            "optimizer_id": manifest.get("optimizer_id", ""),
            "model_id": manifest.get("model_id", ""),
            "tau": manifest.get("tau", ""),
            "constraint_id": manifest.get("constraint_id") or "",
            "baseline_run_id": manifest.get("baseline_run_id") or "",
            "surrogate_success": stats.get("surrogate_success_count", 0),
            "oracle_confirmed": stats.get("oracle_confirmed_count", 0),
            "false_success": stats.get("false_success_count", 0),
            "no_crossing": stats.get("no_crossing_count", 0),
            "validity_failures": stats.get("validity_failure_count", 0),
            "mean_distance": dist.get("all_mean", 0.0),
            "median_distance": dist.get("all_median", 0.0),
            "mean_active_coordinates": sparse.get("active_coordinate_count_mean", ""),
            "mean_l1_distance": sparse.get("l1_distance_mean", ""),
            "oracle_mean_distance": dist.get("oracle_confirmed_mean", 0.0),
            "wall_time_s": manifest.get("wall_time_s", 0.0),
        })
    rows.sort(key=lambda r: (-int(r["oracle_confirmed"]), int(r["false_success"]), float(r["mean_distance"])))
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_md(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Chapter 5 Optimizer Comparison",
        "",
        "Sorted by oracle-confirmed successes, then false successes, then mean normalized distance.",
        "",
        "| Run | Optimizer | Model | Tau | Constraint | Oracle OK | Surr. OK | False OK | No Crossing | Invalid | Mean Dist | Median Dist | Mean Active | Mean L1 |",
        "|-----|-----------|-------|-----|------------|-----------|----------|----------|-------------|---------|-----------|-------------|-------------|---------|",
    ]
    for r in rows:
        constraint = r.get("constraint_id") or "none"
        lines.append(
            f"| {r['run_id']} | {r['optimizer_id']} | {r['model_id']} | {float(r['tau']):.2f} | {constraint} | "
            f"{r['oracle_confirmed']} | {r['surrogate_success']} | {r['false_success']} | "
            f"{r['no_crossing']} | {r['validity_failures']} | "
            f"{float(r['mean_distance']):.4f} | {float(r['median_distance']):.4f} | "
            f"{_fmt_optional(r.get('mean_active_coordinates'))} | {_fmt_optional(r.get('mean_l1_distance'))} |"
        )
    lines.append("")
    path.write_text("\n".join(lines))


def _fmt_optional(value: Any) -> str:
    if value in ("", None):
        return "n/a"
    return f"{float(value):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-ids", default=None, help="Comma-separated run IDs. Defaults to all runs.")
    parser.add_argument("--out-dir", type=Path, default=COMPARISONS_DIR)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR, help="Runs root to scan")
    parser.add_argument("--include-constrained", action="store_true", help="Include parameter-lock constrained runs")
    args = parser.parse_args()

    run_ids = [r.strip() for r in args.run_ids.split(",")] if args.run_ids else None
    rows = collect_rows(run_ids, include_constrained=args.include_constrained, runs_dir=args.runs_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out_dir / "optimizer_comparison.csv")
    write_md(rows, args.out_dir / "optimizer_comparison.md")
    print(f"Wrote {len(rows)} rows to {args.out_dir}")


if __name__ == "__main__":
    main()

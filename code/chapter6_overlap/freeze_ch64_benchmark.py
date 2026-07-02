"""Freeze the canonical Chapter 6.4 overlap-only repair benchmark (Chapter 6 #5f).

The Chapter 6.4 redo uses *exactly* the first 50 rows of the existing strict
overlap start set, the same ordering Chapter 6.5 already consumes. This removes
the historical 72 / 60 / 50 confusion: every Chapter 6.4 metric is reported on a
single frozen benchmark of 50 designs.

Output (under ``data/benchmark_starts/strict_overlap_repair_ch64_v1/``):

- ``starts.csv``   -- the 50 frozen rows (every source column preserved); this is
  what ``--starts-csv`` consumes in :mod:`strict_repair_ch64`.
- ``manifest.csv`` -- byte-identical alias of ``starts.csv`` for the name used in
  the Chapter 6 spec.
- ``summary.json`` -- category / magnitude-bin counts and the strict threshold.
- ``README.md``    -- human-readable composition table.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from ._public_helpers import _read_csv
from .label_cache import DEFAULT_THRESHOLD_NORM, STRICT_THRESHOLD_NORM
from .paths import DATA_DIR
from .release_paths import (
    ACQUIRED_RANDOM_15K,
    CH64_STARTS_CSV,
    CH65_STARTS_CSV,
    HOLDOUT_5K_CSV,
    HOLDOUT_LABELED_5K,
    POOL_100K_CSV,
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> Path:
    out_dir = DATA_DIR / args.name
    rows = _read_csv(args.source_csv)
    frozen = rows[: args.n]
    if len(frozen) < args.n:
        raise SystemExit(f"source has only {len(frozen)} rows, need {args.n}")

    _write_csv(out_dir / "starts.csv", frozen)
    _write_csv(out_dir / "manifest.csv", frozen)

    start_norms = [float(r["total_overlap_norm"]) for r in frozen]
    category_counts = dict(Counter(r.get("strict_category", "") for r in frozen))
    magnitude_counts = dict(Counter(r.get("magnitude_bin", "") for r in frozen))
    dominant_counts = dict(Counter(r.get("dominant_pair", "") for r in frozen))
    # "Near-zero band" starts: strictly overlapping (> 1e-6) yet already below the
    # tau0 = 5e-5 acquisition/clean scale, where the regressor gradient vanishes.
    near_zero = sum(1 for v in start_norms if STRICT_THRESHOLD_NORM < v < DEFAULT_THRESHOLD_NORM)

    summary = {
        "name": args.name,
        "source_csv": str(args.source_csv),
        "n_starts": len(frozen),
        "strict_threshold_norm": STRICT_THRESHOLD_NORM,
        "acquisition_threshold_norm": DEFAULT_THRESHOLD_NORM,
        "category_counts": category_counts,
        "magnitude_bin_counts": magnitude_counts,
        "dominant_pair_counts": dominant_counts,
        "near_zero_band_starts": near_zero,
        "start_overlap_norm_min": min(start_norms),
        "start_overlap_norm_max": max(start_norms),
        "purpose": "Chapter 6.4 overlap-only strict repair benchmark (Chapter 6 #5f).",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Strict Overlap Repair Benchmark (Chapter 6.4, ch64_v1)",
        "",
        "Canonical 50-start overlap-only repair benchmark for the Chapter 6.4 redo",
        "(Chapter 6 #5f). These are the first 50 rows of the existing strict start set,",
        "the same ordering Chapter 6.5 consumes, so the two chapters share a benchmark",
        "prefix. Success is strict analytic non-overlap (`total_overlap_norm <= 1e-6`);",
        "the goal is overlap *removal* only, not kinematic assemblability.",
        "",
        f"- Source: `{args.source_csv}`",
        f"- Starts: `{len(frozen)}`",
        f"- Strict success threshold (normalized): `{STRICT_THRESHOLD_NORM}`",
        f"- Acquisition / clean threshold tau0 (normalized): `{DEFAULT_THRESHOLD_NORM}`",
        f"- Start overlap range (normalized): `{min(start_norms):.3g}` .. `{max(start_norms):.3g}`",
        f"- Near-zero-band starts (1e-6 < O < 5e-5, vanishing-gradient regime): `{near_zero}`",
        "",
        "## Category counts",
        "",
        "| Category | Count |",
        "|---|---:|",
    ]
    for key, count in sorted(category_counts.items()):
        lines.append(f"| {key} | {count} |")
    lines += ["", "## Magnitude-bin counts", "", "| Magnitude bin | Count |", "|---|---:|"]
    for key, count in sorted(magnitude_counts.items()):
        lines.append(f"| {key} | {count} |")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_dir)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="strict_overlap_repair_ch64_v1")
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=CH64_STARTS_CSV,
    )
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

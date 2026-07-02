"""Select strict-overlap starts that are kinematically blocked at the start."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from ._public_helpers import _read_csv
from .label_cache import DEFAULT_PITCH_MM, STRICT_THRESHOLD_NORM, LabelCache, params_from_row
from .paths import DATA_DIR
from .release_paths import (
    ACQUIRED_RANDOM_15K,
    CH64_STARTS_CSV,
    CH65_STARTS_CSV,
    HOLDOUT_5K_CSV,
    HOLDOUT_LABELED_5K,
    POOL_100K_CSV,
)
from .regression_metrics import PAIR_COLUMNS, continuous_label_from_payload, magnitude_bin
from .sampler import compute_margins, validate_relaxed_params
from ._public_helpers import _head_category
from ._public_helpers import _assemblability_margin, evaluate_exact_assemblability


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row})
    preferred = [
        "strict_benchmark_index",
        "sample_id",
        "source_csv",
        "source_row_index",
        "stream",
        "intended_relaxed_rules",
        "relaxed_ok",
        "start_kinematic_assemblable",
        "start_assemblability_reason",
        "start_assemblability_margin",
        "total_overlap_norm",
        "total_overlap_volume",
        "total_part_volume_analytic",
        "dominant_pair",
        "strict_category",
        "magnitude_bin",
        "strict_score",
    ]
    ordered = [k for k in preferred if k in fieldnames] + [k for k in fieldnames if k not in preferred]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def _score(row: dict) -> float:
    overlap = float(row["total_overlap_norm"])
    category_bonus = {
        "splint_head_wall_candidate": 50.0,
        "pin_head_or_shaft_wall_candidate": 45.0,
        "pin_splint_cross_overlap": 40.0,
        "bracket_splint_other": 30.0,
        "bracket_pin_other": 25.0,
        "other_overlap": 10.0,
    }
    blocked_bonus = 1000.0 if str(row["start_kinematic_assemblable"]) == "False" else 0.0
    margin = float(row.get("start_assemblability_margin", 0.0))
    blocked_depth_bonus = min(max(-margin, 0.0), 10.0)
    overlap_bonus = min(np.log1p(overlap / STRICT_THRESHOLD_NORM), 20.0)
    return blocked_bonus + category_bonus.get(str(row["strict_category"]), 0.0) + blocked_depth_bonus + overlap_bonus


def _annotate(row: dict, *, source_csv: Path, source_row_index: int, cache: LabelCache) -> dict | None:
    p = params_from_row(row)
    relaxed = validate_relaxed_params(p)
    if not relaxed.ok:
        return None
    payload = cache.label(p)
    label = continuous_label_from_payload(payload)
    if float(label["total_overlap_norm"]) <= STRICT_THRESHOLD_NORM:
        return None
    exact = evaluate_exact_assemblability(p, require_validity=False)
    out = {**row, **label}
    out["source_csv"] = str(source_csv)
    out["source_row_index"] = source_row_index
    out["relaxed_ok"] = True
    out["hard_failures"] = "|".join(relaxed.hard_failures)
    out["relaxed_violations"] = "|".join(relaxed.relaxed_violations)
    out.update({f"margin_{k}": float(v) for k, v in compute_margins(p).items()})
    out["strict_category"] = _head_category(row, label)
    out["strict_overlap_binary"] = True
    out["magnitude_bin"] = magnitude_bin(float(label["total_overlap_norm"]), threshold_norm=STRICT_THRESHOLD_NORM)
    out["start_kinematic_assemblable"] = bool(exact.kinematic_assemblable)
    out["start_assemblability_reason"] = exact.label_reason
    out["start_assemblability_margin"] = _assemblability_margin(p)
    out["strict_score"] = _score(out)
    return out


def _balanced_select(rows: list[dict], n: int, *, max_controls: int) -> list[dict]:
    blocked = [r for r in rows if str(r["start_kinematic_assemblable"]) == "False"]
    controls = [r for r in rows if str(r["start_kinematic_assemblable"]) == "True"]
    blocked_n = min(len(blocked), n)
    control_n = min(max_controls, max(0, n - blocked_n), len(controls))
    targets = [(blocked, blocked_n), (controls, control_n)]
    selected: list[dict] = []
    seen: set[str] = set()
    preferred = [
        "splint_head_wall_candidate",
        "pin_head_or_shaft_wall_candidate",
        "pin_splint_cross_overlap",
        "bracket_splint_other",
        "bracket_pin_other",
        "other_overlap",
    ]
    for candidates, target_n in targets:
        buckets: dict[str, list[dict]] = {}
        for row in candidates:
            buckets.setdefault(str(row["strict_category"]), []).append(row)
        for vals in buckets.values():
            vals.sort(key=lambda r: float(r["strict_score"]), reverse=True)
        while sum(1 for r in selected if r in candidates) < target_n and any(buckets.get(k) for k in preferred):
            for key in preferred:
                vals = buckets.get(key, [])
                if not vals:
                    continue
                row = vals.pop(0)
                row_key = str(row["sample_id"])
                if row_key in seen:
                    continue
                selected.append(row)
                seen.add(row_key)
                if sum(1 for r in selected if r in candidates) >= target_n:
                    break
    if len(selected) < n:
        leftovers = [r for r in rows if str(r["sample_id"]) not in seen]
        leftovers.sort(key=lambda r: float(r["strict_score"]), reverse=True)
        selected.extend(leftovers[: n - len(selected)])
    return selected[:n]


def run(args: argparse.Namespace) -> Path:
    out_dir = DATA_DIR / args.name
    cache = LabelCache(pitch_mm=DEFAULT_PITCH_MM, threshold_norm=STRICT_THRESHOLD_NORM)
    annotated: list[dict] = []
    scanned_by_source = {}
    for source_csv in args.source_csv:
        rows = _read_csv(source_csv)
        scan_n = min(args.scan_n, len(rows))
        scanned_by_source[str(source_csv)] = scan_n
        for idx, row in enumerate(rows[:scan_n]):
            item = _annotate(row, source_csv=source_csv, source_row_index=idx, cache=cache)
            if item is not None:
                annotated.append(item)
    selected = _balanced_select(annotated, args.n, max_controls=args.max_controls)
    for i, row in enumerate(selected):
        row["strict_benchmark_index"] = i
    blocked_count = sum(str(r["start_kinematic_assemblable"]) == "False" for r in selected)
    assemblable_count = len(selected) - blocked_count
    _write_csv(out_dir / "starts.csv", selected)
    summary = {
        "name": args.name,
        "source_csv": [str(p) for p in args.source_csv],
        "scanned_by_source": scanned_by_source,
        "candidate_count_strict_overlap_relaxed_valid": len(annotated),
        "candidate_count_blocked": sum(str(r["start_kinematic_assemblable"]) == "False" for r in annotated),
        "candidate_count_assemblable_under_overlap": sum(
            str(r["start_kinematic_assemblable"]) == "True" for r in annotated
        ),
        "n_selected": len(selected),
        "selected_blocked_at_start": blocked_count,
        "selected_assemblable_under_overlap": assemblable_count,
        "strict_threshold_norm": STRICT_THRESHOLD_NORM,
        "category_counts": dict(Counter(r["strict_category"] for r in selected)),
        "dominant_pair_counts": dict(Counter(r["dominant_pair"] for r in selected)),
        "magnitude_bin_counts": dict(Counter(r["magnitude_bin"] for r in selected)),
        "start_reason_counts": dict(Counter(r["start_assemblability_reason"] for r in selected)),
        "pair_columns": list(PAIR_COLUMNS),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Strict Overlap Blocked Starts",
        "",
        "A 50-start benchmark selected for strict analytic overlap and start-time kinematic blocking.",
        "",
        f"- Selected starts: `{len(selected)}`",
        f"- Blocked at start: `{blocked_count}/{len(selected)}`",
        f"- Assemblable despite overlap: `{assemblable_count}/{len(selected)}`",
        f"- Strict threshold norm: `{STRICT_THRESHOLD_NORM}`",
        "",
        "## Category Counts",
        "",
        "| Category | Count |",
        "|---|---:|",
    ]
    for key, count in summary["category_counts"].items():
        lines.append(f"| {key} | {count} |")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_dir)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="strict_overlap_blocked_v2")
    parser.add_argument(
        "--source-csv",
        type=Path,
        action="append",
        default=None,
        help="Pool CSV to scan. May be passed multiple times.",
    )
    parser.add_argument("--scan-n", type=int, default=5000)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--max-controls", type=int, default=5)
    args = parser.parse_args()
    if args.source_csv is None:
        args.source_csv = [HOLDOUT_5K_CSV]
    run(args)


if __name__ == "__main__":
    main()

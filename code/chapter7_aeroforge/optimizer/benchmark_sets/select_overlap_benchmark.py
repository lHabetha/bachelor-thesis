#!/usr/bin/env python3
"""Select the ranked 100 overlapping ADV starts for Chapter 7 optimization."""

from __future__ import annotations

from chapter7_aeroforge.release_paths import ensure_code_importable

ensure_code_importable()

import argparse
import json
import math
from pathlib import Path

from chapter7_aeroforge.release_paths import BENCHMARK_JSON, LABELS_CLOUD

DEFAULT_LABELS = LABELS_CLOUD
DEFAULT_OUT = BENCHMARK_JSON
DEFAULT_MAX_OK_ROWS = 100_000
DEFAULT_N = 100
THRESHOLD_MM3 = 1.0


def _overlap_value(row: dict) -> float:
    val = row.get("overlap_mm3_cut_each", row.get("overlap_mm3", 0.0))
    return float(val or 0.0)


def load_candidates(labels_path: Path, *, max_ok_rows: int) -> list[dict]:
    """Load overlapping rows from the first max_ok_rows ok labels."""
    candidates: list[dict] = []
    ok_seen = 0
    with labels_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("ok"):
                continue
            ok_seen += 1
            if ok_seen > max_ok_rows:
                break
            overlap = _overlap_value(row)
            if overlap <= THRESHOLD_MM3:
                continue
            candidates.append(
                {
                    "line_no": line_no,
                    "ok_rank": ok_seen,
                    "sample_idx": int(row.get("sample_idx", ok_seen - 1)),
                    "family": row.get("family"),
                    "overlap_mm3_cut_each": overlap,
                    "overlap_mm3_raw": row.get("overlap_mm3_raw"),
                    "is_overlap": True,
                    "adv": row["adv"],
                }
            )
    candidates.sort(key=lambda r: r["overlap_mm3_cut_each"], reverse=True)
    return candidates


def _select_by_log_overlap(candidates: list[dict], n: int) -> list[dict]:
    if len(candidates) <= n:
        return list(candidates)

    logs = [math.log10(c["overlap_mm3_cut_each"]) for c in candidates]
    hi, lo = max(logs), min(logs)
    targets = [hi - i * (hi - lo) / (n - 1) for i in range(n)]

    selected_indices: set[int] = set()
    selected: list[dict] = []
    for target in targets:
        best_idx = None
        best_dist = float("inf")
        for idx, log_val in enumerate(logs):
            if idx in selected_indices:
                continue
            dist = abs(log_val - target)
            if dist < best_dist:
                best_idx = idx
                best_dist = dist
        if best_idx is None:
            break
        selected_indices.add(best_idx)
        selected.append(candidates[best_idx])

    selected.sort(key=lambda r: r["overlap_mm3_cut_each"], reverse=True)
    return selected


def _select_by_rank(candidates: list[dict], n: int) -> list[dict]:
    if len(candidates) <= n:
        return list(candidates)
    idxs = [round(i * (len(candidates) - 1) / (n - 1)) for i in range(n)]
    selected = [candidates[i] for i in idxs]
    selected.sort(key=lambda r: r["overlap_mm3_cut_each"], reverse=True)
    return selected


def build_benchmark(
    labels_path: Path, *, n: int, max_ok_rows: int, spacing: str, benchmark_id: str = "overlap_ranked100_v1"
) -> dict:
    candidates = load_candidates(labels_path, max_ok_rows=max_ok_rows)
    if len(candidates) < n:
        raise RuntimeError(
            f"Need {n} overlapping rows, found {len(candidates)} in first {max_ok_rows} ok labels"
        )

    if spacing == "log_overlap":
        selected = _select_by_log_overlap(candidates, n)
    elif spacing == "rank":
        selected = _select_by_rank(candidates, n)
    else:
        raise ValueError(f"Unknown spacing: {spacing}")

    for rank, row in enumerate(selected, start=1):
        row["rank"] = rank
        row["start_id"] = f"rank_{rank:03d}_sample_{row['sample_idx']}"

    overlaps = [r["overlap_mm3_cut_each"] for r in selected]
    return {
        "benchmark_id": benchmark_id,
        "source_labels": str(labels_path),
        "source_pool": f"first {max_ok_rows} ok rows from labels.jsonl",
        "selection_rule": (
            "filter overlap_mm3_cut_each > 1 mm3, select n log-spaced overlap "
            "targets, then sort by original overlap descending"
        ),
        "spacing": spacing,
        "threshold_mm3": THRESHOLD_MM3,
        "n_requested": n,
        "n_candidates": len(candidates),
        "n_selected": len(selected),
        "overlap_range_mm3": {
            "max": max(overlaps),
            "min": min(overlaps),
        },
        "starts": selected,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build Chapter 7 overlap optimization benchmark set")
    ap.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--max-ok-rows", type=int, default=DEFAULT_MAX_OK_ROWS)
    ap.add_argument("--spacing", choices=["log_overlap", "rank"], default="log_overlap")
    ap.add_argument(
        "--benchmark-id",
        type=str,
        default="overlap_ranked100_v1",
        help="Stored benchmark_id (use overlap_ranked100_v2 for the 100k cloud-pool rebuild).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_benchmark(
        args.labels_path,
        n=args.n,
        max_ok_rows=args.max_ok_rows,
        spacing=args.spacing,
        benchmark_id=args.benchmark_id,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {manifest['n_selected']} starts to {args.out}")
    print(
        "Overlap range: "
        f"{manifest['overlap_range_mm3']['max']:,.1f} -> "
        f"{manifest['overlap_range_mm3']['min']:,.1f} mm3"
    )


if __name__ == "__main__":
    main()

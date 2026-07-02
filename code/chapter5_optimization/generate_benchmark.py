"""Generate the active 200-start blocked benchmark set for Chapter 5.

The benchmark is deliberately not sampled from the Chapter 4 dense pool. It draws
fresh valid DummyParams from the design-space sampler, formula-labels them, and
keeps only leakage-free non-assemblable rows.

Usage:
    python -m generate_benchmark
"""
from __future__ import annotations

import sys
from pathlib import Path
_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))
from shared.paths import ensure_chapter3_importable, ensure_mlp_lib_importable

ensure_chapter3_importable()
ensure_mlp_lib_importable()

import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from export_model import _get_train_ids
from lib.io import param_hash
from shared.paths import (
    BENCHMARK_DIR,
    DENSE50K_POOL,
    TASK22_HOLDOUT,
)
from shared.oracle import FEATURE_NAMES
from chapter3_clevis_setup.exact_assemblability import evaluate_exact_assemblability
from chapter3_clevis_setup.smart_sampler import (
    stream_boundary_pushed,
    stream_extreme,
    stream_lhs_batch,
    stream_uniform,
)

BENCHMARK_ID = "blocked_200_v1"
N_STARTS = 200
SEED = 6202001
MAX_ATTEMPTS = 250_000
DEFAULT_MODEL_TRAIN_SPLIT = 2

QUOTAS = {
    "near_boundary": 70,
    "typical_blocked": 50,
    "extreme_roof_cage": 50,
    "stress_extreme_corner": 30,
}

SOURCE_STREAMS = ("uniform", "boundary", "extreme", "lhs")


def _load_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    df = pd.read_parquet(path, columns=["param_id"])
    return set(df["param_id"].astype(str))


def _load_legacy_benchmark_ids() -> set[str]:
    legacy = BENCHMARK_DIR.parent / "blocked_100_v1" / "blocked_100_v1.parquet"
    return _load_ids(legacy)


def _exclusion_ids() -> dict[str, set[str]]:
    dense_ids = _load_ids(DENSE50K_POOL)
    holdout_ids = _load_ids(TASK22_HOLDOUT)
    legacy_ids = _load_legacy_benchmark_ids()
    train_ids = set(map(str, _get_train_ids(DEFAULT_MODEL_TRAIN_SPLIT)))
    return {
        "dense50k_pool": dense_ids,
        "task22_holdout": holdout_ids,
        "legacy_blocked_100": legacy_ids,
        "default_model_train_ids": train_ids,
    }


def _draw_candidate(stream: str, rng: np.random.Generator, lhs_buffer: list[Any]) -> Any:
    if stream == "uniform":
        return stream_uniform(rng)
    if stream == "boundary":
        return stream_boundary_pushed(rng)
    if stream == "extreme":
        return stream_extreme(rng, alpha=0.25)
    if stream == "lhs":
        if not lhs_buffer:
            lhs_buffer.extend(stream_lhs_batch(128, rng))
        return lhs_buffer.pop()
    raise ValueError(f"unknown stream: {stream}")


def _terms_dict(p) -> dict[str, Any]:
    result = evaluate_exact_assemblability(p)
    out = result.to_dict()
    terms = asdict(result.terms)
    out.update(terms)
    return out


def _roof_margin(terms: dict[str, Any]) -> float:
    return float(terms["y_overhang_outer_neg"]) - float(terms["y_splint_head_pos_edge"])


def _blocked_metrics(terms: dict[str, Any]) -> dict[str, float]:
    roof = _roof_margin(terms)
    splint = float(terms["splint_clearance_margin"])
    inward = float(terms["inward_movement_margin"])
    margins = {
        "roof_clearance_margin": roof,
        "splint_clearance_margin": splint,
        "inward_movement_margin": inward,
    }
    closest_name, closest_value = max(margins.items(), key=lambda item: item[1])
    overhang_ratio = float(terms["overhang_span_y"]) / max(float(terms["outer_span"]), 1e-9)
    return {
        **margins,
        "closest_margin": closest_value,
        "closest_margin_name": closest_name,
        "overhang_ratio": overhang_ratio,
    }


def _candidate_record(p, source_stream: str, attempt: int) -> dict[str, Any] | None:
    terms = _terms_dict(p)
    if not bool(terms["validity_ok"]):
        return None
    if int(terms["assemblable"]) != 0 or terms["label_reason"] != "blocked":
        return None

    params = {name: float(getattr(p, name)) for name in FEATURE_NAMES}
    metrics = _blocked_metrics({**terms, **params})
    rec: dict[str, Any] = {
        **params,
        "param_id": param_hash(params),
        "source_stream": source_stream,
        "generation_attempt": int(attempt),
        "label": 0,
        "validity_ok": True,
        "formula_reason": "blocked",
        "label_class": "blocked",
        "label_subclass": "",
        **metrics,
        "vertical_extraction_margin": float(terms["vertical_extraction_margin"]),
        "lateral_escape_margin": float(terms["lateral_escape_margin"]),
        "roof_blocks_splint": bool(terms["roof_blocks_splint"]),
    }
    return rec


def _blocked_subgroup(rec: dict[str, Any]) -> str:
    name = str(rec["closest_margin_name"])
    return f"nearest_{name.replace('_margin', '')}"


def _blocked_depth(rec: dict[str, Any]) -> str:
    cm = float(rec["closest_margin"])
    if cm >= -2.0:
        return "near_boundary"
    if cm >= -15.0:
        return "moderate"
    return "deep_blocked"


def _assign_bucket(rec: dict[str, Any]) -> str | None:
    cm = float(rec["closest_margin"])
    overhang = float(rec["overhang_ratio"])
    stream = str(rec["source_stream"])
    if -2.0 <= cm < 0.0:
        return "near_boundary"
    if stream == "extreme" and cm < -2.0:
        return "stress_extreme_corner"
    if overhang >= 1.55 and cm < -8.0:
        return "extreme_roof_cage"
    if -15.0 <= cm < -2.0:
        return "typical_blocked"
    if stream == "extreme":
        return "stress_extreme_corner"
    if overhang >= 1.45:
        return "extreme_roof_cage"
    return None


def _rank_for_bucket(bucket: str, rec: dict[str, Any]) -> tuple[float, ...]:
    cm = float(rec["closest_margin"])
    overhang = float(rec["overhang_ratio"])
    if bucket == "near_boundary":
        return (abs(cm), -overhang)
    if bucket == "typical_blocked":
        return (abs(cm + 8.0), abs(overhang - 1.25))
    if bucket == "extreme_roof_cage":
        return (-overhang, cm)
    if bucket == "stress_extreme_corner":
        return (0 if rec["source_stream"] == "extreme" else 1, cm)
    return (0.0,)


def _build_candidate_pool(exclusions: dict[str, set[str]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    rng = np.random.default_rng(SEED)
    lhs_buffer: list[Any] = []
    excluded_all = set().union(*exclusions.values())
    seen = set(excluded_all)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counters: Counter[str] = Counter()
    attempts = 0
    t0 = time.time()

    while attempts < MAX_ATTEMPTS:
        attempts += 1
        stream = SOURCE_STREAMS[(attempts - 1) % len(SOURCE_STREAMS)]
        counters[f"attempt_{stream}"] += 1
        try:
            p = _draw_candidate(stream, rng, lhs_buffer)
            rec = _candidate_record(p, stream, attempts)
        except Exception:
            counters["exceptions"] += 1
            continue
        if rec is None:
            counters["not_blocked_or_invalid"] += 1
            continue
        param_id = str(rec["param_id"])
        if param_id in seen:
            counters["leakage_or_duplicate_rejected"] += 1
            continue
        seen.add(param_id)
        bucket = _assign_bucket(rec)
        if bucket is None:
            counters["unbucketed_blocked"] += 1
            continue
        rec["benchmark_bucket"] = bucket
        rec["blocked_depth"] = _blocked_depth(rec)
        rec["subgroup"] = _blocked_subgroup(rec)
        buckets[bucket].append(rec)
        counters[f"accepted_{bucket}"] += 1
        if all(len(buckets[name]) >= quota * 3 for name, quota in QUOTAS.items()):
            break
        if attempts % 10_000 == 0:
            print(
                "[benchmark] "
                + " ".join(f"{name}={len(buckets[name])}/{quota}" for name, quota in QUOTAS.items()),
                flush=True,
            )

    audit = {
        "seed": SEED,
        "max_attempts": MAX_ATTEMPTS,
        "attempts": attempts,
        "elapsed_s": time.time() - t0,
        "counters": dict(counters),
        "bucket_candidates": {name: len(rows) for name, rows in buckets.items()},
        "exclusion_counts": {name: len(ids) for name, ids in exclusions.items()},
    }
    return buckets, audit


def _select_benchmark_rows(buckets: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bucket, quota in QUOTAS.items():
        candidates = sorted(buckets[bucket], key=lambda rec: _rank_for_bucket(bucket, rec))
        if len(candidates) < quota:
            raise RuntimeError(f"bucket {bucket} has {len(candidates)} candidates, need {quota}")
        rows.extend(candidates[:quota])

    # Interleave buckets so the viewer list does not show one regime in a block.
    grouped = {bucket: [r for r in rows if r["benchmark_bucket"] == bucket] for bucket in QUOTAS}
    interleaved: list[dict[str, Any]] = []
    while any(grouped.values()):
        for bucket in QUOTAS:
            if grouped[bucket]:
                interleaved.append(grouped[bucket].pop(0))

    for idx, rec in enumerate(interleaved):
        rec["start_id"] = f"blocked_{idx:03d}"
        rec["blocked_explanation"] = _build_explanation(rec)
    return interleaved


def _normalization(rows: list[dict[str, Any]]) -> dict[str, Any]:
    features = np.array([[float(row[name]) for name in FEATURE_NAMES] for row in rows], dtype=np.float64)
    medians = np.median(features, axis=0)
    stds = np.std(features, axis=0, ddof=0)
    stds[stds < 1e-9] = 1.0
    return {
        "benchmark_id": BENCHMARK_ID,
        "parameter_order": list(FEATURE_NAMES),
        "medians": medians.tolist(),
        "stds": stds.tolist(),
        "description": "x_norm = (x - median) / std; distance = ||x_norm_1 - x_norm_0||_2",
        "n_starts": len(rows),
        "seed": SEED,
    }


def _build_explanation(rec: dict[str, Any]) -> str:
    return (
        "Valid geometry, formula-blocked under all analytic routes:\n"
        f"  - Closest blocked margin: {rec['closest_margin_name']} = {float(rec['closest_margin']):.3f} mm\n"
        f"  - Roof clearance margin = {float(rec['roof_clearance_margin']):.3f} mm (need >= 0)\n"
        f"  - Splint clearance margin = {float(rec['splint_clearance_margin']):.3f} mm (need >= 0)\n"
        f"  - Inward movement margin = {float(rec['inward_movement_margin']):.3f} mm (need >= 0)\n"
        f"  - Benchmark bucket: {rec['benchmark_bucket']}; source stream: {rec['source_stream']}"
    )


def _build_readme(rows: list[dict[str, Any]], normalization: dict[str, Any], audit: dict[str, Any]) -> str:
    bucket_counts = Counter(row["benchmark_bucket"] for row in rows)
    stream_counts = Counter(row["source_stream"] for row in rows)
    subgroup_counts = Counter(row["subgroup"] for row in rows)
    closest = [float(row["closest_margin"]) for row in rows]
    overhangs = [float(row["overhang_ratio"]) for row in rows]

    lines = [
        "# Blocked 200 Benchmark Set v1",
        "",
        "## Provenance",
        "",
        "- Source: fresh `chapter3_clevis_setup.smart_sampler` draws, not rows sampled from the Chapter 4 dense pool.",
        f"- Seed: `{SEED}`.",
        "- Geometry version: Chapter 3 corrected (D5 outer wall, D7, slack-aware D4, 5% head lengths).",
        "- Acceptance: `validate_params=True`, exact formula `label=0`, and `formula_reason=blocked`.",
        "- Leakage exclusions: dense50k pool, Chapter 4 holdout, legacy `blocked_100_v1`, and default MLP training IDs.",
        "",
        "## Quotas",
        "",
        "| Bucket | Target | Actual | Interpretation |",
        "|--------|--------|--------|----------------|",
        "| near_boundary | 70 | " + str(bucket_counts["near_boundary"]) + " | Valid blocked rows closest to an assemblable formula route |",
        "| typical_blocked | 50 | " + str(bucket_counts["typical_blocked"]) + " | Moderate blocked rows from normal sampler coverage |",
        "| extreme_roof_cage | 50 | " + str(bucket_counts["extreme_roof_cage"]) + " | Large-overhang blocked rows with strong roof caging |",
        "| stress_extreme_corner | 30 | " + str(bucket_counts["stress_extreme_corner"]) + " | Extreme-stream corner cases that remain valid and blocked |",
        "",
        "## Source Streams",
        "",
        "| Stream | Count |",
        "|--------|-------|",
    ]
    for stream, count in sorted(stream_counts.items()):
        lines.append(f"| {stream} | {count} |")

    lines += [
        "",
        "## Blocked Subgroups",
        "",
        "| Closest failed route | Count |",
        "|----------------------|-------|",
    ]
    for subgroup, count in sorted(subgroup_counts.items()):
        lines.append(f"| {subgroup} | {count} |")

    lines += [
        "",
        "## Margin Summary",
        "",
        f"- Closest-margin range: `{min(closest):.4f}` to `{max(closest):.4f}` mm.",
        f"- Closest-margin median: `{float(np.median(closest)):.4f}` mm.",
        f"- Overhang ratio range: `{min(overhangs):.4f}` to `{max(overhangs):.4f}`.",
        f"- Overhang ratio median: `{float(np.median(overhangs)):.4f}`.",
        "",
        "## Normalization",
        "",
        f"- Parameters: {len(normalization['parameter_order'])}",
        "- Method: median=0, std=1 computed over these 200 starts.",
        "- Distance: L2 in normalized space.",
        "",
        "## Files",
        "",
        "- `blocked_200_v1.parquet` — 200 rows with params, formula terms, buckets, and metadata.",
        "- `blocked_200_v1.jsonl` — same data, one JSON object per line.",
        "- `normalization.json` — medians, stds, parameter order.",
        "- `generation_audit.json` — generation counters, leakage exclusion counts, and zero-overlap proof.",
        "",
        "## Generation Audit Snapshot",
        "",
        f"- Attempts: `{audit['attempts']}`.",
        f"- Candidate bucket pool before quota selection: `{audit['bucket_candidates']}`.",
        f"- Exclusion counts: `{audit['exclusion_counts']}`.",
        "",
    ]
    return "\n".join(lines)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _generate_benchmark() -> None:
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    exclusions = _exclusion_ids()
    buckets, audit = _build_candidate_pool(exclusions)
    rows = _select_benchmark_rows(buckets)
    norm = _normalization(rows)

    selected_ids = {str(row["param_id"]) for row in rows}
    overlaps = {name: len(selected_ids & ids) for name, ids in exclusions.items()}
    audit.update({
        "benchmark_id": BENCHMARK_ID,
        "n_starts": len(rows),
        "quotas": QUOTAS,
        "selected_bucket_counts": dict(Counter(row["benchmark_bucket"] for row in rows)),
        "selected_stream_counts": dict(Counter(row["source_stream"] for row in rows)),
        "selected_subgroup_counts": dict(Counter(row["subgroup"] for row in rows)),
        "selected_overlap_counts": overlaps,
        "all_selected_valid_formula_blocked": True,
    })
    if any(overlaps.values()):
        raise RuntimeError(f"leakage overlap detected: {overlaps}")

    out_df = pd.DataFrame(rows)
    parquet_path = BENCHMARK_DIR / "blocked_200_v1.parquet"
    jsonl_path = BENCHMARK_DIR / "blocked_200_v1.jsonl"
    norm_path = BENCHMARK_DIR / "normalization.json"
    audit_path = BENCHMARK_DIR / "generation_audit.json"
    readme_path = BENCHMARK_DIR / "README.md"

    out_df.to_parquet(parquet_path, index=False)
    with jsonl_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, default=_json_default, sort_keys=True) + "\n")
    norm_path.write_text(json.dumps(norm, indent=2))
    audit_path.write_text(json.dumps(audit, indent=2, default=_json_default))
    readme_path.write_text(_build_readme(rows, norm, audit))

    print(f"Wrote {parquet_path}")
    print(f"Wrote {jsonl_path}")
    print(f"Wrote {norm_path}")
    print(f"Wrote {audit_path}")
    print(f"Wrote {readme_path}")
    print(f"Done. {len(rows)} leakage-free formula-blocked starts generated.")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite datasets/chapter5_blocked200/ (default: refuse)",
    )
    args = parser.parse_args()
    if not args.force:
        parser.print_help()
        print(
            "\nRefusing to regenerate the shipped benchmark without --force. "
            "Use the frozen files under datasets/chapter5_blocked200/ for thesis reproduction."
        )
        return
    _generate_benchmark()


if __name__ == "__main__":
    main()

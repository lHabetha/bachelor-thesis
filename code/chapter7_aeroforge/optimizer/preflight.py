#!/usr/bin/env python3
"""Pre-flight checks for the Chapter 7 optimization grid.

Brings the data domain in line with the production 100k cloud-trained checkpoints
and freezes the benchmark, so the overnight grid runs in-distribution:

  1. Rebuild `normalization_bounds.json` from the **100k cloud** label pool
     (`dataset/labels_cloud/labels.jsonl`) — the domain the surrogates trained on.
  2. Build `overlap_ranked100_v2.json` (100 log-spaced overlapping starts) from the
     same cloud pool.
  3. **Clip-check gate:** every one of the 100 v2 starts must map to u in [0,1]
     under the rebuilt bounds with zero clipping. Fails (non-zero exit) otherwise,
     so `run_sweep.py` refuses to start on an out-of-domain benchmark.

This is data-prep only — it never builds geometry and never runs an optimizer.
"""

from __future__ import annotations

from chapter7_aeroforge.release_paths import ensure_code_importable

ensure_code_importable()

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path


from chapter7_aeroforge.release_paths import BENCHMARK_JSON, LABELS_CLOUD, NORMALIZATION_JSON, REPO_ROOT, subprocess_env

DEFAULT_CLOUD_LABELS = LABELS_CLOUD
DEFAULT_BENCHMARK_OUT = BENCHMARK_JSON
NORMALIZATION_PATH = NORMALIZATION_JSON
BENCHMARK_ID = "overlap_ranked100_v2"


def rebuild_normalization(cloud_labels: Path) -> None:
    print(f"[preflight] rebuilding normalization_bounds.json from {cloud_labels}")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "chapter7_aeroforge.optimizer.build_normalization",
            "--labels",
            str(cloud_labels),
        ],
        cwd=str(REPO_ROOT),
        env=subprocess_env(),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise SystemExit("[preflight] build_normalization failed")
    payload = json.loads(NORMALIZATION_PATH.read_text(encoding="utf-8"))
    print(
        f"[preflight] normalization domain = {payload.get('n_rows')} rows from "
        f"{Path(payload.get('source_labels', '')).name}"
    )


def build_benchmark_v2(cloud_labels: Path, out: Path, *, n: int, max_ok_rows: int) -> dict:
    from chapter7_aeroforge.optimizer.benchmark_sets.select_overlap_benchmark import (
        build_benchmark,
    )

    print(f"[preflight] building {BENCHMARK_ID} ({n} starts) from {cloud_labels}")
    manifest = build_benchmark(
        cloud_labels,
        n=n,
        max_ok_rows=max_ok_rows,
        spacing="log_overlap",
        benchmark_id=BENCHMARK_ID,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    rng = manifest["overlap_range_mm3"]
    print(
        f"[preflight] saved {manifest['n_selected']} starts -> {out} "
        f"(overlap {rng['max']:,.1f} -> {rng['min']:,.1f} mm^3)"
    )
    return manifest


def clip_check(benchmark: dict, *, eps: float) -> tuple[int, list[dict]]:
    """Count drivers whose pre-clip normalized value falls outside [0,1]."""
    # Import AFTER the bounds json is rewritten so NUMERIC_BOUNDS reflects the cloud
    # domain (reload guards against an earlier cached import).
    from chapter7_aeroforge.optimizer import model_io

    importlib.reload(model_io)

    n_clipped = 0
    offenders: list[dict] = []
    for start in benchmark["starts"]:
        adv = start["adv"]
        for key in model_io.opt_keys_for_adv(adv):
            lo, hi = model_io.NUMERIC_BOUNDS[key]
            span = hi - lo
            if span <= 0:
                continue
            raw_u = (float(adv.get(key)) - lo) / span
            if raw_u < -eps or raw_u > 1.0 + eps:
                n_clipped += 1
                offenders.append(
                    {
                        "rank": start.get("rank"),
                        "start_id": start.get("start_id"),
                        "key": key,
                        "raw_u": round(raw_u, 6),
                        "value": float(adv.get(key)),
                        "bounds": [lo, hi],
                    }
                )
    return n_clipped, offenders


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Chapter 7 optimization-grid pre-flight (domain + benchmark v2)")
    ap.add_argument("--cloud-labels", type=Path, default=DEFAULT_CLOUD_LABELS)
    ap.add_argument("--benchmark-out", type=Path, default=DEFAULT_BENCHMARK_OUT)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-ok-rows", type=int, default=100_000)
    ap.add_argument("--eps", type=float, default=1e-6, help="Clip-check tolerance on u")
    ap.add_argument("--skip-normalization", action="store_true")
    ap.add_argument("--skip-benchmark", action="store_true")
    ap.add_argument("--allow-clip", action="store_true", help="Warn instead of failing on clipping")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if not args.cloud_labels.exists():
        raise SystemExit(f"[preflight] cloud labels not found: {args.cloud_labels}")

    if not args.skip_normalization:
        rebuild_normalization(args.cloud_labels)
    else:
        print("[preflight] skipping normalization rebuild (--skip-normalization)")

    if not args.skip_benchmark:
        benchmark = build_benchmark_v2(
            args.cloud_labels, args.benchmark_out, n=args.n, max_ok_rows=args.max_ok_rows
        )
    else:
        print(f"[preflight] reusing existing benchmark {args.benchmark_out}")
        benchmark = json.loads(args.benchmark_out.read_text(encoding="utf-8"))

    n_clipped, offenders = clip_check(benchmark, eps=args.eps)
    n_starts = len(benchmark["starts"])
    if n_clipped == 0:
        print(f"[preflight] CLIP-CHECK PASS: all {n_starts} starts map to u in [0,1] (0 clips)")
        print("[preflight] OK — benchmark v2 frozen; grid is cleared to start.")
        return 0

    print(f"[preflight] CLIP-CHECK: {n_clipped} driver(s) clipped across {n_starts} starts")
    for off in offenders[:20]:
        print(
            f"    rank={off['rank']} {off['key']}: u={off['raw_u']} "
            f"value={off['value']} bounds={off['bounds']}"
        )
    if args.allow_clip:
        print("[preflight] WARNING: proceeding despite clipping (--allow-clip).")
        return 0
    print("[preflight] FAIL — fix the domain/benchmark before running the grid (or pass --allow-clip).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

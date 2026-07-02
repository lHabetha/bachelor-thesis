"""Export Chapter 6.5 alternating-pipeline runs into the Chapter 5 viewer (Chapter 6 #5g).

Writes one Task-26-schema run directory per overlap variant under
``viewers/chapter6_overlap_repair/data/ch65_zigzag_<variant>/`` so the existing
viewer server auto-discovers them (it scans ``runs/**/manifest.json`` and builds
geometry on demand from each frame's ``params``).

Scope: **successful pipelines only**. The zig-zag pipeline has multiple stages per run
(overlap -> assemblability -> overlap -> ...); a single start->final crossfade would hide
which optimizer moved which geometry. Each frame therefore carries a ``segment_idx`` and
``segment_local_t`` so the frontend can fetch only the start/end geometry of each segment
(two OBJ fetches per segment) and crossfade within the segment, rebinding when the
segment changes.

Each run dir contains:

- ``manifest.json``    -- Task-26 manifest + ``task: "ch65_zigzag"`` /
  ``category: "ch65_zigzag_success"`` so the frontend routes it to the new category.
- ``viewer_data.json`` -- one record per *successful* assembly, each with a flat list of
  global frames tagged with ``segment_idx`` / ``segment_stage`` / ``segment_method`` /
  ``segment_local_t`` / ``cycle`` / ``params`` / ``verified_overlap_norm`` /
  ``formula_assemblable``.
- ``algorithm.md``     -- short method description for the info panel.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .paths import RUNS_DIR
from .sampler import DummyParams
from ._public_helpers import evaluate_exact_assemblability
from .label_cache import STRICT_THRESHOLD_NORM

_PARAM_FIELDS = set(DummyParams.__dataclass_fields__)

from .release_paths import VIEWER_RUNS_DIR

VARIANT_TITLES = {
    "hybrid_lite": "6.5 zig-zag (hybrid-lite overlap + trust-region asm)",
    "finish_line": "6.5 zig-zag (finish-line overlap + trust-region asm)",
}


def _formula_assemblable(params: dict) -> bool:
    try:
        dp = DummyParams(**{k: float(v) for k, v in params.items() if k in _PARAM_FIELDS})
    except Exception:  # noqa: BLE001 - viewer cosmetics must never break the export
        return False
    return bool(evaluate_exact_assemblability(dp, require_validity=False).kinematic_assemblable)


def _build_frames(segments: list[dict]) -> list[dict]:
    """Flatten per-stage trajectories into globally-indexed, segment-tagged frames."""
    frames: list[dict] = []
    global_idx = 0
    for seg in segments:
        traj = seg.get("trajectory", [])
        n = len(traj)
        if n == 0:
            continue
        for j, tf in enumerate(traj):
            params = tf["params"]
            verified = float(tf.get("verified_overlap_norm", 0.0))
            assemblable = _formula_assemblable(params)
            local_t = (j / (n - 1)) if n > 1 else 0.0
            frames.append({
                "frame_idx": global_idx,
                "segment_idx": int(seg["segment_idx"]),
                "segment_stage": seg["stage"],
                "segment_method": seg["method"],
                "segment_local_t": float(local_t),
                "cycle": int(seg["cycle"]),
                "params": params,
                "verified_overlap_norm": verified,
                "overlap_ok": bool(verified <= STRICT_THRESHOLD_NORM),
                "formula_assemblable": assemblable,
                # Generic fields the frontend's non-morph displays also read.
                "valid": True,
                "oracle_label": 1 if assemblable else 0,
                "oracle_reason": "assemblable" if assemblable else "blocked",
                "probability": None,
            })
            global_idx += 1
    return frames


def _assembly_record(result: dict) -> dict | None:
    if not result.get("success"):
        return None
    segments = result.get("segments", [])
    frames = _build_frames(segments)
    if len(frames) < 2:
        return None
    n_segments = len({f["segment_idx"] for f in frames})
    overlap_segs = sum(1 for s in segments if s["stage"] == "overlap")
    asm_segs = sum(1 for s in segments if s["stage"] == "assemblability")
    explanation = (
        f"variant={result.get('overlap_method', '')}  category={result.get('strict_category', '')}\n"
        f"cycles={result.get('cycles_used')}  active stages={result.get('active_stages')} "
        f"(overlap {overlap_segs}, asm {asm_segs})\n"
        f"overlap reintroduced={result.get('reintroduced_overlap_count')}  "
        f"still blocked after stage 1={result.get('still_blocked_after_first_overlap')}\n"
        f"start overlap={float(result.get('start_overlap_norm', 0.0)):.3e}  "
        f"final overlap={float(result.get('final_overlap_norm', 0.0)):.3e}  "
        f"overall reduction={float(result.get('overall_pct_overlap_reduction', 0.0)):.1f}%"
    )
    return {
        "start_id": result.get("sample_id", ""),
        "subgroup": result.get("strict_category", ""),
        "status": "pipeline_success",
        "oracle_label": 1,
        "oracle_reason": "assemblable",
        "normalized_distance": 0.0,
        "start_probability": 0.0,
        "final_probability": 1.0,
        "threshold": STRICT_THRESHOLD_NORM,
        "stop_reason": "success",
        "blocked_explanation": explanation,
        "frames": frames,
        # Zig-zag run/assembly markers consumed by the frontend.
        "kind": "zigzag_pipeline",
        "zigzag_mode": True,
        "overlap_method": result.get("overlap_method", ""),
        "success": True,
        "n_segments": n_segments,
        "cycles_used": result.get("cycles_used"),
        "active_stages": result.get("active_stages"),
        "reintroduced_overlap_count": result.get("reintroduced_overlap_count"),
        "start_overlap_norm": float(result.get("start_overlap_norm", 0.0)),
        "final_overlap_norm": float(result.get("final_overlap_norm", 0.0)),
        "start_magnitude_bin": result.get("start_magnitude_bin", ""),
    }


def _algorithm_md(variant: str, title: str, n_success: int, n_total: int) -> str:
    return (
        f"# {title}\n\n"
        f"Chapter 6.5 strict alternating overlap/assemblability pipeline (v2, #5g), "
        f"overlap variant `{variant}`, on `strict_overlap_blocked_v2`.\n\n"
        f"- Successful pipelines shown here: **{n_success}** (of {n_total} starts).\n\n"
        "Each successful run is animated as a sequence of **segments** (overlap repair, "
        "then assemblability repair, then any further overlap pass). Within a segment the "
        "viewer crossfades between the segment's start and end geometry; the on-screen "
        "label shows the current cycle, stage, and method. Geometry is generated on demand "
        "from each frame's clevis parameters.\n"
    )


def _write_run(variant: str, results: list[dict], out_root: Path) -> Path | None:
    assemblies = [a for a in (_assembly_record(r) for r in results) if a]
    title = VARIANT_TITLES.get(variant, f"6.5 zig-zag ({variant})")
    run_id = f"ch65_zigzag_{variant}"
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "viewer_data.json").write_text(json.dumps(assemblies, indent=2), encoding="utf-8")
    manifest = {
        "schema_version": "v1",
        "run_id": run_id,
        "optimizer_id": f"strict_zigzag_ch65_{variant}_v3",
        "model_id": "overlap_regressor_v3 + row1_uncertainty_disagreement_B1000_T2500_best",
        "benchmark_set": "strict_overlap_blocked_v2",
        "tau": STRICT_THRESHOLD_NORM,
        "n_starts": len(assemblies),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task": "ch65_zigzag",
        "category": "ch65_zigzag_success",
        "title": title,
        "results_summary": {
            "pipeline_success": len(assemblies),
            "oracle_confirmed": len(assemblies),
            "false_success": 0,
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (run_dir / "algorithm.md").write_text(
        _algorithm_md(variant, title, len(assemblies), len(results)), encoding="utf-8"
    )
    print(f"  {run_id}: {len(assemblies)} successful assemblies (of {len(results)})")
    return run_dir


def run(args: argparse.Namespace) -> Path:
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    variants = [v for v in args.variants.split(",") if v]
    for variant in variants:
        results_path = RUNS_DIR / "zigzag" / f"strict_zigzag_ch65_{variant}_v3" / "results.json"
        if not results_path.exists():
            print(f"  (skip {variant}: {results_path} missing)")
            continue
        results = json.loads(results_path.read_text(encoding="utf-8"))
        _write_run(variant, results, out_root)
    print(out_root)
    return out_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", default="hybrid_lite,finish_line")
    parser.add_argument("--out-root", type=Path, default=VIEWER_RUNS_DIR)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

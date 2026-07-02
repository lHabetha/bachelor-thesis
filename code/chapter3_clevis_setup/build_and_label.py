#!/usr/bin/env python3
"""Build and label a small number of clevis designs (Chapter 3 release).

Examples::

    cd bachelor-thesis/code
    python -m chapter3_clevis_setup.build_and_label \\
        --params-json ../results/chapter3_clevis_setup/examples/demo01_params.json

    python -m chapter3_clevis_setup.build_and_label \\
        --stream-demo 5 --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_CHAPTER3_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CHAPTER3_DIR.parent.parent
_DEFAULT_OUT = _REPO_ROOT / "results" / "chapter3_clevis_setup"


def _default_out_dir() -> Path:
    return _DEFAULT_OUT


def _load_params_json(path: Path):
    from .labels import params_from_dict

    raw = json.loads(path.read_text(encoding="utf-8"))
    raw.pop("_derived", None)
    return params_from_dict(raw)


def _process_one(
    *,
    name: str,
    params,
    out_root: Path,
    skip_mesh: bool,
) -> dict:
    from .clevis_generator import generate
    from .design_space import validate_params
    from .labels import label_params
    from .overlap_check import overlap_check

    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    validity_ok, validity_reasons = validate_params(params)
    record = label_params(params)
    record["sample_name"] = name

    overlap_ok = None
    overlap_report = None
    generate_ok = None

    if not skip_mesh:
        generate_ok = generate(params, out_dir, verbose=False)
        overlap_ok, overlap_report = overlap_check(out_dir)
        record["mesh_generate_ok"] = bool(generate_ok)
        record["overlap_ok"] = bool(overlap_ok)
        record["overlap_status"] = overlap_report.get("status")
    else:
        record["mesh_generate_ok"] = None
        record["overlap_ok"] = None
        record["overlap_status"] = "skipped"

    record["validity_ok"] = validity_ok
    record["validity_reasons"] = validity_reasons

    summary_path = out_dir / "label_summary.json"
    summary_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate clevis CAD and analytic labels for Chapter 3.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"Output directory (default: {_DEFAULT_OUT})",
    )
    parser.add_argument(
        "--params-json",
        type=Path,
        default=None,
        help="Load one design from a params JSON file.",
    )
    parser.add_argument(
        "--stream-demo",
        type=int,
        default=0,
        metavar="N",
        help="Sample N designs using the thesis Appendix A.1 five-mode cycle.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Starting seed for --stream-demo (increments by 1 per sample).",
    )
    parser.add_argument(
        "--skip-mesh",
        action="store_true",
        help="Label only; skip CadQuery mesh export and overlap check.",
    )
    args = parser.parse_args(argv)

    out_root = (args.out_dir or _default_out_dir()).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []

    if args.params_json is not None:
        params = _load_params_json(args.params_json.resolve())
        records.append(
            _process_one(
                name="from_params_json",
                params=params,
                out_root=out_root,
                skip_mesh=args.skip_mesh,
            )
        )
    elif args.stream_demo > 0:
        from .smart_sampler import sample_by_thesis_cycle

        for i in range(args.stream_demo):
            seed = args.seed + i
            stream_name, params = sample_by_thesis_cycle(seed)
            records.append(
                _process_one(
                    name=f"seed_{seed}_{stream_name}",
                    params=params,
                    out_root=out_root,
                    skip_mesh=args.skip_mesh,
                )
            )
    else:
        from .design_space import DummyParams

        records.append(
            _process_one(
                name="default_params",
                params=DummyParams(),
                out_root=out_root,
                skip_mesh=args.skip_mesh,
            )
        )

    manifest = {
        "out_dir": str(out_root.relative_to(_REPO_ROOT)) if out_root.is_relative_to(_REPO_ROOT) else str(out_root),
        "n_samples": len(records),
        "records": [
            {
                "sample_name": r["sample_name"],
                "label_reason": r["label_reason"],
                "validity_ok": r["validity_ok"],
                "assemblable": r["assemblable"],
                "overlap_ok": r.get("overlap_ok"),
            }
            for r in records
        ],
    }
    manifest_path = out_root / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for r in records:
        print(
            f"{r['sample_name']}: validity={r['validity_ok']} "
            f"label={r['label_reason']} assemblable={r['assemblable']} "
            f"overlap={r.get('overlap_ok')}"
        )
    print(f"Wrote manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

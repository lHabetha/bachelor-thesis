"""Inspect the exported Chapter 6 overlap checkpoints shipped under ``results/``."""

from __future__ import annotations

import json
from pathlib import Path

from .release_paths import CHECKPOINTS_DIR


def summarize_checkpoints(root: Path = CHECKPOINTS_DIR) -> list[dict]:
    rows = []
    for d in sorted(root.iterdir() if root.exists() else []):
        if not d.is_dir():
            continue
        card = d / "model_card.md"
        arch = d / "architecture.json"
        rows.append({
            "model_id": d.name,
            "has_model_state": (d / "model_state.pt").exists(),
            "has_standardizer": (d / "standardizer.npz").exists(),
            "architecture": json.loads(arch.read_text()) if arch.exists() else None,
            "model_card": card.read_text() if card.exists() else "",
        })
    return rows


def main() -> None:
    for row in summarize_checkpoints():
        print(f"{row['model_id']}: model_state={row['has_model_state']} standardizer={row['has_standardizer']}")


if __name__ == "__main__":
    main()

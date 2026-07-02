"""I/O utilities for Chapter 4: hashing, file locks, checkpoints, parquet helpers."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .formula_oracle import FEATURE_NAMES


# ─── Deterministic param_id ──────────────────────────────────────────────


def param_hash(record: dict[str, Any], ndigits: int = 4) -> str:
    """Stable hash from the 13 DummyParams values, rounded to ndigits."""
    parts = []
    for name in FEATURE_NAMES:
        parts.append(f"{name}={round(float(record[name]), ndigits)}")
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ─── Atomic JSON write ───────────────────────────────────────────────────


def atomic_json_write(path: Path, obj: Any) -> None:
    """Write JSON atomically via temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ─── File-lock append (for log shards during parallel runs) ──────────────


def append_json_line(path: Path, obj: dict[str, Any]) -> None:
    """Append one JSON line to a file with an exclusive file lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, sort_keys=True) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(line)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# ─── Checkpoint helpers ──────────────────────────────────────────────────


def write_checkpoint(traj_dir: Path, state: dict[str, Any]) -> None:
    """Write checkpoint.json atomically."""
    atomic_json_write(traj_dir / "checkpoint.json", state)


def read_checkpoint(traj_dir: Path) -> dict[str, Any] | None:
    return read_json(traj_dir / "checkpoint.json")


# ─── Parquet helpers ─────────────────────────────────────────────────────


def records_to_parquet(records: list[dict[str, Any]], path: Path) -> None:
    """Write a list of flat dicts to a parquet file."""
    import pandas as pd
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_parquet(path, index=False)


def read_parquet(path: Path) -> list[dict[str, Any]]:
    """Read parquet file as list of dicts."""
    import pandas as pd
    df = pd.read_parquet(path)
    return df.to_dict(orient="records")


# ─── SHA256 for manifest ─────────────────────────────────────────────────


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# ─── Env setup for workers ───────────────────────────────────────────────


def set_single_thread() -> None:
    """Force single-thread for parallel workers."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)

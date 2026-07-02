"""Data loading and seed-set helpers for Chapter 4."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .paths import HOLDOUT_PATH, MLP_LIB_ROOT, POOL_PATH, SEED_SETS_DIR

if str(MLP_LIB_ROOT) not in sys.path:
    sys.path.insert(0, str(MLP_LIB_ROOT))

from lib.formula_oracle import FEATURE_NAMES, REASON_BLOCKED, REASON_INWARD_MOVEMENT  # noqa: E402


@dataclass(frozen=True)
class DataBundle:
    pool: pd.DataFrame
    holdout: pd.DataFrame

    @property
    def pool_by_id(self) -> dict[str, int]:
        return {str(pid): i for i, pid in enumerate(self.pool["param_id"].astype(str).tolist())}


def set_single_thread() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass


def load_bundle() -> DataBundle:
    pool = pd.read_parquet(POOL_PATH)
    holdout = pd.read_parquet(HOLDOUT_PATH)
    pool["param_id"] = pool["param_id"].astype(str)
    holdout["param_id"] = holdout["param_id"].astype(str)
    return DataBundle(pool=pool, holdout=holdout)


def features(df: pd.DataFrame) -> np.ndarray:
    return df[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)


def labels(df: pd.DataFrame) -> np.ndarray:
    return df["label"].to_numpy(dtype=np.int64)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def make_stratified_ids(df: pd.DataFrame, n: int, rng: np.random.Generator) -> list[str]:
    if n <= 0:
        return []
    pos = df[df["label"] == 1]["param_id"].astype(str).to_numpy()
    neg = df[df["label"] == 0]["param_id"].astype(str).to_numpy()
    rng.shuffle(pos)
    rng.shuffle(neg)
    neg_rate = len(neg) / max(len(df), 1)
    n_neg = int(round(n * neg_rate))
    n_neg = min(max(n_neg, 1 if len(neg) > 0 and n > 1 else 0), len(neg), n)
    n_pos = min(n - n_neg, len(pos))
    if n_pos + n_neg < n:
        extra = n - n_pos - n_neg
        if len(pos) - n_pos >= extra:
            n_pos += extra
        else:
            n_neg = min(n - n_pos, len(neg))
    ids = np.concatenate([pos[:n_pos], neg[:n_neg]])
    rng.shuffle(ids)
    return ids.astype(str).tolist()


def build_seed_sets(bundle: DataBundle, base_sizes: list[int], n_splits: int, seed: int) -> None:
    max_base = max(base_sizes)
    for split in range(1, n_splits + 1):
        rng = np.random.default_rng(seed + split)
        master_ids = make_stratified_ids(bundle.pool, max_base, rng)
        split_dir = SEED_SETS_DIR / f"R{split:03d}"
        write_json(
            split_dir / f"master_{max_base}.json",
            {"seed_split": split, "base_size": max_base, "param_ids": master_ids},
        )
        for base in base_sizes:
            write_json(
                split_dir / f"base_prefix_{base}.json",
                {"seed_split": split, "base_size": base, "param_ids": master_ids[:base]},
            )


def load_seed_ids(base_size: int, seed_split: int) -> list[str]:
    path = SEED_SETS_DIR / f"R{seed_split:03d}" / f"base_prefix_{base_size}.json"
    return read_json(path)["param_ids"]


def records_for_ids(df: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    if not ids:
        return df.iloc[[]].copy()
    id_df = pd.DataFrame({"param_id": [str(i) for i in ids], "_order": np.arange(len(ids))})
    out = id_df.merge(df, on="param_id", how="left").sort_values("_order")
    if out[list(FEATURE_NAMES)].isna().any().any():
        missing = out[out[list(FEATURE_NAMES)].isna().any(axis=1)]["param_id"].tolist()[:5]
        raise KeyError(f"Missing param IDs in pool: {missing}")
    return out.drop(columns=["_order"])


def acquisition_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f_margin = df.get("formula_margin_min", pd.Series(np.zeros(len(df)))).to_numpy(dtype=np.float64)
    reasons = df.get("formula_reason", pd.Series([""] * len(df))).astype(str)
    is_blocked = (reasons == REASON_BLOCKED).to_numpy(dtype=np.float64)
    is_inward = (reasons == REASON_INWARD_MOVEMENT).to_numpy(dtype=np.float64)
    return f_margin, is_blocked, is_inward


def enabled_model_ids(protocol: dict[str, Any]) -> list[str]:
    model_ids: list[str] = []
    for model_id, cfg in protocol["models"].items():
        if cfg.get("enabled", False):
            model_ids.append(model_id)
        elif cfg.get("enabled_if_available", False):
            if model_id == "xgboost":
                try:
                    import xgboost  # noqa: F401

                    model_ids.append(model_id)
                except Exception:
                    continue
    return model_ids


def job_key(*parts: object) -> str:
    return "__".join(str(p) for p in parts).replace("/", "_")

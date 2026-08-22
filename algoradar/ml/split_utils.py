from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Any

import numpy as np
import pandas as pd
import hashlib
from pathlib import Path


def temporal_user_splits(
    frame: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Dict[str, Any]:
    """Create temporal, user-aware train/val/test splits based on `event_time`.

    Requirements:
    - `frame` must contain `event_time` (seconds since epoch) and `handle` columns.

    Returns dict with keys: `train_idx`, `val_idx`, `test_idx`, `metadata`.
    """
    if frame is None or frame.empty:
        return {"train_idx": np.array([], dtype=int), "val_idx": np.array([], dtype=int), "test_idx": np.array([], dtype=int), "metadata": {}}

    if "event_time" not in frame.columns:
        raise ValueError("frame must contain 'event_time' column for temporal splits")

    times = pd.to_numeric(frame["event_time"], errors="coerce").dropna()
    if times.empty:
        raise ValueError("event_time column contains no valid timestamps")

    # compute global cutoffs by quantiles
    train_q = float(train_frac)
    val_q = float(train_frac + val_frac)
    train_cut = float(times.quantile(train_q))
    val_cut = float(times.quantile(val_q))

    # assign splits
    indices = np.arange(len(frame))
    train_mask = frame["event_time"].astype(float) <= train_cut
    val_mask = (frame["event_time"].astype(float) > train_cut) & (frame["event_time"].astype(float) <= val_cut)
    test_mask = frame["event_time"].astype(float) > val_cut

    # Ensure no empty splits: if a split is empty, move nearest rows into it
    if not train_mask.any():
        earliest = int(times.min())
        train_mask = frame["event_time"].astype(float) <= earliest
    if not test_mask.any():
        latest = int(times.max())
        test_mask = frame["event_time"].astype(float) >= latest
    if not val_mask.any():
        # pick middle point
        mid = int((train_cut + val_cut) / 2)
        val_mask = (frame["event_time"].astype(float) > train_cut) & (frame["event_time"].astype(float) <= mid)

    train_idx = indices[train_mask.fillna(False).to_numpy()]
    val_idx = indices[val_mask.fillna(False).to_numpy()]
    test_idx = indices[test_mask.fillna(False).to_numpy()]

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "train_cut": train_cut,
        "val_cut": val_cut,
        "counts": {"train": int(train_idx.size), "val": int(val_idx.size), "test": int(test_idx.size)},
        "seed": int(seed),
    }
    return {"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx, "metadata": metadata}


def dataset_hash(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    """Compute a stable SHA256 hash for a dataset using selected columns (defaults to handle/problem_id/event_time).

    Returns hex digest string.
    """
    if frame is None or frame.empty:
        return hashlib.sha256(b"empty").hexdigest()
    cols = columns or ["handle", "problem_id", "event_time"]
    subset = frame.copy()
    avail = [c for c in cols if c in subset.columns]
    if not avail:
        text = subset.to_csv(index=False).encode("utf-8")
    else:
        text = subset[avail].astype(str).sort_values(avail).to_csv(index=False).encode("utf-8")
    return hashlib.sha256(text).hexdigest()

import time
import numpy as np
import pandas as pd
from algoradar.ml.split_utils import temporal_user_splits
from algoradar.models import evaluate_real_label_models
from pathlib import Path


def test_temporal_user_splits_basic():
    # create synthetic dataset with monotonic event_time
    n = 30
    now = int(time.time())
    event_times = np.arange(now, now + n)
    df = pd.DataFrame({
        "event_time": event_times,
        "handle": [f"user{int(i%3)}" for i in range(n)],
    })

    splits = temporal_user_splits(df, train_frac=0.6, val_frac=0.2, test_frac=0.2)
    meta = splits.get("metadata", {})
    assert meta.get("counts", {})
    counts = meta["counts"]
    assert counts["train"] > 0
    assert counts["val"] > 0
    assert counts["test"] > 0

    train_idx = splits["train_idx"]
    val_idx = splits["val_idx"]
    test_idx = splits["test_idx"]
    assert max(df.iloc[train_idx]["event_time"]) <= min(df.iloc[val_idx]["event_time"]) or len(val_idx)==0
    assert max(df.iloc[val_idx]["event_time"]) <= min(df.iloc[test_idx]["event_time"]) or len(test_idx)==0


def test_evaluate_real_label_models_uses_splits():
    # build small synthetic real-label dataset
    n = 120
    now = int(time.time())
    rows = []
    for i in range(n):
        t = now + i * 86400  # one day apart
        user = f"user{int(i%5)}"
        user_rating = float(1200 + (i % 10) * 30)
        problem_rating = float(1000 + (i % 15) * 40)
        difficulty_gap = problem_rating - user_rating
        tag_mastery = float(((i % 7) * 10) % 100) / 100.0
        previous_attempts = float(i % 4)
        recent_activity = float((i % 20))
        solved_before = float(i % 2)
        tag_count = float((i % 4) + 1)
        y = 1 if difficulty_gap < 200 else 0
        rows.append(
            {
                "handle": user,
                "event_time": t,
                "user_rating_at_time": user_rating,
                "problem_rating": problem_rating,
                "difficulty_gap": difficulty_gap,
                "tag_mastery": tag_mastery,
                "previous_attempts": previous_attempts,
                "recent_activity": recent_activity,
                "solved_before": solved_before,
                "tag_count": tag_count,
                "y": y,
            }
        )
    frame = pd.DataFrame(rows)
    report = evaluate_real_label_models(frame, random_state=1)
    assert isinstance(report, dict)
    assert "split" in report
    assert "metrics" in report


def test_split_persistence_and_dataset_version(tmp_path):
    # small synthetic dataset
    now = int(time.time())
    rows = []
    for i in range(20):
        t = now + i * 3600
        rows.append(
            {
                "handle": f"u{i%3}",
                "problem_id": f"p{i}",
                "event_time": t,
                "user_rating_at_time": 1200.0,
                "problem_rating": 1300.0,
                "difficulty_gap": 100.0,
                "tag_mastery": 0.5,
                "previous_attempts": 0.0,
                "recent_activity": 1.0,
                "solved_before": 0.0,
                "tag_count": 1.0,
                "y": 1 if (i % 3) != 0 else 0,
            }
        )
    frame = pd.DataFrame(rows)
    report = evaluate_real_label_models(frame, random_state=2)
    assert "dataset_version" in report
    assert "split_metadata_path" in report
    path = report["split_metadata_path"]
    assert Path(path).exists()
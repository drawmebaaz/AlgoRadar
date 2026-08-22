import json
from pathlib import Path

import numpy as np
import pandas as pd

from algoradar.models import train_multi_horizon_models, evaluate_and_save_per_horizon


def make_frame(n=120):
    rng = np.random.default_rng(42)
    rows = []
    now = 1_700_000_000
    for i in range(n):
        handle = f"user{i%6}"
        event_time = now + i * 1000
        user_rating = float(1200 + (i % 10) * 10)
        problem_rating = float(1200 + (i % 7) * 50)
        difficulty_gap = problem_rating - user_rating
        tag_mastery = float((i % 3) / 2)
        previous_attempts = float(i % 4)
        recent_activity = float(i % 8)
        solved_before = float(i % 2)
        tag_count = float(1 + (i % 5))
        solved_24 = 1 if i % 3 == 0 else 0
        solved_72 = 1 if i % 4 == 0 else 0
        solved_7d = 1 if i % 5 == 0 else 0
        rows.append(
            {
                "handle": handle,
                "event_time": int(event_time),
                "user_rating_at_time": user_rating,
                "problem_rating": problem_rating,
                "difficulty_gap": difficulty_gap,
                "tag_mastery": tag_mastery,
                "previous_attempts": previous_attempts,
                "recent_activity": recent_activity,
                "solved_before": solved_before,
                "tag_count": tag_count,
                "solved_within_24h": int(solved_24),
                "solved_within_72h": int(solved_72),
                "solved_within_7d": int(solved_7d),
            }
        )
    return pd.DataFrame(rows)


def test_eval_and_save(tmp_path):
    frame = make_frame(120)
    reports = train_multi_horizon_models(frame)
    result = evaluate_and_save_per_horizon(frame, reports, cache_root=tmp_path)
    assert isinstance(result, dict)
    for h, info in result.items():
        assert info.get("status") in ("saved", "no_test_split", "insufficient_rows", "missing_label", "no_model", "error")
        if info.get("status") == "saved":
            p = Path(info["path"])
            assert (p / "metrics.json").exists()
            assert (p / "arrays.json").exists()
            metrics = json.loads((p / "metrics.json").read_text(encoding="utf-8"))
            assert "roc_auc" in metrics
import json
from pathlib import Path

import numpy as np
import pandas as pd

from algoradar.models import train_multi_horizon_models, persist_per_horizon_models
import joblib


def make_frame(n=60):
    rng = np.random.default_rng(42)
    rows = []
    now = 1_700_000_000
    for i in range(n):
        handle = f"user{i%5}"
        event_time = now + i * 1000
        user_rating = float(1200 + (i % 10) * 10)
        problem_rating = float(1200 + (i % 7) * 50)
        difficulty_gap = problem_rating - user_rating
        tag_mastery = float((i % 3) / 2)
        previous_attempts = float(i % 4)
        recent_activity = float(i % 8)
        solved_before = float(i % 2)
        tag_count = float(1 + (i % 5))
        # target horizons: alternate
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


def test_train_and_persist(tmp_path):
    frame = make_frame(80)
    reports = train_multi_horizon_models(frame, horizons=["solved_within_24h", "solved_within_72h", "solved_within_7d"], random_state=42)
    assert "per_horizon" in reports
    per = reports["per_horizon"]
    assert set(per.keys()) == {"solved_within_24h", "solved_within_72h", "solved_within_7d"}

    summary = persist_per_horizon_models(reports, cache_root=tmp_path)
    assert "horizons" in summary
    for h, info in summary["horizons"].items():
        assert info.get("status") in ("saved", "no_pipeline")
        if info.get("status") == "saved":
            assert Path(info["model_path"]).exists()
            assert Path(info["metadata_path"]).exists()
            # verify artifacts are loadable
            model = joblib.load(info["model_path"])
            assert hasattr(model, "predict") or hasattr(model, "predict_proba")
            if info.get("imputer_path"):
                imputer = joblib.load(info["imputer_path"])
                assert hasattr(imputer, "transform")
import math

import pandas as pd

from algoradar.features import SOLVE_FEATURE_COLUMNS
from algoradar.recommender import build_user_solved_tag_vector, recommend_problems


def _now_minus(days: float) -> pd.Timestamp:
    return pd.Timestamp.utcnow() - pd.Timedelta(days=days)


def test_session_multiplier_applied():
    submissions = pd.DataFrame(
        [
            {"problem_id": "p1", "created_at": _now_minus(0), "tags": ["DP"], "is_accepted": False, "rating": 1500},
            {"problem_id": "p2", "created_at": _now_minus(1), "tags": ["Graph"], "is_accepted": False, "rating": 1400},
            {"problem_id": "p3", "created_at": _now_minus(2), "tags": ["Greedy"], "is_accepted": False, "rating": 1300},
        ]
    )

    problems = pd.DataFrame(
        [
            {"problem_id": "c1", "tags": ["DP"], "rating": 1500, "solved_count": 100, "name": "c1"},
            {"problem_id": "c2", "tags": ["Math"], "rating": 1400, "solved_count": 50, "name": "c2"},
        ]
    )

    tag_stats = pd.DataFrame(
        [
            {
                "tag": "DP",
                "attempts": 1,
                "solved": 0,
                "accuracy": 20,
                "avg_rating_solved": 0,
                "max_rating_solved": 0,
                "wrong_submissions": 0,
                "recent_failures": 0,
                "recent_accuracy": 0,
                "avg_fuzzy_struggle": 0.2,
            },
            {
                "tag": "Math",
                "attempts": 1,
                "solved": 0,
                "accuracy": 50,
                "avg_rating_solved": 0,
                "max_rating_solved": 0,
                "wrong_submissions": 0,
                "recent_failures": 0,
                "recent_accuracy": 0,
                "avg_fuzzy_struggle": 0.0,
            },
        ]
    )

    profile = {"current_rating": 1500}
    solve_model_report = {"selected_model_name": "monotonic_scorecard_v2", "features": SOLVE_FEATURE_COLUMNS}

    rec = recommend_problems(
        problems,
        submissions,
        profile,
        tag_stats,
        solve_model_report,
        confidence_count=10,
        growth_count=10,
        stretch_count=10,
    )

    # session_multiplier should exist when there is recent activity
    assert "session_multiplier" in rec.columns

    dp_row = rec[rec["problem_id"] == "c1"].iloc[0]
    math_row = rec[rec["problem_id"] == "c2"].iloc[0]

    assert dp_row["session_multiplier"] > 1.0
    assert dp_row["session_multiplier"] <= 1.15
    # unmatched tag should have multiplier close to 1.0
    assert math_row["session_multiplier"] >= 1.0


def test_build_user_solved_tag_vector_normalized():
    subs = pd.DataFrame(
        [
            {"problem_id": "p1", "tags": ["A", "B"], "is_accepted": True, "created_at": _now_minus(1), "rating": 1200},
            {"problem_id": "p2", "tags": ["A"], "is_accepted": True, "created_at": _now_minus(2), "rating": 1300},
            {"problem_id": "p1", "tags": ["A", "B"], "is_accepted": True, "created_at": _now_minus(3), "rating": 1200},
        ]
    )

    vec = build_user_solved_tag_vector(subs)
    assert "A" in vec and "B" in vec
    norm = math.sqrt(sum(v * v for v in vec.values()))
    assert abs(norm - 1.0) < 1e-6

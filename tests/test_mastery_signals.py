import pandas as pd

from algoradar.features import make_problem_feature_row, tag_feature_frame


def _mk_submission(pid, tags, days_ago, accepted=True, rating=1200):
    return {
        "submission_id": f"s_{pid}_{days_ago}",
        "created_at": pd.Timestamp.utcnow() - pd.Timedelta(days=days_ago),
        "problem_id": pid,
        "tags": tags,
        "rating": rating,
        "is_accepted": accepted,
        "is_wrong": not accepted,
    }


def test_long_term_mastery_vs_recent():
    # create old accepted submissions (>40 days) and recent wrong submissions
    subs = pd.DataFrame(
        [
            _mk_submission("p1", ["X"], days_ago=120, accepted=True),
            _mk_submission("p2", ["X"], days_ago=110, accepted=True),
            _mk_submission("p3", ["X"], days_ago=10, accepted=False),
            _mk_submission("p4", ["X"], days_ago=5, accepted=False),
        ]
    )

    tag_stats = tag_feature_frame(subs)
    row = tag_stats[tag_stats["tag"] == "X"].iloc[0]
    # long-term mastery should reflect older accepted submissions and be higher than recent accuracy
    assert "long_term_mastery" in tag_stats.columns
    assert row["long_term_mastery"] >= 0.0
    assert row["recent_accuracy"] <= row["long_term_mastery"]


def test_make_problem_feature_row_includes_mastery():
    subs = pd.DataFrame([
        _mk_submission("p1", ["A"], days_ago=60, accepted=True),
    ])
    tag_stats = tag_feature_frame(subs)
    profile = {"current_rating": 1200, "recent_accuracy": 100.0}
    problem = pd.Series({"problem_id": "x", "tags": ["A"], "rating": 1300})
    features = make_problem_feature_row(problem, profile, tag_stats)
    assert "recent_mastery" in features
    assert "long_term_mastery" in features
    assert features["recent_mastery"] >= 0.0
    assert features["long_term_mastery"] >= 0.0

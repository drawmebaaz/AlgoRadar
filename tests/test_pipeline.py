from __future__ import annotations

import pandas as pd

from algoradar.features import (
    make_problem_feature_row,
    problemset_to_frame,
    submissions_to_frame,
    tag_feature_frame,
)
from algoradar.models import train_contest_score_predictor, train_solve_probability_model
from algoradar.pipeline import run_analysis
from algoradar.recommender import score_custom_problem
from algoradar.sample_data import make_sample_bundle


def test_sample_submissions_become_feature_frame() -> None:
    bundle = make_sample_bundle("unit_test")
    submissions = submissions_to_frame(bundle["submissions"])
    tags = tag_feature_frame(submissions)

    assert not submissions.empty
    assert {"problem_id", "rating", "tags", "is_accepted", "verdict"}.issubset(submissions.columns)
    assert not tags.empty
    assert {"tag", "attempts", "accuracy", "recent_failures"}.issubset(tags.columns)


def test_full_pipeline_returns_models_and_recommendations() -> None:
    result = run_analysis("unit_test", prefer_transformer=False, submission_limit=600, use_sample=True)

    assert result.profile["problems_solved"] > 0
    assert result.contest_model["predicted_band"]
    assert result.solve_model["training_rows"] > 0
    assert not result.weakness.empty
    assert not result.recommendations.empty
    assert result.semantic_method in {"tfidf-fallback", "sentence-transformers/all-MiniLM-L6-v2"}


def test_custom_problem_probability_is_bucketed() -> None:
    result = run_analysis("probability_test", prefer_transformer=False, submission_limit=600, use_sample=True)
    score = score_custom_problem(
        rating=1500,
        tags=["dp", "greedy"],
        solved_count=5000,
        name="Unit custom problem",
        profile=result.profile,
        tag_stats=result.weakness,
        solve_model_report=result.solve_model,
    )

    assert 0 <= score["solve_probability"] <= 1
    assert score["bucket"] in {"confidence", "growth", "stretch", "avoid"}
    assert score["recent_failures_used"] >= 0


def test_problem_rating_is_estimated_when_codeforces_rating_is_missing() -> None:
    problems = [{"contestId": 1900, "index": "C", "name": "Unrated C", "tags": ["dp"]}]
    stats = [{"contestId": 1900, "index": "C", "solvedCount": 750}]

    frame = problemset_to_frame(problems, stats)

    assert frame.iloc[0]["rating_source"] == "estimated"
    assert 800 <= frame.iloc[0]["rating"] <= 3500


def test_solve_probability_decreases_as_problem_rating_increases() -> None:
    result = run_analysis("monotonic_test", prefer_transformer=False, submission_limit=600, use_sample=True)
    scores = [
        score_custom_problem(
            rating=rating,
            tags=["dp", "greedy"],
            solved_count=5000,
            name=f"Rating {rating}",
            profile=result.profile,
            tag_stats=result.weakness,
            solve_model_report=result.solve_model,
        )["solve_probability"]
        for rating in [1200, 1600, 2000, 2400]
    ]

    assert scores == sorted(scores, reverse=True)


def test_contest_predictor_handles_imbalanced_training_bands() -> None:
    profile = {
        "problems_solved": 1300,
        "average_rating": 1950,
        "tags_attempted": 22,
        "wrong_submissions": 650,
        "submissions": 2500,
        "current_rating": 3800,
        "max_rating": 3900,
        "contest_count": 120,
        "contest_rank_mean_last5": 300,
        "contest_rank_best": 1,
        "rating_volatility": 85,
        "recent_accuracy": 60,
    }

    report = train_contest_score_predictor(profile)

    assert report["predicted_band"]
    assert report["selected_model_name"] in {"contest_scorecard", "random_forest", "logistic_regression", "constant_baseline"}


def test_problem_feature_row_includes_strength_signals() -> None:
    profile = {"current_rating": 1200, "recent_accuracy": 55, "problems_solved": 80}
    tag_stats = tag_feature_frame(
        submissions_to_frame(make_sample_bundle("feature_unit")["submissions"])
    )
    problem = problemset_to_frame(
        [{"contestId": 1, "index": "C", "name": "Tagged", "rating": 1400, "tags": ["dp"]}],
        [{"contestId": 1, "index": "C", "solvedCount": 2500}],
    ).iloc[0]

    features = make_problem_feature_row(problem, profile, tag_stats)

    assert features["tag_solved_count"] >= 0
    assert features["tag_avg_rating_solved"] >= 0
    assert features["tag_max_rating_solved"] >= 0
    assert features["solved_volume_log"] > 0
    assert features["rating_confidence"] == 1.0


def test_real_label_dataset_trains_logistic_model() -> None:
    rows = [
        {
            "user_rating_at_time": 1200,
            "problem_rating": 1200,
            "difficulty_gap": 0,
            "tag_mastery": 0.7,
            "previous_attempts": 0,
            "recent_activity": 1,
            "solved_before": 0,
            "tag_count": 2,
            "y": 1,
        },
        {
            "user_rating_at_time": 1200,
            "problem_rating": 1800,
            "difficulty_gap": 600,
            "tag_mastery": 0.2,
            "previous_attempts": 3,
            "recent_activity": 4,
            "solved_before": 0,
            "tag_count": 1,
            "y": 0,
        },
        {
            "user_rating_at_time": 1350,
            "problem_rating": 1400,
            "difficulty_gap": 50,
            "tag_mastery": 0.8,
            "previous_attempts": 1,
            "recent_activity": 2,
            "solved_before": 1,
            "tag_count": 2,
            "y": 1,
        },
        {
            "user_rating_at_time": 1350,
            "problem_rating": 2200,
            "difficulty_gap": 850,
            "tag_mastery": 0.1,
            "previous_attempts": 5,
            "recent_activity": 5,
            "solved_before": 0,
            "tag_count": 3,
            "y": 0,
        },
        {
            "user_rating_at_time": 1400,
            "problem_rating": 1500,
            "difficulty_gap": 100,
            "tag_mastery": 0.9,
            "previous_attempts": 0,
            "recent_activity": 1,
            "solved_before": 1,
            "tag_count": 2,
            "y": 1,
        },
        {
            "user_rating_at_time": 1400,
            "problem_rating": 2000,
            "difficulty_gap": 600,
            "tag_mastery": 0.25,
            "previous_attempts": 2,
            "recent_activity": 3,
            "solved_before": 0,
            "tag_count": 2,
            "y": 0,
        },
    ]

    report = train_solve_probability_model(pd.DataFrame(rows))

    assert report["selected_model_name"] == "logistic_regression_real_labels"
    assert report["metrics"]["roc_auc"] >= 0.5
    assert "precision@5" in report["metrics"]
    assert "recall@5" in report["metrics"]
    assert "brier_score" in report["metrics"]

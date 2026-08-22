from __future__ import annotations

from algoradar.features import (
    make_problem_feature_row,
    problemset_to_frame,
    submissions_to_frame,
    tag_feature_frame,
)
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


def test_full_pipeline_returns_model_and_recommendations() -> None:
    result = run_analysis("unit_test", submission_limit=600, use_sample=True)

    assert result.profile["problems_solved"] > 0
    assert result.solve_model["selected_model_name"] == "solve_probability_scorecard"
    assert not result.weakness.empty
    assert not result.recommendations.empty


def test_custom_problem_probability_is_bucketed() -> None:
    result = run_analysis("probability_test", submission_limit=600, use_sample=True)
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
    result = run_analysis("monotonic_test", submission_limit=600, use_sample=True)
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

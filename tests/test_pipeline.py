from __future__ import annotations

from algoradar.features import submissions_to_frame, tag_feature_frame
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
        recent_failures=4,
    )

    assert 0 <= score["solve_probability"] <= 1
    assert score["bucket"] in {"confidence", "growth", "stretch", "avoid"}

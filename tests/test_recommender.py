from __future__ import annotations

import pytest
import pandas as pd

from algoradar.features import SOLVE_FEATURE_COLUMNS
from algoradar.recommender import (
    build_user_solved_tag_vector,
    gaussian_rating_fit,
    recommend_problems,
    tag_vector_cosine_similarity,
)


def test_tag_cosine_similarity_is_higher_for_matching_tags() -> None:
    submissions = pd.DataFrame(
        [
            {"problem_id": "1A", "tags": ["dp", "graphs"], "is_accepted": True},
            {"problem_id": "2B", "tags": ["dp"], "is_accepted": True},
            {"problem_id": "3C", "tags": ["graphs"], "is_accepted": True},
        ]
    )

    vector = build_user_solved_tag_vector(submissions)
    matching = tag_vector_cosine_similarity(["dp", "graphs"], vector)
    unrelated = tag_vector_cosine_similarity(["geometry"], vector)

    assert matching > unrelated
    assert 0 <= matching <= 1
    assert unrelated == 0.0


def test_tag_cosine_similarity_handles_empty_user_history() -> None:
    submissions = pd.DataFrame(columns=["problem_id", "tags", "is_accepted"])

    vector = build_user_solved_tag_vector(submissions)

    assert vector == {}
    assert tag_vector_cosine_similarity(["dp"], vector) == 0.0


def test_rating_fit_is_highest_at_user_rating() -> None:
    exact = gaussian_rating_fit(1200, 1200)

    assert exact > gaussian_rating_fit(1600, 1200)
    assert exact > gaussian_rating_fit(800, 1200)
    assert exact == pytest.approx(1.0)


def test_rating_fit_is_symmetric_around_user_rating() -> None:
    assert gaussian_rating_fit(1000, 1200) == pytest.approx(gaussian_rating_fit(1400, 1200))


def test_recommender_still_filters_solved_problems() -> None:
    problems = pd.DataFrame(
        [
            {
                "problem_id": "1A",
                "contest_id": 1,
                "index": "A",
                "name": "Solved Warmup",
                "official_rating": 1200.0,
                "rating": 1200.0,
                "rating_source": "official",
                "tags": ["dp"],
                "tag_count": 1,
                "solved_count": 5000,
            },
            {
                "problem_id": "2B",
                "contest_id": 2,
                "index": "B",
                "name": "Unsolved Growth",
                "official_rating": 1300.0,
                "rating": 1300.0,
                "rating_source": "official",
                "tags": ["dp", "graphs"],
                "tag_count": 2,
                "solved_count": 4200,
            },
            {
                "problem_id": "3C",
                "contest_id": 3,
                "index": "C",
                "name": "Unsolved Geometry",
                "official_rating": 1500.0,
                "rating": 1500.0,
                "rating_source": "official",
                "tags": ["geometry"],
                "tag_count": 1,
                "solved_count": 1800,
            },
        ]
    )
    submissions = pd.DataFrame(
        [
            {"submission_id": 1, "problem_id": "1A", "tags": ["dp"], "rating": 1200, "is_accepted": True, "is_wrong": False},
            {"submission_id": 2, "problem_id": "4D", "tags": ["graphs"], "rating": 1100, "is_accepted": True, "is_wrong": False},
        ]
    )
    tag_stats = pd.DataFrame(
        [
            {
                "tag": "dp",
                "level": "Weak",
                "priority_score": 95,
                "attempts": 12,
                "solved": 6,
                "accuracy": 50.0,
                "avg_rating_solved": 1150.0,
                "max_rating_solved": 1300.0,
                "recent_failures": 2,
            },
            {
                "tag": "graphs",
                "level": "Weak",
                "priority_score": 80,
                "attempts": 8,
                "solved": 4,
                "accuracy": 50.0,
                "avg_rating_solved": 1100.0,
                "max_rating_solved": 1250.0,
                "recent_failures": 1,
            },
        ]
    )
    profile = {"current_rating": 1200, "recent_accuracy": 55, "problems_solved": 80}
    solve_model = {"selected_model_name": "monotonic_scorecard_v2", "model": None, "features": SOLVE_FEATURE_COLUMNS}

    recs = recommend_problems(
        problems,
        submissions,
        profile,
        tag_stats,
        solve_model,
        confidence_count=1,
        growth_count=1,
        stretch_count=1,
        candidate_limit=10,
    )

    assert "1A" not in set(recs["problem_id"])
    assert {"tag_cosine_similarity", "rating_fit_score"}.issubset(recs.columns)
    assert recs["tag_cosine_similarity"].between(0, 1).all()
    assert recs["rating_fit_score"].between(0, 1).all()

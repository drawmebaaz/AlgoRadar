from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .codeforces import CodeforcesClient
from .config import DEFAULT_SUBMISSION_LIMIT
from .features import (
    build_solve_examples,
    contest_trend_frame,
    problemset_to_frame,
    rating_accuracy_frame,
    rating_history_to_frame,
    solved_difficulty_frame,
    submissions_to_frame,
    tag_feature_frame,
    user_profile_features,
    verdict_frame,
)
from .models import train_solve_probability_model
from .sample_data import make_sample_bundle
from .weakness import classify_weakness


@dataclass
class AnalysisResult:
    handle: str
    source: str
    profile: dict[str, Any]
    submissions: pd.DataFrame
    ratings: pd.DataFrame
    problems: pd.DataFrame
    rating_accuracy: pd.DataFrame
    tag_stats: pd.DataFrame
    weakness: pd.DataFrame
    verdicts: pd.DataFrame
    solved_difficulty: pd.DataFrame
    contest_trend: pd.DataFrame
    solve_examples: pd.DataFrame
    solve_model: dict[str, Any]
    recommendations: pd.DataFrame
    progress: pd.DataFrame


def run_analysis(
    handle: str,
    force_refresh: bool = False,
    submission_limit: int = DEFAULT_SUBMISSION_LIMIT,
    use_sample: bool = False,
    include_recommendations: bool = True,
) -> AnalysisResult:
    if use_sample:
        bundle = make_sample_bundle(handle)
    else:
        client = CodeforcesClient()
        bundle = client.bundle(handle, count=submission_limit, force_refresh=force_refresh)

    submissions = submissions_to_frame(bundle["submissions"])
    ratings = rating_history_to_frame(bundle["ratings"])
    problems = problemset_to_frame(bundle["problems"], bundle["problem_statistics"])
    tag_stats = tag_feature_frame(submissions, problems)
    profile = user_profile_features(submissions, ratings)
    profile["handle"] = handle

    weakness = classify_weakness(tag_stats)

    solve_examples = build_solve_examples(submissions, ratings, problems, weakness)
    solve_model = train_solve_probability_model(solve_examples)

    if include_recommendations:
        from .recommender import recommend_problems

        recommendations = recommend_problems(problems, submissions, profile, weakness, solve_model)
    else:
        recommendations = pd.DataFrame()

    return AnalysisResult(
        handle=handle,
        source=bundle["source"],
        profile=profile,
        submissions=submissions,
        ratings=ratings,
        problems=problems,
        rating_accuracy=rating_accuracy_frame(submissions),
        tag_stats=tag_stats,
        weakness=weakness,
        verdicts=verdict_frame(submissions),
        solved_difficulty=solved_difficulty_frame(submissions),
        contest_trend=contest_trend_frame(ratings),
        solve_examples=solve_examples,
        solve_model=solve_model,
        recommendations=recommendations,
        progress=build_progress_frame(submissions, profile),
    )


def build_progress_frame(submissions: pd.DataFrame, profile: dict[str, Any] | None = None) -> pd.DataFrame:
    if submissions.empty:
        return pd.DataFrame(columns=["week", "solved", "attempts", "accuracy", "growth_attempts"])

    frame = submissions.copy()
    frame["week"] = frame["created_at"].dt.tz_convert(None).dt.to_period("W").astype(str)
    profile = profile or {}
    growth_low = float(profile.get("growth_rating_low", 1400))
    growth_high = float(profile.get("growth_rating_high", 1900))
    frame["growth_attempt"] = frame["rating"].between(growth_low, growth_high)
    progress = (
        frame.groupby("week")
        .agg(
            solved=("is_accepted", "sum"),
            attempts=("submission_id", "count"),
            accuracy=("is_accepted", "mean"),
            growth_attempts=("growth_attempt", "sum"),
        )
        .reset_index()
        .tail(10)
    )
    progress["accuracy"] = (progress["accuracy"] * 100).round(1)
    return progress

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
from .models import train_contest_score_predictor, train_solve_probability_model
from .recommender import recommend_problems
from .sample_data import make_sample_bundle
from .semantic import build_semantic_index, similar_problems
from .weakness import classify_weakness, predict_weakness_with_model, train_weakness_model


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
    contest_model: dict[str, Any]
    solve_model: dict[str, Any]
    weakness_model: dict[str, Any]
    recommendations: pd.DataFrame
    semantic_method: str
    similar_harder: pd.DataFrame
    roadmap: pd.DataFrame
    progress: pd.DataFrame


def run_analysis(
    handle: str,
    force_refresh: bool = False,
    submission_limit: int = DEFAULT_SUBMISSION_LIMIT,
    prefer_transformer: bool = False,
    use_sample: bool = False,
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
    weakness_model = train_weakness_model()
    weakness = predict_weakness_with_model(weakness, weakness_model)

    solve_examples = build_solve_examples(submissions, ratings, problems, weakness)
    contest_model = train_contest_score_predictor(profile)
    solve_model = train_solve_probability_model(solve_examples)
    recommendations = recommend_problems(problems, submissions, profile, weakness, solve_model)

    semantic_index = build_semantic_index(problems, prefer_transformer=prefer_transformer)
    if not recommendations.empty:
        reference_id = recommendations.iloc[0]["problem_id"]
        reference = problems[problems["problem_id"] == reference_id].iloc[0]
        similar_harder = similar_problems(reference, problems, semantic_index, top_n=8, harder_only=True)
    else:
        similar_harder = pd.DataFrame()

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
        contest_model=contest_model,
        solve_model=solve_model,
        weakness_model=weakness_model,
        recommendations=recommendations,
        semantic_method=semantic_index.method,
        similar_harder=similar_harder,
        roadmap=build_weekly_roadmap(weakness, recommendations),
        progress=build_progress_frame(submissions, profile),
    )


def build_weekly_roadmap(weakness: pd.DataFrame, recommendations: pd.DataFrame) -> pd.DataFrame:
    focus_tags = weakness.head(5)["tag"].tolist() if not weakness.empty else ["implementation", "math", "greedy"]
    growth_count = int((recommendations["bucket"] == "growth").sum()) if not recommendations.empty else 0
    stretch_count = int((recommendations["bucket"] == "stretch").sum()) if not recommendations.empty else 0
    days = [
        ("Mon", "Calibration", f"2 confidence problems, tag audit: {focus_tags[0] if focus_tags else 'implementation'}", 42),
        ("Tue", "Weakness drill", f"Focused practice on {focus_tags[1] if len(focus_tags) > 1 else focus_tags[0]}", 68),
        ("Wed", "Growth queue", f"{max(2, growth_count // 3)} growth problems with notes", 58),
        ("Thu", "Contest sim", "A-C sprint, 90 minutes, no editorials", 74),
        ("Fri", "Repair block", f"Review failures in {focus_tags[2] if len(focus_tags) > 2 else focus_tags[0]}", 62),
        ("Sat", "Stretch work", f"{max(1, stretch_count // 2)} stretch attempts plus upsolve", 82),
        ("Sun", "Review", "Progress log, flashcards, next queue pruning", 35),
    ]
    return pd.DataFrame(days, columns=["day", "theme", "focus", "load"])


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

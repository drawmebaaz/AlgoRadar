from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .features import SOLVE_FEATURE_COLUMNS, make_problem_feature_row
from .models import bucket_probability, predict_solve_probability
from .semantic import normalize, tag_similarity_score


def recommend_problems(
    problems: pd.DataFrame,
    submissions: pd.DataFrame,
    profile: dict[str, Any],
    tag_stats: pd.DataFrame,
    solve_model_report: dict[str, Any],
    confidence_count: int = 5,
    growth_count: int = 10,
    stretch_count: int = 5,
    candidate_limit: int = 1200,
) -> pd.DataFrame:
    if problems.empty:
        return pd.DataFrame()

    solved_ids = set(submissions[submissions["is_accepted"]]["problem_id"].unique()) if not submissions.empty else set()
    candidate = problems[~problems["problem_id"].isin(solved_ids)].copy()
    candidate = candidate[candidate["rating"].between(800, 3500)].copy()
    if candidate.empty:
        candidate = problems.copy()

    weak_tags = (
        tag_stats[tag_stats.get("level", pd.Series(dtype=str)).isin(["Weak", "Over-attempted but low accuracy", "Untouched"])]
        if "level" in tag_stats.columns
        else tag_stats
    )
    target_tags = weak_tags.sort_values("priority_score", ascending=False)["tag"].head(8).tolist() if "priority_score" in weak_tags.columns else []
    candidate = _prefilter_candidates(candidate, profile, target_tags, candidate_limit)

    feature_rows = [make_problem_feature_row(row, profile, tag_stats) for _, row in candidate.iterrows()]
    feature_frame = pd.DataFrame(feature_rows)
    probabilities = predict_solve_probability(solve_model_report, feature_frame[SOLVE_FEATURE_COLUMNS])
    candidate["solve_probability"] = probabilities
    candidate["bucket"] = [bucket_probability(probability) for probability in probabilities]
    candidate["rating_distance"] = (candidate["rating"] - float(profile.get("current_rating", 1200))).abs()
    candidate["tag_similarity"] = candidate["tags"].apply(lambda tags: tag_similarity_score(tags, target_tags))
    candidate["popularity_score"] = normalize(np.log1p(candidate["solved_count"].fillna(0)))

    probability = candidate["solve_probability"]
    growth_center = 0.6
    candidate["learning_value"] = (
        1.2 * (1 - (probability - growth_center).abs())
        + 0.42 * candidate["tag_similarity"]
        + 0.22 * candidate["popularity_score"]
        - 0.00038 * candidate["rating_distance"]
    )
    candidate["rank_score"] = candidate["learning_value"] + candidate["solve_probability"] * 0.25

    confidence = _pick_bucket(candidate, "confidence", confidence_count, sort_by=["solve_probability", "rank_score"])
    growth = _pick_bucket(candidate, "growth", growth_count, sort_by=["rank_score", "tag_similarity"])
    stretch = _pick_bucket(candidate, "stretch", stretch_count, sort_by=["rank_score", "tag_similarity"])

    recommendations = pd.concat([confidence, growth, stretch], ignore_index=True)
    if recommendations.empty:
        return recommendations
    recommendations["solve_probability_pct"] = (recommendations["solve_probability"] * 100).round(1)
    recommendations["rating"] = recommendations["rating"].astype(int)
    return recommendations.reset_index(drop=True)


def score_custom_problem(
    rating: int,
    tags: list[str],
    solved_count: int,
    name: str,
    profile: dict[str, Any],
    tag_stats: pd.DataFrame,
    solve_model_report: dict[str, Any],
    recent_failures: float | None = None,
) -> dict[str, Any]:
    problem = pd.Series(
        {
            "problem_id": "custom",
            "name": name or "Custom problem",
            "rating": rating,
            "tags": tags,
            "tag_count": len(tags),
            "solved_count": solved_count,
        }
    )
    inferred_recent_failures = _infer_recent_failures(tags, tag_stats) if recent_failures is None else recent_failures
    features = make_problem_feature_row(problem, profile, tag_stats, recent_failures_override=inferred_recent_failures)
    probability = float(predict_solve_probability(solve_model_report, pd.DataFrame([features]))[0])
    return {
        "problem_id": "custom",
        "name": problem["name"],
        "rating": rating,
        "tags": tags,
        "solved_count": solved_count,
        "solve_probability": probability,
        "solve_probability_pct": round(probability * 100, 1),
        "bucket": bucket_probability(probability),
        "features": features,
        "recent_failures_used": round(float(inferred_recent_failures), 1),
    }


def _pick_bucket(candidate: pd.DataFrame, bucket: str, count: int, sort_by: list[str]) -> pd.DataFrame:
    frame = candidate[candidate["bucket"] == bucket].copy()
    if frame.empty and bucket == "confidence":
        frame = candidate[candidate["solve_probability"] > 0.68].copy()
    if frame.empty and bucket == "growth":
        frame = candidate[candidate["solve_probability"].between(0.38, 0.78)].copy()
    if frame.empty and bucket == "stretch":
        frame = candidate[candidate["solve_probability"].between(0.18, 0.5)].copy()
    return frame.sort_values(sort_by, ascending=False).head(count)


def _infer_recent_failures(tags: list[str], tag_stats: pd.DataFrame) -> float:
    if not tags or tag_stats.empty or "recent_failures" not in tag_stats.columns:
        return 0.0
    lookup = tag_stats.set_index("tag")["recent_failures"].to_dict()
    values = [float(lookup.get(tag, 0.0) or 0.0) for tag in tags]
    return float(np.mean(values)) if values else 0.0


def _prefilter_candidates(
    candidate: pd.DataFrame,
    profile: dict[str, Any],
    target_tags: list[str],
    candidate_limit: int,
) -> pd.DataFrame:
    if candidate.empty:
        return candidate

    user_rating = float(profile.get("current_rating", 1200) or 1200)
    if user_rating >= 3000:
        lower, upper = max(2200, int((user_rating - 900) // 100 * 100)), 3500
    elif user_rating >= 2400:
        lower, upper = 1800, 3500
    else:
        lower = max(800, int((user_rating - 500) // 100 * 100))
        upper = min(3500, int((user_rating + 700) // 100 * 100))

    frame = candidate[candidate["rating"].between(lower, upper)].copy()
    if len(frame) < min(500, len(candidate)):
        fallback_lower = max(800, int((user_rating - 800) // 100 * 100))
        fallback_upper = min(3500, int((user_rating + 900) // 100 * 100))
        frame = candidate[candidate["rating"].between(fallback_lower, fallback_upper)].copy()
    if frame.empty:
        frame = candidate.copy()

    frame["prefilter_tag_score"] = frame["tags"].apply(lambda tags: tag_similarity_score(tags, target_tags))
    frame["prefilter_rating_distance"] = (frame["rating"] - user_rating).abs()
    frame["prefilter_score"] = (
        frame["prefilter_tag_score"] * 2.2
        + np.log1p(frame["solved_count"].fillna(0)) * 0.16
        - frame["prefilter_rating_distance"] * 0.00055
    )
    return frame.sort_values("prefilter_score", ascending=False).head(candidate_limit).reset_index(drop=True)

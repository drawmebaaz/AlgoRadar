from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .features import SOLVE_FEATURE_COLUMNS, make_problem_feature_row
from .models import bucket_probability, predict_solve_probability
from .semantic import normalize, tag_similarity_score

TAG_COSINE_WEIGHT = 0.12
RATING_FIT_WEIGHT = 0.14


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
    target_weights = _target_tag_weights(weak_tags)
    target_tags = list(target_weights)
    candidate = _prefilter_candidates(candidate, profile, target_tags, candidate_limit)

    feature_rows = [make_problem_feature_row(row, profile, tag_stats) for _, row in candidate.iterrows()]
    feature_frame = pd.DataFrame(feature_rows)
    probabilities = predict_solve_probability(solve_model_report, feature_frame[SOLVE_FEATURE_COLUMNS])
    candidate["solve_probability"] = probabilities
    candidate["bucket"] = [bucket_probability(probability) for probability in probabilities]
    candidate["probability_bucket"] = candidate["bucket"]
    candidate["rating_distance"] = (candidate["rating"] - float(profile.get("current_rating", 1200))).abs()
    candidate["tag_similarity"] = candidate["tags"].apply(lambda tags: _weighted_tag_score(tags, target_weights))
    user_tag_vector = build_user_solved_tag_vector(submissions)
    user_rating = float(profile.get("current_rating", 1200) or 1200)
    candidate["tag_cosine_similarity"] = candidate["tags"].apply(lambda tags: tag_vector_cosine_similarity(tags, user_tag_vector))
    candidate["rating_fit_score"] = candidate["rating"].apply(lambda rating: gaussian_rating_fit(rating, user_rating))
    candidate["tag_solved_count"] = feature_frame["tag_solved_count"].to_numpy()
    candidate["tag_ceiling_gap"] = feature_frame["tag_max_rating_solved"].to_numpy() - candidate["rating"].astype(float)
    candidate["rating_confidence"] = feature_frame["rating_confidence"].to_numpy()
    candidate["popularity_score"] = normalize(np.log1p(candidate["solved_count"].fillna(0)))
    candidate["evidence_score"] = normalize(np.log1p(candidate["tag_solved_count"].fillna(0)))
    candidate["ceiling_score"] = 1 / (1 + np.exp(-(candidate["tag_ceiling_gap"] + 100) / 300))

    probability = candidate["solve_probability"]
    growth_center = 0.6
    growth_fit = (1 - (probability - growth_center).abs() / 0.6).clip(lower=0)
    quality_score = candidate["popularity_score"] * candidate["rating_confidence"]
    candidate["learning_value"] = (
        1.15 * growth_fit
        + 0.52 * candidate["tag_similarity"]
        + 0.32 * candidate["ceiling_score"]
        + 0.22 * candidate["evidence_score"]
        + 0.18 * quality_score
        + TAG_COSINE_WEIGHT * candidate["tag_cosine_similarity"]
        + RATING_FIT_WEIGHT * candidate["rating_fit_score"]
        - 0.00028 * candidate["rating_distance"]
    )
    candidate["rank_score"] = candidate["learning_value"] + candidate["solve_probability"] * 0.16

    selected_ids: set[str] = set()
    user_rating = float(profile.get("current_rating", 1200) or 1200)
    confidence = _pick_bucket(candidate, "confidence", confidence_count, sort_by=["solve_probability", "rank_score"], excluded_ids=selected_ids, user_rating=user_rating)
    selected_ids.update(confidence["problem_id"].astype(str).tolist())
    growth = _pick_bucket(candidate, "growth", growth_count, sort_by=["rank_score", "tag_similarity"], excluded_ids=selected_ids, user_rating=user_rating)
    selected_ids.update(growth["problem_id"].astype(str).tolist())
    stretch = _pick_bucket(candidate, "stretch", stretch_count, sort_by=["rank_score", "tag_similarity"], excluded_ids=selected_ids, user_rating=user_rating)

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


def _pick_bucket(
    candidate: pd.DataFrame,
    bucket: str,
    count: int,
    sort_by: list[str],
    excluded_ids: set[str] | None = None,
    user_rating: float = 1200.0,
) -> pd.DataFrame:
    excluded_ids = excluded_ids or set()
    available = candidate[~candidate["problem_id"].astype(str).isin(excluded_ids)].copy()
    frame = available[available["probability_bucket"] == bucket].copy()
    if frame.empty and bucket == "confidence":
        frame = available[available["solve_probability"] > 0.68].copy()
    if frame.empty and bucket == "growth":
        frame = available[available["solve_probability"].between(0.38, 0.78)].copy()
    if frame.empty and bucket == "stretch":
        frame = available[available["solve_probability"].between(0.18, 0.5)].copy()
    sorted_frame = frame.sort_values(sort_by, ascending=False)
    picked = _diversified_head(sorted_frame, count)

    if len(picked) < count:
        picked_ids = set(picked["problem_id"].astype(str).tolist()) if not picked.empty else set()
        filler_pool = available[~available["problem_id"].astype(str).isin(picked_ids)].copy()
        filler_pool = _bucket_fallback_pool(filler_pool, bucket, user_rating)
        if not filler_pool.empty:
            target_probability = {"confidence": 0.82, "growth": 0.6, "stretch": 0.34}.get(bucket, 0.6)
            filler_pool["bucket_fit"] = (1 - (filler_pool["solve_probability"] - target_probability).abs() / 0.7).clip(lower=0)
            filler_pool["fallback_score"] = (
                filler_pool["bucket_fit"] * 0.78
                + filler_pool["rank_score"] * 0.28
                + filler_pool["tag_similarity"] * 0.18
                + filler_pool.get("tag_cosine_similarity", 0) * 0.08
                + filler_pool.get("rating_fit_score", 0) * 0.08
            )
            filler = _diversified_head(filler_pool.sort_values(["fallback_score", "rank_score"], ascending=False), count - len(picked))
            picked = pd.concat([picked, filler], ignore_index=True)

    if not picked.empty:
        picked = picked.copy()
        picked["bucket"] = bucket
    return picked.head(count)


def _bucket_fallback_pool(pool: pd.DataFrame, bucket: str, user_rating: float) -> pd.DataFrame:
    if pool.empty:
        return pool
    if bucket == "confidence":
        constrained = pool[(pool["solve_probability"] >= 0.62) & (pool["rating"] <= user_rating + 150)].copy()
        return constrained if not constrained.empty else pool[pool["solve_probability"] >= 0.62].copy()
    if bucket == "growth":
        constrained = pool[
            (pool["rating"].between(user_rating - 200, user_rating + 550))
            & (pool["solve_probability"].between(0.42, 0.86))
        ].copy()
        return constrained if not constrained.empty else pool[pool["rating"].between(user_rating - 250, user_rating + 650)].copy()
    if bucket == "stretch":
        constrained = pool[
            (pool["rating"] >= user_rating + 150)
            & (pool["solve_probability"].between(0.18, 0.62))
        ].copy()
        return constrained
    return pool


def _infer_recent_failures(tags: list[str], tag_stats: pd.DataFrame) -> float:
    if not tags or tag_stats.empty or "recent_failures" not in tag_stats.columns:
        return 0.0
    lookup = tag_stats.set_index("tag")["recent_failures"].to_dict()
    values = [float(lookup.get(tag, 0.0) or 0.0) for tag in tags]
    return float(np.mean(values)) if values else 0.0


def build_user_solved_tag_vector(submissions: pd.DataFrame) -> dict[str, float]:
    if submissions is None or submissions.empty or "tags" not in submissions.columns:
        return {}

    frame = submissions.copy()
    if "is_accepted" in frame.columns:
        frame = frame[frame["is_accepted"]]
    if "problem_id" in frame.columns:
        frame = frame.drop_duplicates("problem_id")
    if frame.empty:
        return {}

    counts: dict[str, float] = {}
    for tags in frame["tags"].tolist():
        for tag in _clean_tags(tags):
            counts[tag] = counts.get(tag, 0.0) + 1.0

    norm = float(np.sqrt(sum(value * value for value in counts.values())))
    if norm <= 0:
        return {}
    return {tag: value / norm for tag, value in counts.items()}


def tag_vector_cosine_similarity(problem_tags: Any, user_tag_vector: dict[str, float] | None) -> float:
    tags = set(_clean_tags(problem_tags))
    if not tags or not user_tag_vector:
        return 0.0

    problem_norm = float(np.sqrt(len(tags)))
    if problem_norm <= 0:
        return 0.0
    dot = sum(float(user_tag_vector.get(tag, 0.0) or 0.0) for tag in tags)
    return float(np.clip(dot / problem_norm, 0.0, 1.0))


def gaussian_rating_fit(problem_rating: Any, user_rating: Any) -> float:
    rating = pd.to_numeric(pd.Series([problem_rating]), errors="coerce").iloc[0]
    if pd.isna(rating):
        return 0.0

    user = pd.to_numeric(pd.Series([user_rating]), errors="coerce").iloc[0]
    sigma = float(user) * 0.3 if pd.notna(user) and float(user) > 0 else 300.0
    if sigma <= 0:
        sigma = 300.0

    score = np.exp(-((float(rating) - float(user if pd.notna(user) and float(user) > 0 else 1200.0)) ** 2) / (2 * sigma**2))
    return float(np.clip(score, 0.0, 1.0))


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


def _target_tag_weights(weak_tags: pd.DataFrame) -> dict[str, float]:
    if weak_tags.empty or "tag" not in weak_tags.columns:
        return {}
    frame = weak_tags.copy()
    if "priority_score" not in frame.columns:
        frame["priority_score"] = 50.0
    sort_columns = ["priority_score"]
    if "attempts" in frame.columns:
        sort_columns.append("attempts")
    frame = frame.sort_values(sort_columns, ascending=False).head(10)
    weights: dict[str, float] = {}
    max_priority = float(frame["priority_score"].max() or 1)
    for row in frame.to_dict("records"):
        tag = str(row.get("tag", "")).strip()
        if not tag:
            continue
        level = str(row.get("level", ""))
        level_boost = 1.25 if level in {"Weak", "Over-attempted but low accuracy"} else 0.85
        priority = float(row.get("priority_score", 0) or 0)
        weights[tag] = max(0.15, priority / max_priority) * level_boost
    return weights


def _weighted_tag_score(problem_tags: list[str], target_weights: dict[str, float]) -> float:
    if not problem_tags or not target_weights:
        return 0.0
    problem_set = set(problem_tags or [])
    matched = sum(weight for tag, weight in target_weights.items() if tag in problem_set)
    total = sum(target_weights.values()) or 1.0
    jaccard = tag_similarity_score(problem_tags, list(target_weights))
    return float(min(1.0, matched / total * 0.75 + jaccard * 0.25))


def _clean_tags(tags: Any) -> list[str]:
    if tags is None:
        return []
    if isinstance(tags, (list, tuple, set)):
        values = tags
    elif isinstance(tags, str):
        values = [item.strip() for item in tags.split(",")]
    else:
        return []
    return [str(tag).strip() for tag in values if str(tag).strip()]


def _diversified_head(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    if frame.empty or len(frame) <= count:
        return frame.head(count)

    chosen = []
    tag_counts: dict[str, int] = {}
    bucket_counts: dict[int, int] = {}
    for _, row in frame.iterrows():
        tags = row.get("tags", []) or []
        primary_tag = str(tags[0]) if tags else "untagged"
        rating_bucket = int(float(row.get("rating", 0) or 0) // 200 * 200)
        tag_limit = 3 if count >= 10 else 2
        bucket_limit = 4 if count >= 10 else 2
        if tag_counts.get(primary_tag, 0) >= tag_limit or bucket_counts.get(rating_bucket, 0) >= bucket_limit:
            continue
        chosen.append(row)
        tag_counts[primary_tag] = tag_counts.get(primary_tag, 0) + 1
        bucket_counts[rating_bucket] = bucket_counts.get(rating_bucket, 0) + 1
        if len(chosen) >= count:
            break

    if len(chosen) < count:
        chosen_ids = {str(row.get("problem_id")) for row in chosen}
        for _, row in frame.iterrows():
            if str(row.get("problem_id")) in chosen_ids:
                continue
            chosen.append(row)
            if len(chosen) >= count:
                break

    return pd.DataFrame(chosen).reset_index(drop=True)

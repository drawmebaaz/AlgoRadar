from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .sample_data import TAGS

SOLVE_FEATURE_COLUMNS = [
    "problem_rating",
    "user_rating",
    "rating_gap",
    "tag_accuracy",
    "attempts_on_tag",
    "tag_solved_count",
    "tag_avg_rating_solved",
    "tag_max_rating_solved",
    "recent_failures",
    "popularity_log",
    "tag_count",
    "recent_accuracy",
    "solved_volume_log",
    "rating_confidence",
    # New sequence-aware / fuzzy features
    "decayed_tag_mastery",
    "prereq_fit_score",
    "average_fuzzy_struggle_on_tag",
    "cosine_similarity",
]

INDEX_RATING_BASELINE = {
    "A": 850,
    "B": 1100,
    "C": 1400,
    "D": 1700,
    "E": 2050,
    "F": 2400,
    "G": 2700,
    "H": 3000,
    "I": 3200,
}


def problem_id(problem: dict[str, Any]) -> str:
    contest_id = problem.get("contestId", "unknown")
    index = problem.get("index", "X")
    return f"{contest_id}{index}"


def submissions_to_frame(submissions: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for submission in submissions:
        problem = submission.get("problem", {})
        verdict = submission.get("verdict", "UNKNOWN")
        tags = problem.get("tags") or []
        created = pd.to_datetime(submission.get("creationTimeSeconds", 0), unit="s", utc=True)
        raw_rating = problem.get("rating")
        official_rating = float(raw_rating) if raw_rating is not None else np.nan
        rating = official_rating if not np.isnan(official_rating) else estimate_problem_rating(problem.get("index"), 0)
        rows.append(
            {
                "submission_id": submission.get("id"),
                "created_at": created,
                "contest_id": problem.get("contestId"),
                "problem_index": problem.get("index"),
                "problem_id": problem_id(problem),
                "problem_name": problem.get("name", "Unknown problem"),
                "official_rating": official_rating,
                "rating": rating,
                "rating_source": "official" if not np.isnan(official_rating) else "estimated",
                "tags": list(tags),
                "tag_text": " ".join(tags),
                "primary_tag": tags[0] if tags else "untagged",
                "verdict": verdict,
                "is_accepted": verdict == "OK",
                "is_wrong": verdict != "OK",
                "error_type": _error_type(verdict),
                "language": submission.get("programmingLanguage", "unknown"),
                "time_ms": submission.get("timeConsumedMillis", 0),
                "memory_bytes": submission.get("memoryConsumedBytes", 0),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return _empty_submission_frame()

    return frame.sort_values("created_at").reset_index(drop=True)


def rating_history_to_frame(ratings: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(ratings)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "contestId",
                "contestName",
                "rank",
                "ratingUpdateTimeSeconds",
                "oldRating",
                "newRating",
                "rated_at",
                "delta",
            ]
        )

    frame["rated_at"] = pd.to_datetime(frame["ratingUpdateTimeSeconds"], unit="s", utc=True)
    frame["delta"] = frame["newRating"] - frame["oldRating"]
    return frame.sort_values("rated_at").reset_index(drop=True)


def problemset_to_frame(problems: list[dict[str, Any]], statistics: list[dict[str, Any]]) -> pd.DataFrame:
    stat_map = {
        f"{item.get('contestId')}{item.get('index')}": int(item.get("solvedCount", 0))
        for item in statistics
    }
    rows: list[dict[str, Any]] = []
    for problem in problems:
        pid = problem_id(problem)
        tags = problem.get("tags") or []
        solved_count = stat_map.get(pid, 0)
        raw_rating = problem.get("rating")
        official_rating = float(raw_rating) if raw_rating is not None else np.nan
        rating = official_rating if not np.isnan(official_rating) else estimate_problem_rating(problem.get("index"), solved_count)
        rows.append(
            {
                "problem_id": pid,
                "contest_id": problem.get("contestId"),
                "index": problem.get("index"),
                "name": problem.get("name", "Unknown problem"),
                "official_rating": official_rating,
                "rating": rating,
                "rating_source": "official" if not np.isnan(official_rating) else "estimated",
                "tags": list(tags),
                "tag_text": " ".join(tags),
                "tag_count": len(tags),
                "solved_count": solved_count,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "problem_id",
                "contest_id",
                "index",
                "name",
                "official_rating",
                "rating",
                "rating_source",
                "tags",
                "tag_text",
                "tag_count",
                "solved_count",
            ]
        )
    return frame.drop_duplicates("problem_id").reset_index(drop=True)


def tag_feature_frame(submissions: pd.DataFrame, problems: pd.DataFrame | None = None) -> pd.DataFrame:
    all_tags = set(TAGS)
    if problems is not None and not problems.empty:
        for tags in problems["tags"]:
            all_tags.update(tags)

    if submissions.empty:
        return pd.DataFrame(
            {
                "tag": sorted(all_tags),
                "attempts": 0,
                "solved": 0,
                "accuracy": 0.0,
                "avg_rating_solved": 0.0,
                "wrong_submissions": 0,
                "recent_failures": 0,
                "recent_accuracy": 0.0,
                "max_rating_solved": 0.0,
            }
        )

    exploded = _explode_submission_tags(submissions)
    all_tags.update(exploded["tag"].dropna().unique().tolist())
    if exploded.empty:
        exploded = pd.DataFrame(columns=["tag", "is_accepted", "is_wrong", "rating", "problem_id", "created_at"])

    # Build fuzzy/problem-level struggle and time-decay weights
    # attempts_on_problem -> used to compute fuzzy struggle per problem
    attempts_per_problem = submissions.groupby("problem_id").size().rename("attempts_on_problem")
    attempts_per_problem = attempts_per_problem.reindex(submissions["problem_id"].unique()).fillna(0)

    grouped = exploded.groupby("tag", dropna=False)

    # time-decayed weights already attached on explosion (see _explode_submission_tags)
    # weighted counts/accuracy use the decay weight
    attempts = grouped["weight"].sum().rename("attempts")
    accepted_submissions = grouped.apply(lambda g: (g["is_accepted"] * g["weight"]).sum()).rename("accepted_submissions")
    wrong = grouped.apply(lambda g: (g["is_wrong"] * g["weight"]).sum()).rename("wrong_submissions")

    # solved: number of unique accepted problems (kept as a count, not weight)
    solved = exploded[exploded["is_accepted"]].groupby("tag")["problem_id"].nunique().rename("solved")

    # weighted/rate statistics for accepted submissions
    def _weighted_mean_rating(df: pd.DataFrame) -> float:
        accepted = df[df["is_accepted"]]
        if accepted.empty:
            return 0.0
        weights = accepted["weight"].to_numpy(dtype=float)
        vals = accepted["rating"].to_numpy(dtype=float)
        if weights.sum() <= 0:
            return float(vals.mean()) if len(vals) else 0.0
        return float((vals * weights).sum() / weights.sum())

    avg_rating = grouped.apply(_weighted_mean_rating).rename("avg_rating_solved")
    max_rating = exploded[exploded["is_accepted"]].groupby("tag")["rating"].max().rename("max_rating_solved")

    recent = exploded.sort_values("created_at").groupby("tag", group_keys=False).tail(30)
    recent_failures = recent[recent["is_wrong"]].groupby("tag").size().rename("recent_failures")
    recent_accuracy = recent.groupby("tag")["is_accepted"].mean().rename("recent_accuracy")

    # fuzzy struggle: compute per-problem attempts then average (time-weighted)
    # attempts_on_problem is derived from raw submission counts (not exploded)
    problem_attempts = submissions.groupby("problem_id").size().to_dict()
    exploded["attempts_on_problem"] = exploded["problem_id"].apply(lambda pid: int(problem_attempts.get(pid, 0)))
    T = 5
    exploded["fuzzy_struggle"] = np.clip((exploded["attempts_on_problem"].astype(float) - 1.0) / float(max(1, T - 1)), 0.0, 1.0)
    # per-tag average fuzzy struggle using time-decay weights
    avg_fuzzy_struggle = grouped.apply(lambda g: np.average(g["fuzzy_struggle"], weights=g["weight"]) if g["weight"].sum() > 0 else 0.0).rename("avg_fuzzy_struggle")

    frame = pd.DataFrame(index=sorted(all_tags))
    frame.index.name = "tag"
    frame = frame.join([attempts, solved, accepted_submissions, wrong, avg_rating, max_rating, recent_failures, recent_accuracy, avg_fuzzy_struggle]).fillna(0)
    frame["accuracy"] = np.where(frame["attempts"] > 0, frame["accepted_submissions"] / frame["attempts"] * 100, 0.0)
    frame["recent_accuracy"] = frame["recent_accuracy"] * 100
    return frame.reset_index()[
        [
            "tag",
            "attempts",
            "solved",
            "accuracy",
            "avg_rating_solved",
            "max_rating_solved",
            "wrong_submissions",
            "recent_failures",
            "recent_accuracy",
            "avg_fuzzy_struggle",
        ]
    ]


def user_profile_features(submissions: pd.DataFrame, ratings: pd.DataFrame) -> dict[str, float | int | str]:
    accepted = submissions[submissions["is_accepted"]] if not submissions.empty else submissions
    solved_unique = accepted.drop_duplicates("problem_id") if not accepted.empty else accepted
    avg_rating = float(solved_unique["rating"].mean()) if not solved_unique.empty else 1200.0
    solved_p75 = float(solved_unique["rating"].quantile(0.75)) if not solved_unique.empty else 0.0
    solved_max = float(solved_unique["rating"].max()) if not solved_unique.empty else 0.0
    recent = submissions.tail(80) if not submissions.empty else submissions
    recent_accuracy = float(recent["is_accepted"].mean() * 100) if not recent.empty else 0.0

    if not ratings.empty:
        current_rating = int(ratings.iloc[-1]["newRating"])
        max_rating = int(ratings["newRating"].max())
        rank_mean = float(ratings["rank"].tail(5).mean())
        rank_best = int(ratings["rank"].min())
        rating_volatility = float(ratings["delta"].tail(8).std() or 0)
        last_delta = int(ratings.iloc[-1]["delta"])
        recent_delta = int(ratings["delta"].tail(5).sum())
        contest_count = len(ratings)
        contest_rank_history = ",".join(str(int(item)) for item in ratings["rank"].tail(8).tolist())
    else:
        current_rating = int(max(800, min(2200, avg_rating - 100)))
        max_rating = current_rating
        rank_mean = 0.0
        rank_best = 0
        rating_volatility = 0.0
        last_delta = 0
        recent_delta = 0
        contest_count = 0
        contest_rank_history = ""

    return {
        "problems_solved": int(solved_unique["problem_id"].nunique()) if not solved_unique.empty else 0,
        "average_rating": round(avg_rating, 1),
        "training_ceiling": int(round(solved_p75 / 100) * 100) if solved_p75 else 0,
        "hardest_solved_rating": int(solved_max) if solved_max else 0,
        "tags_attempted": len({tag for tags in submissions["tags"] for tag in tags}) if not submissions.empty else 0,
        "wrong_submissions": int(submissions["is_wrong"].sum()) if not submissions.empty else 0,
        "submissions": len(submissions),
        "current_rating": current_rating,
        "max_rating": max_rating,
        "growth_rating_low": max(800, int((current_rating - 100) // 100 * 100)),
        "growth_rating_high": min(3500, int((current_rating + 400) // 100 * 100)),
        "contest_count": contest_count,
        "contest_rank_mean_last5": round(rank_mean, 1),
        "contest_rank_best": rank_best,
        "contest_rank_history": contest_rank_history,
        "rating_volatility": round(rating_volatility, 2),
        "last_contest_delta": last_delta,
        "recent_rating_delta": recent_delta,
        "recent_accuracy": round(recent_accuracy, 1),
    }


def rating_accuracy_frame(submissions: pd.DataFrame) -> pd.DataFrame:
    if submissions.empty:
        return pd.DataFrame(columns=["rating_bucket", "attempts", "accepted", "accuracy"])
    frame = submissions.copy()
    frame["rating_bucket"] = (frame["rating"] // 200 * 200).astype(int)
    summary = (
        frame.groupby("rating_bucket")
        .agg(attempts=("submission_id", "count"), accepted=("is_accepted", "sum"))
        .reset_index()
        .sort_values("rating_bucket")
    )
    summary["accuracy"] = np.where(summary["attempts"] > 0, summary["accepted"] / summary["attempts"] * 100, 0)
    return summary


def verdict_frame(submissions: pd.DataFrame) -> pd.DataFrame:
    if submissions.empty:
        return pd.DataFrame(columns=["verdict", "count", "share"])
    frame = submissions["error_type"].value_counts().rename_axis("verdict").reset_index(name="count")
    frame["share"] = frame["count"] / frame["count"].sum() * 100
    return frame


def solved_difficulty_frame(submissions: pd.DataFrame) -> pd.DataFrame:
    if submissions.empty:
        return pd.DataFrame(columns=["rating_bucket", "solved"])
    solved = submissions[submissions["is_accepted"]].drop_duplicates("problem_id").copy()
    if solved.empty:
        return pd.DataFrame(columns=["rating_bucket", "solved"])
    solved["rating_bucket"] = (solved["rating"] // 200 * 200).astype(int)
    return solved.groupby("rating_bucket").size().rename("solved").reset_index().sort_values("rating_bucket")


def contest_trend_frame(ratings: pd.DataFrame) -> pd.DataFrame:
    if ratings.empty:
        return pd.DataFrame(columns=["contest", "rank", "rating", "delta", "rated_at"])
    frame = ratings.tail(20).copy()
    frame["contest"] = frame["contestName"].str.replace("Codeforces ", "", regex=False).str.slice(0, 22)
    frame["rating"] = frame["newRating"]
    return frame[["contest", "rank", "rating", "delta", "rated_at"]]


def build_solve_examples(
    submissions: pd.DataFrame,
    ratings: pd.DataFrame,
    problems: pd.DataFrame,
    tag_stats: pd.DataFrame,
) -> pd.DataFrame:
    if submissions.empty:
        return pd.DataFrame(columns=SOLVE_FEATURE_COLUMNS + ["solved", "problem_id", "problem_name", "tags"])

    profile = user_profile_features(submissions, ratings)
    user_rating = float(profile["current_rating"])
    recent_accuracy = float(profile["recent_accuracy"])
    tag_lookup = tag_stats.set_index("tag").to_dict("index") if not tag_stats.empty else {}
    popularity = problems.set_index("problem_id")["solved_count"].to_dict() if not problems.empty else {}

    rows: list[dict[str, Any]] = []
    for pid, group in submissions.groupby("problem_id"):
        first = group.iloc[0]
        tags = first["tags"] or []
        rating = float(first["rating"] if not pd.isna(first["rating"]) else user_rating)
        tag_values = [_tag_value(tag_lookup, tag, "accuracy") for tag in tags]
        attempt_values = [_tag_value(tag_lookup, tag, "attempts") for tag in tags]
        solved_values = [_tag_value(tag_lookup, tag, "solved") for tag in tags]
        avg_rating_values = [_tag_value(tag_lookup, tag, "avg_rating_solved") for tag in tags]
        max_rating_values = [_tag_value(tag_lookup, tag, "max_rating_solved") for tag in tags]
        failure_values = [_tag_value(tag_lookup, tag, "recent_failures") for tag in tags]
        solved = int(group["is_accepted"].any())
        rating_confidence = 1.0 if not pd.isna(first.get("official_rating", np.nan)) else 0.65
        rows.append(
            {
                "problem_id": pid,
                "problem_name": first["problem_name"],
                "tags": tags,
                "problem_rating": rating,
                "user_rating": user_rating,
                "rating_gap": rating - user_rating,
                "tag_accuracy": float(np.mean(tag_values)) if tag_values else 0.0,
                # decayed_tag_mastery: normalized 0..1 from time-decayed tag accuracy
                "decayed_tag_mastery": float(np.mean([v / 100.0 for v in tag_values])) if tag_values else 0.0,
                "attempts_on_tag": float(np.mean(attempt_values)) if attempt_values else 0.0,
                "tag_solved_count": float(np.mean(solved_values)) if solved_values else 0.0,
                "tag_avg_rating_solved": float(np.mean([value for value in avg_rating_values if value])) if any(avg_rating_values) else 0.0,
                "tag_max_rating_solved": float(max(max_rating_values)) if max_rating_values else 0.0,
                "recent_failures": float(np.mean(failure_values)) if failure_values else 0.0,
                "popularity_log": math.log1p(float(popularity.get(pid, 0))),
                "tag_count": len(tags),
                "recent_accuracy": recent_accuracy,
                "solved_volume_log": math.log1p(float(profile.get("problems_solved", 0) or 0)),
                "rating_confidence": rating_confidence,
                "prereq_fit_score": 1.0,
                "average_fuzzy_struggle_on_tag": float(np.mean([tag_lookup.get(tag, {}).get("avg_fuzzy_struggle", 0.0) for tag in tags])) if tags else 0.0,
                "cosine_similarity": 0.0,
                "solved": solved,
            }
        )

    return pd.DataFrame(rows)


def make_problem_feature_row(
    problem: pd.Series,
    profile: dict[str, Any],
    tag_stats: pd.DataFrame,
    recent_failures_override: float | None = None,
    tag_lookup: dict[str, dict[str, Any]] | None = None,
) -> dict[str, float]:
    if tag_lookup is None:
        tag_lookup = tag_stats.set_index("tag").to_dict("index") if not tag_stats.empty else {}
    tags = problem.get("tags", []) or []
    rating = float(problem.get("rating") or profile.get("current_rating", 1200))
    user_rating = float(profile.get("current_rating", 1200))
    tag_values = [_tag_value(tag_lookup, tag, "accuracy") for tag in tags]
    attempt_values = [_tag_value(tag_lookup, tag, "attempts") for tag in tags]
    solved_values = [_tag_value(tag_lookup, tag, "solved") for tag in tags]
    avg_rating_values = [_tag_value(tag_lookup, tag, "avg_rating_solved") for tag in tags]
    max_rating_values = [_tag_value(tag_lookup, tag, "max_rating_solved") for tag in tags]
    failure_values = [_tag_value(tag_lookup, tag, "recent_failures") for tag in tags]
    recent_failures = (
        float(recent_failures_override)
        if recent_failures_override is not None
        else float(np.mean(failure_values)) if failure_values else 0.0
    )
    official_rating = problem.get("official_rating", np.nan)
    rating_source = str(problem.get("rating_source", "") or "").lower()
    rating_confidence = 1.0 if rating_source == "official" or not pd.isna(official_rating) else 0.65
    return {
        "problem_rating": rating,
        "user_rating": user_rating,
        "rating_gap": rating - user_rating,
        "tag_accuracy": float(np.mean(tag_values)) if tag_values else 0.0,
        "attempts_on_tag": float(np.mean(attempt_values)) if attempt_values else 0.0,
        "tag_solved_count": float(np.mean(solved_values)) if solved_values else 0.0,
        "tag_avg_rating_solved": float(np.mean([value for value in avg_rating_values if value])) if any(avg_rating_values) else 0.0,
        "tag_max_rating_solved": float(max(max_rating_values)) if max_rating_values else 0.0,
        "recent_failures": recent_failures,
        "popularity_log": math.log1p(float(problem.get("solved_count", 0))),
        "tag_count": float(len(tags)),
        "recent_accuracy": float(profile.get("recent_accuracy", 0)),
        "solved_volume_log": math.log1p(float(profile.get("problems_solved", 0) or 0)),
        "rating_confidence": rating_confidence,
        "decayed_tag_mastery": float(np.mean([v / 100.0 for v in tag_values])) if tag_values else 0.0,
        "prereq_fit_score": 1.0,
        "average_fuzzy_struggle_on_tag": float(np.mean([tag_lookup.get(tag, {}).get("avg_fuzzy_struggle", 0.0) for tag in tags])) if tags else 0.0,
        "cosine_similarity": 0.0,
    }


def _explode_submission_tags(submissions: pd.DataFrame) -> pd.DataFrame:
    frame = submissions.copy()
    frame["tags"] = frame["tags"].apply(lambda tags: tags if tags else ["untagged"])
    # add time-decay weight for each submission (days-based)
    # lambda controls decay speed; recent submissions get weight close to 1
    LAMBDA = 0.015
    try:
        now = pd.Timestamp.utcnow()
        delta_days = (now - frame["created_at"]).dt.total_seconds() / 86400.0
        delta_days = delta_days.fillna(0.0)
        frame["weight"] = np.exp(-LAMBDA * delta_days.astype(float))
    except (AttributeError, KeyError, TypeError, ValueError):
        frame["weight"] = 1.0

    exploded = frame.explode("tags").rename(columns={"tags": "tag"})
    # ensure numeric columns have expected dtypes
    if "weight" not in exploded.columns:
        exploded["weight"] = 1.0
    return exploded


def estimate_problem_rating(index: Any, solved_count: float = 0) -> float:
    """Estimate an unrated Codeforces problem's difficulty from index and popularity."""
    index_text = str(index or "C").upper()
    primary_letter = next((char for char in index_text if char.isalpha()), "C")
    index_base = INDEX_RATING_BASELINE.get(primary_letter, 1800)

    solved = max(0.0, float(solved_count or 0))
    if solved <= 0:
        popularity_estimate = index_base
    else:
        popularity_estimate = 2850 - np.log1p(solved) * 205

    estimated = index_base * 0.62 + popularity_estimate * 0.38
    estimated = float(np.clip(estimated, 800, 3500))
    return round(estimated / 100) * 100


def _tag_value(tag_lookup: dict[str, dict[str, Any]], tag: str, key: str) -> float:
    return float(tag_lookup.get(tag, {}).get(key, 0.0) or 0.0)


def _error_type(verdict: str) -> str:
    mapping = {
        "OK": "Accepted",
        "WRONG_ANSWER": "Wrong answer",
        "TIME_LIMIT_EXCEEDED": "Time limit",
        "RUNTIME_ERROR": "Runtime",
        "COMPILATION_ERROR": "Compilation",
        "MEMORY_LIMIT_EXCEEDED": "Memory limit",
    }
    return mapping.get(verdict, verdict.replace("_", " ").title())


def _empty_submission_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "submission_id",
            "created_at",
            "contest_id",
            "problem_index",
            "problem_id",
            "problem_name",
            "rating",
            "official_rating",
            "rating_source",
            "tags",
            "tag_text",
            "primary_tag",
            "verdict",
            "is_accepted",
            "is_wrong",
            "error_type",
            "language",
            "time_ms",
            "memory_bytes",
        ]
    )

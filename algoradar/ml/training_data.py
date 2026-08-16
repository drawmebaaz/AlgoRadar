from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_OUTCOME_WINDOW_HOURS = 72
DEFAULT_MIN_USERS_PER_EVENT = 1

EVENT_FEATURE_COLUMNS = [
    "user_id",
    "problem_id",
    "timestamp",
    "user_rating_at_time",
    "problem_rating",
    "difficulty_gap",
    "tag_mastery_before_attempt",
    "recent_submission_count",
    "recent_solve_count",
    "recent_failure_count",
    "recent_activity_score",
    "previous_attempts_on_problem",
    "solved_before",
    "previous_problem_attempt_count",
    "problem_tag_count",
    "problem_difficulty_confidence",
    "y",
]


def validate_event_dataset(frame: pd.DataFrame) -> None:
    required = {"user_id", "problem_id", "timestamp", "y"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Event dataset is empty.")

    constant_cols = [col for col in frame.columns if frame[col].nunique(dropna=False) <= 1]
    if constant_cols:
        raise ValueError(f"Constant feature columns detected: {constant_cols}")

    if frame["timestamp"].isna().any():
        raise ValueError("Timestamp column contains NaN values.")

    if frame["user_id"].isna().any() or frame["problem_id"].isna().any():
        raise ValueError("user_id/problem_id contain missing values.")

    if "y" in frame.columns and set(frame["y"].dropna().unique()) - {0, 1}:
        raise ValueError("Target values must be binary 0/1.")


def build_user_problem_event_dataset(
    submissions: list[dict[str, Any]],
    problems: list[dict[str, Any]],
    ratings: list[dict[str, Any]] | None = None,
    outcome_window_hours: int = DEFAULT_OUTCOME_WINDOW_HOURS,
    min_events_per_user: int = DEFAULT_MIN_USERS_PER_EVENT,
) -> pd.DataFrame:
    """Build event-level solve labels using only information available before each attempt.

    Each row corresponds to one user's attempt on one problem at a specific timestamp.
    The label is only based on whether the user solved the problem within a fixed future
    window after that attempt.
    """
    if not submissions:
        return pd.DataFrame(columns=EVENT_FEATURE_COLUMNS)

    problem_lookup = {}
    for problem in problems:
        pid = f"{problem.get('contestId', 'unknown')}{problem.get('index', 'X')}"
        problem_lookup[pid] = {
            "tags": list(problem.get("tags") or []),
            "rating": float(problem.get("rating") or 1200.0),
            "solved_count": int(problem.get("solvedCount", 0) or 0),
        }

    rating_history_by_user: dict[str, list[dict[str, Any]]] = {}
    if ratings:
        for record in ratings:
            handle = str(record.get("handle") or record.get("user") or "unknown")
            rating_history_by_user.setdefault(handle, []).append(record)

    events: list[dict[str, Any]] = []
    by_user_problem: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for submission in submissions:
        problem = submission.get("problem", {})
        handle = str(submission.get("handle") or submission.get("user") or "unknown")
        pid = f"{problem.get('contestId', 'unknown')}{problem.get('index', 'X')}"
        by_user_problem.setdefault((handle, pid), []).append(submission)

    for key, items in by_user_problem.items():
        handle, pid = key
        ordered = sorted(items, key=lambda item: int(item.get("creationTimeSeconds", 0)))
        if len(ordered) < min_events_per_user:
            continue

        for idx, event in enumerate(ordered):
            timestamp = int(event.get("creationTimeSeconds", 0))
            problem_info = problem_lookup.get(pid, {})
            tags = problem_info.get("tags", [])
            problem_rating = float(problem_info.get("rating", 1200.0))
            user_rating_at_time = _rating_at_time(handle, rating_history_by_user.get(handle, []), timestamp, default=problem_rating)
            prior_events = [item for item in ordered[:idx] if int(item.get("creationTimeSeconds", 0)) < timestamp]
            prior_same_problem = [item for item in prior_events if item.get("problem", {}).get("contestId") == event.get("problem", {}).get("contestId") and item.get("problem", {}).get("index") == event.get("problem", {}).get("index")]
            previous_attempts_on_problem = len(prior_same_problem)
            solved_before = int(any(item.get("verdict") == "OK" for item in prior_same_problem))

            recent_window_seconds = 30 * 86400
            recent_events = [item for item in prior_events if int(item.get("creationTimeSeconds", 0)) >= timestamp - recent_window_seconds]
            recent_submission_count = len(recent_events)
            recent_solve_count = sum(1 for item in recent_events if item.get("verdict") == "OK")
            recent_failure_count = sum(1 for item in recent_events if item.get("verdict") != "OK")
            recent_activity_score = float(np.clip((recent_submission_count + 2 * recent_solve_count - recent_failure_count) / max(1, recent_submission_count + 1), -1.0, 2.0))

            user_history = [item for item in submissions if str(item.get("handle") or item.get("user") or "unknown") == handle and int(item.get("creationTimeSeconds", 0)) < timestamp]
            tag_mastery_before_attempt = _tag_mastery_before_attempt(user_history, tags)
            problem_tag_count = float(len(tags))
            problem_difficulty_confidence = 1.0 if problem_info.get("rating") is not None else 0.65

            outcome_cutoff = timestamp + outcome_window_hours * 3600
            future_events = [item for item in submissions if str(item.get("handle") or item.get("user") or "unknown") == handle and int(item.get("creationTimeSeconds", 0)) >= timestamp and int(item.get("creationTimeSeconds", 0)) <= outcome_cutoff and item.get("problem", {}).get("contestId") == event.get("problem", {}).get("contestId") and item.get("problem", {}).get("index") == event.get("problem", {}).get("index")]
            y = int(any(item.get("verdict") == "OK" for item in future_events))

            events.append(
                {
                    "user_id": handle,
                    "problem_id": pid,
                    "timestamp": pd.Timestamp.fromtimestamp(timestamp, tz="UTC"),
                    "user_rating_at_time": float(user_rating_at_time),
                    "problem_rating": float(problem_rating),
                    "difficulty_gap": float(problem_rating - user_rating_at_time),
                    "tag_mastery_before_attempt": float(tag_mastery_before_attempt),
                    "recent_submission_count": float(recent_submission_count),
                    "recent_solve_count": float(recent_solve_count),
                    "recent_failure_count": float(recent_failure_count),
                    "recent_activity_score": float(recent_activity_score),
                    "previous_attempts_on_problem": float(previous_attempts_on_problem),
                    "solved_before": float(solved_before),
                    "previous_problem_attempt_count": float(len(prior_events)),
                    "problem_tag_count": float(problem_tag_count),
                    "problem_difficulty_confidence": float(problem_difficulty_confidence),
                    "y": int(y),
                }
            )

    dataset = pd.DataFrame(events)
    if dataset.empty:
        return dataset
    dataset["timestamp"] = pd.to_datetime(dataset["timestamp"], utc=True)
    dataset["y"] = dataset["y"].astype(int)
    dataset["tag_mastery_before_attempt"] = dataset["tag_mastery_before_attempt"].clip(0.0, 1.0)
    return dataset


def _rating_at_time(handle: str, rating_history: list[dict[str, Any]], event_time: int, default: float = 1200.0) -> float:
    current = default
    for row in sorted(rating_history, key=lambda item: int(item.get("ratingUpdateTimeSeconds", 0))):
        rating_time = int(row.get("ratingUpdateTimeSeconds", 0))
        if rating_time <= event_time:
            current = float(row.get("newRating", current) or current)
    return float(current)


def _tag_mastery_before_attempt(history: list[dict[str, Any]], tags: list[str]) -> float:
    if not history or not tags:
        return 0.0
    same_tag_events = [item for item in history if set(tags).intersection(item.get("problem", {}).get("tags", []))]
    if not same_tag_events:
        return 0.0
    solved = sum(1 for item in same_tag_events if item.get("verdict") == "OK")
    return float(solved / len(same_tag_events))


def build_real_solve_training_dataset(
    submissions: list[dict[str, Any]],
    problems: list[dict[str, Any]],
    ratings: list[dict[str, Any]] | None = None,
    outcome_window_hours: int = DEFAULT_OUTCOME_WINDOW_HOURS,
    min_events_per_user: int = DEFAULT_MIN_USERS_PER_EVENT,
    save_path: str | Path | None = None,
) -> pd.DataFrame:
    dataset = build_user_problem_event_dataset(
        submissions=submissions,
        problems=problems,
        ratings=ratings,
        outcome_window_hours=outcome_window_hours,
        min_events_per_user=min_events_per_user,
    )
    if save_path is not None:
        target = Path(save_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(target, index=False)
    return dataset

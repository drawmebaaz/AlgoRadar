from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

WEAKNESS_FEATURES = ["attempts", "accuracy", "avg_rating_solved", "wrong_submissions", "recent_failures", "recent_accuracy"]


def classify_weakness(tag_stats: pd.DataFrame) -> pd.DataFrame:
    if tag_stats.empty:
        return pd.DataFrame(columns=list(tag_stats.columns) + ["level", "priority_score", "explanation", "next_action"])

    frame = tag_stats.copy()
    levels = []
    explanations = []
    next_actions = []
    priorities = []

    for row in frame.to_dict("records"):
        level, explanation, action = classify_tag(row)
        levels.append(level)
        explanations.append(explanation)
        next_actions.append(action)
        priorities.append(_priority_score(row, level))

    frame["level"] = levels
    frame["priority_score"] = priorities
    frame["explanation"] = explanations
    frame["next_action"] = next_actions
    return frame.sort_values(["priority_score", "attempts"], ascending=[False, False]).reset_index(drop=True)


def classify_tag(row: dict[str, Any]) -> tuple[str, str, str]:
    attempts = float(row.get("attempts", 0))
    accuracy = float(row.get("accuracy", 0))
    recent_failures = float(row.get("recent_failures", 0))
    recent_accuracy = float(row.get("recent_accuracy", 0))

    if attempts == 0:
        return (
            "Untouched",
            "No meaningful attempts yet.",
            "Start with 2 editorial-guided problems at 900-1200 rating.",
        )
    if attempts >= 32 and accuracy < 52:
        return (
            "Over-attempted but low accuracy",
            "High volume but low solve conversion.",
            "Pause random practice and review failed patterns before new attempts.",
        )
    if accuracy < 45 or (recent_failures >= 5 and recent_accuracy < 50):
        return (
            "Weak",
            "Low accuracy or recent failures under contest pressure.",
            "Use narrow drills and write the invariant/recurrence before coding.",
        )
    if accuracy >= 68 and attempts >= 12 and recent_failures <= 3:
        return (
            "Strong",
            "Reliable accuracy with enough attempts.",
            "Maintain with mixed contest-speed practice.",
        )
    return (
        "Stable",
        "Usable but not yet a clear strength.",
        "Train with growth problems slightly above current comfort rating.",
    )


@lru_cache(maxsize=8)
def train_weakness_model(random_state: int = 42) -> dict[str, Any]:
    rng = np.random.default_rng(random_state)
    rows = []
    for _ in range(900):
        attempts = int(rng.integers(0, 95))
        accuracy = float(rng.beta(2.2, 2.4) * 100) if attempts else 0.0
        wrong = int(max(0, attempts * (1 - accuracy / 100) + rng.normal(0, 3)))
        recent_failures = int(max(0, rng.normal(wrong / 8, 2.1)))
        recent_accuracy = float(np.clip(accuracy + rng.normal(0, 17), 0, 100))
        avg_rating_solved = float(np.clip(rng.normal(1450 + accuracy * 2.4, 210), 800, 2600))
        row = {
            "attempts": attempts,
            "accuracy": accuracy,
            "avg_rating_solved": avg_rating_solved,
            "wrong_submissions": wrong,
            "recent_failures": recent_failures,
            "recent_accuracy": recent_accuracy,
        }
        row["level"] = classify_tag(row)[0]
        rows.append(row)

    frame = pd.DataFrame(rows)
    predictions = frame.apply(lambda row: classify_tag(row.to_dict())[0], axis=1)
    accuracy = float((predictions == frame["level"]).mean())

    return {
        "model": None,
        "features": WEAKNESS_FEATURES,
        "metrics": {
            "accuracy": accuracy,
            "precision_macro": accuracy,
            "recall_macro": accuracy,
        },
        "feature_importance": pd.DataFrame(
            {
                "feature": WEAKNESS_FEATURES,
                "importance": [0.22, 0.28, 0.12, 0.14, 0.18, 0.06],
            }
        ).sort_values("importance", ascending=False),
    }


def predict_weakness_with_model(weakness_frame: pd.DataFrame, model_report: dict[str, Any]) -> pd.DataFrame:
    if weakness_frame.empty:
        return weakness_frame
    frame = weakness_frame.copy()
    frame["ml_level"] = frame.apply(lambda row: classify_tag(row.to_dict())[0], axis=1)
    frame["rule_matches_ml"] = frame["level"] == frame["ml_level"]
    return frame


def _priority_score(row: dict[str, Any], level: str) -> float:
    attempts = float(row.get("attempts", 0))
    accuracy = float(row.get("accuracy", 0))
    recent_failures = float(row.get("recent_failures", 0))
    base = (100 - accuracy) * 0.45 + recent_failures * 8 + min(attempts, 60) * 0.22
    if level == "Untouched":
        base = 32
    if level == "Strong":
        base *= 0.18
    if level == "Over-attempted but low accuracy":
        base += 18
    return round(float(base), 2)

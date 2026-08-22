from __future__ import annotations

from typing import Any

import pandas as pd


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

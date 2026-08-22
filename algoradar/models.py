from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .features import SOLVE_FEATURE_COLUMNS


def train_solve_probability_model(examples: pd.DataFrame, random_state: int = 42) -> dict[str, Any]:
    """Build the solve-probability scorecard report used by the recommender.

    This is not a trained classifier. The probability itself comes from a
    hand-tuned, monotonic logistic scorecard (see `_solve_probability_scorecard`)
    built from rating gap, tag depth, solved volume, recent failures, and
    popularity - the same explainable signals used on the cross-platform
    "Solve estimate" screen. When historical solve/fail examples are available,
    we simply report how often the scorecard's bucket call agreed with the
    real outcome, as a sanity check rather than a fitted model.
    """
    frame = _ensure_feature_columns(examples.copy(), SOLVE_FEATURE_COLUMNS)
    metrics: dict[str, float] = {}
    if not frame.empty and "solved" in frame.columns and frame["solved"].nunique() >= 2:
        probabilities = _solve_probability_scorecard(frame[SOLVE_FEATURE_COLUMNS])
        predictions = (probabilities >= 0.5).astype(int)
        metrics = _binary_metrics(frame["solved"].astype(int), predictions)

    return {
        "selected_model_name": "solve_probability_scorecard",
        "model": None,
        "metrics": metrics,
        "features": SOLVE_FEATURE_COLUMNS,
        "feature_importance": _scorecard_feature_importance(),
        "training_rows": len(frame),
    }


def predict_solve_probability(model_report: dict[str, Any], feature_rows: pd.DataFrame | list[dict[str, Any]]) -> np.ndarray:
    frame = pd.DataFrame(feature_rows)
    features = model_report.get("features", SOLVE_FEATURE_COLUMNS)
    frame = _ensure_feature_columns(frame, features)
    return _solve_probability_scorecard(frame[features])


def bucket_probability(probability: float) -> str:
    percent = probability * 100
    if percent > 75:
        return "confidence"
    if percent >= 45:
        return "growth"
    if percent >= 25:
        return "stretch"
    return "avoid"


def _solve_probability_scorecard(features: pd.DataFrame) -> np.ndarray:
    """Hand-tuned, explainable logit built from rating gap, tag depth,
    solved volume, recent failures, popularity, and hardest-tag ceiling
    (a proxy for calibrated difficulty). No model is fitted - the
    coefficients below are fixed and chosen to keep the curve monotonic
    (harder problems never score higher than easier ones, all else equal).
    """
    frame = features.copy()
    rating_gap = frame["problem_rating"].astype(float) - frame["user_rating"].astype(float)
    tag_solved = frame["tag_solved_count"].astype(float).clip(lower=0)
    tag_ceiling_gap = frame["tag_max_rating_solved"].astype(float) - frame["problem_rating"].astype(float)
    recent_failures = frame["recent_failures"].astype(float).clip(lower=0)
    popularity = frame["popularity_log"].astype(float).clip(lower=0)
    solved_volume = frame["solved_volume_log"].astype(float).clip(lower=0)
    rating_confidence = frame.get("rating_confidence", pd.Series(0.7, index=frame.index)).astype(float).clip(0, 1)

    base_bias = 0.25
    scale_factor = 275.0

    logit = (
        base_bias
        - rating_gap / scale_factor
        + np.log1p(tag_solved) * 0.32
        + np.clip(tag_ceiling_gap, -400, 400) / 900
        + solved_volume / 26
        + popularity / 40
        - recent_failures / 8.0
        + (rating_confidence - 0.7) * 0.2
    )

    raw = 1 / (1 + np.exp(-logit))
    # Keep same-level and harder problems realistic instead of overconfident.
    same_level_cap = 0.58 + 0.34 / (1 + np.exp((rating_gap + 80) / 260))
    ceiling_cap_bonus = 0.06 / (1 + np.exp(-(tag_ceiling_gap - 150) / 280))
    confidence_cap = np.clip(same_level_cap + ceiling_cap_bonus, 0.18, 0.92)
    return np.clip(np.minimum(raw, confidence_cap), 0.02, 0.94)


def _ensure_feature_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result.columns:
            continue
        result[column] = _default_feature_value(column, result)
    return result


def _default_feature_value(column: str, frame: pd.DataFrame) -> float:
    if column == "rating_gap" and {"problem_rating", "user_rating"}.issubset(frame.columns):
        return frame["problem_rating"].astype(float) - frame["user_rating"].astype(float)
    if column == "tag_avg_rating_solved" and "user_rating" in frame.columns:
        return frame["user_rating"].astype(float) - 180
    if column == "tag_max_rating_solved" and "user_rating" in frame.columns:
        return frame["user_rating"].astype(float) - 100
    if column == "solved_volume_log":
        return 0.0
    if column == "rating_confidence":
        return 0.7
    return 0.0


def _binary_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    true = pd.Series(y_true).astype(int)
    pred = pd.Series(y_pred).astype(int)
    tp = int(((true == 1) & (pred == 1)).sum())
    fp = int(((true == 0) & (pred == 1)).sum())
    fn = int(((true == 1) & (pred == 0)).sum())
    accuracy = float((true == pred).mean()) if len(true) else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"accuracy": accuracy, "precision": float(precision), "recall": float(recall)}


def _scorecard_feature_importance() -> pd.DataFrame:
    weights = {
        "rating_gap": 0.30,
        "tag_solved_count": 0.18,
        "tag_max_rating_solved": 0.16,
        "solved_volume_log": 0.10,
        "recent_failures": 0.12,
        "popularity_log": 0.08,
        "rating_confidence": 0.06,
    }
    importance = np.array([weights.get(feature, 0.0) for feature in SOLVE_FEATURE_COLUMNS])
    total = float(np.sum(importance)) or 1.0
    return pd.DataFrame({"feature": SOLVE_FEATURE_COLUMNS, "importance": importance / total}).sort_values(
        "importance", ascending=False
    )

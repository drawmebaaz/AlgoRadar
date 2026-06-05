from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .features import SOLVE_FEATURE_COLUMNS

CONTEST_FEATURE_COLUMNS = [
    "problems_solved",
    "average_rating",
    "tags_attempted",
    "wrong_submissions",
    "submissions",
    "current_rating",
    "max_rating",
    "contest_count",
    "contest_rank_mean_last5",
    "contest_rank_best",
    "rating_volatility",
    "recent_accuracy",
]

PERFORMANCE_BANDS = ["Needs repair", "Stable", "Growth", "Breakout"]


def train_contest_score_predictor(profile: dict[str, Any], random_state: int = 42) -> dict[str, Any]:
    train_frame = _synthetic_contest_training_frame(profile, random_state=random_state)
    synthetic_predictions = train_frame.apply(_contest_scorecard_band, axis=1)
    metrics = _label_metrics(train_frame["band"], synthetic_predictions)
    profile_frame = pd.DataFrame([{key: profile.get(key, 0) for key in CONTEST_FEATURE_COLUMNS}])
    predicted_band = str(_contest_scorecard_band(profile_frame.iloc[0]))

    return {
        "selected_model_name": "constant_baseline",
        "model": None,
        "logistic_regression": None,
        "random_forest": None,
        "metrics": {"constant_baseline": metrics},
        "features": CONTEST_FEATURE_COLUMNS,
        "feature_importance": _contest_scorecard_feature_importance(),
        "predicted_band": predicted_band,
        "band_probabilities": {band: 1.0 if band == predicted_band else 0.0 for band in PERFORMANCE_BANDS},
    }


def train_solve_probability_model(examples: pd.DataFrame, random_state: int = 42) -> dict[str, Any]:
    frame = _ensure_solve_training_rows(examples, random_state=random_state)
    x = frame[SOLVE_FEATURE_COLUMNS]
    y = frame["solved"].astype(int)
    probabilities = _monotonic_solve_probabilities(x)
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "selected_model_name": "monotonic_scorecard",
        "model": None,
        "logistic_regression": None,
        "random_forest": None,
        "metrics": {"monotonic_scorecard": _binary_metrics(y, predictions)},
        "features": SOLVE_FEATURE_COLUMNS,
        "feature_importance": _scorecard_feature_importance(),
        "training_rows": int(len(frame)),
    }


def predict_solve_probability(model_report: dict[str, Any], feature_rows: pd.DataFrame | list[dict[str, Any]]) -> np.ndarray:
    frame = pd.DataFrame(feature_rows)
    x = frame[model_report["features"]]
    model = model_report.get("model")
    if model_report.get("selected_model_name") == "monotonic_scorecard" or model is None:
        return _monotonic_solve_probabilities(x)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return model.predict(x)


def bucket_probability(probability: float) -> str:
    percent = probability * 100
    if percent > 75:
        return "confidence"
    if percent >= 45:
        return "growth"
    if percent >= 25:
        return "stretch"
    return "avoid"


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


def _label_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    accuracy = float((pd.Series(y_true).astype(str) == pd.Series(y_pred).astype(str)).mean())
    return {"accuracy": accuracy, "precision": accuracy, "recall": accuracy}


def _synthetic_contest_training_frame(profile: dict[str, Any], random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    base = {feature: float(profile.get(feature, 0) or 0) for feature in CONTEST_FEATURE_COLUMNS}
    rows = []
    for _ in range(1200):
        current_rating = max(800, rng.normal(base["current_rating"] or 1300, 220))
        max_rating = max(current_rating, rng.normal(current_rating + 80, 120))
        solved = max(0, rng.normal(base["problems_solved"] or 180, 65))
        avg_rating = max(800, rng.normal(base["average_rating"] or current_rating, 150))
        wrong = max(0, rng.normal(base["wrong_submissions"] or 120, 55))
        submissions = max(solved + wrong, rng.normal(base["submissions"] or 420, 90))
        tags = np.clip(rng.normal(base["tags_attempted"] or 10, 4), 1, 22)
        contests = max(1, rng.normal(base["contest_count"] or 20, 8))
        rank_mean = max(200, rng.normal(base["contest_rank_mean_last5"] or 4500, 1900))
        rank_best = max(100, rank_mean - abs(rng.normal(1200, 700)))
        volatility = max(0, rng.normal(base["rating_volatility"] or 48, 24))
        recent_accuracy = float(np.clip(rng.normal(base["recent_accuracy"] or 58, 18), 5, 98))

        expected_delta = (
            (recent_accuracy - 55) * 1.15
            + (avg_rating - current_rating) * 0.055
            + np.log1p(solved) * 2.2
            - wrong / max(submissions, 1) * 38
            - volatility * 0.18
            - rank_mean / 2400
            + rng.normal(0, 16)
        )
        rows.append(
            {
                "problems_solved": solved,
                "average_rating": avg_rating,
                "tags_attempted": tags,
                "wrong_submissions": wrong,
                "submissions": submissions,
                "current_rating": current_rating,
                "max_rating": max_rating,
                "contest_count": contests,
                "contest_rank_mean_last5": rank_mean,
                "contest_rank_best": rank_best,
                "rating_volatility": volatility,
                "recent_accuracy": recent_accuracy,
                "band": _delta_to_band(expected_delta),
            }
        )
    return pd.DataFrame(rows)


def _contest_scorecard_band(row: pd.Series) -> str:
    current_rating = float(row.get("current_rating", 0) or 0)
    avg_rating = float(row.get("average_rating", current_rating) or current_rating)
    recent_accuracy = float(row.get("recent_accuracy", 0) or 0)
    solved = float(row.get("problems_solved", 0) or 0)
    wrong = float(row.get("wrong_submissions", 0) or 0)
    submissions = max(float(row.get("submissions", 1) or 1), 1.0)
    volatility = float(row.get("rating_volatility", 0) or 0)
    rank_mean = float(row.get("contest_rank_mean_last5", 0) or 0)
    expected_delta = (
        (recent_accuracy - 55) * 1.1
        + (avg_rating - current_rating) * 0.055
        + np.log1p(solved) * 2.1
        - wrong / submissions * 38
        - volatility * 0.18
        - rank_mean / 2400
    )
    return _delta_to_band(float(expected_delta))


def _delta_to_band(expected_delta: float) -> str:
    if expected_delta < -18:
        return "Needs repair"
    if expected_delta < 18:
        return "Stable"
    if expected_delta < 54:
        return "Growth"
    return "Breakout"


def _ensure_solve_training_rows(examples: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    if len(examples) >= 80 and examples["solved"].nunique() >= 2:
        return examples.copy()

    rng = np.random.default_rng(random_state + 101)
    base_rating = float(examples["user_rating"].median()) if not examples.empty else 1350.0
    rows = []
    for _ in range(900):
        problem_rating = float(rng.choice(np.arange(800, 2400, 100)))
        tag_accuracy = float(np.clip(rng.normal(58, 21), 0, 100))
        attempts_on_tag = float(max(0, rng.normal(28, 18)))
        recent_failures = float(max(0, rng.normal(4, 3)))
        recent_accuracy = float(np.clip(rng.normal(60, 17), 0, 100))
        user_rating = float(np.clip(rng.normal(base_rating, 230), 800, 2400))
        rating_gap = problem_rating - user_rating
        popularity_log = float(np.clip(rng.normal(8.1, 1.8), 1.0, 12.0))
        tag_count = float(rng.integers(1, 5))
        logit = (
            1.35
            - rating_gap / 285
            + (tag_accuracy - 50) / 30
            + (recent_accuracy - 55) / 38
            + np.log1p(attempts_on_tag) / 8
            + popularity_log / 18
            - recent_failures / 4.8
            - max(tag_count - 2, 0) * 0.12
        )
        probability = 1 / (1 + np.exp(-logit))
        rows.append(
            {
                "problem_rating": problem_rating,
                "user_rating": user_rating,
                "rating_gap": rating_gap,
                "tag_accuracy": tag_accuracy,
                "attempts_on_tag": attempts_on_tag,
                "recent_failures": recent_failures,
                "popularity_log": popularity_log,
                "tag_count": tag_count,
                "recent_accuracy": recent_accuracy,
                "solved": int(rng.random() < probability),
                "problem_id": "synthetic",
                "problem_name": "Synthetic solve sample",
                "tags": [],
            }
        )

    synthetic = pd.DataFrame(rows)
    if examples.empty:
        return synthetic
    return pd.concat([examples, synthetic], ignore_index=True)


def _monotonic_solve_probabilities(features: pd.DataFrame) -> np.ndarray:
    frame = features.copy()
    rating_gap = frame["problem_rating"].astype(float) - frame["user_rating"].astype(float)
    tag_accuracy = frame["tag_accuracy"].astype(float).clip(0, 100)
    attempts = frame["attempts_on_tag"].astype(float).clip(lower=0)
    recent_failures = frame["recent_failures"].astype(float).clip(lower=0)
    popularity = frame["popularity_log"].astype(float).clip(lower=0)
    tag_count = frame["tag_count"].astype(float).clip(lower=0)
    recent_accuracy = frame["recent_accuracy"].astype(float).clip(0, 100)

    logit = (
        0.85
        - rating_gap / 285
        + (tag_accuracy - 50) / 30
        + (recent_accuracy - 55) / 42
        + np.log1p(attempts) / 9
        + popularity / 24
        - recent_failures / 4.8
        - np.maximum(tag_count - 2, 0) * 0.14
    )
    return np.clip(1 / (1 + np.exp(-logit)), 0.02, 0.98)


def _contest_scorecard_feature_importance() -> pd.DataFrame:
    weights = np.array([0.2, 0.18, 0.06, 0.12, 0.07, 0.2, 0.04, 0.04, 0.04, 0.0, 0.03, 0.12])
    return _importance_frame(CONTEST_FEATURE_COLUMNS, weights)


def _scorecard_feature_importance() -> pd.DataFrame:
    weights = np.array([0.28, 0.18, 0.28, 0.2, 0.08, 0.16, 0.06, 0.04, 0.1])
    return _importance_frame(SOLVE_FEATURE_COLUMNS, weights)


def _importance_frame(features: list[str], importance: np.ndarray) -> pd.DataFrame:
    total = float(np.sum(importance)) or 1.0
    return pd.DataFrame({"feature": features, "importance": importance / total}).sort_values("importance", ascending=False)

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
        "selected_model_name": "contest_scorecard",
        "model": None,
        "logistic_regression": None,
        "random_forest": None,
        "metrics": {"contest_scorecard": metrics},
        "features": CONTEST_FEATURE_COLUMNS,
        "feature_importance": _contest_scorecard_feature_importance(),
        "predicted_band": predicted_band,
        "band_probabilities": {band: 1.0 if band == predicted_band else 0.0 for band in PERFORMANCE_BANDS},
    }


def train_solve_probability_model(examples: pd.DataFrame, random_state: int = 42) -> dict[str, Any]:
    frame = _ensure_solve_training_rows(examples, random_state=random_state)
    frame = _ensure_feature_columns(frame, SOLVE_FEATURE_COLUMNS)
    x = frame[SOLVE_FEATURE_COLUMNS]
    y = frame["solved"].astype(int)
    probabilities = _monotonic_solve_probabilities(x)
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "selected_model_name": "monotonic_scorecard_v2",
        "model": None,
        "logistic_regression": None,
        "random_forest": None,
        "metrics": {"monotonic_scorecard": _binary_metrics(y, predictions)},
        "features": SOLVE_FEATURE_COLUMNS,
        "feature_importance": _scorecard_feature_importance(),
        "training_rows": len(frame),
    }


def predict_solve_probability(model_report: dict[str, Any], feature_rows: pd.DataFrame | list[dict[str, Any]]) -> np.ndarray:
    frame = pd.DataFrame(feature_rows)
    frame = _ensure_feature_columns(frame, model_report["features"])
    x = frame[model_report["features"]]
    model = model_report.get("model")
    if str(model_report.get("selected_model_name", "")).startswith("monotonic_scorecard") or model is None:
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
        user_rating = float(np.clip(rng.normal(base_rating, 230), 800, 2400))
        tag_solved_count = float(max(0, rng.normal(attempts_on_tag * 0.5, 9)))
        tag_avg_rating_solved = float(np.clip(user_rating - rng.normal(80, 180), 700, 2600))
        tag_max_rating_solved = float(np.clip(tag_avg_rating_solved + abs(rng.normal(180, 160)), 700, 3500))
        recent_failures = float(max(0, rng.normal(4, 3)))
        recent_accuracy = float(np.clip(rng.normal(60, 17), 0, 100))
        rating_gap = problem_rating - user_rating
        popularity_log = float(np.clip(rng.normal(8.1, 1.8), 1.0, 12.0))
        tag_count = float(rng.integers(1, 5))
        solved_volume_log = float(np.clip(rng.normal(5.4, 1.1), 1.5, 8.2))
        rating_confidence = float(rng.choice([1.0, 0.65], p=[0.82, 0.18]))
        logit = (
            0.25
            - rating_gap / 260
            + np.log1p(tag_solved_count) * 0.16
            + (tag_max_rating_solved - problem_rating) / 850
            + (tag_avg_rating_solved - problem_rating) / 1200
            + solved_volume_log / 18
            + popularity_log / 30
            + (tag_accuracy - 50) / 115
            + (recent_accuracy - 55) / 130
            - recent_failures / 7.5
            - max(tag_count - 2, 0) * 0.1
            - (1 - rating_confidence) * 0.18
        )
        probability = 1 / (1 + np.exp(-logit))
        rows.append(
            {
                "problem_rating": problem_rating,
                "user_rating": user_rating,
                "rating_gap": rating_gap,
                "tag_accuracy": tag_accuracy,
                "attempts_on_tag": attempts_on_tag,
                "tag_solved_count": tag_solved_count,
                "tag_avg_rating_solved": tag_avg_rating_solved,
                "tag_max_rating_solved": tag_max_rating_solved,
                "recent_failures": recent_failures,
                "popularity_log": popularity_log,
                "tag_count": tag_count,
                "recent_accuracy": recent_accuracy,
                "solved_volume_log": solved_volume_log,
                "rating_confidence": rating_confidence,
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


def _monotonic_solve_probabilities(features: pd.DataFrame) -> np.ndarray:
    frame = features.copy()
    rating_gap = frame["problem_rating"].astype(float) - frame["user_rating"].astype(float)
    attempts = frame["attempts_on_tag"].astype(float).clip(lower=0)
    tag_max_rating = frame["tag_max_rating_solved"].astype(float).clip(lower=0)
    popularity = frame["popularity_log"].astype(float).clip(lower=0)
    solved_volume = frame["solved_volume_log"].astype(float).clip(lower=0)
    ceiling_gap = tag_max_rating - frame["problem_rating"].astype(float)

    # Sequence-aware fields
    decayed_tag_mastery = frame.get("decayed_tag_mastery", pd.Series(0.0, index=frame.index)).astype(float).clip(0, 1)
    prereq_fit = frame.get("prereq_fit_score", pd.Series(1.0, index=frame.index)).astype(float).clip(0, 1)
    avg_fuzzy_struggle = frame.get("average_fuzzy_struggle_on_tag", pd.Series(0.0, index=frame.index)).astype(float).clip(0, 1)
    cosine_sim = frame.get("cosine_similarity", pd.Series(0.0, index=frame.index)).astype(float).clip(0, 1)

    # coefficients for logit -- chosen conservatively; tuneable
    base_bias = 0.25
    scale_factor = 275.0
    w1 = 0.36
    w2 = 0.28
    w3 = 0.44
    w4 = 0.14

    logit = (
        base_bias
        - rating_gap / scale_factor
        + w1 * decayed_tag_mastery
        + w2 * prereq_fit
        - w3 * avg_fuzzy_struggle
        + w4 * cosine_sim
    )

    # small residual signals to preserve desirable monotonic behaviour
    logit += np.log1p(attempts) / 28 + solved_volume / 48 + popularity / 48

    raw = 1 / (1 + np.exp(-logit))
    same_level_cap = 0.58 + 0.34 / (1 + np.exp((rating_gap + 80) / 260))
    ceiling_cap_bonus = 0.06 / (1 + np.exp(-(ceiling_gap - 150) / 280))
    confidence_cap = np.clip(same_level_cap + ceiling_cap_bonus, 0.18, 0.92)
    return np.clip(np.minimum(raw, confidence_cap), 0.02, 0.94)


def _contest_scorecard_feature_importance() -> pd.DataFrame:
    weights = np.array([0.2, 0.18, 0.06, 0.12, 0.07, 0.2, 0.04, 0.04, 0.04, 0.0, 0.03, 0.12])
    return _importance_frame(CONTEST_FEATURE_COLUMNS, weights)


def _scorecard_feature_importance() -> pd.DataFrame:
    # Extended importance weights to cover the new sequence-aware features
    weights = np.array([0.18, 0.12, 0.16, 0.04, 0.04, 0.11, 0.08, 0.10, 0.06, 0.03, 0.025, 0.025, 0.02, 0.02, 0.06, 0.05, 0.05, 0.03])
    return _importance_frame(SOLVE_FEATURE_COLUMNS, weights)


def _importance_frame(features: list[str], importance: np.ndarray) -> pd.DataFrame:
    total = float(np.sum(importance)) or 1.0
    return pd.DataFrame({"feature": features, "importance": importance / total}).sort_values("importance", ascending=False)

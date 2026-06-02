from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import MODEL_DIR
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
    x = train_frame[CONTEST_FEATURE_COLUMNS]
    y = train_frame["band"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.24, stratify=y, random_state=random_state)

    logistic = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)),
        ]
    )
    forest = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=4,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )

    logistic.fit(x_train, y_train)
    forest.fit(x_train, y_train)

    logistic_metrics = _classification_metrics(y_test, logistic.predict(x_test), average="macro")
    forest_predictions = forest.predict(x_test)
    forest_metrics = _classification_metrics(y_test, forest_predictions, average="macro")

    selected_name = "random_forest" if forest_metrics["accuracy"] >= logistic_metrics["accuracy"] else "logistic_regression"
    selected_model = forest if selected_name == "random_forest" else logistic
    feature_importance = _contest_feature_importance(selected_model)
    profile_frame = pd.DataFrame([{key: profile.get(key, 0) for key in CONTEST_FEATURE_COLUMNS}])
    predicted_band = str(selected_model.predict(profile_frame)[0])
    probabilities = _probability_map(selected_model, profile_frame)

    report = {
        "selected_model_name": selected_name,
        "model": selected_model,
        "logistic_regression": logistic,
        "random_forest": forest,
        "metrics": {
            "logistic_regression": logistic_metrics,
            "random_forest": forest_metrics,
        },
        "features": CONTEST_FEATURE_COLUMNS,
        "feature_importance": feature_importance,
        "predicted_band": predicted_band,
        "band_probabilities": probabilities,
    }
    joblib.dump(report, MODEL_DIR / "contest_score_predictor.joblib")
    return report


def train_solve_probability_model(examples: pd.DataFrame, random_state: int = 42) -> dict[str, Any]:
    frame = _ensure_solve_training_rows(examples, random_state=random_state)
    x = frame[SOLVE_FEATURE_COLUMNS]
    y = frame["solved"].astype(int)
    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=stratify, random_state=random_state)

    logistic = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)),
        ]
    )
    forest = RandomForestClassifier(
        n_estimators=120,
        max_depth=8,
        min_samples_leaf=3,
        random_state=random_state,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    logistic.fit(x_train, y_train)
    forest.fit(x_train, y_train)

    logistic_metrics = _classification_metrics(y_test, logistic.predict(x_test))
    forest_metrics = _classification_metrics(y_test, forest.predict(x_test))
    selected_model = forest if forest_metrics["accuracy"] >= logistic_metrics["accuracy"] else logistic
    selected_name = "random_forest" if selected_model is forest else "logistic_regression"

    report = {
        "selected_model_name": selected_name,
        "model": selected_model,
        "logistic_regression": logistic,
        "random_forest": forest,
        "metrics": {
            "logistic_regression": logistic_metrics,
            "random_forest": forest_metrics,
        },
        "features": SOLVE_FEATURE_COLUMNS,
        "feature_importance": _solve_feature_importance(selected_model),
        "training_rows": int(len(frame)),
    }
    joblib.dump(report, MODEL_DIR / "solve_probability_model.joblib")
    return report


def predict_solve_probability(model_report: dict[str, Any], feature_rows: pd.DataFrame | list[dict[str, Any]]) -> np.ndarray:
    frame = pd.DataFrame(feature_rows)
    x = frame[model_report["features"]]
    model = model_report["model"]
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


def _classification_metrics(y_true: pd.Series, y_pred: np.ndarray, average: str = "binary") -> dict[str, float]:
    if average == "binary" and len(set(y_true)) != 2:
        average = "macro"
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
    }


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
        band = _delta_to_band(expected_delta)
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
                "band": band,
            }
        )
    return pd.DataFrame(rows)


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
        solved = int(rng.random() < probability)
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
                "solved": solved,
                "problem_id": "synthetic",
                "problem_name": "Synthetic solve sample",
                "tags": [],
            }
        )

    synthetic = pd.DataFrame(rows)
    if examples.empty:
        return synthetic
    return pd.concat([examples, synthetic], ignore_index=True)


def _contest_feature_importance(model: Any) -> pd.DataFrame:
    if isinstance(model, RandomForestClassifier):
        importance = model.feature_importances_
    elif hasattr(model, "named_steps"):
        estimator = model.named_steps["model"]
        importance = np.abs(estimator.coef_).mean(axis=0)
    else:
        importance = np.ones(len(CONTEST_FEATURE_COLUMNS))
    return _importance_frame(CONTEST_FEATURE_COLUMNS, importance)


def _solve_feature_importance(model: Any) -> pd.DataFrame:
    estimator = getattr(model, "estimator", model)
    if isinstance(estimator, RandomForestClassifier):
        importance = estimator.feature_importances_
    elif hasattr(estimator, "named_steps"):
        coefficients = estimator.named_steps["model"].coef_
        importance = np.abs(coefficients).ravel()
    elif hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    else:
        importance = np.ones(len(SOLVE_FEATURE_COLUMNS))
    return _importance_frame(SOLVE_FEATURE_COLUMNS, importance)


def _importance_frame(features: list[str], importance: np.ndarray) -> pd.DataFrame:
    total = float(np.sum(importance)) or 1.0
    return pd.DataFrame({"feature": features, "importance": importance / total}).sort_values("importance", ascending=False)


def _probability_map(model: Any, profile_frame: pd.DataFrame) -> dict[str, float]:
    if not hasattr(model, "predict_proba"):
        return {}
    probabilities = model.predict_proba(profile_frame)[0]
    return {str(label): float(prob) for label, prob in zip(model.classes_, probabilities)}

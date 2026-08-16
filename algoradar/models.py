from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

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


REAL_LABEL_FEATURE_COLUMNS = [
    "user_rating_at_time",
    "problem_rating",
    "difficulty_gap",
    "tag_mastery",
    "previous_attempts",
    "recent_activity",
    "solved_before",
    "tag_count",
]


def train_solve_probability_model(examples: pd.DataFrame, random_state: int = 42) -> dict[str, Any]:
    frame = examples.copy()
    if _has_real_label_columns(frame):
        return _train_real_label_model(frame, random_state=random_state)

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


def build_real_solve_training_dataset(
    cache_dir: str | Path | None = None,
    max_users: int | None = None,
    min_examples_per_user: int = 2,
    save_path: str | Path | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    cache_root = Path(cache_dir) if cache_dir is not None else Path(__file__).resolve().parent.parent / "data" / "cache"
    if not cache_root.exists():
        empty = pd.DataFrame(columns=REAL_LABEL_FEATURE_COLUMNS + ["y", "problem_id", "handle", "result", "time_to_solve"])
        if save_path is not None:
            target = Path(save_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            empty.to_csv(target, index=False)
        return empty

    problemset = json.loads((cache_root / "problemset.json").read_text(encoding="utf-8")) if (cache_root / "problemset.json").exists() else {"problems": []}
    problem_lookup = {}
    for problem in problemset.get("problems", []):
        pid = f"{problem.get('contestId', 'unknown')}{problem.get('index', 'X')}"
        tags = problem.get("tags") or []
        rating = problem.get("rating")
        problem_lookup[pid] = {"tags": list(tags), "rating": float(rating) if rating is not None else None}

    rows: list[dict[str, Any]] = []
    handles = sorted({path.name.split("user_status_")[1].rsplit("_", 1)[0] for path in cache_root.glob("user_status_*.json")})
    if max_users is not None:
        handles = handles[:max_users]

    for handle in handles:
        status_path = cache_root / f"user_status_{handle}_1200.json"
        if not status_path.exists():
            matches = sorted(cache_root.glob(f"user_status_{handle}_*.json"))
            if not matches:
                continue
            status_path = matches[0]
        submissions = json.loads(status_path.read_text(encoding="utf-8"))
        if not submissions:
            continue
        sorted_submissions = sorted(submissions, key=lambda s: int(s.get("creationTimeSeconds", 0)))
        ratings = []
        rating_path = cache_root / f"user_rating_{handle}.json"
        if rating_path.exists():
            ratings = json.loads(rating_path.read_text(encoding="utf-8"))
        rating_history = sorted(ratings, key=lambda item: int(item.get("ratingUpdateTimeSeconds", 0)))

        by_problem: dict[str, list[dict[str, Any]]] = {}
        for submission in sorted_submissions:
            problem = submission.get("problem", {})
            pid = f"{problem.get('contestId', 'unknown')}{problem.get('index', 'X')}"
            by_problem.setdefault(pid, []).append(submission)

        for pid, items in by_problem.items():
            if len(items) < min_examples_per_user:
                continue
            first_seen = min(int(item.get("creationTimeSeconds", 0)) for item in items)
            first_accepted = next((int(item.get("creationTimeSeconds", 0)) for item in items if item.get("verdict") == "OK"), None)
            problem_info = items[0].get("problem", {})
            problem_tags = list(problem_info.get("tags") or problem_lookup.get(pid, {}).get("tags", []))
            problem_rating = float(problem_info.get("rating") or problem_lookup.get(pid, {}).get("rating") or 1200.0)
            user_rating_at_time = _rating_at_time(handle, rating_history, first_seen, default=problem_rating)
            prior = [item for item in sorted_submissions if int(item.get("creationTimeSeconds", 0)) < first_seen]
            prior_same_tags = [item for item in prior if set(problem_tags).intersection(item.get("problem", {}).get("tags", []))]
            prior_accepted_same_tag = sum(1 for item in prior_same_tags if item.get("verdict") == "OK")
            tag_mastery = 0.0
            if prior_same_tags:
                tag_mastery = sum(1.0 for item in prior_same_tags if item.get("verdict") == "OK") / max(len(prior_same_tags), 1)
            recent_window_start = first_seen - 30 * 86400
            recent_activity = sum(1 for item in prior if int(item.get("creationTimeSeconds", 0)) >= recent_window_start)
            solved_before = 1 if any(item.get("verdict") == "OK" for item in prior if item.get("problem", {}).get("contestId") == problem_info.get("contestId") and item.get("problem", {}).get("index") == problem_info.get("index")) else 0
            previous_attempts = sum(1 for item in prior if item.get("problem", {}).get("contestId") == problem_info.get("contestId") and item.get("problem", {}).get("index") == problem_info.get("index"))
            difficulty_gap = problem_rating - user_rating_at_time
            time_to_solve = (first_accepted - first_seen) / 86400.0 if first_accepted is not None else np.nan

            rows.append(
                {
                    "handle": handle,
                    "problem_id": pid,
                    "y": 1 if first_accepted is not None else 0,
                    "result": "solved" if first_accepted is not None else "unsolved",
                    "user_rating_at_time": float(user_rating_at_time),
                    "problem_rating": float(problem_rating),
                    "difficulty_gap": float(difficulty_gap),
                    "tag_mastery": float(tag_mastery),
                    "previous_attempts": float(previous_attempts),
                    "recent_activity": float(recent_activity),
                    "solved_before": float(solved_before),
                    "tag_count": float(len(problem_tags)),
                    "time_to_solve": float(time_to_solve) if not pd.isna(time_to_solve) else np.nan,
                }
            )

    dataset = pd.DataFrame(rows)
    if dataset.empty:
        if save_path is not None:
            target = Path(save_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            dataset.to_csv(target, index=False)
        return dataset
    dataset["y"] = dataset["y"].astype(int)
    dataset["tag_mastery"] = dataset["tag_mastery"].clip(0.0, 1.0)
    if save_path is not None:
        target = Path(save_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(target, index=False)
    return dataset


def load_or_build_real_label_dataset(
    cache_dir: str | Path | None = None,
    max_users: int | None = None,
    min_examples_per_user: int = 2,
    dataset_path: str | Path | None = None,
) -> pd.DataFrame:
    cache_root = Path(cache_dir) if cache_dir is not None else Path(__file__).resolve().parent.parent / "data" / "cache"
    default_path = Path(dataset_path) if dataset_path is not None else cache_root / "real_solve_training_dataset.csv"
    if default_path.exists():
        return pd.read_csv(default_path)
    return build_real_solve_training_dataset(
        cache_dir=str(cache_root),
        max_users=max_users,
        min_examples_per_user=min_examples_per_user,
        save_path=default_path,
        overwrite=True,
    )


def _rating_at_time(handle: str, rating_history: list[dict[str, Any]], event_time: int, default: float = 1200.0) -> float:
    current = default
    for row in rating_history:
        rating_time = int(row.get("ratingUpdateTimeSeconds", 0))
        if rating_time <= event_time:
            current = float(row.get("newRating", current) or current)
    return float(current)


def _has_real_label_columns(frame: pd.DataFrame) -> bool:
    return {"y", "user_rating_at_time", "problem_rating", "difficulty_gap", "tag_mastery"}.issubset(frame.columns)


def evaluate_real_label_models(frame: pd.DataFrame, random_state: int = 42) -> dict[str, Any]:
    working = frame.copy()
    working = working[REAL_LABEL_FEATURE_COLUMNS + ["y"]].dropna(subset=REAL_LABEL_FEATURE_COLUMNS + ["y"]).copy()
    if working.empty or working["y"].nunique() < 2:
        return {
            "selected_model_name": "heuristic_fallback",
            "model": None,
            "logistic_regression": None,
            "calibrated_logistic": None,
            "random_forest": None,
            "metrics": {"status": "insufficient_real_labels"},
            "candidate_metrics": {},
            "features": REAL_LABEL_FEATURE_COLUMNS,
            "training_rows": len(working),
        }

    x = working[REAL_LABEL_FEATURE_COLUMNS]
    y = working["y"].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=random_state,
        stratify=y,
    )
    imputer = SimpleImputer(strategy="median")
    x_train_imputed = imputer.fit_transform(x_train)
    x_test_imputed = imputer.transform(x_test)

    candidate_models: dict[str, Any] = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state),
    }
    if y_train.nunique() == 2 and int(y_train.value_counts().min()) >= 3:
        candidate_models["calibrated_logistic"] = CalibratedClassifierCV(
            estimator=LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state),
            method="sigmoid",
            cv=3,
        )

    candidate_metrics: dict[str, dict[str, float]] = {}
    fitted_models: dict[str, Any] = {}
    for name, model in candidate_models.items():
        model.fit(x_train_imputed, y_train)
        fitted_models[name] = model
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(x_test_imputed)[:, 1]
        else:
            probs = model.predict(x_test_imputed).astype(float)
        candidate_metrics[name] = _classifier_metrics(y_test.to_numpy(), probs)

    best_name = max(
        candidate_metrics,
        key=lambda name: (
            candidate_metrics[name].get("roc_auc", 0.0),
            candidate_metrics[name].get("pr_auc", 0.0),
            candidate_metrics[name].get("ndcg@10", 0.0),
        ),
    )
    best_model = fitted_models[best_name]
    pipeline = {
        "imputer": imputer,
        "model": best_model,
        "feature_columns": REAL_LABEL_FEATURE_COLUMNS,
    }

    return {
        "selected_model_name": f"{best_name}_real_labels",
        "model": pipeline,
        "logistic_regression": fitted_models.get("logistic_regression"),
        "calibrated_logistic": fitted_models.get("calibrated_logistic"),
        "random_forest": None,
        "metrics": candidate_metrics[best_name],
        "candidate_metrics": candidate_metrics,
        "features": REAL_LABEL_FEATURE_COLUMNS,
        "training_rows": len(working),
    }


def _train_real_label_model(frame: pd.DataFrame, random_state: int = 42) -> dict[str, Any]:
    return evaluate_real_label_models(frame, random_state=random_state)


def _classifier_metrics(y_true: np.ndarray, y_prob: np.ndarray, ks: tuple[int, ...] = (5, 10)) -> dict[str, float]:
    true = np.asarray(y_true, dtype=int)
    prob = np.asarray(y_prob, dtype=float).clip(0.0, 1.0)
    pred = (prob >= 0.5).astype(int)
    if len(np.unique(true)) < 2:
        return {"roc_auc": 0.5, "pr_auc": 0.5, "brier_score": float(brier_score_loss(true, prob)), "calibration_error": 0.0, "precision@5": 0.0, "recall@5": 0.0, "precision@10": 0.0, "recall@10": 0.0, "ndcg@5": 0.0, "ndcg@10": 0.0}

    roc = float(roc_auc_score(true, prob))
    pr_auc = float(average_precision_score(true, prob))
    brier = float(brier_score_loss(true, prob))
    calibration_error = _calibration_error(true, prob)
    metrics: dict[str, float] = {
        "roc_auc": roc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "calibration_error": calibration_error,
    }
    for k in ks:
        precision, recall = _precision_recall_at_k(true, prob, k)
        ndcg = _ndcg_at_k(true, prob, k)
        metrics[f"precision@{k}"] = float(precision)
        metrics[f"recall@{k}"] = float(recall)
        metrics[f"ndcg@{k}"] = float(ndcg)
    return metrics


def _calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    bins_edges = np.linspace(0.0, 1.0, bins + 1)
    deltas: list[float] = []
    for left, right in zip(bins_edges[:-1], bins_edges[1:]):
        mask = (y_prob >= left) & (y_prob < right) if right < 1.0 else (y_prob >= left) & (y_prob <= right)
        if not np.any(mask):
            continue
        avg_pred = float(y_prob[mask].mean())
        avg_true = float(y_true[mask].mean())
        deltas.append(abs(avg_pred - avg_true))
    return float(np.mean(deltas)) if deltas else 0.0


def _precision_recall_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: int) -> tuple[float, float]:
    order = np.argsort(-y_prob)[: k]
    if order.size == 0:
        return 0.0, 0.0
    hits = int(np.sum(y_true[order] == 1))
    precision = hits / max(len(order), 1)
    recall = hits / max(np.sum(y_true == 1), 1)
    return precision, recall


def _ndcg_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: int) -> float:
    order = np.argsort(-y_prob)[:k]
    if order.size == 0:
        return 0.0
    labels = y_true[order].astype(float)
    dcg = float(np.sum((2 ** labels - 1) / np.log2(np.arange(2, len(labels) + 2))))
    ideal = np.sort(y_true)[::-1][:k]
    idcg = float(np.sum((2 ** ideal - 1) / np.log2(np.arange(2, len(ideal) + 2)))) if len(ideal) else 0.0
    return dcg / idcg if idcg > 0 else 0.0


def predict_solve_probability(model_report: dict[str, Any], feature_rows: pd.DataFrame | list[dict[str, Any]]) -> np.ndarray:
    frame = pd.DataFrame(feature_rows)
    if "real_labels" in str(model_report.get("selected_model_name", "")):
        pipeline = model_report.get("model") or {}
        model = pipeline.get("model") if isinstance(pipeline, dict) else None
        imputer = pipeline.get("imputer") if isinstance(pipeline, dict) else None
        features = model_report.get("features", REAL_LABEL_FEATURE_COLUMNS)
        frame = frame[features].copy()
        if imputer is not None:
            frame = imputer.transform(frame)
        if model is not None and hasattr(model, "predict_proba"):
            return model.predict_proba(frame)[:, 1]
        if model is not None:
            return np.asarray(model.predict(frame), dtype=float)

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
    recent_mastery = frame.get("recent_mastery", pd.Series(0.0, index=frame.index)).astype(float).clip(0, 1)
    long_term_mastery = frame.get("long_term_mastery", pd.Series(0.0, index=frame.index)).astype(float).clip(0, 1)

    # coefficients for logit -- chosen conservatively; tuneable
    base_bias = 0.25
    scale_factor = 275.0
    # expose module-level tunable weights (defaults chosen conservatively)
    try:
        from . import config as _cfg
        W_DECAY = float(getattr(_cfg, "W_DECAY", 0.36))
        W_PREREQ = float(getattr(_cfg, "W_PREREQ", 0.28))
        W_FUZZY = float(getattr(_cfg, "W_FUZZY", 0.44))
        W_COSINE = float(getattr(_cfg, "W_COSINE", 0.14))
        W_RECENT = float(getattr(_cfg, "W_RECENT", 0.22))
        W_LONGTERM = float(getattr(_cfg, "W_LONGTERM", 0.18))
    except Exception:
        W_DECAY = 0.36
        W_PREREQ = 0.28
        W_FUZZY = 0.44
        W_COSINE = 0.14
        W_RECENT = 0.22
        W_LONGTERM = 0.18

    logit = (
        base_bias
        - rating_gap / scale_factor
        + W_DECAY * decayed_tag_mastery
        + W_PREREQ * prereq_fit
        - W_FUZZY * avg_fuzzy_struggle
        + W_COSINE * cosine_sim
        + W_RECENT * recent_mastery
        + W_LONGTERM * long_term_mastery
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
    weights = np.array([0.18, 0.12, 0.16, 0.04, 0.04, 0.11, 0.08, 0.10, 0.06, 0.03, 0.025, 0.025, 0.02, 0.02, 0.06, 0.05, 0.05, 0.03, 0.04, 0.03])
    return _importance_frame(SOLVE_FEATURE_COLUMNS, weights)


def _importance_frame(features: list[str], importance: np.ndarray) -> pd.DataFrame:
    total = float(np.sum(importance)) or 1.0
    return pd.DataFrame({"feature": features, "importance": importance / total}).sort_values("importance", ascending=False)

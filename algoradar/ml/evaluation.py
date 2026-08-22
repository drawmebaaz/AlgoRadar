from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def temporal_split_user_aware(frame: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return frame.copy(), frame.iloc[0:0].copy(), frame.iloc[0:0].copy()

    sorted_frame = frame.sort_values(["user_id", "timestamp"]).copy()
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for _, user_df in sorted_frame.groupby("user_id", sort=True):
        n = len(user_df)
        if n < 3:
            train_parts.append(user_df)
            continue
        train_end = max(1, int(n * train_ratio))
        val_end = max(train_end + 1, int(n * (train_ratio + val_ratio)))
        train_parts.append(user_df.iloc[:train_end])
        val_parts.append(user_df.iloc[train_end:val_end])
        test_parts.append(user_df.iloc[val_end:])

    train = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame(columns=frame.columns)
    val = pd.concat(val_parts, ignore_index=True) if val_parts else pd.DataFrame(columns=frame.columns)
    test = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=frame.columns)
    return train, val, test


def _precision_recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> tuple[float, float]:
    if len(y_true) == 0:
        return 0.0, 0.0
    order = np.argsort(-y_score)[:k]
    if len(order) == 0:
        return 0.0, 0.0
    hits = int(np.sum(y_true[order] == 1))
    precision = hits / len(order)
    recall = hits / max(np.sum(y_true == 1), 1)
    return precision, recall


def _ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    if len(y_true) == 0:
        return 0.0
    labels = np.asarray(y_true, dtype=float)
    scores = np.asarray(y_score, dtype=float)
    order = np.argsort(-scores)
    sorted_labels = labels[order][:k]
    ideal = np.sort(labels)[::-1][:k]
    if ideal.size == 0:
        return 0.0
    dcg = np.sum((2 ** sorted_labels - 1) / np.log2(np.arange(2, len(sorted_labels) + 2)))
    idcg = np.sum((2 ** ideal - 1) / np.log2(np.arange(2, len(ideal) + 2)))
    return float(dcg / idcg) if idcg > 0 else 0.0


def _calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    if len(y_true) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    errors: list[float] = []
    for left, right in zip(edges[:-1], edges[1:]):
        if right < 1.0:
            mask = (y_prob >= left) & (y_prob < right)
        else:
            mask = (y_prob >= left) & (y_prob <= right)
        if not np.any(mask):
            continue
        avg_pred = float(y_prob[mask].mean())
        avg_true = float(y_true[mask].mean())
        errors.append(abs(avg_pred - avg_true))
    return float(np.mean(errors)) if errors else 0.0


def evaluate_probability_model(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    true = np.asarray(y_true, dtype=int)
    prob = np.asarray(y_prob, dtype=float).clip(0.0, 1.0)
    if len(np.unique(true)) < 2:
        return {
            "roc_auc": 0.5,
            "pr_auc": 0.5,
            "brier_score": float(brier_score_loss(true, prob)),
            "calibration_error": 0.0,
            "precision@5": 0.0,
            "recall@5": 0.0,
            "ndcg@5": 0.0,
            "precision@10": 0.0,
            "recall@10": 0.0,
            "ndcg@10": 0.0,
        }

    metrics: dict[str, float] = {
        "roc_auc": float(roc_auc_score(true, prob)),
        "pr_auc": float(average_precision_score(true, prob)),
        "brier_score": float(brier_score_loss(true, prob)),
        "calibration_error": _calibration_error(true, prob),
    }
    for k in (5, 10):
        precision, recall = _precision_recall_at_k(true, prob, k)
        ndcg = _ndcg_at_k(true, prob, k)
        metrics[f"precision@{k}"] = float(precision)
        metrics[f"recall@{k}"] = float(recall)
        metrics[f"ndcg@{k}"] = float(ndcg)
    return metrics


def evaluate_model_suite(models: dict[str, Any], feature_frame: pd.DataFrame, target_col: str = "y") -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        if model is None:
            continue
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(feature_frame)[:, 1]
        elif hasattr(model, "predict"):
            probabilities = model.predict(feature_frame).astype(float)
        else:
            continue
        results[name] = evaluate_probability_model(feature_frame[target_col].to_numpy(), probabilities)
    return results


def run_ablation_study(frame: pd.DataFrame, model_factory: Any) -> dict[str, dict[str, float]]:
    feature_groups = {
        "rating_gap_only": ["difficulty_gap"],
        "rating_gap_plus_tag_mastery": ["difficulty_gap", "tag_mastery_before_attempt"],
        "rating_gap_plus_tag_mastery_plus_recent": ["difficulty_gap", "tag_mastery_before_attempt", "recent_activity_score"],
        "all_features": [col for col in frame.columns if col not in {"user_id", "problem_id", "timestamp", "y"}],
    }
    results: dict[str, dict[str, float]] = {}
    for name, columns in feature_groups.items():
        subset = frame[[*columns, "y"]].copy()
        model = model_factory(subset)
        probs = model.predict_proba(subset[columns])[:, 1]
        results[name] = evaluate_probability_model(subset["y"].to_numpy(), probs)
    return results


def summarize_feature_importance(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_col: str = "y",
    random_state: int = 42,
) -> pd.DataFrame:
    """Return the most informative features from a logistic model trained on the frame."""
    if frame.empty or not feature_columns:
        return pd.DataFrame(columns=["feature", "absolute_weight", "weight"])

    model_frame = frame[feature_columns + [target_col]].dropna().copy()
    if model_frame.empty or model_frame[target_col].nunique() < 2:
        return pd.DataFrame(columns=["feature", "absolute_weight", "weight"])

    x = model_frame[feature_columns]
    y = model_frame[target_col].astype(int)
    imputer = SimpleImputer(strategy="median")
    logistic = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)
    logistic.fit(imputer.fit_transform(x), y)
    coefs = logistic.coef_[0]
    importance = pd.DataFrame({"feature": feature_columns, "weight": coefs})
    importance["absolute_weight"] = importance["weight"].abs()
    return importance.sort_values("absolute_weight", ascending=False).reset_index(drop=True)


def build_model_comparison_report(
    frame: pd.DataFrame,
    feature_columns: list[str] | None = None,
    target_col: str = "y",
    test_split: str = "test",
    random_state: int = 42,
) -> dict[str, Any]:
    """Train a few simple baselines and return a model comparison across temporal splits."""
    working = frame.copy()
    if working.empty:
        return {
            "split": {"train": 0, "val": 0, "test": 0},
            "best_model": "none",
            "by_model": {},
            "selected_model": None,
        }

    required = {"user_id", "timestamp", target_col}
    missing = required - set(working.columns)
    if missing:
        raise ValueError(f"Comparison report requires temporal columns: {sorted(missing)}")

    if feature_columns is None:
        feature_columns = [col for col in working.columns if col not in {"user_id", "problem_id", "timestamp", target_col}]
    feature_columns = [col for col in feature_columns if col in working.columns]
    if not feature_columns:
        raise ValueError("No feature columns available for temporal comparison.")

    train, val, test = temporal_split_user_aware(working, train_ratio=0.7, val_ratio=0.15)
    candidate_sets = {"train": train, "val": val, "test": test}
    split_counts = {name: len(df) for name, df in candidate_sets.items()}

    models: dict[str, Any] = {}
    constant_rate = float(train[target_col].mean()) if not train.empty else 0.0
    for name, split in candidate_sets.items():
        if split.empty:
            continue
        base_probs = np.full(len(split), fill_value=constant_rate, dtype=float)
        models[f"constant_{name}"] = base_probs

        if "difficulty_gap" in split.columns and target_col in split.columns:
            gap = split["difficulty_gap"].astype(float)
            heuristic = 1.0 / (1.0 + np.exp(gap / 350.0))
            models[f"heuristic_{name}"] = heuristic

    if not train.empty:
        x_train = train[feature_columns].copy()
        y_train = train[target_col].astype(int)
        imputer = SimpleImputer(strategy="median")
        x_fit = imputer.fit_transform(x_train)
        logistic = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)
        logistic.fit(x_fit, y_train)
        for name, split in candidate_sets.items():
            if split.empty:
                continue
            x_split = imputer.transform(split[feature_columns])
            models[f"logistic_{name}"] = logistic.predict_proba(x_split)[:, 1]

        if y_train.nunique() == 2 and int(y_train.value_counts().min()) >= 3:
            calibrated = CalibratedClassifierCV(
                estimator=LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state),
                method="sigmoid",
                cv=3,
            )
            calibrated.fit(x_fit, y_train)
            for name, split in candidate_sets.items():
                if split.empty:
                    continue
                x_split = imputer.transform(split[feature_columns])
                models[f"calibrated_logistic_{name}"] = calibrated.predict_proba(x_split)[:, 1]

    by_model: dict[str, dict[str, float]] = {}
    for label, scores in models.items():
        split_name = label.rsplit("_", 1)[-1]
        if split_name not in candidate_sets:
            continue
        split_df = candidate_sets[split_name]
        metrics = evaluate_probability_model(split_df[target_col].to_numpy(), np.asarray(scores, dtype=float).clip(0.0, 1.0))
        by_model[label] = metrics

    best_model_name = "constant_test"
    best_score = -np.inf
    target_key = test_split
    target_frame = candidate_sets[target_key]
    if not target_frame.empty:
        for name, metrics in by_model.items():
            if name.endswith(f"_{target_key}"):
                score = metrics.get("roc_auc", 0.0) + 0.5 * metrics.get("pr_auc", 0.0)
                if score > best_score:
                    best_score = score
                    best_model_name = name

    report = {
        "split": split_counts,
        "feature_columns": feature_columns,
        "best_model": best_model_name,
        "selected_model": best_model_name,
        "by_model": by_model,
        "target_split": test_split,
    }
    return report

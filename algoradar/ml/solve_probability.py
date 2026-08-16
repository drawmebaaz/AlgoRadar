from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


@dataclass
class SolveModelBundle:
    kind: str
    model: Any
    imputer: Any | None = None
    feature_columns: list[str] | None = None


def fit_logistic_regression_model(frame: pd.DataFrame, features: list[str], target: str = "y", random_state: int = 42) -> SolveModelBundle:
    frame = frame[features + [target]].dropna().copy()
    if frame.empty or frame[target].nunique() < 2:
        raise ValueError("Insufficient data to fit logistic regression.")

    x = frame[features]
    y = frame[target].astype(int)
    x_train, x_test, y_train, _ = train_test_split(x, y, test_size=0.2, random_state=random_state, stratify=y)
    imputer = SimpleImputer(strategy="median")
    x_train_imputed = imputer.fit_transform(x_train)
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)
    model.fit(x_train_imputed, y_train)
    return SolveModelBundle(kind="logistic_regression", model=model, imputer=imputer, feature_columns=features)


def predict_probabilities(bundle: SolveModelBundle, frame: pd.DataFrame) -> np.ndarray:
    features = bundle.feature_columns or list(frame.columns)
    x = frame[features].copy()
    if bundle.imputer is not None:
        x = bundle.imputer.transform(x)
    if hasattr(bundle.model, "predict_proba"):
        return bundle.model.predict_proba(x)[:, 1]
    return bundle.model.predict(x).astype(float)

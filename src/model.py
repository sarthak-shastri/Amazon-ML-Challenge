"""Binary match classifier: training, prediction, persistence."""
from __future__ import annotations

from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .features import FEATURE_COLUMNS


def train_model(
    X: pd.DataFrame, y: np.ndarray, random_seed: int
) -> HistGradientBoostingClassifier:
    """Train a HistGradientBoostingClassifier with class-balanced sample weights.

    HGBC is used because it is permissively licensed (scikit-learn / BSD),
    handles feature scale differences well without preprocessing, and is
    fast enough to train on tens of millions of candidate pairs.
    """
    y = np.asarray(y)
    n_pos = max(int(y.sum()), 1)
    n_neg = max(int((1 - y).sum()), 1)
    pos_weight = n_neg / n_pos
    sample_weight = np.where(y == 1, pos_weight, 1.0)

    clf = HistGradientBoostingClassifier(
        random_state=random_seed,
        max_iter=300,
        learning_rate=0.08,
        max_depth=8,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
    )
    clf.fit(X[FEATURE_COLUMNS], y, sample_weight=sample_weight)
    return clf


def predict_proba(clf: HistGradientBoostingClassifier, X: pd.DataFrame) -> np.ndarray:
    return clf.predict_proba(X[FEATURE_COLUMNS])[:, 1]


def save_model(clf, transformers, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "transformers": transformers}, path)


def load_model(path: Path):
    bundle = joblib.load(path)
    return bundle["model"], bundle["transformers"]

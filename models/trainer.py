from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from config import DEFAULT_MODEL, MODELS, RESULTS_DIR

log = logging.getLogger(__name__)


def build_model(name: str = DEFAULT_MODEL):
    cfg = MODELS[name]
    if name == "random_forest":
        return RandomForestClassifier(**cfg)
    if name == "gradient_boosting":
        return GradientBoostingClassifier(**cfg)
    if name == "logistic_regression":
        return LogisticRegression(**cfg)
    raise ValueError(f"Unknown model: {name}")


def train(model, X_train: pd.DataFrame, y_train: pd.Series):
    log.info("Training %s …", type(model).__name__)
    model.fit(X_train, y_train)
    log.info("Training complete.")
    return model

def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else y_pred.astype(float)
    )

    metrics = {
        "roc_auc":          round(roc_auc_score(y_test, y_prob), 4),
        "f1_weighted":      round(f1_score(y_test, y_pred, average="weighted"), 4),
        "f1_defect":        round(f1_score(y_test, y_pred, pos_label=1), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "report":           classification_report(y_test, y_pred),
    }
    log.info(
        "ROC-AUC=%.4f | F1(defect)=%.4f | F1(weighted)=%.4f",
        metrics["roc_auc"], metrics["f1_defect"], metrics["f1_weighted"],
    )
    print(metrics["report"])
    return metrics

def save_model(model, name: str = DEFAULT_MODEL) -> Path:
    path = RESULTS_DIR / f"{name}_model.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    log.info("Model saved → %s", path)
    return path

def load_model(name: str = DEFAULT_MODEL):
    path = RESULTS_DIR / f"{name}_model.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def feature_importance(model, feature_names: list[str]) -> pd.DataFrame:
    if not hasattr(model, "feature_importances_"):
        log.warning("Model has no feature_importances_; skipping.")
        return pd.DataFrame()
    fi = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    return fi
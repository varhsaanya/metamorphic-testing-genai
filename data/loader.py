from __future__ import annotations

import logging
import urllib.request
from pathlib import Path 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import (
    DATA_DIR, JM1_LOCAL, JM1_URL,
    RANDOM_STATE, TARGET_COLUMN, TEST_SIZE,
)

log = logging.getLogger(__name__)

JM1_COLUMNS = [
    "loc", "v(g)", "ev(g)", "iv(g)", "n", "v", "l", "d", "i", "e",
    "b", "t", "lOCode", "lOComment", "lOBlank", "locCodeAndComment",
    "uniq_Op", "uniq_Opnd", "total_Op", "total_Opnd", "branchCount",
    "defects",
]

def download_jm1(force: bool = False) -> Path:
    if not JM1_LOCAL.exists():
        raise FileNotFoundError(
            f"JM1 dataset not found at {JM1_LOCAL}. Please place jm1.csv there."
        )
    log.info("Using local JM1 dataset: %s", JM1_LOCAL)
    return JM1_LOCAL

def load_raw(path: Path | None = None) -> pd.DataFrame:
    path = path or JM1_LOCAL

    if not Path(path).exists():
        download_jm1()

    df = pd.read_csv(path)

    df.columns = [c.strip().lower() for c in df.columns]

    if TARGET_COLUMN not in df.columns:
        if len(df.columns) == len(JM1_COLUMNS):
            df.columns = [c.lower() for c in JM1_COLUMNS]
        else:
            raise ValueError(
                f"Cannot find '{TARGET_COLUMN}' column. "
                f"Found: {list(df.columns)}"
            )
    return df

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.drop_duplicates(inplace=True)
    df.dropna(how="all", inplace=True)

    target = df[TARGET_COLUMN].astype(str).str.strip().str.lower()

    df[TARGET_COLUMN] = target.map({
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0,
        "yes": 1,
        "no": 0
    })

    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df[TARGET_COLUMN] = df[TARGET_COLUMN].fillna(0).astype(int)

    # Feature columns
    feat_cols = [c for c in df.columns if c != TARGET_COLUMN]
    for col in feat_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    for col in feat_cols:
        median = df[col].median()
        df[col] = df[col].fillna(median)

    # Drop constant columns
    constant_cols = [c for c in feat_cols if df[c].nunique() <= 1]
    if constant_cols:
        log.info("Dropping constant columns: %s", constant_cols)
        df.drop(columns=constant_cols, inplace=True)

    log.info(
        "Preprocessed: %d rows × %d features | defect rate=%.1f%%",
        len(df),
        df.shape[1] - 1,
        100 * df[TARGET_COLUMN].mean(),
    )
    return df


def split_and_scale(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame, pd.DataFrame,
    pd.Series,   pd.Series,
    StandardScaler, list[str],
]:
   
    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    X = df[feature_cols]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_cols,
        index=X_train.index,
    )
    X_test_s = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_cols,
        index=X_test.index,
    )
    return X_train_s, X_test_s, y_train, y_test, scaler, feature_cols


def get_feature_stats(X_train: pd.DataFrame) -> dict:
    return {
        col: {"mean": X_train[col].mean(), "std": X_train[col].std()}
        for col in X_train.columns
    }
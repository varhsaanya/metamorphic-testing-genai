from __future__ import annotations

import logging
import random
import time

import numpy as np
import pandas as pd

from metamorphic.executor import MRResult

log = logging.getLogger(__name__)


def mr_summary_table(results: list[MRResult]) -> pd.DataFrame:
    rows = [
        {
            "mr_name":        r.mr_name,
            "n_samples":      r.n_samples,
            "n_violations":   r.n_violations,
            "violation_rate": round(r.violation_rate, 4),
            "exec_time_s":    round(r.exec_time_s, 4),
        }
        for r in results
    ]
    return pd.DataFrame(rows).sort_values("violation_rate", ascending=False)


def overall_violation_rate(results: list[MRResult]) -> float:
    if not results:
        return 0.0
    return float(np.mean([r.violation_rate for r in results]))


def fault_detection_rate(
    results: list[MRResult],
    y_true: np.ndarray,
) -> pd.DataFrame:
    
    rows = []
    true_defect_idx = set(np.where(y_true == 1)[0].tolist())
    n_defects = len(true_defect_idx)

    for r in results:
        viol_set = set(r.violated_indices)
        true_positives = len(viol_set & true_defect_idx)
        fdr = true_positives / n_defects if n_defects else 0.0
        rows.append(
            {
                "mr_name":          r.mr_name,
                "fault_detect_rate": round(fdr, 4),
                "true_positives":   true_positives,
                "total_defects":    n_defects,
            }
        )
    return pd.DataFrame(rows).sort_values("fault_detect_rate", ascending=False)


def random_testing_baseline(
    model,
    X: pd.DataFrame,
    y_true: np.ndarray,
    n_runs: int = 500,
    noise_scale: float = 0.05,
    seed: int = 0,
) -> dict:
   
    rng = np.random.default_rng(seed)
    true_defect_idx = set(np.where(y_true == 1)[0].tolist())
    n_defects = len(true_defect_idx)

    all_violations:   list[bool] = []
    detected_defects: set[int]   = set()

    t0 = time.perf_counter()
    y_orig = model.predict(X)

    for _ in range(n_runs):
        noise = rng.normal(0, noise_scale, size=X.shape)
        X_noisy = X + noise
        y_noisy  = model.predict(X_noisy)
        flips = y_orig != y_noisy
        all_violations.extend(flips.tolist())
        flipped_idx = set(np.where(flips)[0].tolist())
        detected_defects |= flipped_idx & true_defect_idx

    elapsed = time.perf_counter() - t0

    viol_rate = float(np.mean(all_violations))
    fdr       = len(detected_defects) / n_defects if n_defects else 0.0

    return {
        "method":           "random_testing",
        "n_runs":           n_runs,
        "violation_rate":   round(viol_rate, 4),
        "fault_detect_rate": round(fdr, 4),
        "exec_time_s":      round(elapsed, 4),
    }


def build_comparison_table(
    mr_results: list[MRResult],
    y_true: np.ndarray,
    random_baseline: dict,
) -> pd.DataFrame:
  
    fdr_df = fault_detection_rate(mr_results, y_true)
    fdr_map = dict(zip(fdr_df["mr_name"], fdr_df["fault_detect_rate"]))

    rows = []
    for r in mr_results:
        rows.append(
            {
                "method":            r.mr_name,
                "violation_rate":    r.violation_rate,
                "fault_detect_rate": fdr_map.get(r.mr_name, 0.0),
                "exec_time_s":       r.exec_time_s,
            }
        )
    rows.append(
        {
            "method":            "random_testing",
            "violation_rate":    random_baseline["violation_rate"],
            "fault_detect_rate": random_baseline["fault_detect_rate"],
            "exec_time_s":       random_baseline["exec_time_s"],
        }
    )
    df = pd.DataFrame(rows)
    df["viol_rate_pct"]   = (df["violation_rate"]    * 100).round(2)
    df["fdr_pct"]         = (df["fault_detect_rate"] * 100).round(2)
    return df
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import DEFAULT_MODEL, MODELS, RESULTS_DIR
from data.loader import (
    get_feature_stats,
    load_raw,
    preprocess,
    split_and_scale,
)
from evaluation.metrics import (
    build_comparison_table,
    mr_summary_table,
    overall_violation_rate,
    random_testing_baseline,
)
from evaluation.visualizer import (
    plot_exec_times,
    plot_fault_detection,
    plot_model_comparison_heatmap,
    plot_prob_shift,
    plot_violation_rates,
)
from genai.mr_generator import generate_mrs_from_genai, print_genai_mrs
from metamorphic.executor import run_all_mrs
from metamorphic.relations import ALL_MRS
from models.trainer import (
    build_model,
    evaluate,
    feature_importance,
    save_model,
    train,
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def run_single(model_name: str, use_genai: bool = False) -> dict:
    log.info("=" * 60)
    log.info("Experiment: model=%s", model_name)
    log.info("=" * 60)

    # 1. Data
    df    = preprocess(load_raw())
    X_train, X_test, y_train, y_test, scaler, feat_cols = split_and_scale(df)
    stats = get_feature_stats(X_train)
    y_arr = y_test.values

    # 2. Train & evaluate
    model   = build_model(model_name)
    model   = train(model, X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    save_model(model, model_name)

    fi = feature_importance(model, feat_cols)
    if not fi.empty:
        fi_path = RESULTS_DIR / f"{model_name}_feature_importance.csv"
        fi.to_csv(fi_path, index=False)
        log.info("Feature importance → %s", fi_path)

    if use_genai:
        genai_mrs = generate_mrs_from_genai(n=5)
        print_genai_mrs(genai_mrs)
        (RESULTS_DIR / f"{model_name}_genai_mrs.json").write_text(
            json.dumps(genai_mrs, indent=2)
        )

    log.info("Running %d MRs on %d test samples …", len(ALL_MRS), len(X_test))
    mr_results = run_all_mrs(ALL_MRS, model, X_test, stats)

    log.info("Running random testing baseline …")
    random_bl = random_testing_baseline(model, X_test, y_arr)
    log.info("Random baseline: %s", random_bl)

    summary   = mr_summary_table(mr_results)
    comp_df   = build_comparison_table(mr_results, y_arr, random_bl)
    overall   = overall_violation_rate(mr_results)

    summary.to_csv(RESULTS_DIR / f"{model_name}_mr_summary.csv",    index=False)
    comp_df.to_csv(RESULTS_DIR / f"{model_name}_comparison.csv",    index=False)

    log.info("\n%s", summary.to_string(index=False))
    log.info("\nComparison Table:\n%s", comp_df[
        ["method", "viol_rate_pct", "fdr_pct", "exec_time_s"]
    ].to_string(index=False))
    log.info("Overall MR violation rate: %.2f%%", overall * 100)

    plot_violation_rates(mr_results)
    plot_fault_detection(comp_df)
    plot_prob_shift(mr_results)
    plot_exec_times(mr_results)
    log.info("Plots saved to %s", RESULTS_DIR)

    return {
        "model":          model_name,
        "metrics":        metrics,
        "mr_results":     mr_results,
        "random_bl":      random_bl,
        "overall_vr":     overall,
        "comparison_df":  comp_df,
    }


def run_all_models(use_genai: bool = False) -> None:
    all_mr_results: dict = {}
    for model_name in MODELS:
        res = run_single(model_name, use_genai=use_genai)
        all_mr_results[model_name] = res["mr_results"]

    plot_model_comparison_heatmap(all_mr_results)
    log.info("Model comparison heatmap saved.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MT Framework Experiment Runner")
    p.add_argument("--model",      default=DEFAULT_MODEL, choices=list(MODELS))
    p.add_argument("--all-models", action="store_true")
    p.add_argument("--genai",      action="store_true",
                   help="Call OpenAI to generate additional MR suggestions")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.all_models:
        run_all_models(use_genai=args.genai)
    else:
        run_single(args.model, use_genai=args.genai)
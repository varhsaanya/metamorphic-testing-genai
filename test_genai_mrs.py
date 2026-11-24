from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from config import DEFAULT_MODEL, OPENAI_MODEL, RESULTS_DIR
from data.loader import (
    get_feature_stats, load_raw, preprocess, split_and_scale,
)
from evaluation.metrics import (
    build_comparison_table,
    fault_detection_rate,
    mr_summary_table,
    overall_violation_rate,
    random_testing_baseline,
)
from evaluation.visualizer import (
    plot_exec_times,
    plot_fault_detection,
    plot_prob_shift,
    plot_violation_rates,
)
from genai.dynamic_mr_builder import build_all_dynamic_mrs
from genai.mock_responses import MOCK_GENAI_MRS
from genai.mr_generator import (
    generate_mrs_from_genai,
    print_genai_mrs,
)
from metamorphic.executor import run_all_mrs
from metamorphic.relations import ALL_MRS
from models.trainer import build_model, load_model, train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

W = 66   # separator width


def sep(title: str = "") -> None:
    if title:
        pad = (W - len(title) - 2) // 2
        print(f"\n{'-'*pad} {title} {'-'*pad}")
    else:
        print("-" * W)


def main(use_mock: bool, use_cache: bool, n_mrs: int) -> None:

    t_start = time.perf_counter()

    sep("STEP 1 · Load data & model")
    df = preprocess(load_raw())
    X_train, X_test, y_train, y_test, scaler, feat_cols = split_and_scale(df)
    stats = get_feature_stats(X_train)
    y_arr = y_test.values
    print(f"  Test set : {len(X_test)} samples  |  {len(feat_cols)} features")
    print(f"  Defect rate (test) : {y_arr.mean()*100:.1f}%")

    try:
        model = load_model(DEFAULT_MODEL)
        print("  Model    : loaded from disk ✓")
    except FileNotFoundError:
        print("  Model    : not found on disk — training now …")
        model = train(build_model(DEFAULT_MODEL), X_train, y_train)

    sep("STEP 2 · Fetch GenAI MR proposals")

    if use_mock:
        print("  Mode: MOCK (no API call)")
        raw_dicts = MOCK_GENAI_MRS
    else:
        print(f"  Mode: LIVE HuggingFace call  (model={OPENAI_MODEL})")
        raw_dicts = generate_mrs_from_genai(n=n_mrs, use_cache=use_cache)
        if not raw_dicts:
            print("\n  ⚠  OpenAI returned nothing — falling back to mock MRs.")
            print("     Check your OPENAI_API_KEY or run with --mock.\n")
            raw_dicts = MOCK_GENAI_MRS

    sep("STEP 3 · Raw GenAI output (what GPT returned)")
    print_genai_mrs(raw_dicts)

    sep("STEP 4 · Validate & build executable MR classes")
    genai_mrs, failed = build_all_dynamic_mrs(raw_dicts)

    if failed:
        print(f"\n  ⚠  {len(failed)} MR(s) FAILED validation:")
        for f in failed:
            print(f"     ✗ {f.get('name','?'):40s}  →  {f['_error']}")

    print(f"\n  ✓ {len(genai_mrs)} MR(s) built and ready:\n")
    for mr in genai_mrs:
        print(f"     • {mr.name}")

    # Save audit trail
    audit = {
        "raw_proposals": raw_dicts,
        "built":   [m.name for m in genai_mrs],
        "failed":  failed,
    }
    audit_path = RESULTS_DIR / "genai_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2))
    print(f"\n  Audit JSON → {audit_path}")

    if not genai_mrs:
        print("\n  No GenAI MRs to execute. Exiting.")
        sys.exit(1)

    sep("STEP 5 · Execute GenAI MRs on test set")
    t5 = time.perf_counter()
    genai_results = run_all_mrs(genai_mrs, model, X_test, stats)
    print(f"\n  Completed in {time.perf_counter()-t5:.2f}s")

    genai_sum = mr_summary_table(genai_results)
    print(f"\n{genai_sum.to_string(index=False)}\n")

    # Per-MR fault detection
    genai_fdr = fault_detection_rate(genai_results, y_arr)
    print("  Fault Detection Rate (GenAI MRs):")
    print(genai_fdr.to_string(index=False))

    sep("STEP 6 · Execute hard-coded MRs (reference baseline)")
    hc_results = run_all_mrs(ALL_MRS, model, X_test, stats)
    hc_sum = mr_summary_table(hc_results)
    print(f"\n{hc_sum.to_string(index=False)}\n")

    sep("STEP 7 · Random testing baseline (200 runs)")
    t7 = time.perf_counter()
    random_bl = random_testing_baseline(model, X_test, y_arr, n_runs=200)
    print(f"  Completed in {time.perf_counter()-t7:.2f}s")
    print(f"  violation_rate = {random_bl['violation_rate']*100:.3f}%")
    print(f"  fault_detect   = {random_bl['fault_detect_rate']*100:.3f}%")

    sep("STEP 8 · Comparison: GenAI vs Hard-coded vs Random")

    all_results   = genai_results + hc_results
    comparison_df = build_comparison_table(all_results, y_arr, random_bl)

    genai_names = {mr.name for mr in genai_mrs}
    hc_names    = {mr.name for mr in ALL_MRS}

    def tag(m: str) -> str:
        if m in genai_names:  return "GenAI"
        if m in hc_names:     return "Hard-coded"
        return "Random"

    comparison_df["source"] = comparison_df["method"].apply(tag)

    print(f"\n{'Source':12} {'Method':38} {'ViolRate%':>10} {'FDR%':>8} {'Time(s)':>8}")
    print("─" * 78)
    for _, row in comparison_df.iterrows():
        print(
            f"  {row['source']:10} {row['method']:38} "
            f"{row['viol_rate_pct']:>9.2f}  "
            f"{row['fdr_pct']:>7.2f}  "
            f"{row['exec_time_s']:>7.3f}"
        )

    sep("STEP 9 · Summary statistics by source")

    non_random = comparison_df[comparison_df["source"] != "Random"]
    avg = (
        non_random.groupby("source")[["viol_rate_pct", "fdr_pct"]]
        .agg(["mean", "max"])
        .round(2)
    )
    print(f"\n{avg.to_string()}\n")

    genai_ovr = overall_violation_rate(genai_results)
    hc_ovr    = overall_violation_rate(hc_results)
    rand_vr   = random_bl["violation_rate"]

    print(f"  GenAI MRs    avg violation rate : {genai_ovr*100:.2f}%")
    print(f"  Hard-coded   avg violation rate : {hc_ovr*100:.2f}%")
    print(f"  Random test  violation rate     : {rand_vr*100:.3f}%")

    improvement = genai_ovr / max(rand_vr, 1e-9)
    print(f"\n  GenAI MRs are {improvement:.0f}× more effective than random testing")

    sep("STEP 10 · Save results & plots")

    genai_sum.to_csv(     RESULTS_DIR / "genai_mr_summary.csv",     index=False)
    hc_sum.to_csv(        RESULTS_DIR / "hardcoded_mr_summary.csv", index=False)
    pd.concat([genai_sum, hc_sum], ignore_index=True).to_csv(RESULTS_DIR / "GenAI_vs_Manual_mr_summary.csv", index=False)
    comparison_df.to_csv( RESULTS_DIR / "GenAI_vs_Manual_comparison.csv",      index=False)
    genai_fdr.to_csv(     RESULTS_DIR / "genai_fdr.csv",            index=False)

    plot_violation_rates(genai_results)
    plot_prob_shift(genai_results)
    plot_exec_times(genai_results)

    chart_df = comparison_df[
        comparison_df["source"].isin(["GenAI", "Random"])
    ].copy()
    chart_df["violation_rate"]    = chart_df["viol_rate_pct"] / 100
    chart_df["fault_detect_rate"] = chart_df["fdr_pct"]       / 100
    plot_fault_detection(chart_df)

    total = time.perf_counter() - t_start
    print(f"\n  All outputs → {RESULTS_DIR}")
    print(f"  Total wall time : {total:.1f}s")
    sep()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full GenAI MR pipeline")
    p.add_argument("--mock",     action="store_true",
                   help="Use built-in mock MRs (no API call)")
    p.add_argument("--no-cache", dest="no_cache", action="store_true",
                   help="Force fresh OpenAI call (ignore disk cache)")
    p.add_argument("--n",        type=int, default=5,
                   help="Number of MRs to request from OpenAI (default 5)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        use_mock  = args.mock,
        use_cache = not args.no_cache,
        n_mrs     = args.n,
    )
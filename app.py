from flask import Flask, render_template, send_from_directory, jsonify
import pandas as pd
import json
from pathlib import Path

app = Flask(__name__, static_folder="results")
RESULTS_DIR = Path(__file__).parent / "results"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/results")
def get_results():
    if not RESULTS_DIR.exists():
        return jsonify({"error": "No results directory found. Please run the experiment first."}), 404

    models_data = {}
    for p in RESULTS_DIR.glob("*_comparison.csv"):
        model_name = p.name.replace("_comparison.csv", "")
        summary_path = RESULTS_DIR / f"{model_name}_mr_summary.csv"
        comp_df   = pd.read_csv(p)
        summary_df = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
        models_data[model_name] = {
            "comparison": comp_df.to_dict(orient="records"),
            "summary":    summary_df.to_dict(orient="records"),
        }

    return jsonify({
        "models": models_data,
        "images": [p.name for p in RESULTS_DIR.glob("*.png")],
    })


@app.route("/api/genai-comparison")
def get_genai_comparison():
    """Return the full GenAI vs Manual vs Random comparison data."""
    cmp_path = RESULTS_DIR / "GenAI_vs_Manual_comparison.csv"
    sum_path = RESULTS_DIR / "GenAI_vs_Manual_mr_summary.csv"
    fdr_path = RESULTS_DIR / "genai_fdr.csv"
    audit_path = RESULTS_DIR / "genai_audit.json"

    if not cmp_path.exists():
        return jsonify({"error": "Run python test_genai_mrs.py first."}), 404

    cmp_df = pd.read_csv(cmp_path)
    sum_df = pd.read_csv(sum_path) if sum_path.exists() else pd.DataFrame()
    fdr_df = pd.read_csv(fdr_path) if fdr_path.exists() else pd.DataFrame()

    audit = {}
    if audit_path.exists():
        with open(audit_path) as f:
            audit = json.load(f)

    # Aggregate stats per source (flatten multi-index so JSON serialises)
    if "source" in cmp_df.columns:
        raw_agg = (
            cmp_df[cmp_df["source"] != "Random"]
            .groupby("source")[["viol_rate_pct", "fdr_pct"]]
            .agg(["mean", "max"])
            .round(2)
        )
        # Flatten tuple column keys → "viol_rate_pct_mean" etc.
        raw_agg.columns = ["_".join(c) for c in raw_agg.columns]
        agg = raw_agg.reset_index().to_dict(orient="records")
    else:
        agg = []

    # Random row
    random_rows = cmp_df[cmp_df["method"] == "random_testing"].to_dict(orient="records")
    random_stats = random_rows[0] if random_rows else {}

    return jsonify({
        "comparison":   cmp_df.to_dict(orient="records"),
        "summary":      sum_df.to_dict(orient="records"),
        "fdr":          fdr_df.to_dict(orient="records"),
        "aggregated":   agg,
        "random_stats": random_stats,
        "audit":        audit,
        "images":       [p.name for p in RESULTS_DIR.glob("*.png")],
    })


@app.route("/results/<path:filename>")
def serve_results(filename):
    return send_from_directory(RESULTS_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True, port=5000)

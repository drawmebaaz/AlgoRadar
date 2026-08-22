"""Evaluate per-horizon models and produce JSON + HTML plots for inspection.

Usage: python scripts/eval_multi_horizon.py

This script:
- Builds the real-label dataset from `data/cache`
- Trains per-horizon models (GridSearch + calibration)
- Evaluates and saves JSON arrays (ROC/PR/calibration)
- Writes interactive HTML ROC/PR plots under the evaluation folder
"""
from pathlib import Path
import json

import plotly.graph_objects as go

from algoradar.models import (
    build_real_solve_training_dataset,
    train_multi_horizon_models,
    evaluate_and_save_per_horizon,
)


def make_plots_for_dir(out_dir: Path):
    arrays_path = out_dir / "arrays.json"
    metrics_path = out_dir / "metrics.json"
    if not arrays_path.exists() or not metrics_path.exists():
        return False
    arrays = json.loads(arrays_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    # ROC
    fpr = arrays.get("fpr", [])
    tpr = arrays.get("tpr", [])
    if fpr and tpr:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")))
        fig.update_layout(title=f"ROC AUC: {metrics.get('roc_auc'):.3f}", xaxis_title="FPR", yaxis_title="TPR")
        (out_dir / "roc.html").write_text(fig.to_html(include_plotlyjs='cdn'))

    # PR
    prec = arrays.get("precision", [])
    rec = arrays.get("recall", [])
    if prec and rec:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name="PR"))
        fig2.update_layout(title=f"PR AUC: {metrics.get('pr_auc'):.3f}", xaxis_title="Recall", yaxis_title="Precision")
        (out_dir / "pr.html").write_text(fig2.to_html(include_plotlyjs='cdn'))

    # calibration table
    calib = arrays.get("calibration_bins", [])
    if calib:
        table = go.Figure(data=[go.Table(header=dict(values=["left", "right", "count", "avg_pred", "avg_true"]), cells=dict(values=[
            [b.get("left") for b in calib],
            [b.get("right") for b in calib],
            [b.get("count") for b in calib],
            [b.get("avg_pred") for b in calib],
            [b.get("avg_true") for b in calib],
        ]))])
        (out_dir / "calibration.html").write_text(table.to_html(include_plotlyjs='cdn'))

    return True


def main():
    cache_root = Path(__file__).resolve().parents[1] / "data" / "cache"
    dataset = build_real_solve_training_dataset(cache_dir=cache_root)
    if dataset.empty:
        print("No dataset found in cache/data; run dataset build first.")
        return

    print("Training per-horizon models...")
    reports = train_multi_horizon_models(dataset)
    print("Evaluating and saving artifacts...")
    eval_results = evaluate_and_save_per_horizon(dataset, reports, cache_root=cache_root)
    print(json.dumps(eval_results, indent=2))

    # generate plots for saved directories
    for h, info in eval_results.items():
        if info.get("status") == "saved":
            out_dir = Path(info.get("path"))
            ok = make_plots_for_dir(out_dir)
            print(f"Plotted {h}: {ok} -> {out_dir}")


if __name__ == "__main__":
    main()

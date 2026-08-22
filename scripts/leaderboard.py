"""Aggregate per-horizon evaluation metrics into a leaderboard CSV/JSON.

Usage: python scripts/leaderboard.py

Scans `data/cache/eval_per_horizon/*/*/metrics.json` and produces
`data/cache/leaderboard/latest_leaderboard.csv` and JSON.
"""
from pathlib import Path
import json
from datetime import datetime
import pandas as pd


def main():
    base = Path(__file__).resolve().parents[1] / "data" / "cache" / "eval_per_horizon"
    if not base.exists():
        print("No evaluation artifacts found under data/cache/eval_per_horizon")
        return

    rows = []
    for horizon_dir in base.iterdir():
        if not horizon_dir.is_dir():
            continue
        for ts_dir in horizon_dir.iterdir():
            metrics_file = ts_dir / "metrics.json"
            if not metrics_file.exists():
                continue
            try:
                m = json.loads(metrics_file.read_text(encoding="utf-8"))
                m["horizon"] = horizon_dir.name
                m["path"] = str(ts_dir)
                rows.append(m)
            except Exception:
                continue

    if not rows:
        print("No metrics.json files found.")
        return

    df = pd.DataFrame(rows)
    # natural sort by horizon then recent timestamp
    df["timestamp_parsed"] = pd.to_datetime(df["timestamp"], format="%Y%m%dT%H%M%SZ", errors="coerce")
    df = df.sort_values(["horizon", "roc_auc", "pr_auc", "timestamp_parsed"], ascending=[True, False, False, False])

    out_dir = Path(__file__).resolve().parents[1] / "data" / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    csv_path = out_dir / f"leaderboard_{ts}.csv"
    json_path = out_dir / f"leaderboard_{ts}.json"
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)

    latest_csv = out_dir / "latest_leaderboard.csv"
    latest_json = out_dir / "latest_leaderboard.json"
    df.to_csv(latest_csv, index=False)
    df.to_json(latest_json, orient="records", indent=2)

    print(f"Wrote leaderboard: {csv_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()

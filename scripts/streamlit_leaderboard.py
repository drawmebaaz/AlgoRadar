import json
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


def load_leaderboard(base: Path):
    csv = base / "latest_leaderboard.csv"
    jsonp = base / "latest_leaderboard.json"
    if csv.exists():
        return pd.read_csv(csv)
    if jsonp.exists():
        return pd.DataFrame(json.loads(jsonp.read_text(encoding="utf-8")))
    return pd.DataFrame()


def embed_html(path: Path):
    if not path.exists():
        st.warning(f"Missing plot: {path}")
        return
    html = path.read_text(encoding="utf-8")
    components.html(html, height=400)


def main():
    st.set_page_config(page_title="AlgoRadar — Evaluation Leaderboard", layout="wide")
    st.title("AlgoRadar — Per-horizon Evaluation Leaderboard")

    base = Path(__file__).resolve().parents[1] / "data" / "cache"
    lb = load_leaderboard(base)
    if lb.empty:
        st.info("No leaderboard found. Run the evaluation runner to generate artifacts: scripts/eval_multi_horizon.py")
        return

    st.sidebar.header("Filters")
    horizons = sorted(lb["horizon"].unique()) if "horizon" in lb.columns else []
    sel_horizon = st.sidebar.selectbox("Horizon", ["all"] + horizons, index=0)

    if sel_horizon != "all":
        df = lb[lb["horizon"] == sel_horizon]
    else:
        df = lb.copy()

    st.write("**Leaderboard (latest)**")
    st.dataframe(df[ [c for c in df.columns if c != "timestamp_parsed"] ].sort_values(["horizon", "roc_auc"], ascending=[True, False]))

    st.write("---")
    st.header("Inspect evaluation artifacts")
    st.write("Select a row to view metrics and plots.")

    st.write("**Compare two evaluation runs**")
    sel = st.multiselect("Select up to two rows (by index)", options=list(range(len(df))), default=[0], max_selections=2)
    if not sel:
        st.info("Select one or two rows to inspect plots and metrics.")
        return

    cols = st.columns(len(sel))
    for i, idx in enumerate(sel):
        with cols[i]:
            row = df.iloc[int(idx)]
            st.subheader(f"Horizon: {row.get('horizon')} — ROC AUC: {row.get('roc_auc')}")
            path = Path(row.get("path")) if row.get("path") else None
            if path and path.exists():
                st.markdown("**Metrics**")
                metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8")) if (path / "metrics.json").exists() else {}
                st.json(metrics)

                st.markdown("**ROC curve**")
                embed_html(path / "roc.html")

                st.markdown("**Precision-Recall curve**")
                embed_html(path / "pr.html")

                st.markdown("**Calibration**")
                embed_html(path / "calibration.html")
            else:
                st.warning("Evaluation artifacts not found for selected row.")

    # If two rows selected, show compact metric deltas
    if len(sel) == 2:
        r0 = df.iloc[int(sel[0])]
        r1 = df.iloc[int(sel[1])]
        m0_path = Path(r0.get("path")) if r0.get("path") else None
        m1_path = Path(r1.get("path")) if r1.get("path") else None
        if m0_path and m1_path and m0_path.exists() and m1_path.exists():
            m0 = json.loads((m0_path / "metrics.json").read_text(encoding="utf-8")) if (m0_path / "metrics.json").exists() else {}
            m1 = json.loads((m1_path / "metrics.json").read_text(encoding="utf-8")) if (m1_path / "metrics.json").exists() else {}
            keys = ["roc_auc", "pr_auc", "brier"]
            st.write("---")
            st.subheader("Metric deltas (selected run 2 minus run 1)")
            delta_rows = []
            for k in keys:
                v0 = m0.get(k)
                v1 = m1.get(k)
                if v0 is None or v1 is None:
                    continue
                try:
                    diff = float(v1) - float(v0)
                    pct = (diff / float(v0)) * 100.0 if float(v0) != 0 else None
                except Exception:
                    diff = None
                    pct = None
                delta_rows.append({"metric": k, "run1": v0, "run2": v1, "diff": diff, "pct_change": pct})
            if delta_rows:
                st.table(pd.DataFrame(delta_rows))
            else:
                st.info("No comparable metrics found for the selected runs.")


if __name__ == "__main__":
    main()

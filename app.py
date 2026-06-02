from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from algoradar import run_analysis
from algoradar.recommender import score_custom_problem

st.set_page_config(
    page_title="AlgoRadar",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def cached_analysis(handle: str, force_refresh: bool, prefer_transformer: bool, use_sample: bool, submission_limit: int):
    return run_analysis(
        handle=handle,
        force_refresh=force_refresh,
        prefer_transformer=prefer_transformer,
        use_sample=use_sample,
        submission_limit=submission_limit,
    )


def main() -> None:
    inject_css()
    render_sidebar()

    with st.sidebar:
        handle = st.text_input("Codeforces handle", value="tourist", help="Use any public Codeforces handle.")
        force_refresh = st.toggle("Refresh Codeforces cache", value=False)
        prefer_transformer = st.toggle(
            "Use MiniLM embeddings",
            value=False,
            help="If sentence-transformers is installed, this uses all-MiniLM-L6-v2. Otherwise the app uses TF-IDF fallback.",
        )
        analyze = st.button("Analyze handle", width="stretch")

    current_args = {
        "handle": handle.strip() or "tourist",
        "force_refresh": force_refresh,
        "prefer_transformer": prefer_transformer,
        "use_sample": False,
        "submission_limit": 2500,
    }
    if "analysis_args" not in st.session_state:
        st.session_state.analysis_args = current_args
    if analyze:
        st.session_state.analysis_args = current_args
        cached_analysis.clear()

    analysis_args = st.session_state.analysis_args
    settings_changed = current_args != analysis_args
    with st.spinner("Running AlgoRadar on live Codeforces data..."):
        result = cached_analysis(**analysis_args)

    source_label = "Live Codeforces API" if result.source == "codeforces" else result.source
    st.markdown(
        f"""
        <div class="hero">
          <div>
            <p class="eyebrow">AlgoRadar / Competitive programming intelligence</p>
            <h1>{result.handle}</h1>
            <p class="subcopy">ML-powered weakness analysis, solve probability, recommendations, semantic retrieval, and weekly training plans.</p>
          </div>
          <div class="source-pill">{source_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if settings_changed:
        st.info("Sidebar settings changed. Click Analyze handle to run the pipeline with the new settings.")

    screen = st.sidebar.radio(
        "Screens",
        [
            "1. User profile analytics",
            "2. Weakness map",
            "3. Problem recommendations",
            "4. Solve probability",
            "5. Weekly roadmap",
            "6. Progress tracking",
        ],
    )

    if screen.startswith("1."):
        render_profile(result)
    elif screen.startswith("2."):
        render_weakness(result)
    elif screen.startswith("3."):
        render_recommendations(result)
    elif screen.startswith("4."):
        render_probability(result)
    elif screen.startswith("5."):
        render_roadmap(result)
    elif screen.startswith("6."):
        render_progress(result)


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div class="brand">
              <div class="brand-mark">AR</div>
              <div>
                <strong>AlgoRadar</strong>
                <span>CP weakness analyzer</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_profile(result) -> None:
    profile = result.profile
    cols = st.columns(4)
    metric_card(cols[0], "Current rating", f"{profile['current_rating']}", "official Codeforces rating")
    metric_card(cols[1], "Max rating", f"{profile['max_rating']}", "peak Codeforces rating")
    metric_card(cols[2], "Solved", f"{profile['problems_solved']}", "unique accepted problems")
    metric_card(cols[3], "Recent accuracy", f"{profile['recent_accuracy']}%", "last 80 submissions")

    left, right = st.columns([1.45, 1])
    with left:
        panel_title("Contest performance trend", "Codeforces rating and rank history")
        if result.contest_trend.empty:
            st.info("No contest rating history found for this handle.")
        else:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=result.contest_trend["contest"],
                    y=result.contest_trend["rating"],
                    mode="lines+markers",
                    name="Rating",
                    line=dict(color="#5ee0a0", width=2),
                )
            )
            fig.add_trace(
                go.Bar(
                    x=result.contest_trend["contest"],
                    y=result.contest_trend["delta"],
                    name="Delta",
                    marker_color="#75a7ff",
                    opacity=0.42,
                    yaxis="y2",
                )
            )
            fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False), **chart_layout())
            st.plotly_chart(fig, width="stretch")

    with right:
        panel_title("Verdict distribution", "submission failure patterns")
        verdicts = result.verdicts.copy()
        if verdicts.empty:
            st.info("No submissions found.")
        else:
            fig = px.pie(verdicts, names="verdict", values="count", hole=0.58, color_discrete_sequence=palette())
            fig.update_layout(**chart_layout(height=360))
            st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    with left:
        panel_title("Rating-wise accuracy", "success rate by official or estimated problem rating")
        frame = result.rating_accuracy.copy()
        if not frame.empty:
            fig = go.Figure()
            fig.add_bar(x=frame["rating_bucket"], y=frame["attempts"], name="Attempts", marker_color="#2b3340")
            fig.add_trace(
                go.Scatter(
                    x=frame["rating_bucket"],
                    y=frame["accuracy"],
                    mode="lines+markers",
                    name="Accuracy %",
                    line=dict(color="#f8c76f", width=2),
                    yaxis="y2",
                )
            )
            fig.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0, 100], showgrid=False), **chart_layout())
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No rating data available.")

    with right:
        panel_title("Solved difficulty distribution", "where accepted problems cluster")
        frame = result.solved_difficulty
        if not frame.empty:
            fig = px.bar(frame, x="rating_bucket", y="solved", color_discrete_sequence=["#5ee0a0"])
            fig.update_layout(**chart_layout())
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No accepted submissions available.")


def render_weakness(result) -> None:
    cols = st.columns(4)
    weakness = result.weakness
    metric_card(cols[0], "Weak tags", str((weakness["level"] == "Weak").sum()), "rule baseline")
    metric_card(cols[1], "Untouched tags", str((weakness["level"] == "Untouched").sum()), "coverage gap")
    metric_card(cols[2], "Rule vs ML agreement", f"{weakness['rule_matches_ml'].mean() * 100:.1f}%", "baseline comparison")
    metric_card(cols[3], "ML accuracy", f"{result.weakness_model['metrics']['accuracy'] * 100:.1f}%", "synthetic validation")

    left, right = st.columns([1.55, 1])
    with left:
        panel_title("Weakness classifier", "accuracy, attempts, rating, recency")
        show = weakness[
            [
                "tag",
                "level",
                "attempts",
                "accuracy",
                "max_rating_solved",
                "recent_failures",
                "priority_score",
                "next_action",
            ]
        ].copy()
        show["accuracy"] = show["accuracy"].round(1)
        show["max_rating_solved"] = show["max_rating_solved"].round(0)
        show = show.rename(
            columns={
                "tag": "Tag",
                "level": "Level",
                "attempts": "Attempts",
                "accuracy": "Accuracy %",
                "max_rating_solved": "Hardest solved",
                "recent_failures": "Recent fails",
                "priority_score": "Repair priority",
                "next_action": "Next action",
            }
        )
        st.dataframe(show, width="stretch", height=520)

    with right:
        panel_title("Most failed tags", "repair priority")
        top = weakness.head(10).sort_values("priority_score", ascending=True)
        fig = px.bar(
            top,
            x="priority_score",
            y="tag",
            orientation="h",
            color="level",
            color_discrete_map={
                "Strong": "#5ee0a0",
                "Stable": "#75a7ff",
                "Weak": "#f28b82",
                "Untouched": "#6f7886",
                "Over-attempted but low accuracy": "#f8c76f",
            },
        )
        fig.update_layout(**chart_layout(height=520), showlegend=False)
        st.plotly_chart(fig, width="stretch")

    panel_title("Explainability", "random forest feature importance for weakness model")
    st.dataframe(result.weakness_model["feature_importance"], width="stretch", hide_index=True)


def render_recommendations(result) -> None:
    recs = result.recommendations
    if recs.empty:
        st.warning("No recommendations could be generated from the current data.")
        return

    counts = recs["bucket"].value_counts().to_dict()
    cols = st.columns(4)
    metric_card(cols[0], "Confidence builders", str(counts.get("confidence", 0)), ">75% solve chance")
    metric_card(cols[1], "Growth problems", str(counts.get("growth", 0)), "45-75% solve chance")
    metric_card(cols[2], "Stretch problems", str(counts.get("stretch", 0)), "25-45% solve chance")
    metric_card(cols[3], "Semantic method", result.semantic_method, "similar problem finder")

    for bucket, label in [
        ("confidence", "5 confidence builders"),
        ("growth", "10 growth problems"),
        ("stretch", "5 stretch problems"),
    ]:
        panel_title(label, "ranked by probability, tag fit, popularity, and rating distance")
        frame = recs[recs["bucket"] == bucket].copy()
        if frame.empty:
            st.info(f"No {bucket} recommendations found for this handle yet.")
            continue

        frame["open"] = frame.apply(_codeforces_problem_url, axis=1)
        frame = frame[
            [
                "open",
                "problem_id",
                "name",
                "rating",
                "rating_source",
                "tags",
                "solve_probability_pct",
                "tag_similarity",
                "solved_count",
                "rank_score",
            ]
        ].copy()
        frame["tags"] = frame["tags"].apply(_format_tags)
        frame["tag_similarity"] = frame["tag_similarity"].round(2)
        frame["rank_score"] = frame["rank_score"].round(3)
        st.dataframe(
            frame,
            width="stretch",
            hide_index=True,
            column_config={
                "open": st.column_config.LinkColumn("Open", display_text="Codeforces"),
                "rating_source": st.column_config.TextColumn("Rating source"),
                "solve_probability_pct": st.column_config.NumberColumn("Solve %", format="%.1f%%"),
                "tag_similarity": st.column_config.NumberColumn("Tag fit", format="%.2f"),
                "rank_score": st.column_config.NumberColumn("Rank score", format="%.3f"),
            },
        )

    panel_title("Similar-but-harder retrieval", "embeddings/vector search layer")
    if result.similar_harder.empty:
        st.info("No similar harder problems found.")
    else:
        frame = result.similar_harder.copy()
        frame["open"] = frame.apply(_codeforces_problem_url, axis=1)
        frame = frame[["open", "problem_id", "name", "rating", "rating_source", "tags", "semantic_score", "solved_count"]].copy()
        frame["tags"] = frame["tags"].apply(_format_tags)
        frame["semantic_score"] = frame["semantic_score"].round(3)
        st.dataframe(
            frame,
            width="stretch",
            hide_index=True,
            column_config={
                "open": st.column_config.LinkColumn("Open", display_text="Codeforces"),
                "rating_source": st.column_config.TextColumn("Rating source"),
                "semantic_score": st.column_config.NumberColumn("Similarity", format="%.3f"),
            },
        )


def render_probability(result) -> None:
    left, right = st.columns([0.9, 1.1])
    with left:
        panel_title("Problem lookup", "enter Codeforces code like 1900C or 1497E2")
        default_code = _default_problem_code(result)
        problem_code = st.text_input("Codeforces problem code", value=default_code)
        problem = _find_problem_by_code(result.problems, problem_code)

        if problem is not None:
            rating = int(problem["rating"])
            tags = list(problem["tags"] or [])
            solved_count = int(problem["solved_count"])
            name = str(problem["name"])
            rating_source = str(problem.get("rating_source", "official"))
            st.markdown(
                f"""
                <div class="rule-list">
                  <p><strong>{problem['problem_id']} - {name}</strong></p>
                  <p>Rating: <strong>{rating}</strong> ({rating_source})</p>
                  <p>Solved by: <strong>{solved_count}</strong> users</p>
                  <p>Tags: {_format_tags(tags) or "untagged"}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if rating_source == "estimated":
                st.info("This problem has no official Codeforces rating, so AlgoRadar estimated it from problem index and solved count.")
        else:
            st.warning("Problem code not found in the Codeforces problemset cache. Use manual fallback below.")
            with st.expander("Manual fallback", expanded=True):
                rating = st.slider("Estimated problem rating", min_value=800, max_value=3500, value=int(result.profile["current_rating"] + 200), step=100)
                tags = st.multiselect("Problem tags", options=sorted(result.tag_stats["tag"].unique()), default=result.weakness["tag"].head(2).tolist())
                solved_count = st.number_input("Problem solved count", min_value=0, max_value=200000, value=4500, step=100)
                name = st.text_input("Problem name", value="Custom training target")

        score = score_custom_problem(
            rating=rating,
            tags=tags,
            solved_count=int(solved_count),
            name=name,
            profile=result.profile,
            tag_stats=result.weakness,
            solve_model_report=result.solve_model,
        )
        st.caption(
            f"AlgoRadar inferred recent tag failures from your history: {score['recent_failures_used']}"
        )

    with right:
        panel_title("Prediction", "classification bucket")
        probability = score["solve_probability_pct"]
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability,
                number={"suffix": "%", "font": {"color": "#eef2f6"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#6f7886"},
                    "bar": {"color": bucket_color(score["bucket"])},
                    "bgcolor": "#151922",
                    "bordercolor": "#252b35",
                    "steps": [
                        {"range": [0, 25], "color": "#221719"},
                        {"range": [25, 45], "color": "#261c17"},
                        {"range": [45, 75], "color": "#292416"},
                        {"range": [75, 100], "color": "#15251d"},
                    ],
                },
            )
        )
        fig.update_layout(**chart_layout(height=360))
        st.plotly_chart(fig, width="stretch")
        st.markdown(f"<div class='bucket-label'>{score['bucket'].upper()}</div>", unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        panel_title("Solve model metrics", result.solve_model["selected_model_name"])
        metrics = pd.DataFrame(result.solve_model["metrics"]).T.reset_index().rename(columns={"index": "model"})
        st.dataframe(metrics.round(3), width="stretch", hide_index=True)
    with right:
        panel_title("Feature importance", "model interpretation")
        st.dataframe(result.solve_model["feature_importance"].round(3), width="stretch", hide_index=True)


def render_roadmap(result) -> None:
    panel_title("Weekly roadmap", "adaptive plan from weakness map and recommendation queue")
    roadmap = result.roadmap.copy()
    cols = st.columns(7)
    for col, row in zip(cols, roadmap.to_dict("records")):
        with col:
            st.markdown(
                f"""
                <div class="day-card">
                  <code>{row['day']}</code>
                  <h3>{row['theme']}</h3>
                  <p>{row['focus']}</p>
                  <div class="meter"><span style="width:{row['load']}%"></span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    left, right = st.columns(2)
    with left:
        panel_title("Focus load", "weekly intensity")
        fig = px.bar(roadmap, x="day", y="load", color_discrete_sequence=["#f8c76f"])
        fig.update_layout(**chart_layout())
        st.plotly_chart(fig, width="stretch")

    with right:
        panel_title("Operating rules", "keeps the ML project honest")
        st.markdown(
            """
            <div class="rule-list">
              <p><strong>Baseline first:</strong> compare every ML label with the rule classifier.</p>
              <p><strong>Timebox:</strong> stop after 75 minutes and mark failure mode.</p>
              <p><strong>Upsolve:</strong> every failed contest problem gets a mistake note.</p>
              <p><strong>Measure:</strong> track accuracy, precision, recall, calibration, and overfitting.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_progress(result) -> None:
    progress = result.progress
    cols = st.columns(4)
    if progress.empty:
        metric_card(cols[0], "Weeks tracked", "0", "no submissions")
        st.info("No submission history available.")
        return

    latest = progress.iloc[-1]
    metric_card(cols[0], "Latest solves", str(int(latest["solved"])), "accepted submissions")
    metric_card(cols[1], "Attempts", str(int(latest["attempts"])), "latest week")
    metric_card(cols[2], "Accuracy", f"{latest['accuracy']:.1f}%", "latest week")
    metric_card(
        cols[3],
        "Growth attempts",
        str(int(latest["growth_attempts"])),
        f"{result.profile['growth_rating_low']}-{result.profile['growth_rating_high']} rating",
    )

    panel_title("Progress tracking", "weekly solve volume and accuracy")
    fig = go.Figure()
    fig.add_bar(x=progress["week"], y=progress["attempts"], name="Attempts", marker_color="#2b3340")
    fig.add_trace(go.Scatter(x=progress["week"], y=progress["solved"], mode="lines+markers", name="Solved", line=dict(color="#5ee0a0")))
    fig.add_trace(
        go.Scatter(
            x=progress["week"],
            y=progress["accuracy"],
            mode="lines+markers",
            name="Accuracy %",
            line=dict(color="#f8c76f"),
            yaxis="y2",
        )
    )
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0, 100], showgrid=False), **chart_layout(height=460))
    st.plotly_chart(fig, width="stretch")


def metric_card(column, label: str, value: str, detail: str) -> None:
    with column:
        st.markdown(
            f"""
            <div class="metric-card">
              <p>{label}</p>
              <strong>{value}</strong>
              <span>{detail}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def panel_title(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="panel-title">
          <p>{subtitle}</p>
          <h2>{title}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )


def chart_layout(height: int = 390) -> dict:
    return {
        "height": height,
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": "#a0a8b5", "family": "Inter, sans-serif"},
        "margin": {"l": 20, "r": 20, "t": 22, "b": 30},
        "legend": {"orientation": "h", "y": 1.08, "x": 0},
        "xaxis": {"gridcolor": "#242a33", "zerolinecolor": "#242a33"},
        "yaxis": {"gridcolor": "#242a33", "zerolinecolor": "#242a33"},
    }


def palette() -> list[str]:
    return ["#5ee0a0", "#f28b82", "#f8c76f", "#75a7ff", "#6f7886"]


def bucket_color(bucket: str) -> str:
    return {
        "confidence": "#5ee0a0",
        "growth": "#f8c76f",
        "stretch": "#f28b82",
        "avoid": "#6f7886",
    }.get(bucket, "#75a7ff")


def _default_problem_code(result) -> str:
    if not result.recommendations.empty:
        growth = result.recommendations[result.recommendations["bucket"] == "growth"]
        source = growth if not growth.empty else result.recommendations
        return str(source.iloc[0]["problem_id"])
    return "1900C"


def _find_problem_by_code(problems: pd.DataFrame, code: str) -> pd.Series | None:
    normalized = _normalize_problem_code(code)
    if not normalized or problems.empty:
        return None
    matches = problems[problems["problem_id"].astype(str).str.upper() == normalized]
    if matches.empty:
        return None
    return matches.iloc[0]


def _normalize_problem_code(code: str) -> str:
    return "".join(str(code or "").upper().split())


def _format_tags(tags) -> str:
    if isinstance(tags, list):
        return ", ".join(str(tag) for tag in tags)
    if pd.isna(tags):
        return ""
    return str(tags)


def _codeforces_problem_url(row: pd.Series) -> str:
    contest_id = row.get("contest_id")
    index = row.get("index")
    if pd.notna(contest_id) and pd.notna(index):
        return f"https://codeforces.com/problemset/problem/{int(contest_id)}/{index}"
    problem_id = str(row.get("problem_id", ""))
    split_at = 0
    while split_at < len(problem_id) and problem_id[split_at].isdigit():
        split_at += 1
    digits = problem_id[:split_at]
    suffix = problem_id[split_at:]
    if digits and suffix:
        return f"https://codeforces.com/problemset/problem/{digits}/{suffix}"
    return "https://codeforces.com/problemset"


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
          --bg: #08090b;
          --surface: #101216;
          --surface-2: #151922;
          --border: #252b35;
          --text: #eef2f6;
          --muted: #a0a8b5;
          --faint: #6f7886;
          --green: #5ee0a0;
          --amber: #f8c76f;
          --red: #f28b82;
          --blue: #75a7ff;
        }
        .stApp {
          background:
            linear-gradient(90deg, rgba(255,255,255,.035) 1px, transparent 1px) 0 0 / 72px 72px,
            linear-gradient(0deg, rgba(255,255,255,.028) 1px, transparent 1px) 0 0 / 72px 72px,
            var(--bg);
          color: var(--text);
        }
        section[data-testid="stSidebar"] {
          background: #0a0c0f;
          border-right: 1px solid var(--border);
        }
        .block-container {
          max-width: 1480px;
          padding-top: 1.4rem;
          padding-bottom: 3rem;
        }
        .brand {
          display: flex;
          align-items: center;
          gap: 12px;
          padding: 8px 0 18px;
        }
        .brand-mark {
          display: grid;
          place-items: center;
          width: 38px;
          height: 38px;
          border: 1px solid #38414e;
          border-radius: 8px;
          background: #131720;
          color: var(--green);
          font-family: ui-monospace, Consolas, monospace;
          font-weight: 800;
          letter-spacing: 0;
        }
        .brand strong {
          display: block;
          color: var(--text);
          font-size: 15px;
        }
        .brand span {
          display: block;
          color: var(--faint);
          font-family: ui-monospace, Consolas, monospace;
          font-size: 11px;
          text-transform: uppercase;
          letter-spacing: 0;
        }
        .hero {
          display: flex;
          align-items: flex-start;
          justify-content: space-between;
          gap: 20px;
          margin-bottom: 18px;
          padding: 20px;
          border: 1px solid var(--border);
          border-radius: 8px;
          background: rgba(16,18,22,.92);
          box-shadow: inset 0 1px 0 rgba(255,255,255,.035);
        }
        .hero h1 {
          margin: 4px 0 6px;
          color: var(--text);
          font-size: 34px;
          line-height: 1.05;
          letter-spacing: 0;
        }
        .eyebrow,
        .panel-title p {
          margin: 0;
          color: var(--faint);
          font-family: ui-monospace, Consolas, monospace;
          font-size: 11px;
          text-transform: uppercase;
          letter-spacing: 0;
        }
        .subcopy {
          max-width: 760px;
          color: var(--muted);
          font-size: 14px;
          line-height: 1.55;
        }
        .source-pill,
        .bucket-label {
          display: inline-flex;
          align-items: center;
          min-height: 28px;
          padding: 4px 10px;
          border: 1px solid rgba(94,224,160,.3);
          border-radius: 999px;
          background: rgba(94,224,160,.06);
          color: var(--green);
          font-family: ui-monospace, Consolas, monospace;
          font-size: 12px;
          letter-spacing: 0;
          white-space: nowrap;
        }
        .metric-card,
        .day-card,
        .rule-list {
          min-height: 112px;
          margin-bottom: 14px;
          padding: 15px;
          border: 1px solid var(--border);
          border-radius: 8px;
          background: rgba(16,18,22,.92);
          box-shadow: inset 0 1px 0 rgba(255,255,255,.035);
        }
        .metric-card p {
          margin: 0;
          color: var(--muted);
          font-size: 12px;
          font-weight: 650;
        }
        .metric-card strong {
          display: block;
          margin-top: 10px;
          color: var(--text);
          font-family: ui-monospace, Consolas, monospace;
          font-size: 26px;
          line-height: 1.05;
          letter-spacing: 0;
          overflow-wrap: anywhere;
        }
        .metric-card span {
          display: block;
          margin-top: 9px;
          color: var(--faint);
          font-size: 12px;
        }
        .panel-title {
          margin: 18px 0 10px;
        }
        .panel-title h2 {
          margin: 4px 0 0;
          color: var(--text);
          font-size: 18px;
          font-weight: 760;
          letter-spacing: 0;
        }
        .stDataFrame,
        div[data-testid="stPlotlyChart"] {
          padding: 10px;
          border: 1px solid var(--border);
          border-radius: 8px;
          background: rgba(16,18,22,.92);
          box-shadow: inset 0 1px 0 rgba(255,255,255,.035);
        }
        .day-card {
          min-height: 188px;
        }
        .day-card code {
          display: inline-flex;
          padding: 2px 7px;
          border: 1px solid #2b313c;
          border-radius: 5px;
          background: #0b0d11;
          color: #d9e1ea;
          font-family: ui-monospace, Consolas, monospace;
          font-size: 12px;
        }
        .day-card h3 {
          margin: 16px 0 8px;
          color: var(--text);
          font-size: 15px;
        }
        .day-card p,
        .rule-list p {
          color: var(--muted);
          font-size: 12px;
          line-height: 1.5;
        }
        .meter {
          overflow: hidden;
          height: 7px;
          margin-top: 16px;
          border-radius: 999px;
          background: #222832;
        }
        .meter span {
          display: block;
          height: 100%;
          border-radius: inherit;
          background: var(--amber);
        }
        button[kind="primary"],
        .stButton > button {
          border: 1px solid #3a4452;
          border-radius: 7px;
          background: #18201f;
          color: var(--green);
          font-weight: 750;
        }
        .stTextInput input,
        .stNumberInput input,
        .stSelectbox div,
        .stMultiSelect div {
          border-color: #343b47;
          background-color: #0f1217;
          color: var(--text);
        }
        @media (max-width: 900px) {
          .hero {
            display: block;
          }
          .source-pill {
            margin-top: 12px;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

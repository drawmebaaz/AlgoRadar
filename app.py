from __future__ import annotations

import inspect

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from algoradar.platforms import analyze_external_platforms, build_combined_overview, lookup_leetcode_problem
from algoradar.solve_probability import (
    available_probability_tags,
    score_saved_profile_problem,
)

st.set_page_config(
    page_title="AlgoRadar",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)

_STRETCH_SUPPORT_CACHE: dict[str, bool] = {}


def stretch_kwargs(component) -> dict[str, bool | str]:
    key = getattr(component, "__qualname__", repr(component))
    if key not in _STRETCH_SUPPORT_CACHE:
        try:
            _STRETCH_SUPPORT_CACHE[key] = "width" in inspect.signature(component).parameters
        except (TypeError, ValueError):
            _STRETCH_SUPPORT_CACHE[key] = False
    return {"width": "stretch"} if _STRETCH_SUPPORT_CACHE[key] else {"use_container_width": True}


@st.cache_resource(show_spinner=False)
def cached_analysis(
    handle: str,
    force_refresh: bool,
    prefer_transformer: bool,
    use_sample: bool,
    submission_limit: int,
    include_recommendations: bool,
    include_semantic: bool,
):
    from algoradar.pipeline import run_analysis

    return run_analysis(
        handle=handle,
        force_refresh=force_refresh,
        prefer_transformer=prefer_transformer,
        use_sample=use_sample,
        submission_limit=submission_limit,
        include_recommendations=include_recommendations,
        include_semantic=include_semantic,
    )


@st.cache_resource(show_spinner=False)
def cached_external_analysis(
    leetcode_handle: str,
    codechef_handle: str,
    force_refresh: bool,
    include_recommendations: bool,
):
    return analyze_external_platforms(
        leetcode_handle=leetcode_handle,
        codechef_handle=codechef_handle,
        force_refresh=force_refresh,
        include_recommendations=include_recommendations,
    )


@st.cache_data(show_spinner=False)
def cached_leetcode_problem_lookup(slug_or_url: str, force_refresh: bool):
    return lookup_leetcode_problem(slug_or_url, force_refresh=force_refresh)


def main() -> None:
    inject_css()
    render_sidebar()

    with st.sidebar:
        codeforces_handle = st.text_input("Codeforces handle", key="codeforces_handle_input", help="Optional. Enables the deepest verdict-level ML analysis.")
        codechef_handle = st.text_input("CodeChef handle", key="codechef_handle_input", help="Optional. Enables CodeChef rating and practice analysis.")
        leetcode_handle = st.text_input("LeetCode username", key="leetcode_handle_input", help="Optional. Enables LeetCode topic and contest analysis.")
        analyze = st.button("Analyze handles", **stretch_kwargs(st.button))
        force_refresh = st.toggle("Refresh platform caches", value=False)
        prefer_transformer = st.toggle(
            "Use MiniLM embeddings",
            value=False,
            help="If sentence-transformers is installed, this uses all-MiniLM-L6-v2. Otherwise the app uses TF-IDF fallback.",
        )

        if "screen" not in st.session_state:
            st.session_state.screen = "Combined analysis"
        screen = st.radio(
            "Sections",
            [
                "Combined analysis",
                "Codeforces",
                "CodeChef",
                "LeetCode",
                "Recommendations",
                "Solve probability",
            ],
            key="screen",
        )

    current_args = {
        "codeforces": codeforces_handle.strip(),
        "leetcode": leetcode_handle.strip(),
        "codechef": codechef_handle.strip(),
        "force_refresh": force_refresh,
        "prefer_transformer": prefer_transformer,
    }
    if "active_handle_args" not in st.session_state:
        st.session_state.active_handle_args = current_args
    if analyze:
        st.session_state.active_handle_args = current_args
        cached_analysis.clear()
        cached_external_analysis.clear()

    active_args = st.session_state.active_handle_args
    settings_changed = current_args != active_args
    result = None
    external_results: dict = {}

    if _needs_codeforces(screen, active_args) and active_args["codeforces"]:
        include_recommendations = screen == "Recommendations"
        cf_args = {
            "handle": active_args["codeforces"],
            "force_refresh": active_args["force_refresh"],
            "prefer_transformer": active_args["prefer_transformer"],
            "use_sample": False,
            "submission_limit": 2500 if screen in {"Codeforces", "Recommendations"} else 1200,
            "include_recommendations": include_recommendations,
            "include_semantic": include_recommendations,
        }
        with st.spinner("Loading Codeforces profile and submissions..."):
            result = cached_analysis(**cf_args)

    external_handles = _external_handles_for_screen(screen, active_args)
    if external_handles["leetcode"] or external_handles["codechef"]:
        with st.spinner("Loading platform profile data..."):
            external_results = cached_external_analysis(
                external_handles["leetcode"],
                external_handles["codechef"],
                active_args["force_refresh"],
                screen == "Recommendations",
            )

    source_label = _source_label(result, external_results)
    hero_title = _fallback_profile_title(active_args)
    st.markdown(
        f"""
        <div class="hero">
          <div>
            <p class="eyebrow">AlgoRadar / Multi-platform competitive programming intelligence</p>
            <h1>{hero_title}</h1>
            <p class="subcopy">Add any combination of Codeforces, CodeChef, and LeetCode handles. Platform sections stay focused; recommendations are separated by platform, and solve probability uses the handles you provide.</p>
          </div>
          <div class="source-pill">{source_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if settings_changed:
        st.info("Sidebar handles/settings changed. Click Analyze handles to run with the new values.")

    if screen == "Combined analysis":
        render_combined_entry(result, external_results, active_args)
    elif screen == "Codeforces":
        if require_codeforces(result):
            render_codeforces_section(result)
    elif screen == "CodeChef":
        render_platform_detail(external_results.get("codechef"), "CodeChef")
    elif screen == "LeetCode":
        render_platform_detail(external_results.get("leetcode"), "LeetCode")
    elif screen == "Recommendations":
        render_combined_recommendations(result, external_results, active_args)
    elif screen == "Solve probability":
        render_general_probability(result, external_results, active_args)


def _needs_codeforces(screen: str, active_args: dict[str, str]) -> bool:
    if screen == "Combined analysis":
        return _provided_handle_count(active_args) >= 2
    return screen in {"Codeforces", "Recommendations", "Solve probability"}


def _external_handles_for_screen(screen: str, active_args: dict[str, str]) -> dict[str, str]:
    combined_ready = screen == "Combined analysis" and _provided_handle_count(active_args) >= 2
    needs_leetcode = screen in {"LeetCode", "Recommendations", "Solve probability"} or combined_ready
    needs_codechef = screen in {"CodeChef", "Recommendations", "Solve probability"} or combined_ready
    return {
        "leetcode": active_args["leetcode"] if needs_leetcode else "",
        "codechef": active_args["codechef"] if needs_codechef else "",
    }


def _source_label(result, external_results: dict) -> str:
    sources = []
    if result is not None:
        sources.append("Codeforces")
    sources.extend(analysis.platform for analysis in external_results.values() if analysis.status == "ok")
    if not sources:
        return "Add handles"
    return " + ".join(sources)


def _fallback_profile_title(active_args: dict[str, str]) -> str:
    provided = [value for key, value in active_args.items() if key in {"codeforces", "leetcode", "codechef"} and value]
    if len(provided) >= 2:
        return "Combined CP profile"
    for key in ["codeforces", "leetcode", "codechef"]:
        if active_args.get(key):
            return active_args[key]
    return "AlgoRadar profile"


def require_codeforces(result) -> bool:
    if result is not None:
        return True
    st.info("Add a Codeforces handle in the sidebar, then click Analyze handles.")
    return False


def _provided_handle_count(active_args: dict[str, str]) -> int:
    return sum(1 for key in ["codeforces", "leetcode", "codechef"] if active_args.get(key))


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


def render_combined_entry(result, external_results: dict, active_args: dict[str, str]) -> None:
    if _provided_handle_count(active_args) < 2:
        st.info("Add at least two handles in the sidebar to unlock combined analysis.")
        return
    render_combined_profile(result, external_results)


def render_codeforces_section(result) -> None:
    render_profile(result)
    render_weakness(result)
    render_progress(result)


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
            st.plotly_chart(fig, **stretch_kwargs(st.plotly_chart))

    with right:
        panel_title("Verdict distribution", "submission failure patterns")
        verdicts = result.verdicts.copy()
        if verdicts.empty:
            st.info("No submissions found.")
        else:
            fig = go.Figure(
                go.Pie(
                    labels=verdicts["verdict"],
                    values=verdicts["count"],
                    hole=0.58,
                    marker=dict(colors=palette()),
                )
            )
            fig.update_layout(**chart_layout(height=360))
            st.plotly_chart(fig, **stretch_kwargs(st.plotly_chart))

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
            st.plotly_chart(fig, **stretch_kwargs(st.plotly_chart))
        else:
            st.info("No rating data available.")

    with right:
        panel_title("Solved difficulty distribution", "where accepted problems cluster")
        frame = result.solved_difficulty
        if not frame.empty:
            fig = go.Figure(go.Bar(x=frame["rating_bucket"], y=frame["solved"], marker_color="#5ee0a0"))
            fig.update_layout(**chart_layout())
            st.plotly_chart(fig, **stretch_kwargs(st.plotly_chart))
        else:
            st.info("No accepted submissions available.")


def render_combined_profile(result, external_results: dict) -> None:
    overview = build_combined_overview(result, external_results)
    summary = overview["summary"]
    platforms = overview["platforms"]
    focus = overview["focus"]
    trend = overview["trend"]

    cols = st.columns(4)
    metric_card(cols[0], "Total solved", str(summary["total_solved"]), "all connected platforms")
    metric_card(cols[1], "Platforms", f"{summary['platforms_connected']}/3", "active public profiles")
    metric_card(cols[2], "Focus areas", str(summary["focus_areas"]), "combined weakness signals")
    metric_card(cols[3], "Priority platform", summary["attention_platform"], "highest repair signal")

    if platforms.empty:
        st.info("Add handles in the sidebar, then click Analyze handles.")
        return

    left, right = st.columns([1.1, 1])
    with left:
        panel_title("Platform breakdown", "provided handles and cached public-data signals")
        show = platforms.rename(
            columns={
                "platform": "Platform",
                "handle": "Handle",
                "status": "Status",
                "solved": "Solved",
                "current_rating": "Native rating",
                "max_rating": "Native max",
                "contests": "Contests",
                "accuracy": "Accuracy %",
                "signal": "Data signal",
            }
        )
        st.dataframe(show, **stretch_kwargs(st.dataframe), hide_index=True)

    with right:
        panel_title("Solved distribution", "cross-platform practice volume")
        ok = platforms[platforms["status"] == "ok"].copy()
        if ok.empty:
            st.info("No successful platform pulls yet.")
        else:
            fig = go.Figure(
                go.Bar(
                    x=ok["platform"],
                    y=ok["solved"],
                    marker_color=_series_colors(ok["platform"]),
                )
            )
            fig.update_layout(**chart_layout(height=360), showlegend=False)
            st.plotly_chart(fig, **stretch_kwargs(st.plotly_chart))

    left, right = st.columns([1.2, 1])
    with left:
        panel_title("Combined focus map", "highest-priority areas across platforms")
        if focus.empty:
            st.info("No weakness signals available yet.")
        else:
            show = focus.rename(
                columns={
                    "platform": "Platform",
                    "area": "Area",
                    "level": "Level",
                    "priority": "Priority",
                    "next_action": "Next action",
                }
            )
            st.dataframe(show.head(12), **stretch_kwargs(st.dataframe), hide_index=True)

    with right:
        panel_title("Native rating trend", "contest signals in each platform's own scale")
        if trend.empty:
            st.info("No contest rating history found.")
        else:
            trend = trend.copy()
            trend["point"] = trend.groupby("platform").cumcount()
            fig = go.Figure()
            for platform, group in trend.groupby("platform"):
                fig.add_trace(
                    go.Scatter(
                        x=group["point"],
                        y=group["rating"],
                        mode="lines+markers",
                        name=str(platform),
                        customdata=group[["contest", "delta"]],
                        hovertemplate="Contest: %{customdata[0]}<br>Rating: %{y}<br>Delta: %{customdata[1]}<extra></extra>",
                    )
                )
            fig.update_layout(**chart_layout(height=420))
            st.plotly_chart(fig, **stretch_kwargs(st.plotly_chart))


def render_platform_detail(analysis, platform_name: str) -> None:
    if analysis is None:
        st.info(f"Add your {platform_name} handle in the sidebar, then click Analyze handles.")
        return
    if analysis.status != "ok":
        st.warning(f"{platform_name} could not be loaded: {analysis.error or analysis.status}")
        return
    if platform_name == "LeetCode":
        render_leetcode_detail(analysis)
    else:
        render_codechef_detail(analysis)


def render_leetcode_detail(analysis) -> None:
    profile = analysis.profile
    cols = st.columns(4)
    metric_card(cols[0], "Solved", str(profile.get("total_solved", 0)), "Easy + Medium + Hard")
    metric_card(cols[1], "Contest rating", str(profile.get("contest_rating", 0)), "LeetCode native rating")
    metric_card(cols[2], "Acceptance", f"{profile.get('acceptance_rate', 0)}%", "accepted submissions / submissions")
    metric_card(cols[3], "Ranking", str(profile.get("ranking", 0)), "global profile rank")

    left, right = st.columns([1.1, 1])
    with left:
        panel_title("Difficulty accuracy", "accepted and attempted submissions by difficulty")
        difficulty = analysis.difficulty.copy()
        if difficulty.empty:
            st.info("No public difficulty stats available.")
        else:
            fig = go.Figure()
            fig.add_bar(x=difficulty["difficulty"], y=difficulty["solved"], name="Solved", marker_color="#5ee0a0")
            fig.add_trace(
                go.Scatter(
                    x=difficulty["difficulty"],
                    y=difficulty["accuracy"],
                    mode="lines+markers",
                    name="Accuracy %",
                    line=dict(color="#f8c76f", width=2),
                    yaxis="y2",
                )
            )
            fig.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0, 100], showgrid=False), **chart_layout())
            st.plotly_chart(fig, **stretch_kwargs(st.plotly_chart))

    with right:
        panel_title("Contest trend", "LeetCode rating history")
        trend = analysis.contest_trend.copy()
        if trend.empty:
            st.info("No public LeetCode contest history found.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=trend["contest"], y=trend["rating"], mode="lines+markers", name="Rating", line=dict(color="#75a7ff")))
            fig.add_bar(x=trend["contest"], y=trend["delta"], name="Delta", marker_color="#2b3340", yaxis="y2")
            fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False), **chart_layout())
            st.plotly_chart(fig, **stretch_kwargs(st.plotly_chart))

    left, right = st.columns([1.2, 1])
    with left:
        panel_title("LeetCode weakness map", "coverage-based topic classifier")
        weakness = analysis.weakness.copy()
        if weakness.empty:
            st.info("No tag coverage data available.")
        else:
            show = weakness[["tag", "level", "solved", "priority_score", "next_action", "source"]].head(18)
            show = show.rename(
                columns={
                    "tag": "Topic",
                    "level": "Level",
                    "solved": "Solved",
                    "priority_score": "Priority",
                    "next_action": "Next action",
                    "source": "Source",
                }
            )
            st.dataframe(show, **stretch_kwargs(st.dataframe), hide_index=True)

    with right:
        panel_title("Top solved tags", "public tag counters")
        tags = analysis.tags.head(12).copy()
        if tags.empty:
            st.info("No public tag counters available.")
        else:
            sorted_tags = tags.sort_values("solved")
            fig = go.Figure(go.Bar(x=sorted_tags["solved"], y=sorted_tags["tag"], orientation="h", marker_color="#5ee0a0"))
            fig.update_layout(**chart_layout(height=480), showlegend=False)
            st.plotly_chart(fig, **stretch_kwargs(st.plotly_chart))


def render_codechef_detail(analysis) -> None:
    profile = analysis.profile
    cols = st.columns(4)
    metric_card(cols[0], "Current rating", str(profile.get("current_rating", 0)), profile.get("division", "CodeChef rating"))
    metric_card(cols[1], "Max rating", str(profile.get("max_rating", 0)), f"{profile.get('stars', 0)} star profile")
    metric_card(cols[2], "Solved", str(profile.get("total_solved", 0)), "public profile count")
    metric_card(cols[3], "Contests", str(profile.get("contest_count", 0)), "rated history entries")

    left, right = st.columns([1.25, 1])
    with left:
        panel_title("CodeChef rating trend", "contest rating and delta")
        trend = analysis.contest_trend.copy()
        if trend.empty:
            st.info("No CodeChef rating history found.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=trend["contest"], y=trend["rating"], mode="lines+markers", name="Rating", line=dict(color="#5ee0a0")))
            fig.add_bar(x=trend["contest"], y=trend["delta"], name="Delta", marker_color="#75a7ff", opacity=0.42, yaxis="y2")
            fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False), **chart_layout(height=430))
            st.plotly_chart(fig, **stretch_kwargs(st.plotly_chart))

    with right:
        panel_title("Solved sections", "what the public profile exposes")
        difficulty = analysis.difficulty.copy()
        if difficulty.empty:
            st.info("No solved section data available.")
        else:
            fig = go.Figure(go.Bar(x=difficulty["difficulty"], y=difficulty["solved"], marker_color="#f8c76f"))
            fig.update_layout(**chart_layout(height=430), showlegend=False)
            st.plotly_chart(fig, **stretch_kwargs(st.plotly_chart))

    panel_title("CodeChef weakness signals", "rating-history and practice-volume signals")
    weakness = analysis.weakness.copy()
    if weakness.empty:
        st.info("No CodeChef weakness signals available.")
    else:
        show = weakness[["tag", "level", "attempts", "priority_score", "next_action", "source"]].rename(
            columns={
                "tag": "Area",
                "level": "Level",
                "attempts": "Signal value",
                "priority_score": "Priority",
                "next_action": "Next action",
                "source": "Source",
            }
        )
        st.dataframe(show, **stretch_kwargs(st.dataframe), hide_index=True)


def render_combined_recommendations(result, external_results: dict, active_args: dict[str, str]) -> None:
    overview = build_combined_overview(result, external_results)
    recommendations = overview["recommendations"]
    cols = st.columns(4)
    if recommendations.empty:
        metric_card(cols[0], "Recommendations", "0", "add handles first")
        metric_card(cols[1], "Codeforces", "0", "confidence/growth/stretch")
        metric_card(cols[2], "CodeChef", "0", "confidence/growth/stretch")
        metric_card(cols[3], "LeetCode", "0", "confidence/growth/stretch")
    else:
        counts = recommendations["bucket"].value_counts().to_dict()
        metric_card(cols[0], "Confidence", str(counts.get("confidence", 0)), "warm-up problems")
        metric_card(cols[1], "Growth", str(counts.get("growth", 0)), "main training queue")
        metric_card(cols[2], "Stretch", str(counts.get("stretch", 0)), "harder attempts")
        metric_card(cols[3], "Platforms", str(recommendations["platform"].nunique()), "sources in queue")

    panel_title("Platform practice queues", "recommendations separated by platform and training bucket")
    tabs = st.tabs(["Codeforces", "CodeChef", "LeetCode"])
    for tab, platform in zip(tabs, ["Codeforces", "CodeChef", "LeetCode"]):
        with tab:
            _render_platform_recommendation_group(platform, recommendations, result, external_results, active_args)


def _render_platform_recommendation_group(
    platform: str,
    recommendations: pd.DataFrame,
    result,
    external_results: dict,
    active_args: dict[str, str],
) -> None:
    status = _recommendation_platform_status(platform, result, external_results, active_args)
    if status:
        st.info(status)
        return

    frame = recommendations[recommendations["platform"] == platform].copy() if not recommendations.empty else pd.DataFrame()
    if frame.empty:
        st.info(f"No {platform} recommendations are available yet.")
        return

    counts = frame["bucket"].value_counts().to_dict()
    cols = st.columns(4)
    metric_card(cols[0], "Confidence", str(counts.get("confidence", 0)), "warm-up problems")
    metric_card(cols[1], "Growth", str(counts.get("growth", 0)), "main training queue")
    metric_card(cols[2], "Stretch", str(counts.get("stretch", 0)), "harder attempts")
    metric_card(cols[3], "Total", str(len(frame)), f"{platform} queue")

    bucket_order = [
        ("confidence", "Confidence builders", "problems that should feel controlled"),
        ("growth", "Growth problems", "main practice targets"),
        ("stretch", "Stretch problems", "harder attempts for expansion"),
        ("avoid", "Avoid for now", "come back after stronger prep"),
    ]
    bucket_tabs = st.tabs([label for bucket, label, _ in bucket_order if bucket in counts or bucket != "avoid"])
    visible_buckets = [(bucket, label, subtitle) for bucket, label, subtitle in bucket_order if bucket in counts or bucket != "avoid"]
    for bucket_tab, (bucket, label, subtitle) in zip(bucket_tabs, visible_buckets):
        with bucket_tab:
            bucket_frame = frame[frame["bucket"] == bucket].copy()
            panel_title(label, subtitle)
            if bucket_frame.empty:
                st.info(f"No {bucket} {platform} recommendations found right now.")
                continue
            _show_recommendations_table(bucket_frame, show_platform=False, show_bucket=False)


def _recommendation_platform_status(platform: str, result, external_results: dict, active_args: dict[str, str]) -> str:
    key = {"Codeforces": "codeforces", "CodeChef": "codechef", "LeetCode": "leetcode"}[platform]
    if not active_args.get(key):
        return f"Add your {platform} handle in the sidebar, then click Analyze handles."
    if platform == "Codeforces":
        return "" if result is not None else "Codeforces analysis is not loaded yet. Click Analyze handles."
    analysis = external_results.get(key)
    if analysis is None:
        return f"{platform} analysis is not loaded yet. Click Analyze handles."
    if analysis.status != "ok":
        return f"{platform} could not be loaded: {analysis.error or analysis.status}"
    return ""


def render_general_probability(result, external_results: dict, active_args: dict[str, str]) -> None:
    available_platforms = [
        platform
        for platform, key in [("Codeforces", "codeforces"), ("CodeChef", "codechef"), ("LeetCode", "leetcode")]
        if active_args.get(key)
    ]
    if not available_platforms:
        st.info("Add at least one handle in the sidebar, then click Analyze handles.")
        return

    left, right = st.columns([0.95, 1.05])
    with left:
        panel_title("Problem context", "works with whichever handles you provided")
        platform = st.selectbox("Problem platform", options=available_platforms)
        tags_options = available_probability_tags(result, external_results)
        default_tags = _default_probability_tags(result, external_results)

        context = _probability_problem_context(
            platform=platform,
            result=result,
            tags_options=tags_options,
            default_tags=default_tags,
            force_refresh=bool(active_args.get("force_refresh", False)),
        )

        score = score_saved_profile_problem(
            platform=platform,
            target_rating=context["target_rating"],
            tags=context["tags"],
            popularity=context["popularity"],
            codeforces_result=result,
            external_results=external_results,
            leetcode_difficulty=context.get("leetcode_difficulty", ""),
            leetcode_contest_slot=context.get("leetcode_contest_slot", "Unknown"),
        )
        st.caption(
            "Ratings are first calibrated onto a CF-equivalent difficulty scale using the mapping CSV. "
            "The score prioritizes solved volume, hardest solved difficulty, and selected-tag strength; accuracy is not a primary signal."
        )

    with right:
        panel_title("Solve probability", "calibrated handle-aware estimate")
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
        st.plotly_chart(fig, **stretch_kwargs(st.plotly_chart))
        st.markdown(f"<div class='bucket-label'>{score['bucket'].upper()}</div>", unsafe_allow_html=True)

    cols = st.columns(4)
    metric_card(cols[0], "User CF-eq anchor", str(int(score["anchor_cf_equivalent"])), "calibrated from provided handles")
    metric_card(cols[1], "Target CF-eq", str(int(score["target_cf_equivalent"])), str(score["native_target"]))
    metric_card(cols[2], "Tag solves", str(int(score["tag_solved"])), "selected tags / aliases")
    metric_card(cols[3], "Tag ceiling", str(int(score["tag_rating_ceiling"])), "hardest solved signal")
    st.caption(f"Calibration: {score['calibration_source']} | confidence: {score['calibration_confidence']} | weight: {score['calibration_weight']}")

    panel_title("Why this probability", "volume and rating-strength inputs")
    st.dataframe(score["factors"], **stretch_kwargs(st.dataframe), hide_index=True)


def _probability_problem_context(
    platform: str,
    result,
    tags_options: list[str],
    default_tags: list[str],
    force_refresh: bool,
) -> dict:
    if platform == "Codeforces" and result is not None:
        default_code = _default_problem_code(result)
        problem_code = st.text_input("Codeforces problem code", value=default_code)
        problem = _find_problem_by_code(result.problems, problem_code)
        if problem is not None:
            problem_name = str(problem["name"])
            rating = int(problem["rating"])
            fetched_tags = list(problem["tags"] or [])
            popularity = int(problem["solved_count"])
            tag_options = _merge_tag_options(tags_options, fetched_tags)
            tags = st.multiselect(
                "Problem tags",
                options=tag_options,
                default=[tag for tag in fetched_tags if tag in tag_options][:6],
                help="Pulled from Codeforces problem metadata; edit if needed.",
            )
            st.markdown(
                f"""
                <div class="rule-list">
                  <p><strong>{problem['problem_id']} - {problem_name}</strong></p>
                  <p>Difficulty: <strong>{rating}</strong> ({problem.get('rating_source', 'official')})</p>
                  <p>Solved by: <strong>{popularity}</strong> users</p>
                  <p>Tags: {_format_tags(fetched_tags) or "untagged"}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return {
                "target_rating": rating,
                "tags": tags or fetched_tags,
                "popularity": popularity,
                "problem_name": problem_name,
                "leetcode_difficulty": "",
                "leetcode_contest_slot": "Unknown",
            }
        st.warning("Problem code not found. Use manual fields below.")

    if platform == "LeetCode":
        return _leetcode_probability_inputs(tags_options, default_tags, force_refresh)

    return _manual_probability_inputs(platform, tags_options, default_tags)


def _leetcode_probability_inputs(tags_options: list[str], default_tags: list[str], force_refresh: bool) -> dict:
    slug_or_url = st.text_input("LeetCode problem slug or URL", placeholder="two-sum or https://leetcode.com/problems/two-sum/")
    if slug_or_url.strip():
        try:
            problem = cached_leetcode_problem_lookup(slug_or_url.strip(), force_refresh)
        except Exception as exc:
            problem = {"status": "error", "error": str(exc)}
        if problem.get("status") == "ok":
            fetched_tags = list(problem.get("tags", []) or [])
            tag_options = _merge_tag_options(tags_options, fetched_tags)
            tags = st.multiselect(
                "Problem tags",
                options=tag_options,
                default=[tag for tag in fetched_tags if tag in tag_options][:6],
                help="Pulled from LeetCode topic tags; edit if needed.",
            )
            slot = st.selectbox(
                "Contest slot reference",
                options=["Unknown", "Q1", "Q2", "Q3", "Q4"],
                index=0,
                help="LeetCode public problem lookup does not expose the original contest slot. Set this only if you know it.",
            )
            accepted = int(problem.get("accepted", 0) or 0)
            popularity = accepted or 5000
            st.markdown(
                f"""
                <div class="rule-list">
                  <p><strong>#{problem.get('problem_id', '')} - {problem.get('title', '')}</strong></p>
                  <p>Difficulty: <strong>{problem.get('difficulty', 'Medium')}</strong></p>
                  <p>Accepted submissions: <strong>{accepted}</strong></p>
                  <p>Acceptance rate: <strong>{problem.get('acceptance_rate', 0)}%</strong></p>
                  <p>Tags: {_format_tags(fetched_tags) or "untagged"}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption("LeetCode has no official problem rating. AlgoRadar uses difficulty, tags, acceptance, and optional Q1-Q4 slot as calibrated references.")
            return {
                "target_rating": None,
                "tags": tags or fetched_tags,
                "popularity": popularity,
                "problem_name": problem.get("title", ""),
                "leetcode_difficulty": problem.get("difficulty", "Medium"),
                "leetcode_contest_slot": slot,
            }
        st.warning(problem.get("error", "LeetCode problem could not be loaded. Use manual fields below."))
    else:
        st.info("Enter a LeetCode slug or URL to fetch difficulty and tags automatically, or use manual fields below.")
    return _manual_probability_inputs("LeetCode", tags_options, default_tags)


def _manual_probability_inputs(platform: str, tags_options: list[str], default_tags: list[str]) -> dict:
    problem_name = st.text_input("Problem name", value="Custom training target")
    if platform == "LeetCode":
        difficulty = st.selectbox("LeetCode difficulty", options=["Easy", "Medium", "Hard"], index=1)
        slot = st.selectbox(
            "Contest slot reference",
            options=["Unknown", "Q1", "Q2", "Q3", "Q4"],
            index=0,
            help="Optional reference if the problem came from a LeetCode contest.",
        )
        rating = None
        st.caption("LeetCode has no official problem rating. Difficulty and optional contest slot are calibrated onto a CF-equivalent scale.")
    elif platform == "CodeChef":
        rating = st.slider("CodeChef problem rating (native)", min_value=400, max_value=5000, value=1500, step=50)
        difficulty = ""
        slot = "Unknown"
    else:
        rating = st.slider("Codeforces problem rating (official or estimated)", min_value=400, max_value=3500, value=1600, step=100)
        difficulty = ""
        slot = "Unknown"
    tags = st.multiselect(
        "Problem tags",
        options=tags_options,
        default=[tag for tag in default_tags if tag in tags_options][:3],
        help="Use the platform tags when available; otherwise add the closest topic tags manually.",
    )
    popularity = st.number_input("Solved count / popularity", min_value=0, max_value=500000, value=5000, step=100)
    return {
        "target_rating": int(rating) if rating is not None else None,
        "tags": tags,
        "popularity": int(popularity),
        "problem_name": problem_name,
        "leetcode_difficulty": difficulty,
        "leetcode_contest_slot": slot,
    }


def _merge_tag_options(tags_options: list[str], tags: list[str]) -> list[str]:
    return sorted(set(tags_options) | {str(tag) for tag in tags if str(tag).strip()}, key=lambda value: value.lower())


def _default_probability_tags(result, external_results: dict) -> list[str]:
    tags: list[str] = []
    if result is not None and not result.weakness.empty:
        tags.extend(str(tag) for tag in result.weakness["tag"].head(3).tolist())
    leetcode = external_results.get("leetcode")
    if leetcode is not None and getattr(leetcode, "status", "") == "ok" and not leetcode.weakness.empty:
        tags.extend(str(tag) for tag in leetcode.weakness["tag"].head(3).tolist())
    return tags or ["dp", "graphs", "greedy"]


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
        st.dataframe(show, **stretch_kwargs(st.dataframe), height=520)

    with right:
        panel_title("Most failed tags", "repair priority")
        top = weakness.head(10).sort_values("priority_score", ascending=True)
        level_colors = {
            "Strong": "#5ee0a0",
            "Stable": "#75a7ff",
            "Weak": "#f28b82",
            "Untouched": "#6f7886",
            "Over-attempted but low accuracy": "#f8c76f",
        }
        fig = go.Figure(
            go.Bar(
                x=top["priority_score"],
                y=top["tag"],
                orientation="h",
                marker_color=top["level"].map(level_colors).fillna("#75a7ff"),
                customdata=top["level"],
                hovertemplate="Tag: %{y}<br>Priority: %{x}<br>Level: %{customdata}<extra></extra>",
            )
        )
        fig.update_layout(**chart_layout(height=520), showlegend=False)
        st.plotly_chart(fig, **stretch_kwargs(st.plotly_chart))

    panel_title("Explainability", "random forest feature importance for weakness model")
    st.dataframe(result.weakness_model["feature_importance"], **stretch_kwargs(st.dataframe), hide_index=True)


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
            **stretch_kwargs(st.dataframe),
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
            **stretch_kwargs(st.dataframe),
            hide_index=True,
            column_config={
                "open": st.column_config.LinkColumn("Open", display_text="Codeforces"),
                "rating_source": st.column_config.TextColumn("Rating source"),
                "semantic_score": st.column_config.NumberColumn("Similarity", format="%.3f"),
            },
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
    st.plotly_chart(fig, **stretch_kwargs(st.plotly_chart))


def _show_recommendations_table(recommendations: pd.DataFrame, show_platform: bool = True, show_bucket: bool = True) -> None:
    if recommendations is None or recommendations.empty:
        st.info("No recommendations available for this profile yet.")
        return
    frame = recommendations.copy()
    if "tags" in frame.columns:
        frame["tags"] = frame["tags"].apply(_format_tags)
    if "rank_score" in frame.columns:
        frame["rank_score"] = pd.to_numeric(frame["rank_score"], errors="coerce").round(3)
    if "acceptance_rate" in frame.columns:
        frame["acceptance_rate"] = pd.to_numeric(frame["acceptance_rate"], errors="coerce")
    if "solve_probability_pct" in frame.columns:
        frame["solve_probability_pct"] = pd.to_numeric(frame["solve_probability_pct"], errors="coerce")
    if "difficulty" in frame.columns:
        frame["difficulty"] = frame["difficulty"].astype(str)

    columns = []
    if show_platform:
        columns.append("platform")
    if show_bucket:
        columns.append("bucket")
    columns.extend(
        [
            "problem_id",
            "title",
            "difficulty",
            "tags",
            "solve_probability_pct",
            "acceptance_rate",
            "rank_score",
            "url",
            "reason",
        ]
    )
    frame = frame[[column for column in columns if column in frame.columns]]
    st.dataframe(
        frame,
        **stretch_kwargs(st.dataframe),
        hide_index=True,
        column_config={
            "platform": st.column_config.TextColumn("Platform"),
            "bucket": st.column_config.TextColumn("Bucket"),
            "problem_id": st.column_config.TextColumn("Code"),
            "title": st.column_config.TextColumn("Problem"),
            "difficulty": st.column_config.TextColumn("Difficulty"),
            "tags": st.column_config.TextColumn("Tags"),
            "solve_probability_pct": st.column_config.NumberColumn("Solve %", format="%.1f%%"),
            "acceptance_rate": st.column_config.NumberColumn("Accept %", format="%.1f%%"),
            "rank_score": st.column_config.NumberColumn("Rank score", format="%.3f"),
            "url": st.column_config.LinkColumn("Open", display_text="Open"),
            "reason": st.column_config.TextColumn("Why"),
        },
    )


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


def _series_colors(values) -> list[str]:
    colors = palette()
    keys = {value: colors[index % len(colors)] for index, value in enumerate(dict.fromkeys(values))}
    return [keys[value] for value in values]


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

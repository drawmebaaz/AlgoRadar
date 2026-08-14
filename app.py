from __future__ import annotations

import inspect

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from algoradar.platforms import (
    analyze_external_platforms,
    build_combined_overview,
    lookup_leetcode_problem,
)
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
        codeforces_handle = st.text_input("Codeforces handle", key="codeforces_handle_input", help="Optional. Uses Codeforces submissions, ratings, tags, and verdicts.")
        codechef_handle = st.text_input("CodeChef handle", key="codechef_handle_input", help="Optional. Enables CodeChef rating and practice analysis.")
        leetcode_handle = st.text_input("LeetCode username", key="leetcode_handle_input", help="Optional. Enables LeetCode topic and contest analysis.")
        analyze = st.button("Analyze handles", **stretch_kwargs(st.button))
        force_refresh = st.toggle("Refresh platform caches", value=False)
        prefer_transformer = st.toggle(
            "Improve similar-problem matching",
            value=False,
            help="Optional. Uses a local MiniLM model for better similar-problem matching. Run scripts/verify_minilm.py once after installing optional dependencies.",
        )

        screen_options = [
            "Combined analysis",
            "Codeforces",
            "CodeChef",
            "LeetCode",
            "Recommendations",
            "Solve estimate",
        ]
        if st.session_state.get("screen") == "Solve probability":
            st.session_state.screen = "Solve estimate"
        if st.session_state.get("screen") not in screen_options:
            st.session_state.screen = "Combined analysis"
        screen = st.radio(
            "Sections",
            screen_options,
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
            <p class="eyebrow">AlgoRadar / competitive programming analytics</p>
            <h1>{hero_title}</h1>
            <p class="subcopy">Analyze Codeforces, CodeChef, and LeetCode handles in one place. Each platform stays separate, while recommendations and solve estimates use the handles you provide.</p>
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
    elif screen == "Solve estimate":
        render_general_probability(result, external_results, active_args)


def _needs_codeforces(screen: str, active_args: dict[str, str]) -> bool:
    if screen == "Combined analysis":
        return _provided_handle_count(active_args) >= 2
    return screen in {"Codeforces", "Recommendations", "Solve estimate"}


def _external_handles_for_screen(screen: str, active_args: dict[str, str]) -> dict[str, str]:
    combined_ready = screen == "Combined analysis" and _provided_handle_count(active_args) >= 2
    needs_leetcode = screen in {"LeetCode", "Recommendations", "Solve estimate"} or combined_ready
    needs_codechef = screen in {"CodeChef", "Recommendations", "Solve estimate"} or combined_ready
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
                <span>CP analytics suite</span>
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
    metric_card(cols[3], "Recent success", f"{profile['recent_accuracy']}%", "last 80 submissions")

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
                    line={"color": "#4fce8a", "width": 2},
                )
            )
            fig.add_trace(
                go.Bar(
                    x=result.contest_trend["contest"],
                    y=result.contest_trend["delta"],
                    name="Delta",
                    marker_color="#7aa7e8",
                    opacity=0.42,
                    yaxis="y2",
                )
            )
            fig.update_layout(yaxis2={"overlaying": "y", "side": "right", "showgrid": False}, **chart_layout())
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
                    marker={"colors": palette()},
                )
            )
            fig.update_layout(**chart_layout(height=360))
            st.plotly_chart(fig, **stretch_kwargs(st.plotly_chart))

    left, right = st.columns(2)
    with left:
        panel_title("Success by difficulty", "accepted rate by official or estimated problem rating")
        frame = result.rating_accuracy.copy()
        if not frame.empty:
            fig = go.Figure()
            fig.add_bar(x=frame["rating_bucket"], y=frame["attempts"], name="Attempts", marker_color="#242b35")
            fig.add_trace(
                go.Scatter(
                    x=frame["rating_bucket"],
                    y=frame["accuracy"],
                    mode="lines+markers",
                    name="Success %",
                    line={"color": "#d9a441", "width": 2},
                    yaxis="y2",
                )
            )
            fig.update_layout(yaxis2={"overlaying": "y", "side": "right", "range": [0, 100], "showgrid": False}, **chart_layout())
            st.plotly_chart(fig, **stretch_kwargs(st.plotly_chart))
        else:
            st.info("No rating data available.")

    with right:
        panel_title("Solved difficulty distribution", "where accepted problems cluster")
        frame = result.solved_difficulty
        if not frame.empty:
            fig = go.Figure(go.Bar(x=frame["rating_bucket"], y=frame["solved"], marker_color="#4fce8a"))
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
    metric_card(cols[2], "Focus areas", str(summary["focus_areas"]), "topics that need attention")
    metric_card(cols[3], "Needs attention", summary["attention_platform"], "highest improvement need")

    if platforms.empty:
        st.info("Add handles in the sidebar, then click Analyze handles.")
        return

    left, right = st.columns([1.1, 1])
    with left:
        panel_title("Platform breakdown", "what was loaded from each public profile")
        show = platforms.rename(
            columns={
                "platform": "Platform",
                "handle": "Handle",
                "status": "Status",
                "solved": "Solved",
                "current_rating": "Current rating",
                "max_rating": "Best rating",
                "contests": "Contests",
                "accuracy": "Success %",
                "signal": "Data available",
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
        panel_title("What to work on next", "topics and platform areas that need attention")
        if focus.empty:
            st.info("No focus areas available yet.")
        else:
            show = focus.rename(
                columns={
                    "platform": "Platform",
                    "area": "Area",
                    "level": "Level",
                    "priority": "Focus score",
                    "next_action": "Next action",
                }
            )
            st.dataframe(show.head(12), **stretch_kwargs(st.dataframe), hide_index=True)

    with right:
        panel_title("Contest rating trend", "each platform keeps its own rating scale")
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
    metric_card(cols[1], "Contest rating", str(profile.get("contest_rating", 0)), "LeetCode rating")
    metric_card(cols[2], "Acceptance", f"{profile.get('acceptance_rate', 0)}%", "accepted submissions / submissions")
    metric_card(cols[3], "Ranking", str(profile.get("ranking", 0)), "global profile rank")

    left, right = st.columns([1.1, 1])
    with left:
        panel_title("Success by difficulty", "accepted and attempted submissions by difficulty")
        difficulty = analysis.difficulty.copy()
        if difficulty.empty:
            st.info("No public difficulty stats available.")
        else:
            fig = go.Figure()
            fig.add_bar(x=difficulty["difficulty"], y=difficulty["solved"], name="Solved", marker_color="#4fce8a")
            fig.add_trace(
                go.Scatter(
                    x=difficulty["difficulty"],
                    y=difficulty["accuracy"],
                    mode="lines+markers",
                    name="Success %",
                    line={"color": "#d9a441", "width": 2},
                    yaxis="y2",
                )
            )
            fig.update_layout(yaxis2={"overlaying": "y", "side": "right", "range": [0, 100], "showgrid": False}, **chart_layout())
            st.plotly_chart(fig, **stretch_kwargs(st.plotly_chart))

    with right:
        panel_title("Contest trend", "LeetCode rating history")
        trend = analysis.contest_trend.copy()
        if trend.empty:
            st.info("No public LeetCode contest history found.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=trend["contest"], y=trend["rating"], mode="lines+markers", name="Rating", line={"color": "#7aa7e8"}))
            fig.add_bar(x=trend["contest"], y=trend["delta"], name="Delta", marker_color="#242b35", yaxis="y2")
            fig.update_layout(yaxis2={"overlaying": "y", "side": "right", "showgrid": False}, **chart_layout())
            st.plotly_chart(fig, **stretch_kwargs(st.plotly_chart))

    left, right = st.columns([1.2, 1])
    with left:
        panel_title("LeetCode focus areas", "topics with low public solved coverage")
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
                    "priority_score": "Focus score",
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
            fig = go.Figure(go.Bar(x=sorted_tags["solved"], y=sorted_tags["tag"], orientation="h", marker_color="#4fce8a"))
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
            fig.add_trace(go.Scatter(x=trend["contest"], y=trend["rating"], mode="lines+markers", name="Rating", line={"color": "#4fce8a"}))
            fig.add_bar(x=trend["contest"], y=trend["delta"], name="Delta", marker_color="#7aa7e8", opacity=0.42, yaxis="y2")
            fig.update_layout(yaxis2={"overlaying": "y", "side": "right", "showgrid": False}, **chart_layout(height=430))
            st.plotly_chart(fig, **stretch_kwargs(st.plotly_chart))

    with right:
        panel_title("Solved sections", "what the public profile exposes")
        difficulty = analysis.difficulty.copy()
        if difficulty.empty:
            st.info("No solved section data available.")
        else:
            fig = go.Figure(go.Bar(x=difficulty["difficulty"], y=difficulty["solved"], marker_color="#d9a441"))
            fig.update_layout(**chart_layout(height=430), showlegend=False)
            st.plotly_chart(fig, **stretch_kwargs(st.plotly_chart))

    panel_title("CodeChef focus areas", "rating history and practice volume")
    weakness = analysis.weakness.copy()
    if weakness.empty:
        st.info("No CodeChef focus areas available.")
    else:
        show = weakness[["tag", "level", "attempts", "priority_score", "next_action", "source"]].rename(
            columns={
                "tag": "Area",
                "level": "Level",
                "attempts": "Signal value",
                "priority_score": "Focus score",
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
        panel_title("Problem details", "works with whichever handles you provided")
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
            "Different platforms use different difficulty scales, so AlgoRadar first converts the problem into one shared estimate. "
            "The estimate mainly uses rating gap, solved depth on the selected tags, and your hardest similar solves."
        )

    with right:
        panel_title("Solve estimate", "practical chance for an honest attempt")
        probability = score["solve_probability_pct"]
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability,
                number={"suffix": "%", "font": {"color": "#eef2f6"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#8a95a5"},
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
    metric_card(cols[0], "Estimated level", str(int(score["anchor_cf_equivalent"])), "shared difficulty scale")
    metric_card(cols[1], "Problem level", str(int(score["target_cf_equivalent"])), str(score["native_target"]))
    metric_card(cols[2], "Tag practice", str(int(score["tag_solved"])), "solved on selected tags")
    metric_card(cols[3], "Hardest similar", str(int(score["tag_rating_ceiling"])), "selected tags")

    panel_title("Why this estimate", "the main signals behind the number")
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
        except (RuntimeError, ValueError, KeyError) as exc:
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
            st.caption("LeetCode has no official problem rating. AlgoRadar estimates difficulty from Easy/Medium/Hard, tags, acceptance, and the optional contest slot.")
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
        st.caption("LeetCode has no official problem rating. AlgoRadar estimates difficulty from Easy/Medium/Hard and the optional contest slot.")
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
    focus_count = int(weakness["level"].isin(["Weak", "Over-attempted but low accuracy"]).sum())
    metric_card(cols[0], "Focus tags", str(focus_count), "need structured practice")
    metric_card(cols[1], "Low coverage", str((weakness["level"] == "Untouched").sum()), "not tried enough yet")
    metric_card(cols[2], "High effort tags", str((weakness["level"] == "Over-attempted but low accuracy").sum()), "many tries, low return")
    metric_card(cols[3], "Recent misses", str(int(weakness["recent_failures"].sum())), "latest failed attempts")

    left, right = st.columns([1.55, 1])
    with left:
        panel_title("Tag focus table", "attempts, success rate, hardest solved, and recent misses")
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
        show["level"] = show["level"].apply(_friendly_level_label)
        show = show.rename(
            columns={
                "tag": "Tag",
                "level": "Status",
                "attempts": "Attempts",
                "accuracy": "Success %",
                "max_rating_solved": "Hardest solved",
                "recent_failures": "Recent misses",
                "priority_score": "Focus score",
                "next_action": "Next action",
            }
        )
        st.dataframe(show, **stretch_kwargs(st.dataframe), height=520)

    with right:
        panel_title("Top focus tags", "higher score means more attention needed")
        top = weakness.head(10).sort_values("priority_score", ascending=True)
        level_colors = {
            "Strong": "#4fce8a",
            "Stable": "#7aa7e8",
            "Needs focus": "#e67878",
            "Low coverage": "#8a95a5",
            "High effort, low return": "#d9a441",
        }
        top = top.copy()
        top["level_label"] = top["level"].apply(_friendly_level_label)
        fig = go.Figure(
            go.Bar(
                x=top["priority_score"],
                y=top["tag"],
                orientation="h",
                marker_color=top["level_label"].map(level_colors).fillna("#7aa7e8"),
                customdata=top["level_label"],
                hovertemplate="Tag: %{y}<br>Focus score: %{x}<br>Status: %{customdata}<extra></extra>",
            )
        )
        fig.update_layout(**chart_layout(height=520), showlegend=False)
        st.plotly_chart(fig, **stretch_kwargs(st.plotly_chart))


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
    metric_card(cols[3], "Similar matching", _friendly_matching_method(result.semantic_method), "problem finder")

    for bucket, label in [
        ("confidence", "5 confidence builders"),
        ("growth", "10 growth problems"),
        ("stretch", "5 stretch problems"),
    ]:
        panel_title(label, "ranked by solve chance, tag fit, popularity, and difficulty gap")
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
                "tag_cosine_similarity",
                "rating_fit_score",
                "solved_count",
                "rank_score",
            ]
        ].copy()
        frame["tags"] = frame["tags"].apply(_format_tags)
        frame["tag_similarity"] = frame["tag_similarity"].round(2)
        frame["tag_cosine_similarity"] = frame["tag_cosine_similarity"].round(2)
        frame["rating_fit_score"] = frame["rating_fit_score"].round(2)
        frame["rank_score"] = frame["rank_score"].round(3)
        st.dataframe(
            frame,
            **stretch_kwargs(st.dataframe),
            hide_index=True,
            column_config={
                "open": st.column_config.LinkColumn("Open", display_text="Codeforces"),
                "rating_source": st.column_config.TextColumn("Rating source"),
                "solve_probability_pct": st.column_config.NumberColumn("Solve %", format="%.1f%%"),
                "tag_similarity": st.column_config.NumberColumn("Focus tag fit", format="%.2f"),
                "tag_cosine_similarity": st.column_config.NumberColumn("Topic familiarity", format="%.2f"),
                "rating_fit_score": st.column_config.NumberColumn("Rating fit", format="%.2f"),
                "rank_score": st.column_config.NumberColumn("Fit score", format="%.3f"),
            },
        )

    panel_title("Similar harder problems", "matched from title, tags, and difficulty")
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
    metric_card(cols[2], "Success", f"{latest['accuracy']:.1f}%", "latest week")
    metric_card(
        cols[3],
        "Growth attempts",
        str(int(latest["growth_attempts"])),
        f"{result.profile['growth_rating_low']}-{result.profile['growth_rating_high']} rating",
    )

    panel_title("Progress tracking", "weekly solve volume and success rate")
    fig = go.Figure()
    fig.add_bar(x=progress["week"], y=progress["attempts"], name="Attempts", marker_color="#242b35")
    fig.add_trace(go.Scatter(x=progress["week"], y=progress["solved"], mode="lines+markers", name="Solved", line={"color": "#4fce8a"}))
    fig.add_trace(
        go.Scatter(
            x=progress["week"],
            y=progress["accuracy"],
            mode="lines+markers",
            name="Success %",
            line={"color": "#d9a441"},
            yaxis="y2",
        )
    )
    fig.update_layout(yaxis2={"overlaying": "y", "side": "right", "range": [0, 100], "showgrid": False}, **chart_layout(height=460))
    st.plotly_chart(fig, **stretch_kwargs(st.plotly_chart))


def _show_recommendations_table(recommendations: pd.DataFrame, show_platform: bool = True, show_bucket: bool = True) -> None:
    if recommendations is None or recommendations.empty:
        st.info("No recommendations available for this profile yet.")
        return
    frame = recommendations.copy()
    if "tags" in frame.columns:
        frame["tags"] = frame["tags"].apply(_format_tags)
    if "reason" in frame.columns:
        frame["reason"] = frame["reason"].apply(_friendly_recommendation_reason)
    if "rank_score" in frame.columns:
        frame["rank_score"] = pd.to_numeric(frame["rank_score"], errors="coerce").round(3)
    if "tag_cosine_similarity" in frame.columns:
        frame["tag_cosine_similarity"] = pd.to_numeric(frame["tag_cosine_similarity"], errors="coerce").round(2)
    if "rating_fit_score" in frame.columns:
        frame["rating_fit_score"] = pd.to_numeric(frame["rating_fit_score"], errors="coerce").round(2)
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
            "tag_cosine_similarity",
            "rating_fit_score",
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
            "tag_cosine_similarity": st.column_config.NumberColumn("Topic familiarity", format="%.2f"),
            "rating_fit_score": st.column_config.NumberColumn("Rating fit", format="%.2f"),
            "rank_score": st.column_config.NumberColumn("Fit score", format="%.3f"),
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
        "font": {"color": "#a8b3c1", "family": "Inter, sans-serif"},
        "margin": {"l": 22, "r": 18, "t": 24, "b": 32},
        "legend": {"orientation": "h", "y": 1.08, "x": 0},
        "xaxis": {"gridcolor": "#242b35", "zerolinecolor": "#242b35"},
        "yaxis": {"gridcolor": "#242b35", "zerolinecolor": "#242b35"},
    }


def palette() -> list[str]:
    return ["#4fce8a", "#e67878", "#d9a441", "#7aa7e8", "#8a95a5"]


def _series_colors(values) -> list[str]:
    colors = palette()
    keys = {value: colors[index % len(colors)] for index, value in enumerate(dict.fromkeys(values))}
    return [keys[value] for value in values]


def bucket_color(bucket: str) -> str:
    return {
        "confidence": "#4fce8a",
        "growth": "#d9a441",
        "stretch": "#e67878",
        "avoid": "#8a95a5",
    }.get(bucket, "#7aa7e8")


def _friendly_matching_method(method: str) -> str:
    value = str(method or "").lower()
    if "minilm" in value or "sentence" in value:
        return "Deep matching"
    if "tfidf" in value:
        return "Fast matching"
    if "deferred" in value:
        return "On demand"
    return "Available"


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


def _friendly_recommendation_reason(reason) -> str:
    text = str(reason or "")
    replacements = {
        "weak-tag-fit": "good practice for",
        "weak tag fit": "good practice for",
        "focus tag fit": "good practice for",
        "weak-tag": "focus tag",
        "rank score": "fit score",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _friendly_level_label(level) -> str:
    return {
        "Weak": "Needs focus",
        "Untouched": "Low coverage",
        "Over-attempted but low accuracy": "High effort, low return",
    }.get(str(level or ""), str(level or ""))


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
          --bg: #0b0d10;
          --sidebar: #0d1014;
          --surface: #11151b;
          --surface-2: #171c24;
          --surface-3: #1b212a;
          --border: #26303a;
          --text: #f3f6fa;
          --muted: #a8b3c1;
          --faint: #7e8a99;
          --green: #4fce8a;
          --amber: #d9a441;
          --red: #e67878;
          --blue: #7aa7e8;
        }
        .stApp {
          background: var(--bg);
          color: var(--text);
        }
        section[data-testid="stSidebar"] {
          background: var(--sidebar);
          border-right: 1px solid var(--border);
        }
        header[data-testid="stHeader"] {
          background: var(--bg);
        }
        .block-container {
          max-width: 1380px;
          padding-top: 3.2rem;
          padding-bottom: 3rem;
        }
        .brand {
          display: flex;
          align-items: center;
          gap: 12px;
          padding: 6px 0 20px;
        }
        .brand-mark {
          display: grid;
          place-items: center;
          width: 40px;
          height: 40px;
          border: 1px solid #303a46;
          border-radius: 8px;
          background: #121720;
          color: var(--green);
          font-family: ui-monospace, Consolas, monospace;
          font-weight: 800;
          font-size: 15px;
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
          margin-bottom: 20px;
          padding: 18px 20px;
          border: 1px solid var(--border);
          border-radius: 8px;
          background: var(--surface);
          box-shadow: none;
        }
        .hero > div:first-child {
          min-width: 0;
        }
        .hero h1 {
          margin: 6px 0 6px;
          color: var(--text);
          font-size: 30px;
          line-height: 1.12;
          font-weight: 760;
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
          margin: 0;
          max-width: 800px;
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
          border: 1px solid rgba(79,206,138,.34);
          border-radius: 999px;
          background: rgba(79,206,138,.08);
          color: var(--green);
          font-family: ui-monospace, Consolas, monospace;
          font-size: 12px;
          letter-spacing: 0;
          white-space: normal;
          overflow-wrap: anywhere;
          text-align: center;
          flex: 0 1 auto;
          max-width: min(420px, 100%);
        }
        .metric-card,
        .day-card,
        .rule-list {
          min-height: 106px;
          margin-bottom: 14px;
          padding: 16px;
          border: 1px solid var(--border);
          border-radius: 8px;
          background: var(--surface);
          box-shadow: none;
          overflow: hidden;
        }
        .metric-card p {
          margin: 0;
          color: var(--muted);
          font-size: 12px;
          font-weight: 650;
          overflow-wrap: anywhere;
        }
        .metric-card strong {
          display: block;
          margin-top: 10px;
          color: var(--text);
          font-family: ui-monospace, Consolas, monospace;
          font-size: 25px;
          line-height: 1.05;
          letter-spacing: 0;
          overflow-wrap: anywhere;
        }
        .metric-card span {
          display: block;
          margin-top: 9px;
          color: var(--faint);
          font-size: 12px;
          overflow-wrap: anywhere;
        }
        .panel-title {
          margin: 22px 0 10px;
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
          background: var(--surface);
          box-shadow: none;
        }
        .stDataFrame {
          overflow-x: auto;
        }
        .day-card {
          min-height: 188px;
        }
        .day-card code {
          display: inline-flex;
          padding: 2px 7px;
          border: 1px solid #2b313c;
          border-radius: 5px;
          background: #0e1217;
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
          background: #222a34;
        }
        .meter span {
          display: block;
          height: 100%;
          border-radius: inherit;
          background: var(--amber);
        }
        button[kind="primary"],
        .stButton > button {
          min-height: 42px;
          border: 1px solid #344152;
          border-radius: 7px;
          background: #14201a;
          color: var(--green);
          font-weight: 750;
          line-height: 1.2;
          white-space: normal;
        }
        .stButton > button:hover {
          border-color: rgba(79,206,138,.52);
          background: #17271f;
          color: var(--green);
        }
        .stTextInput input,
        .stNumberInput input,
        .stSelectbox [data-baseweb="select"] > div,
        .stMultiSelect [data-baseweb="select"] > div {
          border-color: #303946;
          background-color: #10141a;
          color: var(--text);
        }
        .stTextInput input,
        .stNumberInput input {
          min-height: 42px;
        }
        .stMultiSelect [data-baseweb="tag"] {
          max-width: 100%;
          margin-top: 2px;
          margin-bottom: 2px;
        }
        .stMultiSelect [data-baseweb="tag"] span {
          display: inline-block;
          max-width: min(220px, 56vw);
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }
        div[data-testid="stAlert"] {
          border-radius: 8px;
          border: 1px solid #273343;
          background: #0f1720;
          overflow-wrap: anywhere;
        }
        .stTabs [data-baseweb="tab-list"] {
          gap: 18px;
          border-bottom: 1px solid var(--border);
          overflow-x: auto;
          overflow-y: hidden;
        }
        .stTabs [data-baseweb="tab"] {
          height: 42px;
          padding: 0 2px;
          color: var(--muted);
          font-weight: 650;
          flex: 0 0 auto;
          white-space: nowrap;
        }
        .stTabs [aria-selected="true"] {
          color: var(--green);
        }
        @media (max-width: 900px) {
          .block-container {
            padding-top: 3.6rem;
          }
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

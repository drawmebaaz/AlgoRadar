from __future__ import annotations

import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import requests

from .config import CACHE_DIR

REQUEST_TIMEOUT_SECONDS = 8
CACHE_MAX_AGE_SECONDS = 6 * 3600
CATALOG_CACHE_SECONDS = 24 * 3600
LEETCODE_PROBLEM_LIMIT = 700
LEETCODE_BATCH_SIZE = 100
CODECHEF_PROBLEM_LIMIT = 80

LEETCODE_GRAPHQL_URL = "https://leetcode.com/graphql"
CODECHEF_PROBLEM_URL = "https://www.codechef.com/api/list/problems/all"

USER_AGENT = "Mozilla/5.0 AlgoRadar/1.0"

CORE_LEETCODE_TAGS = {
    "Array": 1.0,
    "Hash Table": 0.95,
    "Two Pointers": 0.85,
    "String": 0.8,
    "Binary Search": 0.9,
    "Sorting": 0.7,
    "Stack": 0.75,
    "Queue": 0.55,
    "Linked List": 0.65,
    "Tree": 0.95,
    "Graph": 1.0,
    "Dynamic Programming": 1.0,
    "Greedy": 0.85,
    "Backtracking": 0.8,
    "Heap (Priority Queue)": 0.75,
    "Trie": 0.65,
    "Union Find": 0.7,
    "Bit Manipulation": 0.7,
    "Math": 0.75,
    "Prefix Sum": 0.8,
    "Sliding Window": 0.8,
}


@dataclass
class PlatformAnalysis:
    platform: str
    handle: str
    source: str
    status: str
    profile: dict[str, Any] = field(default_factory=dict)
    difficulty: pd.DataFrame = field(default_factory=pd.DataFrame)
    tags: pd.DataFrame = field(default_factory=pd.DataFrame)
    activity: pd.DataFrame = field(default_factory=pd.DataFrame)
    contest_trend: pd.DataFrame = field(default_factory=pd.DataFrame)
    weakness: pd.DataFrame = field(default_factory=pd.DataFrame)
    recommendations: pd.DataFrame = field(default_factory=pd.DataFrame)
    capabilities: dict[str, Any] = field(default_factory=dict)
    error: str = ""


def analyze_external_platforms(
    leetcode_handle: str = "",
    codechef_handle: str = "",
    force_refresh: bool = False,
    include_recommendations: bool = True,
) -> dict[str, PlatformAnalysis]:
    tasks: dict[str, tuple[Callable[..., PlatformAnalysis], str]] = {}
    if leetcode_handle.strip():
        tasks["leetcode"] = (analyze_leetcode, leetcode_handle.strip())
    if codechef_handle.strip():
        tasks["codechef"] = (analyze_codechef, codechef_handle.strip())

    if not tasks:
        return {}

    results: dict[str, PlatformAnalysis] = {}
    with ThreadPoolExecutor(max_workers=max(1, len(tasks))) as executor:
        futures = {
            key: executor.submit(fn, handle, force_refresh, include_recommendations)
            for key, (fn, handle) in tasks.items()
        }
        for key, future in futures.items():
            try:
                results[key] = future.result()
            except Exception as exc:
                platform = "LeetCode" if key == "leetcode" else "CodeChef"
                handle = tasks[key][1]
                results[key] = _empty_analysis(platform, handle, "error", str(exc))
    return results


def analyze_leetcode(username: str, force_refresh: bool = False, include_recommendations: bool = True) -> PlatformAnalysis:
    try:
        payload = _cached_json(
            CACHE_DIR / f"leetcode_profile_{_safe_key(username)}.json",
            lambda: _fetch_leetcode(username),
            force_refresh=force_refresh,
        )
        matched = payload.get("matchedUser")
        if not matched:
            return _empty_analysis("LeetCode", username, "not_found", "LeetCode user was not found.")

        profile = _leetcode_profile(username, payload)
        difficulty = _leetcode_difficulty(payload)
        tags = _leetcode_tags(payload)
        activity = _leetcode_activity(payload)
        contest_trend = _leetcode_contest_trend(payload)
        weakness = _leetcode_weakness(tags, difficulty, profile)
        recommendations = (
            _safe_leetcode_recommendations(profile, weakness, activity, force_refresh)
            if include_recommendations
            else pd.DataFrame()
        )

        return PlatformAnalysis(
            platform="LeetCode",
            handle=username,
            source="leetcode_graphql",
            status="ok",
            profile=profile,
            difficulty=difficulty,
            tags=tags,
            activity=activity,
            contest_trend=contest_trend,
            weakness=weakness,
            recommendations=recommendations,
            capabilities={
                "profile": "public GraphQL",
                "difficulty": "public accepted/attempt counters",
                "tags": "public solved tag counters",
                "recommendations": "public problem catalog; solved filter limited to recent accepted submissions",
            },
        )
    except Exception as exc:
        return _empty_analysis("LeetCode", username, "error", str(exc))


def analyze_codechef(handle: str, force_refresh: bool = False, include_recommendations: bool = True) -> PlatformAnalysis:
    try:
        html = _cached_text(
            CACHE_DIR / f"codechef_profile_{_safe_key(handle)}.html",
            lambda: _fetch_codechef_html(handle),
            force_refresh=force_refresh,
        )
        if _looks_like_missing_codechef_profile(html):
            return _empty_analysis("CodeChef", handle, "not_found", "CodeChef user was not found.")

        profile = _codechef_profile(handle, html)
        contest_trend = _codechef_rating_activity(html)
        difficulty = _codechef_difficulty(profile)
        activity = _codechef_solved_sections(html)
        weakness = _codechef_weakness(profile, contest_trend)
        recommendations = _safe_codechef_recommendations(profile, force_refresh) if include_recommendations else pd.DataFrame()

        return PlatformAnalysis(
            platform="CodeChef",
            handle=handle,
            source="codechef_profile_and_practice_api",
            status="ok",
            profile=profile,
            difficulty=difficulty,
            tags=pd.DataFrame(columns=["tag", "solved", "level"]),
            activity=activity,
            contest_trend=contest_trend,
            weakness=weakness,
            recommendations=recommendations,
            capabilities={
                "profile": "public profile page",
                "contest_trend": "public rating history embedded in profile page",
                "recommendations": "public practice problem API by rating band",
                "tags": "not exposed in public solved-profile data",
            },
        )
    except Exception as exc:
        return _empty_analysis("CodeChef", handle, "error", str(exc))


def build_combined_overview(
    codeforces_result: Any | None,
    external_results: dict[str, PlatformAnalysis],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    focus_rows: list[dict[str, Any]] = []
    trend_rows: list[dict[str, Any]] = []

    if codeforces_result is not None:
        profile = codeforces_result.profile
        rows.append(
            {
                "platform": "Codeforces",
                "handle": codeforces_result.handle,
                "status": "ok",
                "solved": int(profile.get("problems_solved", 0) or 0),
                "current_rating": int(profile.get("current_rating", 0) or 0),
                "max_rating": int(profile.get("max_rating", 0) or 0),
                "contests": int(profile.get("contest_count", 0) or 0),
                "accuracy": float(profile.get("recent_accuracy", 0) or 0),
                "signal": "full verdict history",
            }
        )
        if not codeforces_result.weakness.empty:
            for item in codeforces_result.weakness.head(5).to_dict("records"):
                focus_rows.append(
                    {
                        "platform": "Codeforces",
                        "area": item.get("tag", ""),
                        "level": item.get("level", ""),
                        "priority": float(item.get("priority_score", 0) or 0),
                        "next_action": item.get("next_action", ""),
                    }
                )
        if not codeforces_result.contest_trend.empty:
            for index, item in enumerate(codeforces_result.contest_trend.to_dict("records")):
                trend_rows.append(
                    {
                        "platform": "Codeforces",
                        "contest": item.get("contest", ""),
                        "rating": int(item.get("rating", 0) or 0),
                        "delta": int(item.get("delta", 0) or 0),
                        "order": index,
                    }
                )

    for analysis in external_results.values():
        if analysis.status != "ok":
            rows.append(
                {
                    "platform": analysis.platform,
                    "handle": analysis.handle,
                    "status": analysis.status,
                    "solved": 0,
                    "current_rating": 0,
                    "max_rating": 0,
                    "contests": 0,
                    "accuracy": 0.0,
                    "signal": analysis.error,
                }
            )
            continue

        rows.append(_platform_summary_row(analysis))
        if not analysis.weakness.empty:
            for item in analysis.weakness.head(5).to_dict("records"):
                focus_rows.append(
                    {
                        "platform": analysis.platform,
                        "area": item.get("tag", item.get("area", "")),
                        "level": item.get("level", ""),
                        "priority": float(item.get("priority_score", 0) or 0),
                        "next_action": item.get("next_action", ""),
                    }
                )
        if not analysis.contest_trend.empty:
            for index, item in enumerate(analysis.contest_trend.to_dict("records")):
                trend_rows.append(
                    {
                        "platform": analysis.platform,
                        "contest": item.get("contest", ""),
                        "rating": int(float(item.get("rating", 0) or 0)),
                        "delta": int(float(item.get("delta", 0) or 0)),
                        "order": index,
                    }
                )

    platform_rows = pd.DataFrame(rows)
    focus = pd.DataFrame(focus_rows)
    trend = pd.DataFrame(trend_rows)
    recommendations = combined_recommendations(codeforces_result, external_results)

    ok_rows = platform_rows[platform_rows["status"] == "ok"] if not platform_rows.empty else platform_rows
    total_solved = int(ok_rows["solved"].sum()) if not ok_rows.empty else 0
    platforms_connected = int(len(ok_rows)) if not ok_rows.empty else 0
    contest_entries = int(ok_rows["contests"].sum()) if not ok_rows.empty else 0
    attention_platform = _attention_platform(platform_rows, focus)

    summary = {
        "total_solved": total_solved,
        "platforms_connected": platforms_connected,
        "contest_entries": contest_entries,
        "focus_areas": int(len(focus)),
        "attention_platform": attention_platform,
    }
    return {
        "summary": summary,
        "platforms": platform_rows,
        "focus": focus.sort_values("priority", ascending=False).reset_index(drop=True) if not focus.empty else focus,
        "trend": trend,
        "recommendations": recommendations,
    }


def combined_recommendations(
    codeforces_result: Any | None,
    external_results: dict[str, PlatformAnalysis],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if codeforces_result is not None and not codeforces_result.recommendations.empty:
        cf = codeforces_result.recommendations.copy()
        cf["platform"] = "Codeforces"
        cf["title"] = cf["name"]
        cf["difficulty"] = cf["rating"].astype(int).astype(str)
        cf["url"] = cf.apply(_codeforces_url_from_recommendation, axis=1)
        cf["reason"] = cf["tags"].apply(lambda tags: _tag_reason(tags, "Codeforces weak-tag fit"))
        cf["acceptance_rate"] = ""
        frames.append(
            cf[
                [
                    "platform",
                    "bucket",
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
            ].head(20)
        )

    for analysis in external_results.values():
        if analysis.status == "ok" and not analysis.recommendations.empty:
            frame = analysis.recommendations.copy()
            for column in ["solve_probability_pct", "acceptance_rate", "rank_score"]:
                if column not in frame.columns:
                    frame[column] = None
            frames.append(
                frame[
                    [
                        "platform",
                        "bucket",
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
                ]
            )

    if not frames:
        return pd.DataFrame(
            columns=[
                "platform",
                "bucket",
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
    return pd.concat(frames, ignore_index=True)


def _fetch_leetcode(username: str) -> dict[str, Any]:
    query = """
    query AlgoRadarLeetCode($username: String!) {
      matchedUser(username: $username) {
        username
        profile {
          realName
          ranking
          reputation
          countryName
        }
        submitStatsGlobal {
          acSubmissionNum {
            difficulty
            count
            submissions
          }
          totalSubmissionNum {
            difficulty
            count
            submissions
          }
        }
        tagProblemCounts {
          advanced {
            tagName
            problemsSolved
          }
          intermediate {
            tagName
            problemsSolved
          }
          fundamental {
            tagName
            problemsSolved
          }
        }
      }
      userContestRanking(username: $username) {
        attendedContestsCount
        rating
        globalRanking
        totalParticipants
        topPercentage
      }
      userContestRankingHistory(username: $username) {
        attended
        trendDirection
        problemsSolved
        totalProblems
        finishTimeInSeconds
        rating
        ranking
        contest {
          title
          startTime
        }
      }
      recentAcSubmissionList(username: $username, limit: 20) {
        title
        titleSlug
        timestamp
      }
      recentSubmissionList(username: $username, limit: 20) {
        title
        titleSlug
        timestamp
        statusDisplay
        lang
      }
    }
    """
    response = requests.post(
        LEETCODE_GRAPHQL_URL,
        json={"query": query, "variables": {"username": username}},
        headers={
            "User-Agent": USER_AGENT,
            "Referer": f"https://leetcode.com/{username}/",
            "Content-Type": "application/json",
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("errors"):
        raise RuntimeError(payload["errors"][0].get("message", "LeetCode GraphQL error"))
    return payload.get("data", {})


def lookup_leetcode_problem(slug_or_url: str, force_refresh: bool = False) -> dict[str, Any]:
    slug = _leetcode_slug_from_input(slug_or_url)
    if not slug:
        return {"status": "missing", "error": "Enter a LeetCode problem slug or URL."}
    payload = _cached_json(
        CACHE_DIR / f"leetcode_problem_{_safe_key(slug)}.json",
        lambda: _fetch_leetcode_problem(slug),
        force_refresh=force_refresh,
        max_age_seconds=CATALOG_CACHE_SECONDS,
    )
    return _leetcode_problem_from_payload(payload)


def _fetch_leetcode_problem(slug: str) -> dict[str, Any]:
    query = """
    query AlgoRadarLeetCodeProblem($titleSlug: String!) {
      question(titleSlug: $titleSlug) {
        questionFrontendId
        title
        titleSlug
        difficulty
        isPaidOnly
        stats
        topicTags {
          name
          slug
        }
      }
    }
    """
    response = requests.post(
        LEETCODE_GRAPHQL_URL,
        json={"query": query, "variables": {"titleSlug": slug}},
        headers={
            "User-Agent": USER_AGENT,
            "Referer": f"https://leetcode.com/problems/{slug}/",
            "Content-Type": "application/json",
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("errors"):
        raise RuntimeError(payload["errors"][0].get("message", "LeetCode problem lookup error"))
    data = payload.get("data", {})
    data["fetched_at"] = int(time.time())
    return data


def _leetcode_problem_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    question = payload.get("question") or {}
    if not question:
        return {"status": "not_found", "error": "LeetCode problem was not found."}
    stats = _parse_leetcode_problem_stats(question.get("stats"))
    tags = [tag.get("name", "") for tag in question.get("topicTags", []) if tag.get("name")]
    accepted = int(float(stats.get("totalAcceptedRaw", 0) or 0))
    submissions = int(float(stats.get("totalSubmissionRaw", 0) or 0))
    acceptance_rate = _leetcode_acceptance_rate(stats, accepted, submissions)
    slug = question.get("titleSlug", "")
    return {
        "status": "ok",
        "problem_id": str(question.get("questionFrontendId") or slug),
        "title": question.get("title", ""),
        "slug": slug,
        "difficulty": question.get("difficulty", ""),
        "paid_only": bool(question.get("isPaidOnly", False)),
        "tags": tags,
        "accepted": accepted,
        "submissions": submissions,
        "acceptance_rate": acceptance_rate,
        "url": f"https://leetcode.com/problems/{slug}/" if slug else "https://leetcode.com/problemset/",
    }


def _parse_leetcode_problem_stats(raw_stats: Any) -> dict[str, Any]:
    if isinstance(raw_stats, dict):
        return raw_stats
    try:
        return json.loads(raw_stats or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}


def _leetcode_acceptance_rate(stats: dict[str, Any], accepted: int, submissions: int) -> float:
    raw = stats.get("acRate")
    if raw not in (None, ""):
        try:
            return round(float(str(raw).replace("%", "")), 1)
        except ValueError:
            pass
    return round(accepted / submissions * 100, 1) if submissions else 0.0


def _leetcode_slug_from_input(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    match = re.search(r"/problems/([^/?#]+)/?", text)
    slug = match.group(1) if match else text
    slug = slug.split("?")[0].split("#")[0].strip("/")
    return re.sub(r"[^a-zA-Z0-9-]+", "", slug.lower())


def _fetch_leetcode_problemset(force_refresh: bool = False) -> pd.DataFrame:
    payload = _cached_json(
        CACHE_DIR / f"leetcode_problemset_top_{LEETCODE_PROBLEM_LIMIT}.json",
        _fetch_leetcode_problem_pages,
        force_refresh=force_refresh,
        max_age_seconds=CATALOG_CACHE_SECONDS,
    )
    rows = []
    for item in payload.get("questions", []):
        tags = [tag.get("name", "") for tag in item.get("topicTags", []) if tag.get("name")]
        rows.append(
            {
                "problem_id": item.get("frontendQuestionId") or item.get("questionFrontendId") or item.get("titleSlug", ""),
                "title": item.get("title", ""),
                "difficulty": item.get("difficulty", ""),
                "acceptance_rate": float(item.get("acRate", 0) or 0),
                "tags": tags,
                "paid_only": bool(item.get("paidOnly", False)),
                "url": f"https://leetcode.com/problems/{item.get('titleSlug', '')}/",
                "slug": item.get("titleSlug", ""),
            }
        )
    return pd.DataFrame(rows)


def _fetch_leetcode_problem_pages() -> dict[str, Any]:
    query = """
    query problemsetQuestionList(
      $categorySlug: String,
      $limit: Int,
      $skip: Int,
      $filters: QuestionListFilterInput
    ) {
      problemsetQuestionList: questionList(
        categorySlug: $categorySlug,
        limit: $limit,
        skip: $skip,
        filters: $filters
      ) {
        total: totalNum
        questions: data {
          acRate
          difficulty
          freqBar
          frontendQuestionId: questionFrontendId
          paidOnly: isPaidOnly
          status
          title
          titleSlug
          topicTags {
            name
            slug
          }
        }
      }
    }
    """
    questions: list[dict[str, Any]] = []
    for skip in range(0, LEETCODE_PROBLEM_LIMIT, LEETCODE_BATCH_SIZE):
        response = requests.post(
            LEETCODE_GRAPHQL_URL,
            json={
                "query": query,
                "variables": {
                    "categorySlug": "all-code-essentials",
                    "skip": skip,
                    "limit": LEETCODE_BATCH_SIZE,
                    "filters": {},
                },
            },
            headers={
                "User-Agent": USER_AGENT,
                "Referer": "https://leetcode.com/problemset/",
                "Content-Type": "application/json",
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("errors"):
            raise RuntimeError(payload["errors"][0].get("message", "LeetCode problem catalog error"))
        page = payload.get("data", {}).get("problemsetQuestionList", {})
        questions.extend(page.get("questions", []) or [])
        if len(questions) >= int(page.get("total", LEETCODE_PROBLEM_LIMIT) or LEETCODE_PROBLEM_LIMIT):
            break
    return {"questions": questions[:LEETCODE_PROBLEM_LIMIT], "fetched_at": int(time.time())}


def _fetch_codechef_html(handle: str) -> str:
    response = requests.get(
        f"https://www.codechef.com/users/{handle}",
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    if response.status_code == 404:
        return "CODECHEF_PROFILE_404"
    response.raise_for_status()
    return response.text


def _fetch_codechef_problems(low: int, high: int, force_refresh: bool = False) -> pd.DataFrame:
    low = max(0, int(low))
    high = max(low + 1, min(5001, int(high)))
    payload = _cached_json(
        CACHE_DIR / f"codechef_problems_{low}_{high}.json",
        lambda: _fetch_codechef_problem_page(low, high),
        force_refresh=force_refresh,
        max_age_seconds=CATALOG_CACHE_SECONDS,
    )
    rows = []
    for item in payload.get("data", []) or []:
        total = int(item.get("total_submissions") or 0)
        successful = int(item.get("successful_submissions") or 0)
        distinct_successful = int(item.get("distinct_successful_submissions") or 0)
        code = item.get("code", "")
        rows.append(
            {
                "platform": "CodeChef",
                "problem_id": code,
                "title": item.get("name", ""),
                "difficulty": int(float(item.get("difficulty_rating") or 0)),
                "tags": [],
                "acceptance_rate": round(successful / total * 100, 1) if total else 0.0,
                "solved_count": distinct_successful,
                "contest_code": item.get("contest_code", ""),
                "url": f"https://www.codechef.com/problems/{code}",
            }
        )
    return pd.DataFrame(rows)


def _fetch_codechef_problem_page(low: int, high: int) -> dict[str, Any]:
    response = requests.get(
        CODECHEF_PROBLEM_URL,
        params={
            "sort_by": "difficulty_rating",
            "sorting_order": "asc",
            "start_rating": low,
            "end_rating": high,
            "limit": CODECHEF_PROBLEM_LIMIT,
            "offset": 0,
        },
        headers={
            "User-Agent": USER_AGENT,
            "Referer": "https://www.codechef.com/practice-old/tags",
            "Accept": "application/json,text/plain,*/*",
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") not in {None, "success"}:
        raise RuntimeError(payload.get("message", "CodeChef practice API error"))
    return payload


def _leetcode_profile(username: str, payload: dict[str, Any]) -> dict[str, Any]:
    matched = payload.get("matchedUser") or {}
    profile = matched.get("profile") or {}
    stats = matched.get("submitStatsGlobal") or {}
    accepted = {item["difficulty"]: item for item in stats.get("acSubmissionNum", [])}
    totals = {item["difficulty"]: item for item in stats.get("totalSubmissionNum", [])}
    contest = payload.get("userContestRanking") or {}

    total_ac = int(accepted.get("All", {}).get("count", 0) or 0)
    accepted_submissions = int(accepted.get("All", {}).get("submissions", 0) or 0)
    total_submissions = int(totals.get("All", {}).get("submissions", 0) or 0)
    acceptance = round(accepted_submissions / total_submissions * 100, 1) if total_submissions else 0.0
    contest_rating = contest.get("rating")
    hard = int(accepted.get("Hard", {}).get("count", 0) or 0)
    medium = int(accepted.get("Medium", {}).get("count", 0) or 0)
    easy = int(accepted.get("Easy", {}).get("count", 0) or 0)

    return {
        "username": username,
        "display_name": profile.get("realName") or username,
        "ranking": int(profile.get("ranking") or 0),
        "reputation": int(profile.get("reputation") or 0),
        "country": profile.get("countryName") or "",
        "total_solved": total_ac,
        "easy_solved": easy,
        "medium_solved": medium,
        "hard_solved": hard,
        "accepted_submissions": accepted_submissions,
        "total_submissions": total_submissions,
        "acceptance_rate": acceptance,
        "contest_rating": round(float(contest_rating), 1) if contest_rating else 0,
        "current_rating": round(float(contest_rating), 1) if contest_rating else 0,
        "max_rating": round(float(contest_rating), 1) if contest_rating else 0,
        "estimated_cp_anchor": _leetcode_cp_anchor(easy, medium, hard, contest_rating),
        "contests": int(contest.get("attendedContestsCount", 0) or 0),
        "top_percentage": round(float(contest.get("topPercentage", 0) or 0), 2),
        "global_ranking": int(contest.get("globalRanking", 0) or 0),
    }


def _leetcode_difficulty(payload: dict[str, Any]) -> pd.DataFrame:
    matched = payload.get("matchedUser") or {}
    stats = matched.get("submitStatsGlobal") or {}
    accepted = {item["difficulty"]: item for item in stats.get("acSubmissionNum", [])}
    totals = {item["difficulty"]: item for item in stats.get("totalSubmissionNum", [])}
    rows = []
    for difficulty in ["Easy", "Medium", "Hard"]:
        accepted_submissions = int(accepted.get(difficulty, {}).get("submissions", 0) or 0)
        total_submissions = int(totals.get(difficulty, {}).get("submissions", 0) or 0)
        rows.append(
            {
                "difficulty": difficulty,
                "solved": int(accepted.get(difficulty, {}).get("count", 0) or 0),
                "accepted_submissions": accepted_submissions,
                "submissions": total_submissions,
                "accuracy": round(accepted_submissions / total_submissions * 100, 1) if total_submissions else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _leetcode_tags(payload: dict[str, Any]) -> pd.DataFrame:
    matched = payload.get("matchedUser") or {}
    counts = matched.get("tagProblemCounts") or {}
    rows = []
    for group in ["advanced", "intermediate", "fundamental"]:
        for item in counts.get(group, []) or []:
            rows.append(
                {
                    "tag": item.get("tagName", ""),
                    "solved": int(item.get("problemsSolved", 0) or 0),
                    "level": group,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["tag", "solved", "level"])
    return frame.sort_values("solved", ascending=False).reset_index(drop=True)


def _leetcode_activity(payload: dict[str, Any]) -> pd.DataFrame:
    rows = []
    seen: set[str] = set()
    for item in payload.get("recentAcSubmissionList", []) or []:
        slug = item.get("titleSlug", "")
        seen.add(slug)
        rows.append(
            {
                "title": item.get("title", ""),
                "slug": slug,
                "timestamp": int(item.get("timestamp", 0) or 0),
                "status": "Accepted",
                "language": "",
                "url": f"https://leetcode.com/problems/{slug}/",
            }
        )
    for item in payload.get("recentSubmissionList", []) or []:
        slug = item.get("titleSlug", "")
        if slug in seen:
            continue
        rows.append(
            {
                "title": item.get("title", ""),
                "slug": slug,
                "timestamp": int(item.get("timestamp", 0) or 0),
                "status": item.get("statusDisplay", ""),
                "language": item.get("lang", ""),
                "url": f"https://leetcode.com/problems/{slug}/",
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["title", "slug", "timestamp", "status", "language", "url"])
    return frame.sort_values("timestamp", ascending=False).reset_index(drop=True)


def _leetcode_contest_trend(payload: dict[str, Any]) -> pd.DataFrame:
    rows = []
    last_rating: float | None = None
    for item in payload.get("userContestRankingHistory", []) or []:
        if not item.get("attended"):
            continue
        rating = float(item.get("rating", 0) or 0)
        delta = 0 if last_rating is None else int(round(rating - last_rating))
        contest = item.get("contest") or {}
        total = int(item.get("totalProblems", 0) or 0)
        solved = int(item.get("problemsSolved", 0) or 0)
        rows.append(
            {
                "contest": contest.get("title", ""),
                "rank": int(item.get("ranking", 0) or 0),
                "rating": round(rating, 1),
                "delta": delta,
                "solved": solved,
                "total": total,
                "score": f"{solved}/{total}" if total else "",
                "rated_at": int(contest.get("startTime", 0) or 0),
            }
        )
        last_rating = rating
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["contest", "rank", "rating", "delta", "solved", "total", "score", "rated_at"])
    return frame.tail(20).reset_index(drop=True)


def _leetcode_weakness(tags: pd.DataFrame, difficulty: pd.DataFrame, profile: dict[str, Any]) -> pd.DataFrame:
    solved_lookup = tags.set_index("tag")["solved"].to_dict() if not tags.empty else {}
    total_solved = int(profile.get("total_solved", 0) or 0)
    rows = []
    for tag, weight in CORE_LEETCODE_TAGS.items():
        solved = int(solved_lookup.get(tag, 0) or 0)
        level = _leetcode_tag_level(solved, total_solved)
        priority = round((1 - min(solved / max(18, total_solved * 0.08), 1)) * 100 * weight, 1)
        rows.append(
            {
                "platform": "LeetCode",
                "tag": tag,
                "level": level,
                "attempts": solved,
                "accuracy": None,
                "solved": solved,
                "priority_score": priority,
                "next_action": _leetcode_next_action(tag, level),
                "source": "solved tag coverage",
            }
        )

    hard_row = difficulty[difficulty["difficulty"] == "Hard"]
    hard_solved = int(hard_row.iloc[0]["solved"]) if not hard_row.empty else 0
    if total_solved >= 120 and hard_solved < max(8, total_solved * 0.08):
        rows.append(
            {
                "platform": "LeetCode",
                "tag": "Hard problem exposure",
                "level": "Weak",
                "attempts": hard_solved,
                "accuracy": None,
                "solved": hard_solved,
                "priority_score": 82.0,
                "next_action": "Add one reviewed Hard problem every 2-3 sessions.",
                "source": "difficulty distribution",
            }
        )

    frame = pd.DataFrame(rows)
    return frame.sort_values("priority_score", ascending=False).reset_index(drop=True)


def _safe_leetcode_recommendations(
    profile: dict[str, Any],
    weakness: pd.DataFrame,
    activity: pd.DataFrame,
    force_refresh: bool,
) -> pd.DataFrame:
    try:
        return _leetcode_recommendations(profile, weakness, activity, force_refresh)
    except Exception:
        return pd.DataFrame(
            columns=[
                "platform",
                "bucket",
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


def _leetcode_recommendations(
    profile: dict[str, Any],
    weakness: pd.DataFrame,
    activity: pd.DataFrame,
    force_refresh: bool = False,
) -> pd.DataFrame:
    catalog = _fetch_leetcode_problemset(force_refresh=force_refresh)
    if catalog.empty:
        return pd.DataFrame()

    recent_slugs = set(activity["slug"].dropna().astype(str).tolist()) if not activity.empty else set()
    weak_tags = weakness.head(8)["tag"].tolist() if not weakness.empty else []
    weak_tags = [tag for tag in weak_tags if tag in CORE_LEETCODE_TAGS]
    total_solved = int(profile.get("total_solved", 0) or 0)

    frame = catalog[~catalog["paid_only"]].copy()
    if recent_slugs:
        frame = frame[~frame["slug"].isin(recent_slugs)]
    if frame.empty:
        return pd.DataFrame()

    frame["tag_fit"] = frame["tags"].apply(lambda tags: _tag_overlap_score(tags, weak_tags))
    frame["acceptance_component"] = frame["acceptance_rate"].fillna(0) / 100
    frame["difficulty_rank"] = frame["difficulty"].map({"Easy": 1, "Medium": 2, "Hard": 3}).fillna(2)
    target_rank = _leetcode_target_rank(total_solved)
    frame["level_fit"] = 1 - (frame["difficulty_rank"] - target_rank).abs() / 2
    frame["rank_score"] = (
        frame["tag_fit"] * 0.52
        + frame["level_fit"].clip(lower=0) * 0.26
        + frame["acceptance_component"] * 0.22
    )

    buckets = [
        ("confidence", ["Easy"] if total_solved < 120 else ["Medium"], 5, 68),
        ("growth", ["Medium"] if total_solved < 350 else ["Medium", "Hard"], 10, 52),
        ("stretch", ["Hard"] if total_solved >= 80 else ["Medium"], 5, 35),
    ]
    rows: list[pd.DataFrame] = []
    used: set[str] = set()
    for bucket, difficulties, count, probability in buckets:
        subset = frame[frame["difficulty"].isin(difficulties)].copy()
        subset = subset[~subset["slug"].isin(used)]
        if subset.empty:
            subset = frame[~frame["slug"].isin(used)].copy()
        subset = subset.sort_values(["rank_score", "acceptance_rate"], ascending=[False, False]).head(count)
        used.update(subset["slug"].tolist())
        subset["platform"] = "LeetCode"
        subset["bucket"] = bucket
        subset["solve_probability_pct"] = probability
        subset["reason"] = subset["tags"].apply(lambda tags: _tag_reason(tags, "LeetCode weak-tag coverage"))
        rows.append(subset)

    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if result.empty:
        return result
    result["rank_score"] = result["rank_score"].round(3)
    return result[
        [
            "platform",
            "bucket",
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
    ]


def _codechef_profile(handle: str, html: str) -> dict[str, Any]:
    rating = _first_int(html, [r'class="rating-number">\s*(\d+)\s*<', r"class='rating-number'>\s*(\d+)\s*<"])
    highest = _first_int(html, [r"Highest Rating\s*(\d+)", r"Highest Rating</small>\s*<h5[^>]*>\s*(\d+)"])
    stars = _codechef_star_count(html)
    division = _first_text(html, [r"<div>\s*\((Div\s+\d+)\)\s*</div>"])
    global_rank = _first_int(html, [r"Global Rank</strong>\s*<a[^>]*>\s*(\d+)", r"Global Rank[^0-9]*(\d+)"])
    country_rank = _first_int(html, [r"Country Rank</strong>\s*<a[^>]*>\s*(\d+)", r"Country Rank[^0-9]*(\d+)"])
    total_solved = _first_int(html, [r"Total Problems Solved:\s*(\d+)", r"Problems Solved[^0-9]*(\d+)"])
    section_counts = _codechef_section_counts(html)
    contest_count = len(_extract_codechef_rating_history(html)) or section_counts.get("Contests", 0)

    return {
        "username": handle,
        "current_rating": rating,
        "max_rating": highest or rating,
        "stars": stars,
        "division": division,
        "global_rank": global_rank,
        "country_rank": country_rank,
        "total_solved": total_solved,
        "fully_solved": total_solved,
        "partially_solved": 0,
        "contest_count": contest_count,
        "learning_paths": section_counts.get("Learning Paths", 0),
        "practice_paths": section_counts.get("Practice Paths", 0),
        "contest_problem_sets": section_counts.get("Contests", 0),
    }


def _codechef_rating_activity(html: str) -> pd.DataFrame:
    rows = _extract_codechef_rating_history(html)
    if not rows:
        return pd.DataFrame(columns=["contest", "rating", "rank", "delta", "end_date"])
    previous: int | None = None
    for row in rows:
        rating = int(row.get("rating", 0) or 0)
        row["delta"] = 0 if previous is None else rating - previous
        previous = rating
    return pd.DataFrame(rows).tail(20).reset_index(drop=True)


def _codechef_difficulty(profile: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"difficulty": "Profile solved", "solved": int(profile.get("total_solved", 0) or 0)},
            {"difficulty": "Contest sets", "solved": int(profile.get("contest_problem_sets", 0) or 0)},
            {"difficulty": "Practice paths", "solved": int(profile.get("practice_paths", 0) or 0)},
        ]
    )


def _codechef_solved_sections(html: str) -> pd.DataFrame:
    section_counts = _codechef_section_counts(html)
    rows = [{"section": key, "count": value} for key, value in section_counts.items()]
    if not rows:
        return pd.DataFrame(columns=["section", "count"])
    return pd.DataFrame(rows)


def _codechef_weakness(profile: dict[str, Any], contest_trend: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rating = int(profile.get("current_rating", 0) or 0)
    max_rating = int(profile.get("max_rating", 0) or 0)
    total_solved = int(profile.get("total_solved", 0) or 0)

    recent_delta = int(contest_trend["delta"].tail(5).sum()) if not contest_trend.empty else 0
    volatility = float(contest_trend["delta"].tail(8).std() or 0) if not contest_trend.empty else 0.0
    rating_gap = max(0, max_rating - rating)

    rows.append(
        {
            "platform": "CodeChef",
            "tag": "Contest consistency",
            "level": "Weak" if recent_delta < -80 or volatility > 120 else "Stable",
            "attempts": int(profile.get("contest_count", 0) or 0),
            "accuracy": None,
            "priority_score": round(min(100, abs(min(recent_delta, 0)) * 0.7 + volatility * 0.35), 1),
            "next_action": "Give one rated Starters/Long contest, then upsolve the first unsolved problem immediately.",
            "source": "rating history",
        }
    )
    rows.append(
        {
            "platform": "CodeChef",
            "tag": "Rating recovery",
            "level": "Weak" if rating_gap >= 150 else "Stable",
            "attempts": rating_gap,
            "accuracy": None,
            "priority_score": round(min(100, rating_gap * 0.45), 1),
            "next_action": "Practice confidence problems 100-250 below current rating before the next contest.",
            "source": "current vs max rating",
        }
    )
    rows.append(
        {
            "platform": "CodeChef",
            "tag": "Practice volume",
            "level": "Weak" if total_solved < 120 and rating >= 1400 else "Stable",
            "attempts": total_solved,
            "accuracy": None,
            "priority_score": round(max(0, 120 - total_solved) * 0.55, 1),
            "next_action": "Build a 20-problem rated practice block around your current CodeChef rating.",
            "source": "public solved count",
        }
    )
    return pd.DataFrame(rows).sort_values("priority_score", ascending=False).reset_index(drop=True)


def _safe_codechef_recommendations(profile: dict[str, Any], force_refresh: bool) -> pd.DataFrame:
    try:
        return _codechef_recommendations(profile, force_refresh=force_refresh)
    except Exception:
        return pd.DataFrame(
            columns=[
                "platform",
                "bucket",
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


def _codechef_recommendations(profile: dict[str, Any], force_refresh: bool = False) -> pd.DataFrame:
    rating = int(profile.get("current_rating", 0) or profile.get("max_rating", 0) or 1200)
    bands = [
        ("confidence", max(0, rating - 350), max(1, rating - 50), 5, 72),
        ("growth", max(0, rating - 50), min(5001, rating + 250), 10, 58),
        ("stretch", min(5000, rating + 250), min(5001, rating + 550), 5, 38),
    ]
    rows: list[pd.DataFrame] = []
    used: set[str] = set()
    for bucket, low, high, count, probability in bands:
        frame = _fetch_codechef_problems(low, high, force_refresh=force_refresh)
        if frame.empty:
            continue
        frame = frame[~frame["problem_id"].isin(used)].copy()
        if frame.empty:
            continue
        frame["bucket"] = bucket
        frame["solve_probability_pct"] = probability
        frame["rating_gap"] = (frame["difficulty"] - rating).abs()
        frame["rank_score"] = (
            (1 / (1 + frame["rating_gap"])) * 80
            + frame["acceptance_rate"].fillna(0) / 100 * 0.18
            + frame["solved_count"].fillna(0).apply(lambda value: math.log1p(value)) * 0.02
        )
        frame["reason"] = frame["difficulty"].apply(lambda value: f"CodeChef rating band around {int(value)}")
        chosen = frame.sort_values(["rank_score", "acceptance_rate"], ascending=[False, False]).head(count)
        used.update(chosen["problem_id"].tolist())
        rows.append(chosen)

    if not rows:
        return pd.DataFrame()
    result = pd.concat(rows, ignore_index=True)
    result["rank_score"] = result["rank_score"].round(3)
    return result[
        [
            "platform",
            "bucket",
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
    ]


def _extract_codechef_rating_history(html: str) -> list[dict[str, Any]]:
    match = re.search(r"all_rating\s*=\s*(\[.*?\]);", html, flags=re.S)
    if not match:
        return []
    try:
        raw = json.loads(match.group(1))
    except json.JSONDecodeError:
        return []
    rows = []
    for item in raw:
        rows.append(
            {
                "contest": item.get("name") or item.get("code") or "",
                "rating": int(float(item.get("rating", 0) or 0)),
                "rank": int(float(item.get("rank", 0) or 0)),
                "end_date": item.get("end_date") or "",
            }
        )
    return rows


def _empty_analysis(platform: str, handle: str, status: str, error: str) -> PlatformAnalysis:
    return PlatformAnalysis(
        platform=platform,
        handle=handle,
        source="unavailable",
        status=status,
        profile={"username": handle},
        difficulty=pd.DataFrame(),
        tags=pd.DataFrame(),
        activity=pd.DataFrame(),
        contest_trend=pd.DataFrame(),
        weakness=pd.DataFrame(),
        recommendations=pd.DataFrame(),
        capabilities={},
        error=error,
    )


def _platform_summary_row(analysis: PlatformAnalysis) -> dict[str, Any]:
    profile = analysis.profile
    if analysis.platform == "LeetCode":
        solved = int(profile.get("total_solved", 0) or 0)
        current = int(float(profile.get("contest_rating", 0) or 0))
        max_rating = max(current, int(profile.get("estimated_cp_anchor", 0) or 0))
        contests = int(profile.get("contests", 0) or 0)
        accuracy = float(profile.get("acceptance_rate", 0) or 0)
        signal = "difficulty, tags, contests"
    else:
        solved = int(profile.get("total_solved", 0) or 0)
        current = int(profile.get("current_rating", 0) or 0)
        max_rating = int(profile.get("max_rating", 0) or 0)
        contests = int(profile.get("contest_count", 0) or 0)
        accuracy = 0.0
        signal = "rating history and practice API"
    return {
        "platform": analysis.platform,
        "handle": analysis.handle,
        "status": analysis.status,
        "solved": solved,
        "current_rating": current,
        "max_rating": max_rating,
        "contests": contests,
        "accuracy": accuracy,
        "signal": signal,
    }


def _attention_platform(platform_rows: pd.DataFrame, focus: pd.DataFrame) -> str:
    if not focus.empty:
        return str(focus.sort_values("priority", ascending=False).iloc[0]["platform"])
    if not platform_rows.empty:
        ok_rows = platform_rows[platform_rows["status"] == "ok"]
        if not ok_rows.empty:
            return str(ok_rows.sort_values("solved").iloc[0]["platform"])
    return "Add handles"


def _cached_json(
    path: Path,
    loader: Callable[[], dict[str, Any]],
    force_refresh: bool = False,
    max_age_seconds: int = CACHE_MAX_AGE_SECONDS,
) -> dict[str, Any]:
    if not force_refresh and _is_fresh(path, max_age_seconds):
        return json.loads(path.read_text(encoding="utf-8"))
    data = loader()
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


def _cached_text(
    path: Path,
    loader: Callable[[], str],
    force_refresh: bool = False,
    max_age_seconds: int = CACHE_MAX_AGE_SECONDS,
) -> str:
    if not force_refresh and _is_fresh(path, max_age_seconds):
        return path.read_text(encoding="utf-8")
    data = loader()
    path.write_text(data, encoding="utf-8")
    return data


def _is_fresh(path: Path, max_age_seconds: int = CACHE_MAX_AGE_SECONDS) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < max_age_seconds


def _safe_key(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip().lower() or "anonymous")


def _first_int(text: str, patterns: list[str]) -> int:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I | re.S)
        if match:
            return int(str(match.group(1)).replace(",", ""))
    return 0


def _first_text(text: str, patterns: list[str]) -> str:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I | re.S)
        if match:
            return re.sub(r"\s+", " ", match.group(1)).strip()
    return ""


def _looks_like_missing_codechef_profile(html: str) -> bool:
    lowered = html[:4000].lower()
    return "codechef_profile_404" in lowered or "user not found" in lowered or "page not found" in lowered


def _codechef_star_count(html: str) -> int:
    match = re.search(r'class="rating-star">(.*?)</div>', html, flags=re.I | re.S)
    if not match:
        return 0
    block = match.group(1)
    return max(block.count("&#9733;"), block.count("*"), len(re.findall(r"<span", block, flags=re.I)))


def _codechef_section_counts(html: str) -> dict[str, int]:
    section_match = re.search(r'<section class="rating-data-section problems-solved">(.*?)</section>', html, flags=re.I | re.S)
    section = section_match.group(1) if section_match else html
    counts: dict[str, int] = {}
    for label in ["Learning Paths", "Practice Paths", "Contests"]:
        match = re.search(rf"<h3>\s*{re.escape(label)}\s*\((\d+)\)", section, flags=re.I | re.S)
        if match:
            counts[label] = int(match.group(1))
    return counts


def _leetcode_tag_level(solved: int, total_solved: int) -> str:
    if solved <= 0:
        return "Untouched"
    expected = max(8, total_solved * 0.055)
    if solved >= expected * 2.2:
        return "Strong"
    if solved >= expected:
        return "Stable"
    return "Weak"


def _leetcode_next_action(tag: str, level: str) -> str:
    if level == "Untouched":
        return f"Start with 3 Easy/Medium {tag} problems and write the pattern note."
    if level == "Weak":
        return f"Do a 5-problem {tag} block: 3 Medium, 1 review, 1 timed retry."
    return f"Maintain {tag} with one mixed-difficulty problem this week."


def _leetcode_cp_anchor(easy: int, medium: int, hard: int, contest_rating: Any) -> int:
    if contest_rating:
        return int(float(contest_rating))
    score = easy * 2.5 + medium * 7 + hard * 18
    return int(max(900, min(2400, 850 + math.sqrt(score) * 65)))


def _leetcode_target_rank(total_solved: int) -> float:
    if total_solved < 80:
        return 1.4
    if total_solved < 350:
        return 2.0
    return 2.45


def _tag_overlap_score(tags: list[str], weak_tags: list[str]) -> float:
    if not weak_tags:
        return 0.4
    if not tags:
        return 0.0
    overlap = len(set(tags) & set(weak_tags))
    return min(1.0, overlap / min(3, len(weak_tags)))


def _tag_reason(tags: Any, prefix: str) -> str:
    if isinstance(tags, list) and tags:
        return f"{prefix}: {', '.join(str(tag) for tag in tags[:3])}"
    return prefix


def _codeforces_url_from_recommendation(row: pd.Series) -> str:
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

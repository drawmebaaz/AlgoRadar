from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from algoradar import platforms


def test_external_platforms_skip_blank_handles() -> None:
    assert platforms.analyze_external_platforms("", "") == {}


def test_leetcode_payload_becomes_profile_and_weakness() -> None:
    payload = {
        "matchedUser": {
            "username": "unit",
            "profile": {"ranking": 1234, "reputation": 9, "realName": "Unit User", "countryName": "IN"},
            "submitStatsGlobal": {
                "acSubmissionNum": [
                    {"difficulty": "All", "count": 120, "submissions": 210},
                    {"difficulty": "Easy", "count": 40, "submissions": 55},
                    {"difficulty": "Medium", "count": 70, "submissions": 125},
                    {"difficulty": "Hard", "count": 10, "submissions": 30},
                ],
                "totalSubmissionNum": [
                    {"difficulty": "All", "count": 145, "submissions": 300},
                    {"difficulty": "Easy", "count": 45, "submissions": 70},
                    {"difficulty": "Medium", "count": 82, "submissions": 180},
                    {"difficulty": "Hard", "count": 18, "submissions": 50},
                ],
            },
            "tagProblemCounts": {
                "advanced": [{"tagName": "Dynamic Programming", "problemsSolved": 18}],
                "intermediate": [{"tagName": "Graph", "problemsSolved": 3}],
                "fundamental": [{"tagName": "Array", "problemsSolved": 25}],
            },
        },
        "userContestRanking": {
            "attendedContestsCount": 4,
            "rating": 1650.4,
            "globalRanking": 9000,
            "totalParticipants": 100000,
            "topPercentage": 9.0,
        },
        "userContestRankingHistory": [
            {
                "attended": True,
                "problemsSolved": 3,
                "totalProblems": 4,
                "rating": 1600.2,
                "ranking": 400,
                "contest": {"title": "Weekly 1", "startTime": 1700000000},
            },
            {
                "attended": True,
                "problemsSolved": 2,
                "totalProblems": 4,
                "rating": 1650.4,
                "ranking": 300,
                "contest": {"title": "Weekly 2", "startTime": 1701000000},
            },
        ],
        "recentAcSubmissionList": [{"title": "Two Sum", "titleSlug": "two-sum", "timestamp": 1701}],
        "recentSubmissionList": [],
    }

    profile = platforms._leetcode_profile("unit", payload)
    difficulty = platforms._leetcode_difficulty(payload)
    tags = platforms._leetcode_tags(payload)
    weakness = platforms._leetcode_weakness(tags, difficulty, profile)
    trend = platforms._leetcode_contest_trend(payload)

    assert profile["total_solved"] == 120
    assert profile["contest_rating"] == 1650.4
    assert difficulty.loc[difficulty["difficulty"] == "Medium", "accuracy"].iloc[0] == 69.4
    assert "Graph" in weakness["tag"].tolist()
    assert trend.iloc[-1]["delta"] == 50


def test_leetcode_problem_payload_extracts_tags_and_stats() -> None:
    payload = {
        "question": {
            "questionFrontendId": "1",
            "title": "Two Sum",
            "titleSlug": "two-sum",
            "difficulty": "Easy",
            "isPaidOnly": False,
            "stats": '{"totalAcceptedRaw": 12000000, "totalSubmissionRaw": 25000000, "acRate": "48.0%"}',
            "topicTags": [{"name": "Array", "slug": "array"}, {"name": "Hash Table", "slug": "hash-table"}],
        }
    }

    problem = platforms._leetcode_problem_from_payload(payload)

    assert problem["problem_id"] == "1"
    assert problem["difficulty"] == "Easy"
    assert problem["tags"] == ["Array", "Hash Table"]
    assert problem["accepted"] == 12000000
    assert problem["acceptance_rate"] == 48.0


def test_codechef_profile_and_history_parser() -> None:
    html = """
    <div class="rating-number">1512</div>
    <div>(Div 2)</div>
    <div class="rating-star"><span>&#9733;</span><span>&#9733;</span></div>
    <small>(Highest Rating 1603)</small>
    <section class="rating-data-section problems-solved">
      <h3>Learning Paths (1)</h3>
      <h3>Practice Paths (2)</h3>
      <h3>Contests (7)</h3>
      <h3>Total Problems Solved: 42</h3>
    </section>
    <script>
      all_rating = [
        {"rating":"1450","rank":"100","name":"Starter A","end_date":"2024-01-01"},
        {"rating":"1512","rank":"80","name":"Starter B","end_date":"2024-02-01"}
      ];
    </script>
    """

    profile = platforms._codechef_profile("unit", html)
    trend = platforms._codechef_rating_activity(html)
    weakness = platforms._codechef_weakness(profile, trend)

    assert profile["current_rating"] == 1512
    assert profile["max_rating"] == 1603
    assert profile["total_solved"] == 42
    assert trend.iloc[-1]["delta"] == 62
    assert not weakness.empty


def test_codechef_recommendations_use_rating_bands(monkeypatch) -> None:
    def fake_fetch(low: int, high: int, force_refresh: bool = False) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "platform": "CodeChef",
                    "problem_id": f"P{low}",
                    "title": "Band Problem",
                    "difficulty": low + 20,
                    "tags": [],
                    "acceptance_rate": 55.0,
                    "solved_count": 1000,
                    "url": f"https://www.codechef.com/problems/P{low}",
                }
            ]
        )

    monkeypatch.setattr(platforms, "_fetch_codechef_problems", fake_fetch)

    recs = platforms._codechef_recommendations({"current_rating": 1500})

    assert set(recs["bucket"]) == {"confidence", "growth", "stretch"}
    assert recs["url"].str.contains("codechef.com/problems").all()


def test_combined_overview_accepts_codeforces_and_external() -> None:
    cf = SimpleNamespace(
        handle="cf",
        profile={"problems_solved": 10, "current_rating": 1200, "max_rating": 1300, "contest_count": 2, "recent_accuracy": 50},
        weakness=pd.DataFrame([{"tag": "dp", "level": "Weak", "priority_score": 90, "next_action": "Practice dp"}]),
        contest_trend=pd.DataFrame([{"contest": "Round", "rating": 1200, "delta": 20}]),
        recommendations=pd.DataFrame(),
    )
    lc = platforms.PlatformAnalysis(
        platform="LeetCode",
        handle="lc",
        source="test",
        status="ok",
        profile={"total_solved": 20, "contest_rating": 1500, "estimated_cp_anchor": 1500, "contests": 3, "acceptance_rate": 60},
        weakness=pd.DataFrame([{"tag": "Graph", "level": "Weak", "priority_score": 80, "next_action": "Practice graph"}]),
        recommendations=pd.DataFrame(),
    )

    overview = platforms.build_combined_overview(cf, {"leetcode": lc})

    assert overview["summary"]["total_solved"] == 30
    assert overview["summary"]["platforms_connected"] == 2
    assert not overview["focus"].empty

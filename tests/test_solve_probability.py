from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from algoradar.platforms import PlatformAnalysis
from algoradar.solve_probability import native_to_cf_equivalent, score_saved_profile_problem


def test_cross_platform_probability_decreases_with_rating() -> None:
    cf = SimpleNamespace(
        profile={"problems_solved": 300, "current_rating": 1700, "max_rating": 1800},
        tag_stats=pd.DataFrame(
            [
                {"tag": "dp", "solved": 45, "avg_rating_solved": 1550, "max_rating_solved": 1900},
                {"tag": "graphs", "solved": 20, "avg_rating_solved": 1450, "max_rating_solved": 1750},
            ]
        ),
    )

    low = score_saved_profile_problem("Codeforces", 1500, ["dp"], 5000, cf, {})
    high = score_saved_profile_problem("Codeforces", 2300, ["dp"], 5000, cf, {})

    assert low["solve_probability"] > high["solve_probability"]


def test_cross_platform_probability_uses_leetcode_tag_volume() -> None:
    leetcode = PlatformAnalysis(
        platform="LeetCode",
        handle="unit",
        source="test",
        status="ok",
        profile={
            "total_solved": 400,
            "contest_rating": 1900,
            "estimated_cp_anchor": 1900,
            "medium_solved": 230,
            "hard_solved": 45,
        },
        tags=pd.DataFrame(
            [
                {"tag": "Dynamic Programming", "solved": 60},
                {"tag": "Graph", "solved": 5},
            ]
        ),
    )

    strong = score_saved_profile_problem("LeetCode", 1800, ["Dynamic Programming"], 4000, None, {"leetcode": leetcode})
    weak = score_saved_profile_problem("LeetCode", 1800, ["Graph"], 4000, None, {"leetcode": leetcode})

    assert strong["tag_solved"] > weak["tag_solved"]
    assert strong["solve_probability"] > weak["solve_probability"]


def test_same_rating_codeforces_problem_is_not_overconfident() -> None:
    cf = SimpleNamespace(
        profile={"problems_solved": 220, "current_rating": 1200, "max_rating": 1300},
        tag_stats=pd.DataFrame(
            [
                {"tag": "dp", "solved": 24, "avg_rating_solved": 1150, "max_rating_solved": 1400},
            ]
        ),
    )

    score = score_saved_profile_problem("Codeforces", 1200, ["dp"], 5000, cf, {})

    assert score["target_cf_equivalent"] == 1200
    assert 35 <= score["solve_probability_pct"] <= 60


def test_codeforces_probability_curve_is_practical_for_pupil_profile() -> None:
    cf = SimpleNamespace(
        profile={"problems_solved": 220, "current_rating": 1200, "max_rating": 1300},
        tag_stats=pd.DataFrame(
            [
                {"tag": "dp", "solved": 24, "avg_rating_solved": 1150, "max_rating_solved": 1400},
            ]
        ),
    )

    scores = [
        score_saved_profile_problem("Codeforces", rating, ["dp"], 5000, cf, {})["solve_probability_pct"]
        for rating in [1000, 1200, 1400, 1600]
    ]

    assert scores == sorted(scores, reverse=True)
    assert scores[1] <= 60
    assert scores[2] < 35


def test_tag_ceiling_and_volume_raise_probability() -> None:
    strong_cf = SimpleNamespace(
        profile={"problems_solved": 260, "current_rating": 1500, "max_rating": 1600},
        tag_stats=pd.DataFrame(
            [
                {"tag": "dp", "solved": 42, "avg_rating_solved": 1450, "max_rating_solved": 1850},
            ]
        ),
    )
    weak_cf = SimpleNamespace(
        profile={"problems_solved": 260, "current_rating": 1500, "max_rating": 1600},
        tag_stats=pd.DataFrame(
            [
                {"tag": "dp", "solved": 2, "avg_rating_solved": 900, "max_rating_solved": 1000},
            ]
        ),
    )

    strong = score_saved_profile_problem("Codeforces", 1600, ["dp"], 6000, strong_cf, {})
    weak = score_saved_profile_problem("Codeforces", 1600, ["dp"], 6000, weak_cf, {})

    assert strong["solve_probability"] > weak["solve_probability"]
    assert weak["solve_probability_pct"] < 70


def test_platform_difficulty_calibration_uses_mapping_csv() -> None:
    codechef = native_to_cf_equivalent("CodeChef", native_rating=1450)
    leetcode_medium = native_to_cf_equivalent("LeetCode", leetcode_difficulty="Medium")
    leetcode_q4 = native_to_cf_equivalent("LeetCode", leetcode_difficulty="Hard", leetcode_contest_slot="Q4")

    assert codechef["cf_equivalent"] == 1200
    assert 1100 <= leetcode_medium["cf_equivalent"] <= 1500
    assert leetcode_q4["cf_equivalent"] >= 1900

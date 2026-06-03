from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from algoradar.platforms import PlatformAnalysis
from algoradar.solve_probability import score_saved_profile_problem


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

from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd

from .models import bucket_probability

DIFFICULTY_TO_RATING = {
    "Easy": 1000,
    "Medium": 1600,
    "Hard": 2200,
}

TAG_ALIASES = {
    "dynamic programming": {"dp", "dynamic programming"},
    "dp": {"dp", "dynamic programming"},
    "graph": {"graphs", "graph"},
    "graphs": {"graphs", "graph"},
    "hash table": {"hashing", "hash table"},
    "hashing": {"hashing", "hash table"},
    "binary search": {"binary search"},
    "two pointers": {"two pointers"},
    "math": {"math", "number theory"},
    "number theory": {"math", "number theory"},
    "greedy": {"greedy"},
    "tree": {"trees", "tree"},
    "trees": {"trees", "tree"},
    "bit manipulation": {"bitmasks", "bit manipulation"},
    "bitmasks": {"bitmasks", "bit manipulation"},
    "heap": {"heap (priority queue)", "heaps", "heap"},
    "heap (priority queue)": {"heap (priority queue)", "heaps", "heap"},
    "string": {"strings", "string"},
    "strings": {"strings", "string"},
    "array": {"arrays", "array"},
    "arrays": {"arrays", "array"},
    "sliding window": {"sliding window", "two pointers"},
    "union find": {"dsu", "union find"},
    "dsu": {"dsu", "union find"},
}


def score_saved_profile_problem(
    platform: str,
    target_rating: int,
    tags: list[str],
    popularity: int,
    codeforces_result: Any | None,
    external_results: dict[str, Any],
) -> dict[str, Any]:
    target_rating = int(max(800, min(3500, target_rating or 1200)))
    profile_strength = _combined_strength(codeforces_result, external_results)
    tag_strength = _tag_strength(tags, codeforces_result, external_results)

    anchor_rating = profile_strength["anchor_rating"]
    total_solved = profile_strength["total_solved"]
    tag_solved = tag_strength["tag_solved"]
    tag_rating_ceiling = tag_strength["tag_rating_ceiling"]
    tag_avg_rating = tag_strength["tag_avg_rating"]

    rating_gap = target_rating - anchor_rating
    volume_score = _bounded_log(tag_solved, 80)
    overall_volume_score = _bounded_log(total_solved, 1800)
    ceiling_score = _sigmoid((tag_rating_ceiling - target_rating + 140) / 260)
    avg_rating_score = _sigmoid((tag_avg_rating - target_rating + 220) / 320)
    popularity_score = _bounded_log(popularity, 60000)
    platform_fit = _platform_fit(platform, profile_strength)
    tag_penalty = min(len(tags), 5) * 0.055

    logit = (
        0.22
        - rating_gap / 390
        + volume_score * 1.18
        + ceiling_score * 1.0
        + avg_rating_score * 0.62
        + overall_volume_score * 0.38
        + popularity_score * 0.16
        + platform_fit * 0.22
        - tag_penalty
    )
    probability = float(max(0.02, min(0.98, 1 / (1 + math.exp(-logit)))))

    factors = pd.DataFrame(
        [
            {"factor": "Solved on selected tags", "value": round(tag_solved, 1), "impact": round(volume_score, 3)},
            {"factor": "Hardest solved on selected tags", "value": round(tag_rating_ceiling, 0), "impact": round(ceiling_score, 3)},
            {"factor": "Average solved rating on selected tags", "value": round(tag_avg_rating, 0), "impact": round(avg_rating_score, 3)},
            {"factor": "Total solved across saved platforms", "value": round(total_solved, 1), "impact": round(overall_volume_score, 3)},
            {"factor": "Target rating gap", "value": round(rating_gap, 0), "impact": round(-rating_gap / 390, 3)},
            {"factor": "Problem popularity", "value": int(popularity or 0), "impact": round(popularity_score, 3)},
        ]
    )

    return {
        "platform": platform,
        "target_rating": target_rating,
        "tags": tags,
        "popularity": int(popularity or 0),
        "solve_probability": probability,
        "solve_probability_pct": round(probability * 100, 1),
        "bucket": bucket_probability(probability),
        "anchor_rating": round(anchor_rating, 0),
        "total_solved": round(total_solved, 1),
        "tag_solved": round(tag_solved, 1),
        "tag_rating_ceiling": round(tag_rating_ceiling, 0),
        "tag_avg_rating": round(tag_avg_rating, 0),
        "factors": factors,
    }


def target_rating_from_difficulty(platform: str, difficulty: str, fallback_rating: int = 1600) -> int:
    if platform == "LeetCode":
        return DIFFICULTY_TO_RATING.get(difficulty, fallback_rating)
    return int(fallback_rating)


def available_probability_tags(codeforces_result: Any | None, external_results: dict[str, Any]) -> list[str]:
    tags: set[str] = set()
    if codeforces_result is not None and not codeforces_result.tag_stats.empty:
        tags.update(str(tag) for tag in codeforces_result.tag_stats["tag"].dropna().tolist())
    leetcode = external_results.get("leetcode")
    if leetcode is not None and getattr(leetcode, "status", "") == "ok" and not leetcode.tags.empty:
        tags.update(str(tag) for tag in leetcode.tags["tag"].dropna().tolist())
    tags.update(["dp", "graphs", "greedy", "math", "binary search", "Dynamic Programming", "Graph", "Array"])
    return sorted(tags, key=lambda value: value.lower())


def _combined_strength(codeforces_result: Any | None, external_results: dict[str, Any]) -> dict[str, float]:
    ratings: list[tuple[float, float]] = []
    total_solved = 0.0
    platform_solved: dict[str, float] = {}

    if codeforces_result is not None:
        profile = codeforces_result.profile
        solved = float(profile.get("problems_solved", 0) or 0)
        rating = float(profile.get("current_rating", 0) or profile.get("max_rating", 0) or 1200)
        ratings.append((rating, max(1.0, min(4.0, math.log1p(solved) / 2.2))))
        total_solved += solved
        platform_solved["Codeforces"] = solved

    leetcode = external_results.get("leetcode")
    if leetcode is not None and getattr(leetcode, "status", "") == "ok":
        profile = leetcode.profile
        solved = float(profile.get("total_solved", 0) or 0)
        rating = float(profile.get("contest_rating", 0) or profile.get("estimated_cp_anchor", 0) or 1200)
        ratings.append((rating, max(0.8, min(3.2, math.log1p(solved) / 2.4))))
        total_solved += solved
        platform_solved["LeetCode"] = solved

    codechef = external_results.get("codechef")
    if codechef is not None and getattr(codechef, "status", "") == "ok":
        profile = codechef.profile
        solved = float(profile.get("total_solved", 0) or 0)
        rating = float(profile.get("current_rating", 0) or profile.get("max_rating", 0) or 1200)
        ratings.append((rating, max(0.8, min(3.2, math.log1p(solved) / 2.4))))
        total_solved += solved
        platform_solved["CodeChef"] = solved

    if not ratings:
        return {"anchor_rating": 1200.0, "total_solved": 0.0, "platform_solved": {}}

    weighted_sum = sum(rating * weight for rating, weight in ratings)
    total_weight = sum(weight for _, weight in ratings) or 1.0
    return {
        "anchor_rating": weighted_sum / total_weight,
        "total_solved": total_solved,
        "platform_solved": platform_solved,
    }


def _tag_strength(tags: list[str], codeforces_result: Any | None, external_results: dict[str, Any]) -> dict[str, float]:
    normalized = _expanded_tags(tags)
    solved_values: list[float] = []
    avg_rating_values: list[float] = []
    max_rating_values: list[float] = []

    if codeforces_result is not None and not codeforces_result.tag_stats.empty:
        frame = codeforces_result.tag_stats.copy()
        frame["tag_key"] = frame["tag"].apply(_tag_key)
        cf = frame[frame["tag_key"].isin(normalized)]
        if not cf.empty:
            solved_values.append(float(cf["solved"].sum()))
            avg_rating_values.extend(float(value) for value in cf["avg_rating_solved"].tolist() if value)
            max_rating_values.extend(float(value) for value in cf["max_rating_solved"].tolist() if value)

    leetcode = external_results.get("leetcode")
    if leetcode is not None and getattr(leetcode, "status", "") == "ok":
        tag_frame = leetcode.tags.copy()
        if not tag_frame.empty:
            tag_frame["tag_key"] = tag_frame["tag"].apply(_tag_key)
            lc = tag_frame[tag_frame["tag_key"].isin(normalized)]
            if not lc.empty:
                lc_solved = float(lc["solved"].sum())
                solved_values.append(lc_solved)
                profile = leetcode.profile
                anchor = float(profile.get("contest_rating", 0) or profile.get("estimated_cp_anchor", 0) or 1500)
                avg_rating_values.append(anchor)
                if int(profile.get("hard_solved", 0) or 0) >= 20:
                    max_rating_values.append(max(anchor, 2200))
                elif int(profile.get("medium_solved", 0) or 0) >= 50:
                    max_rating_values.append(max(anchor, 1650))
                else:
                    max_rating_values.append(max(anchor, 1200))

    tag_solved = sum(solved_values)
    if tag_solved == 0:
        fallback = _combined_strength(codeforces_result, external_results)
        tag_solved = fallback["total_solved"] * 0.035

    anchor = _combined_strength(codeforces_result, external_results)["anchor_rating"]
    tag_avg_rating = sum(avg_rating_values) / len(avg_rating_values) if avg_rating_values else max(900.0, anchor - 180)
    tag_rating_ceiling = max(max_rating_values) if max_rating_values else max(900.0, anchor - 100)

    return {
        "tag_solved": tag_solved,
        "tag_avg_rating": tag_avg_rating,
        "tag_rating_ceiling": tag_rating_ceiling,
    }


def _expanded_tags(tags: list[str]) -> set[str]:
    expanded: set[str] = set()
    for tag in tags:
        key = _tag_key(tag)
        expanded.add(key)
        expanded.update(_tag_key(alias) for alias in TAG_ALIASES.get(key, {key}))
    return expanded


def _tag_key(tag: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(tag or "").lower()).strip()
    return re.sub(r"\s+", " ", text)


def _platform_fit(platform: str, profile_strength: dict[str, Any]) -> float:
    solved = profile_strength.get("platform_solved", {})
    value = float(solved.get(platform, 0) or solved.get(platform.title(), 0) or 0)
    return _bounded_log(value, 600)


def _bounded_log(value: float, scale: float) -> float:
    return max(0.0, min(1.0, math.log1p(max(0.0, value)) / math.log1p(scale)))


def _sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-value))

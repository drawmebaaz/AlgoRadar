from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Any

import pandas as pd

from .config import DATA_DIR

CALIBRATION_PATH = DATA_DIR / "platform_calibration.csv"

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
    "prefix sum": {"prefix sum"},
    "backtracking": {"backtracking", "brute force"},
    "brute force": {"backtracking", "brute force"},
    "sorting": {"sortings", "sorting"},
    "sortings": {"sortings", "sorting"},
}

LEETCODE_SLOT_PRIORS = {
    "Q1": 850.0,
    "Q2": 1300.0,
    "Q3": 1700.0,
    "Q4": 2250.0,
}


def _bucket_probability(probability: float) -> str:
    if probability >= 0.75:
        return "confidence"
    if probability >= 0.45:
        return "growth"
    if probability >= 0.25:
        return "stretch"
    return "avoid"


def score_saved_profile_problem(
    platform: str,
    target_rating: int | float | None,
    tags: list[str],
    popularity: int,
    codeforces_result: Any | None,
    external_results: dict[str, Any],
    leetcode_difficulty: str = "",
    leetcode_contest_slot: str = "Unknown",
) -> dict[str, Any]:
    calibration = native_to_cf_equivalent(
        platform=platform,
        native_rating=target_rating,
        leetcode_difficulty=leetcode_difficulty,
        leetcode_contest_slot=leetcode_contest_slot,
    )
    target_cf = float(calibration["cf_equivalent"])
    profile_strength = _combined_strength(codeforces_result, external_results)
    tag_strength = _tag_strength(tags, codeforces_result, external_results)

    combined_anchor = float(profile_strength["anchor_rating"])
    anchor_rating = _platform_adjusted_anchor(platform, profile_strength)
    total_solved = float(profile_strength["total_solved"])
    tag_solved = float(tag_strength["tag_solved"])
    tag_rating_ceiling = float(tag_strength["tag_rating_ceiling"])
    tag_avg_rating = float(tag_strength["tag_avg_rating"])

    rating_gap = target_cf - anchor_rating
    ceiling_gap = tag_rating_ceiling - target_cf
    avg_gap = tag_avg_rating - target_cf
    tag_volume_score = _bounded_log(tag_solved, 120)
    overall_volume_score = _bounded_log(total_solved, 1600)
    ceiling_score = _sigmoid((ceiling_gap - 60) / 170)
    avg_rating_score = _sigmoid((avg_gap - 80) / 220)
    popularity_score = _bounded_log(popularity, 70000)
    platform_fit = _platform_fit(platform, profile_strength)
    calibration_weight = float(calibration["training_weight"])
    cold_tag_adjustment = -0.22 if tags and tag_solved < 3 else 0.0

    # A problem around the user's current level is usually a growth attempt,
    # not a guaranteed solve. Keep rating gap as the main driver and let
    # tag/popularity/platform signals nudge the result rather than dominate it.
    logit = (
        -0.38
        - rating_gap / 285
        + 0.34 * (tag_volume_score - 0.45)
        + 0.34 * (ceiling_score - 0.5)
        + 0.18 * (avg_rating_score - 0.5)
        + 0.10 * (overall_volume_score - 0.45)
        + 0.05 * (platform_fit - 0.35)
        + 0.04 * (popularity_score - 0.5)
        + 0.04 * (calibration_weight - 0.6)
        + cold_tag_adjustment
    )
    raw_probability = _sigmoid(logit)
    confidence_cap = _practical_probability_cap(rating_gap, ceiling_gap, tag_solved, bool(tags))
    probability = float(max(0.02, min(raw_probability, confidence_cap)))

    factors = pd.DataFrame(
        [
            {
                "Signal": "Problem level",
                "Value": round(target_cf, 0),
                "Meaning": "shared difficulty scale",
            },
            {
                "Signal": "Your estimated level",
                "Value": round(anchor_rating, 0),
                "Meaning": "mostly this platform; other handles help a little",
            },
            {
                "Signal": "Problem above your level",
                "Value": round(rating_gap, 0),
                "Meaning": "positive means harder than your current level",
            },
            {
                "Signal": "Solved on these tags",
                "Value": round(tag_solved, 1),
                "Meaning": "more real practice on these topics helps",
            },
            {
                "Signal": "Hardest solved on these tags",
                "Value": round(tag_rating_ceiling, 0),
                "Meaning": "strong evidence only if close to the problem level",
            },
            {
                "Signal": "Hardest tag solve vs problem",
                "Value": round(ceiling_gap, 0),
                "Meaning": "positive means you have solved harder similar work",
            },
            {
                "Signal": "Usual level on these tags",
                "Value": round(tag_avg_rating, 0),
                "Meaning": "keeps one lucky hard solve from overrating you",
            },
            {
                "Signal": "Total solved",
                "Value": round(total_solved, 1),
                "Meaning": "small confidence boost, not the main factor",
            },
            {
                "Signal": "Problem popularity",
                "Value": int(popularity or 0),
                "Meaning": "popular problems are usually easier to learn from",
            },
            {
                "Signal": "Tags used",
                "Value": ", ".join(tags) if tags else "None selected",
                "Meaning": "pulled automatically when available",
            },
        ]
    )
    factors["Value"] = factors["Value"].astype(str)
    factors["Meaning"] = factors["Meaning"].astype(str)

    return {
        "platform": platform,
        "target_rating": round(target_cf, 0),
        "target_cf_equivalent": round(target_cf, 0),
        "native_target": calibration["native_target"],
        "calibration_source": calibration["source"],
        "calibration_confidence": calibration["confidence"],
        "calibration_weight": calibration["training_weight"],
        "calibration_context": calibration["context_band"],
        "leetcode_reference": calibration.get("leetcode_reference", ""),
        "tags": tags,
        "popularity": int(popularity or 0),
        "solve_probability": probability,
        "solve_probability_pct": round(probability * 100, 1),
        "bucket": _bucket_probability(probability),
        "anchor_rating": round(anchor_rating, 0),
        "anchor_cf_equivalent": round(anchor_rating, 0),
        "combined_anchor_cf_equivalent": round(combined_anchor, 0),
        "total_solved": round(total_solved, 1),
        "tag_solved": round(tag_solved, 1),
        "tag_rating_ceiling": round(tag_rating_ceiling, 0),
        "tag_avg_rating": round(tag_avg_rating, 0),
        "factors": factors,
    }


def native_to_cf_equivalent(
    platform: str,
    native_rating: int | float | None = None,
    leetcode_difficulty: str = "",
    leetcode_contest_slot: str = "Unknown",
) -> dict[str, Any]:
    platform_key = _platform_key(platform)
    if platform_key == "codeforces":
        target = _clip_rating(float(native_rating or 1200), 400, 3500)
        row = _nearest_calibration_row(target, "cf_problem_rating")
        return _calibration_result(
            platform=platform,
            native_target=str(int(round(target))),
            cf_equivalent=target,
            source="Codeforces official/estimated rating",
            row=row,
        )

    if platform_key == "codechef":
        native = _clip_rating(float(native_rating or 1450), 400, 5000)
        frame = calibration_frame().sort_values("codechef_equiv_mid")
        cf_equivalent = _interpolate(frame["codechef_equiv_mid"].tolist(), frame["cf_problem_rating"].tolist(), native)
        row = _nearest_calibration_row(native, "codechef_equiv_mid")
        return _calibration_result(
            platform=platform,
            native_target=f"CodeChef {int(round(native))}",
            cf_equivalent=cf_equivalent,
            source="CodeChef rating calibrated from mapping CSV",
            row=row,
        )

    if platform_key == "leetcode":
        if native_rating not in (None, "", 0):
            native = _clip_rating(float(native_rating or 1600), 500, 4000)
            frame = calibration_frame().sort_values("leetcode_zerotrac_equiv_mid")
            cf_equivalent = _interpolate(frame["leetcode_zerotrac_equiv_mid"].tolist(), frame["cf_problem_rating"].tolist(), native)
            row = _nearest_calibration_row(native, "leetcode_zerotrac_equiv_mid")
            return _calibration_result(
                platform=platform,
                native_target=f"LeetCode/Zerotrac {int(round(native))}",
                cf_equivalent=cf_equivalent,
                source="LeetCode numeric difficulty calibrated from mapping CSV",
                row=row,
                leetcode_reference="Numeric LeetCode-style difficulty",
            )

        difficulty = _clean_leetcode_difficulty(leetcode_difficulty)
        slot = _clean_leetcode_slot(leetcode_contest_slot)
        difficulty_prior = _leetcode_difficulty_prior(difficulty)
        cf_equivalent = difficulty_prior
        source = "LeetCode difficulty calibrated from mapping CSV"
        reference = difficulty
        if slot != "Unknown":
            slot_prior = LEETCODE_SLOT_PRIORS[slot]
            cf_equivalent = difficulty_prior * 0.55 + slot_prior * 0.45
            source = "LeetCode difficulty + contest-slot prior calibrated from mapping CSV"
            reference = f"{difficulty} / {slot} reference"
        row = _nearest_calibration_row(cf_equivalent, "cf_problem_rating")
        return _calibration_result(
            platform=platform,
            native_target=reference,
            cf_equivalent=cf_equivalent,
            source=source,
            row=row,
            leetcode_reference=reference,
        )

    target = _clip_rating(float(native_rating or 1200), 400, 3500)
    row = _nearest_calibration_row(target, "cf_problem_rating")
    return _calibration_result(
        platform=platform,
        native_target=str(int(round(target))),
        cf_equivalent=target,
        source="Fallback shared difficulty rating",
        row=row,
    )


def target_rating_from_difficulty(platform: str, difficulty: str, fallback_rating: int = 1600) -> int:
    if _platform_key(platform) == "leetcode":
        return int(round(native_to_cf_equivalent("LeetCode", leetcode_difficulty=difficulty)["cf_equivalent"]))
    return int(fallback_rating)


def available_probability_tags(codeforces_result: Any | None, external_results: dict[str, Any]) -> list[str]:
    tags: set[str] = set()
    if codeforces_result is not None and not codeforces_result.tag_stats.empty:
        tags.update(str(tag) for tag in codeforces_result.tag_stats["tag"].dropna().tolist())
    leetcode = external_results.get("leetcode")
    if leetcode is not None and getattr(leetcode, "status", "") == "ok" and not leetcode.tags.empty:
        tags.update(str(tag) for tag in leetcode.tags["tag"].dropna().tolist())
    tags.update(
        [
            "dp",
            "graphs",
            "greedy",
            "math",
            "binary search",
            "Dynamic Programming",
            "Graph",
            "Array",
            "Two Pointers",
            "Hash Table",
            "Prefix Sum",
        ]
    )
    return sorted(tags, key=lambda value: value.lower())


@lru_cache(maxsize=1)
def calibration_frame() -> pd.DataFrame:
    frame = pd.read_csv(CALIBRATION_PATH)
    numeric_columns = [
        "cf_problem_rating",
        "codechef_equiv_mid",
        "codechef_equiv_low",
        "codechef_equiv_high",
        "leetcode_zerotrac_equiv_mid",
        "leetcode_zerotrac_equiv_low",
        "leetcode_zerotrac_equiv_high",
        "recommended_training_weight",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["cf_problem_rating"]).reset_index(drop=True)


def _combined_strength(codeforces_result: Any | None, external_results: dict[str, Any]) -> dict[str, Any]:
    ratings: list[tuple[float, float, str]] = []
    total_solved = 0.0
    platform_solved: dict[str, float] = {}
    platform_anchor: dict[str, float] = {}

    if codeforces_result is not None:
        profile = codeforces_result.profile
        solved = float(profile.get("problems_solved", 0) or 0)
        rating = float(profile.get("current_rating", 0) or profile.get("max_rating", 0) or 1200)
        ratings.append((rating, max(1.0, min(4.0, math.log1p(solved) / 2.2)), "Codeforces"))
        total_solved += solved
        platform_solved["Codeforces"] = solved
        platform_anchor["Codeforces"] = rating

    leetcode = external_results.get("leetcode")
    if leetcode is not None and getattr(leetcode, "status", "") == "ok":
        profile = leetcode.profile
        solved = float(profile.get("total_solved", 0) or 0)
        native_rating = float(profile.get("contest_rating", 0) or 0)
        if native_rating:
            rating = float(native_to_cf_equivalent("LeetCode", native_rating=native_rating)["cf_equivalent"])
        else:
            rating = float(profile.get("estimated_cp_anchor", 0) or _leetcode_profile_anchor(profile))
        ratings.append((rating, max(0.8, min(3.2, math.log1p(solved) / 2.4)), "LeetCode"))
        total_solved += solved
        platform_solved["LeetCode"] = solved
        platform_anchor["LeetCode"] = rating

    codechef = external_results.get("codechef")
    if codechef is not None and getattr(codechef, "status", "") == "ok":
        profile = codechef.profile
        solved = float(profile.get("total_solved", 0) or 0)
        native_rating = float(profile.get("current_rating", 0) or profile.get("max_rating", 0) or 1450)
        rating = float(native_to_cf_equivalent("CodeChef", native_rating=native_rating)["cf_equivalent"])
        ratings.append((rating, max(0.8, min(3.2, math.log1p(solved) / 2.4)), "CodeChef"))
        total_solved += solved
        platform_solved["CodeChef"] = solved
        platform_anchor["CodeChef"] = rating

    if not ratings:
        return {"anchor_rating": 1200.0, "total_solved": 0.0, "platform_solved": {}, "platform_anchor": {}}

    weighted_sum = sum(rating * weight for rating, weight, _ in ratings)
    total_weight = sum(weight for _, weight, _ in ratings) or 1.0
    return {
        "anchor_rating": weighted_sum / total_weight,
        "total_solved": total_solved,
        "platform_solved": platform_solved,
        "platform_anchor": platform_anchor,
    }


def _platform_adjusted_anchor(platform: str, profile_strength: dict[str, Any]) -> float:
    combined_anchor = float(profile_strength.get("anchor_rating", 1200.0) or 1200.0)
    platform_anchor = profile_strength.get("platform_anchor", {}) or {}
    platform_key = _platform_name(platform)
    direct_anchor = platform_anchor.get(platform_key)
    if direct_anchor is None:
        return combined_anchor

    direct_anchor = float(direct_anchor)
    blended = direct_anchor * 0.72 + combined_anchor * 0.28
    return _clip_rating(blended, direct_anchor - 180, direct_anchor + 220)


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
                anchor = _leetcode_profile_anchor(profile)
                avg_rating_values.append(anchor)
                hard = int(profile.get("hard_solved", 0) or 0)
                medium = int(profile.get("medium_solved", 0) or 0)
                if hard >= 20:
                    max_rating_values.append(max(anchor, _leetcode_difficulty_prior("Hard")))
                elif medium >= 50:
                    max_rating_values.append(max(anchor, _leetcode_difficulty_prior("Medium")))
                else:
                    max_rating_values.append(max(anchor, _leetcode_difficulty_prior("Easy")))

    tag_solved = sum(solved_values)
    fallback = _combined_strength(codeforces_result, external_results)
    anchor = float(fallback["anchor_rating"])
    if tag_solved == 0 and not normalized:
        tag_solved = fallback["total_solved"] * 0.025

    missing_tag_penalty = 260 if normalized else 160
    tag_avg_rating = sum(avg_rating_values) / len(avg_rating_values) if avg_rating_values else max(700.0, anchor - missing_tag_penalty)
    tag_rating_ceiling = max(max_rating_values) if max_rating_values else max(700.0, anchor - missing_tag_penalty + 60)

    return {
        "tag_solved": tag_solved,
        "tag_avg_rating": tag_avg_rating,
        "tag_rating_ceiling": tag_rating_ceiling,
    }


def _leetcode_profile_anchor(profile: dict[str, Any]) -> float:
    contest_rating = float(profile.get("contest_rating", 0) or 0)
    if contest_rating:
        return float(native_to_cf_equivalent("LeetCode", native_rating=contest_rating)["cf_equivalent"])
    easy = int(profile.get("easy_solved", 0) or 0)
    medium = int(profile.get("medium_solved", 0) or 0)
    hard = int(profile.get("hard_solved", 0) or 0)
    solved_score = easy * 1.8 + medium * 6.5 + hard * 17
    return float(max(800, min(2300, 780 + math.sqrt(solved_score) * 45)))


def _leetcode_difficulty_prior(difficulty: str) -> float:
    difficulty = _clean_leetcode_difficulty(difficulty)
    frame = calibration_frame().copy()
    label = frame["leetcode_official_label_likely"].fillna("").str.lower()
    if difficulty == "Easy":
        subset = frame[label.str.contains("easy") & ~label.str.contains("hard")]
        fallback = 800.0
    elif difficulty == "Medium":
        subset = frame[label.str.contains("medium")]
        fallback = 1300.0
    else:
        subset = frame[label.str.contains("hard") & ~label.str.contains("q4")]
        fallback = 1750.0
    if subset.empty:
        return fallback
    return _weighted_average(subset["cf_problem_rating"], subset["recommended_training_weight"], fallback)


def _weighted_average(values: pd.Series, weights: pd.Series, fallback: float) -> float:
    cleaned = pd.DataFrame({"value": values, "weight": weights}).dropna()
    if cleaned.empty or float(cleaned["weight"].sum()) <= 0:
        return fallback
    return float((cleaned["value"] * cleaned["weight"]).sum() / cleaned["weight"].sum())


def _calibration_result(
    platform: str,
    native_target: str,
    cf_equivalent: float,
    source: str,
    row: pd.Series,
    leetcode_reference: str = "",
) -> dict[str, Any]:
    return {
        "platform": platform,
        "native_target": native_target,
        "cf_equivalent": int(round(_clip_rating(cf_equivalent, 400, 3500))),
        "source": source,
        "confidence": str(row.get("confidence", "medium") or "medium"),
        "training_weight": float(row.get("recommended_training_weight", 0.5) or 0.5),
        "context_band": str(row.get("cf_context_band", "") or ""),
        "leetcode_reference": leetcode_reference,
    }


def _nearest_calibration_row(value: float, column: str) -> pd.Series:
    frame = calibration_frame()
    distances = (pd.to_numeric(frame[column], errors="coerce") - float(value)).abs()
    return frame.loc[distances.idxmin()]


def _interpolate(x_values: list[float], y_values: list[float], x: float) -> float:
    pairs = sorted((float(xv), float(yv)) for xv, yv in zip(x_values, y_values) if pd.notna(xv) and pd.notna(yv))
    if not pairs:
        return float(x)
    if x <= pairs[0][0]:
        return pairs[0][1]
    if x >= pairs[-1][0]:
        return pairs[-1][1]
    for index in range(1, len(pairs)):
        left_x, left_y = pairs[index - 1]
        right_x, right_y = pairs[index]
        if left_x <= x <= right_x:
            span = right_x - left_x
            if span == 0:
                return right_y
            ratio = (x - left_x) / span
            return left_y + (right_y - left_y) * ratio
    return pairs[-1][1]


def _expanded_tags(tags: list[str]) -> set[str]:
    expanded: set[str] = set()
    for tag in tags:
        key = _tag_key(tag)
        if not key:
            continue
        expanded.add(key)
        expanded.update(_tag_key(alias) for alias in TAG_ALIASES.get(key, {key}))
    return expanded


def _tag_key(tag: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(tag or "").lower()).strip()
    return re.sub(r"\s+", " ", text)


def _platform_fit(platform: str, profile_strength: dict[str, Any]) -> float:
    solved = profile_strength.get("platform_solved", {})
    value = float(solved.get(_platform_name(platform), 0) or solved.get(platform, 0) or solved.get(platform.title(), 0) or 0)
    return _bounded_log(value, 700)


def _practical_probability_cap(rating_gap: float, ceiling_gap: float, tag_solved: float, has_tags: bool) -> float:
    cap = 0.48 + 0.38 * _sigmoid((-rating_gap - 120) / 260)
    cap += 0.05 * _sigmoid((ceiling_gap - 260) / 220)
    if rating_gap >= 0:
        cap = min(cap, 0.62)
    if rating_gap >= 200:
        cap = min(cap, 0.48)
    if rating_gap >= 400:
        cap = min(cap, 0.34)
    if has_tags and tag_solved < 3:
        cap = min(cap, 0.50 if rating_gap <= 0 else 0.42)
    return max(0.10, min(cap, 0.90))


def _bounded_log(value: float, scale: float) -> float:
    return max(0.0, min(1.0, math.log1p(max(0.0, value)) / math.log1p(scale)))


def _sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-value))


def _clip_rating(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _platform_key(platform: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(platform or "").lower())


def _platform_name(platform: str) -> str:
    key = _platform_key(platform)
    return {"codeforces": "Codeforces", "codechef": "CodeChef", "leetcode": "LeetCode"}.get(key, str(platform or "").title())


def _clean_leetcode_difficulty(difficulty: str) -> str:
    value = str(difficulty or "").strip().title()
    return value if value in {"Easy", "Medium", "Hard"} else "Medium"


def _clean_leetcode_slot(slot: str) -> str:
    value = str(slot or "").strip().upper()
    return value if value in {"Q1", "Q2", "Q3", "Q4"} else "Unknown"

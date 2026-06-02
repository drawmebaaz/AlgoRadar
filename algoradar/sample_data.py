from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

TAGS = [
    "implementation",
    "math",
    "greedy",
    "dp",
    "graphs",
    "binary search",
    "constructive algorithms",
    "data structures",
    "trees",
    "strings",
    "number theory",
    "sortings",
    "two pointers",
    "combinatorics",
    "dfs and similar",
]

TITLE_PARTS = [
    "Balanced",
    "Hidden",
    "Circular",
    "Restoring",
    "Maximum",
    "Minimum",
    "Strange",
    "Fast",
    "Lost",
    "Prefix",
    "Segment",
    "Array",
    "Graph",
    "Route",
    "Tournament",
]

VERDICTS = [
    "OK",
    "WRONG_ANSWER",
    "TIME_LIMIT_EXCEEDED",
    "RUNTIME_ERROR",
    "COMPILATION_ERROR",
]


def _handle_seed(handle: str) -> int:
    return sum(ord(char) for char in handle) + 1337


def make_sample_problemset(handle: str = "demo") -> dict[str, list[dict[str, Any]]]:
    rng = np.random.default_rng(_handle_seed(handle))
    problems: list[dict[str, Any]] = []
    statistics: list[dict[str, Any]] = []

    for idx in range(620):
      contest_id = 1200 + idx // 6
      index = chr(65 + idx % 6)
      rating = int(rng.choice(np.arange(800, 2500, 100), p=_rating_distribution()))
      tag_count = int(rng.choice([1, 2, 3], p=[0.35, 0.5, 0.15]))
      tags = rng.choice(TAGS, size=tag_count, replace=False).tolist()
      title = f"{rng.choice(TITLE_PARTS)} {rng.choice(TITLE_PARTS)} {rng.choice(['Path', 'Pairs', 'Game', 'Query', 'Matrix', 'Score'])}"
      solved_count = int(max(120, 32000 * math.exp(-(rating - 800) / 710) + rng.normal(0, 340)))

      problem = {
          "contestId": contest_id,
          "index": index,
          "name": title,
          "type": "PROGRAMMING",
          "rating": rating,
          "tags": tags,
      }
      problems.append(problem)
      statistics.append(
          {
              "contestId": contest_id,
              "index": index,
              "solvedCount": solved_count,
          }
      )

    return {"problems": problems, "problemStatistics": statistics}


def make_sample_submissions(handle: str = "demo", count: int = 560) -> list[dict[str, Any]]:
    rng = np.random.default_rng(_handle_seed(handle) + 7)
    problemset = make_sample_problemset(handle)
    problems = problemset["problems"]
    tag_skill = {tag: float(rng.normal(0.0, 0.45)) for tag in TAGS}
    base_rating = 1320 + (_handle_seed(handle) % 280)
    now = datetime.now(timezone.utc)
    submissions: list[dict[str, Any]] = []

    for submission_id in range(count):
      days_ago = int((count - submission_id) * rng.uniform(0.25, 0.72))
      created_at = now - timedelta(days=days_ago, hours=int(rng.integers(0, 22)))
      problem = dict(rng.choice(problems))
      rating = int(problem.get("rating", 1200))
      tags = problem.get("tags", [])
      skill = np.mean([tag_skill.get(tag, 0.0) for tag in tags]) if tags else 0.0
      rating_gap = rating - base_rating
      logit = 1.15 + skill - rating_gap / 360
      solve_probability = 1 / (1 + math.exp(-logit))
      accepted = rng.random() < solve_probability

      if accepted:
          verdict = "OK"
      else:
          verdict = str(rng.choice(VERDICTS[1:], p=[0.68, 0.18, 0.09, 0.05]))

      submissions.append(
          {
              "id": 10_000_000 + submission_id,
              "contestId": problem["contestId"],
              "creationTimeSeconds": int(created_at.timestamp()),
              "relativeTimeSeconds": 2147483647,
              "problem": problem,
              "programmingLanguage": str(rng.choice(["GNU C++20", "PyPy 3", "GNU C++17"])),
              "verdict": verdict,
              "testset": "TESTS",
              "passedTestCount": int(rng.integers(0, 34)),
              "timeConsumedMillis": int(rng.integers(31, 2000)),
              "memoryConsumedBytes": int(rng.integers(1_000_000, 180_000_000)),
          }
      )

      if not accepted and rng.random() < 0.28:
          retry = submissions[-1].copy()
          retry["id"] = 20_000_000 + submission_id
          retry["creationTimeSeconds"] += int(rng.integers(600, 7200))
          retry["verdict"] = "OK" if rng.random() < solve_probability + 0.18 else verdict
          submissions.append(retry)

    return sorted(submissions, key=lambda item: item["creationTimeSeconds"], reverse=True)


def make_sample_rating_history(handle: str = "demo", contests: int = 34) -> list[dict[str, Any]]:
    rng = np.random.default_rng(_handle_seed(handle) + 19)
    now = datetime.now(timezone.utc)
    rating = 950 + (_handle_seed(handle) % 260)
    history: list[dict[str, Any]] = []

    for idx in range(contests):
      old_rating = int(rating)
      drift = 16 + idx * 0.7
      noise = rng.normal(0, 42)
      rating = max(800, rating + drift + noise)
      rank = int(max(450, 9200 - rating * 4.5 + rng.normal(0, 840)))
      contest_time = now - timedelta(days=(contests - idx) * 8)
      history.append(
          {
              "contestId": 900 + idx,
              "contestName": f"Codeforces Round {900 + idx}",
              "handle": handle,
              "rank": rank,
              "ratingUpdateTimeSeconds": int(contest_time.timestamp()),
              "oldRating": old_rating,
              "newRating": int(rating),
          }
      )

    return history


def make_sample_bundle(handle: str = "demo") -> dict[str, Any]:
    problemset = make_sample_problemset(handle)
    return {
        "handle": handle,
        "source": "sample",
        "submissions": make_sample_submissions(handle),
        "ratings": make_sample_rating_history(handle),
        "problems": problemset["problems"],
        "problem_statistics": problemset["problemStatistics"],
    }


def _rating_distribution() -> list[float]:
    values = np.array([10, 13, 16, 17, 15, 12, 8, 5, 2.8, 1.2, 0.7, 0.3, 0.15, 0.08, 0.04, 0.02, 0.01])
    return (values / values.sum()).tolist()

import json
import time
from pathlib import Path

import pandas as pd

from algoradar.models import build_real_solve_training_dataset


def test_multi_horizon_labels(tmp_path):
    cache_dir = tmp_path
    cache_dir.mkdir(parents=True, exist_ok=True)

    # create problemset.json
    problem = {"contestId": 2000, "index": "B", "tags": ["dp"], "rating": 1500}
    (cache_dir / "problemset.json").write_text(json.dumps({"problems": [problem]}))

    now = int(time.time())
    first_seen = now - 10 * 3600  # 10 hours ago
    first_accepted = now - 2 * 3600  # 2 hours ago
    later_accept = now + 2 * 3600  # in future (shouldn't be counted)

    # submissions: first attempt wrong, then accepted within 24h
    sub1 = {"creationTimeSeconds": first_seen, "verdict": "WRONG_ANSWER", "problem": {"contestId": 2000, "index": "B", "tags": ["dp"], "rating": 1500}}
    sub2 = {"creationTimeSeconds": first_accepted, "verdict": "OK", "problem": {"contestId": 2000, "index": "B", "tags": ["dp"], "rating": 1500}}

    submissions = [sub1, sub2]
    (cache_dir / "user_status_testuser_1200.json").write_text(json.dumps(submissions))

    save_path = tmp_path / "dataset.csv"
    df = build_real_solve_training_dataset(cache_dir=str(cache_dir), max_users=1, save_path=str(save_path))
    assert not df.empty
    row = df[(df["problem_id"] == "2000B") & (df["handle"] == "testuser")].iloc[0]
    assert int(row["solved_within_24h"]) == 1
    assert int(row["solved_within_72h"]) == 1
    assert int(row["solved_within_7d"]) == 1
    assert int(row["first_attempt_solved"]) == 0

    # Now test first-attempt success
    submissions2 = [{"creationTimeSeconds": first_seen, "verdict": "OK", "problem": {"contestId": 2000, "index": "B", "tags": ["dp"], "rating": 1500}}]
    (cache_dir / "user_status_testuser_1200.json").write_text(json.dumps(submissions2))
    df2 = build_real_solve_training_dataset(cache_dir=str(cache_dir), max_users=1, min_examples_per_user=1, save_path=str(save_path))
    row2 = df2[(df2["problem_id"] == "2000B") & (df2["handle"] == "testuser")].iloc[0]
    assert int(row2["first_attempt_solved"]) == 1
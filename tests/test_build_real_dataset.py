import json
import time
from pathlib import Path

import pandas as pd

from algoradar.models import build_real_solve_training_dataset


def test_build_real_dataset_temporal_integrity(tmp_path):
    cache_dir = tmp_path
    cache_dir.mkdir(parents=True, exist_ok=True)

    # create problemset.json
    problem = {"contestId": 1000, "index": "A", "tags": ["graphs"], "rating": 1300}
    problemset = {"problems": [problem]}
    (cache_dir / "problemset.json").write_text(json.dumps(problemset))

    # times
    now = int(time.time())
    prior_time = now - 10 * 86400  # 10 days earlier
    first_seen = now - 5 * 86400  # 5 days earlier
    first_accepted = now - 4 * 86400  # 4 days earlier

    # prior submission on different problem but overlapping tag
    prior_submission = {
        "creationTimeSeconds": prior_time,
        "verdict": "OK",
        "problem": {"contestId": 999, "index": "X", "tags": ["graphs"], "rating": 1200},
    }

    # target problem submissions: first seen (attempt), then accepted
    sub1 = {"creationTimeSeconds": first_seen, "verdict": "WRONG_ANSWER", "problem": {"contestId": 1000, "index": "A", "tags": ["graphs"], "rating": 1300}}
    sub2 = {"creationTimeSeconds": first_accepted, "verdict": "OK", "problem": {"contestId": 1000, "index": "A", "tags": ["graphs"], "rating": 1300}}

    submissions = [prior_submission, sub1, sub2]
    (cache_dir / "user_status_testuser_1200.json").write_text(json.dumps(submissions))

    # no rating file necessary for this test

    # build dataset
    save_path = tmp_path / "dataset.csv"
    df = build_real_solve_training_dataset(cache_dir=str(cache_dir), max_users=1, save_path=str(save_path))

    assert not df.empty
    # find row for problem
    row = df[(df["problem_id"] == "1000A") & (df["handle"] == "testuser")]
    assert len(row) == 1
    r = row.iloc[0]
    # event_time should equal first_seen
    assert int(r["event_time"]) == int(first_seen)
    # tag_mastery should count only prior_same_tags (prior_submission OK => mastery 1.0)
    assert float(r["tag_mastery"]) == 1.0
    # time_to_solve should be (first_accepted - first_seen)/86400
    expected = (first_accepted - first_seen) / 86400.0
    assert abs(float(r["time_to_solve"]) - expected) < 1e-6

    # metadata file should exist
    meta = Path(str(save_path)).with_suffix(".meta.json")
    assert meta.exists()
    meta_obj = json.loads(meta.read_text())
    assert "dataset_version" in meta_obj
import hashlib
import json
from pathlib import Path

import pandas as pd

from algoradar import semantic

CACHE_DIR = Path("data") / "cache"


def _compute_hash(problem_ids):
    text = json.dumps(sorted([str(x) for x in (problem_ids or [])]), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_semantic_persistence_and_hash(tmp_path, monkeypatch):
    # ensure we use a temporary cache dir for the test
    monkeypatch.setattr(semantic, "DEFAULT_CACHE_DIR", tmp_path)

    problems = pd.DataFrame(
        [
            {"problem_id": "lc1", "name": "Two Sum", "rating": 800, "tags": ["array", "hashmap"]},
            {"problem_id": "cf123", "name": "A+B Problem", "rating": 900, "tags": ["math"]},
        ]
    )

    # build index using TF-IDF fallback to avoid requiring transformer downloads
    idx = semantic.build_semantic_index(problems, prefer_transformer=False)
    assert idx is not None

    vec_file = tmp_path / "semantic_vectors.npy"
    meta_file = tmp_path / "semantic_meta.json"

    assert vec_file.exists()
    assert meta_file.exists()

    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    expected_hash = _compute_hash(["lc1", "cf123"])
    assert meta.get("catalog_hash") == expected_hash

    # loading with the same expected hash should succeed
    loaded = semantic.load_semantic_index(tmp_path, expected_catalog_hash=expected_hash)
    assert loaded is not None

    # loading with a different hash should return None
    other_hash = _compute_hash(["lc1"])  # different catalog
    loaded2 = semantic.load_semantic_index(tmp_path, expected_catalog_hash=other_hash)
    assert loaded2 is None

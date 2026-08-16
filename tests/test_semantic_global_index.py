import hashlib
import json

import pandas as pd

from algoradar import semantic


def _compute_hash(problem_ids):
    text = json.dumps(sorted([str(x) for x in (problem_ids or [])]), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_build_global_index_persists(tmp_path, monkeypatch):
    monkeypatch.setattr(semantic, "DEFAULT_CACHE_DIR", tmp_path)

    problems = pd.DataFrame(
        [
            {"problem_id": "g1", "name": "P One", "rating": 1200, "tags": ["dp"]},
            {"problem_id": "g2", "name": "P Two", "rating": 1300, "tags": ["graphs"]},
        ]
    )

    idx = semantic.build_global_index(problems, prefer_transformer=False)
    assert idx is not None
    expected_hash = _compute_hash(["g1", "g2"])
    # persisted dir should start with global_{hash[:8]}
    expected_dir = tmp_path / f"global_{expected_hash[:8]}"
    assert expected_dir.exists()
    assert (expected_dir / "semantic_vectors.npy").exists()
    assert (expected_dir / "semantic_meta.json").exists()

    loaded = semantic.load_semantic_index(dir_path=expected_dir, expected_catalog_hash=expected_hash)
    assert loaded is not None
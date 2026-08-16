import numpy as np

from algoradar import semantic


def test_embeddings_only_load_when_model_unavailable(tmp_path, monkeypatch):
    # create a fake semantic index and persist it with model_name set
    texts = ["one two three", "alpha beta gamma"]
    problem_ids = ["p1", "p2"]
    # simple embeddings (2x3)
    embeddings = np.random.RandomState(0).randn(2, 3)
    idx = semantic.SemanticIndex(
        method=semantic.MINILM_MODEL_NAME,
        texts=texts,
        problem_ids=problem_ids,
        embeddings=embeddings,
        model=None,
    )
    # save to tmp_path
    semantic.save_semantic_index(idx, dir_path=tmp_path)

    # ensure that attempting to import SentenceTransformer raises ImportError
    monkeypatch.setitem(__import__("sys").modules, "sentence_transformers", None)

    loaded = semantic.load_semantic_index(dir_path=tmp_path)
    assert loaded is not None
    assert loaded.model is None
    assert getattr(loaded, "embeddings_only", False) is True

import numpy as np

from algoradar.semantic import SemanticIndex, _cosine_scores, annoy_retrieve_and_rerank


def test_annoy_retrieve_and_rerank_matches_bruteforce():
    rng = np.random.RandomState(42)
    n = 200
    dim = 64
    embeddings = rng.normal(size=(n, dim)).astype(float)
    # normalize rows
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-12)

    # build a SemanticIndex with Annoy if available
    idx = SemanticIndex(method="embeddings-only", texts=[str(i) for i in range(n)], problem_ids=[str(i) for i in range(n)], embeddings=embeddings, model=None, annoy_index=None)

    try:
        from annoy import AnnoyIndex

        a = AnnoyIndex(dim, "angular")
        for i in range(n):
            a.add_item(i, embeddings[i].tolist())
        a.build(10)
        idx.annoy_index = a
    except (ImportError, OSError, RuntimeError, ValueError):
        # if Annoy not available the helper falls back to brute-force, still valid
        idx.annoy_index = None

    # choose a random query vector (one of the embeddings with small noise)
    q = embeddings[10] + 1e-3 * rng.normal(size=(dim,))
    q = q / (np.linalg.norm(q) + 1e-12)

    top_n = 10
    ann_indices, _ann_scores = annoy_retrieve_and_rerank(q, idx, top_n=top_n)

    # brute-force top-k
    bf_scores = _cosine_scores(q, embeddings)
    bf_order = np.argsort(-bf_scores)[:top_n]

    # compare sets (order may differ slightly if scores tie)
    assert set(ann_indices) == set(bf_order.tolist())
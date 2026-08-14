from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

MINILM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class SemanticIndex:
    method: str
    texts: list[str]
    problem_ids: list[str]
    embeddings: Any
    vectorizer: Any | None = None
    model: Any | None = None


def build_problem_text(problem: pd.Series | dict[str, Any]) -> str:
    name = str(problem.get("name", ""))
    rating = str(int(problem.get("rating", 0) or 0))
    tags = " ".join(problem.get("tags", []) or [])
    return f"{name} rating_{rating} {tags}".strip()


def build_semantic_index(problems: pd.DataFrame, prefer_transformer: bool = True) -> SemanticIndex:
    frame = problems.dropna(subset=["problem_id"]).drop_duplicates("problem_id").copy()
    texts = [build_problem_text(row) for _, row in frame.iterrows()]
    problem_ids = frame["problem_id"].astype(str).tolist()

    if prefer_transformer:
        try:
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            from sentence_transformers import SentenceTransformer

            allow_download = os.environ.get("ALGORADAR_ALLOW_MINILM_DOWNLOAD", "").strip() == "1"
            model = SentenceTransformer(MINILM_MODEL_NAME, local_files_only=not allow_download)
            embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return SemanticIndex(
                method=MINILM_MODEL_NAME,
                texts=texts,
                problem_ids=problem_ids,
                embeddings=embeddings,
                model=model,
            )
        except (ImportError, RuntimeError, OSError):
            pass

    vectorizer = _build_vectorizer(texts, max_features=6000)
    embeddings = _vectorize_texts(texts, vectorizer)
    return SemanticIndex(
        method="tfidf-fallback",
        texts=texts,
        problem_ids=problem_ids,
        embeddings=embeddings,
        vectorizer=vectorizer,
    )


def similar_problems(
    query_problem: pd.Series,
    problems: pd.DataFrame,
    index: SemanticIndex,
    top_n: int = 8,
    harder_only: bool = True,
) -> pd.DataFrame:
    if problems.empty or not index.problem_ids:
        return pd.DataFrame()

    query_text = build_problem_text(query_problem)
    if index.method.startswith("sentence-transformers") and index.model is not None:
        query_embedding = index.model.encode([query_text], normalize_embeddings=True, show_progress_bar=False)
    else:
        query_embedding = _vectorize_texts([query_text], index.vectorizer)
    scores = _cosine_scores(np.asarray(query_embedding), np.asarray(index.embeddings))

    score_frame = pd.DataFrame({"problem_id": index.problem_ids, "semantic_score": scores})
    merged = problems.merge(score_frame, on="problem_id", how="inner")
    merged = merged[merged["problem_id"] != query_problem.get("problem_id")]
    if harder_only:
        base_rating = float(query_problem.get("rating", 0) or 0)
        merged = merged[merged["rating"] >= base_rating]
    return merged.sort_values(["semantic_score", "solved_count"], ascending=[False, False]).head(top_n).reset_index(drop=True)


def tag_similarity_score(problem_tags: list[str], target_tags: list[str]) -> float:
    problem_set = set(problem_tags or [])
    target_set = set(target_tags or [])
    if not problem_set or not target_set:
        return 0.0
    intersection = len(problem_set & target_set)
    union = len(problem_set | target_set)
    return intersection / union if union else 0.0


def normalize(values: pd.Series) -> pd.Series:
    if values.empty:
        return values
    min_value = values.min()
    max_value = values.max()
    if np.isclose(min_value, max_value):
        return pd.Series(np.ones(len(values)), index=values.index)
    return (values - min_value) / (max_value - min_value)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def _build_vectorizer(texts: list[str], max_features: int) -> dict[str, Any]:
    doc_counts: Counter[str] = Counter()
    for text in texts:
        doc_counts.update(set(_tokens(text)))
    vocab = {
        token: index
        for index, (token, _) in enumerate(doc_counts.most_common(max_features))
    }
    document_count = max(len(texts), 1)
    idf = {
        token: np.log((1 + document_count) / (1 + doc_counts[token])) + 1
        for token in vocab
    }
    return {"vocab": vocab, "idf": idf}


def _vectorize_texts(texts: list[str], vectorizer: dict[str, Any] | None) -> np.ndarray:
    vectorizer = vectorizer or {"vocab": {}, "idf": {}}
    vocab = vectorizer.get("vocab", {})
    idf = vectorizer.get("idf", {})
    matrix = np.zeros((len(texts), len(vocab)), dtype=float)
    for row_index, text in enumerate(texts):
        counts = Counter(token for token in _tokens(text) if token in vocab)
        for token, count in counts.items():
            matrix[row_index, vocab[token]] = (1 + np.log(count)) * float(idf.get(token, 1.0))
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _cosine_scores(query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    if embeddings.size == 0:
        return np.array([])
    query_norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
    row_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    query_norm[query_norm == 0] = 1.0
    row_norms[row_norms == 0] = 1.0
    return ((query_embedding / query_norm) @ (embeddings / row_norms).T).ravel()


def detect_isomorphic_twins(candidates: pd.DataFrame, solved_problems: pd.DataFrame, prefer_transformer: bool = True, threshold: float = 0.88) -> list[str]:
    """Return a list of candidate problem_ids that semantically match a solved Codeforces problem above threshold.

    This builds an index from the solved_problems and compares candidate texts. If a candidate has a
    cosine similarity greater than `threshold` to any solved problem (assumed Codeforces when platform
    is not explicit), it is flagged as an isomorphic twin.
    """
    if candidates is None or candidates.empty or solved_problems is None or solved_problems.empty:
        return []

    # build an index of solved problems (anchor set)
    solved_index = build_semantic_index(solved_problems, prefer_transformer=prefer_transformer)
    if not solved_index.problem_ids:
        return []

    # prepare candidate texts (only check candidates that look like LeetCode if platform column exists)
    def _is_leetcode_row(row: pd.Series) -> bool:
        if "platform" in row.index:
            return str(row.get("platform", "")).lower() == "leetcode"
        # fallback: if problem_id is non-numeric or slug-like, assume leetcode (best-effort)
        pid = str(row.get("problem_id", ""))
        return not pid.isdigit()

    leetcode_candidates = candidates[candidates.apply(_is_leetcode_row, axis=1)]
    if leetcode_candidates.empty:
        return []

    texts = [build_problem_text(row) for _, row in leetcode_candidates.iterrows()]
    # encode candidate texts with the same method used for the solved index
    if solved_index.method.startswith("sentence-transformers") and solved_index.model is not None:
        try:
            cand_emb = solved_index.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        except (AttributeError, RuntimeError, TypeError):
            cand_emb = _vectorize_texts(texts, solved_index.vectorizer)
    else:
        cand_emb = _vectorize_texts(texts, solved_index.vectorizer)

    # compute pairwise similarities (candidate x solved)
    sims = []
    for i in range(len(texts)):
        scores = _cosine_scores(np.asarray(cand_emb[i]), np.asarray(solved_index.embeddings))
        max_score = float(np.max(scores)) if scores.size else 0.0
        sims.append(max_score)

    flagged: list[str] = []
    for (idx, row), score in zip(leetcode_candidates.iterrows(), sims):
        if score >= threshold:
            flagged.append(str(row.get("problem_id")))
    return flagged

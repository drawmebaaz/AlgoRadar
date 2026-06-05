from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


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
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return SemanticIndex(
                method="sentence-transformers/all-MiniLM-L6-v2",
                texts=texts,
                problem_ids=problem_ids,
                embeddings=embeddings,
                model=model,
            )
        except Exception:
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

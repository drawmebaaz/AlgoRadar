from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=6000)
    embeddings = vectorizer.fit_transform(texts)
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
        scores = cosine_similarity(query_embedding, index.embeddings).ravel()
    else:
        query_embedding = index.vectorizer.transform([query_text])
        scores = cosine_similarity(query_embedding, index.embeddings).ravel()

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

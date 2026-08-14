from __future__ import annotations

import os
import pickle
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MINILM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CACHE_DIR = Path("data") / "cache"
DEFAULT_BATCH_SIZE = 64

# optional Annoy backend
try:
    from annoy import AnnoyIndex  # type: ignore

    ANN_AVAILABLE = True
except ImportError:
    AnnoyIndex = None  # type: ignore
    ANN_AVAILABLE = False


@dataclass
class SemanticIndex:
    method: str
    texts: list[str]
    problem_ids: list[str]
    embeddings: Any
    vectorizer: Any | None = None
    model: Any | None = None
    annoy_index: Any | None = None


def build_problem_text(problem: pd.Series | dict[str, Any]) -> str:
    name = str(problem.get("name", ""))
    rating = str(int(problem.get("rating", 0) or 0))
    tags = " ".join(problem.get("tags", []) or [])
    return f"{name} rating_{rating} {tags}".strip()


def build_semantic_index(problems: pd.DataFrame, prefer_transformer: bool = True) -> SemanticIndex:
    frame = problems.dropna(subset=["problem_id"]).drop_duplicates("problem_id").copy()
    texts = [build_problem_text(row) for _, row in frame.iterrows()]
    problem_ids = frame["problem_id"].astype(str).tolist()

    # Try to load persisted index if present and matching problem ids
    persisted = load_semantic_index()
    if persisted is not None and persisted.problem_ids == problem_ids:
        return persisted

    if prefer_transformer:
        try:
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            from sentence_transformers import SentenceTransformer

            allow_download = os.environ.get("ALGORADAR_ALLOW_MINILM_DOWNLOAD", "").strip() == "1"
            model = SentenceTransformer(MINILM_MODEL_NAME, local_files_only=not allow_download)
            embeddings = _batch_encode(model, texts, batch_size=DEFAULT_BATCH_SIZE, normalize=True)
            index = SemanticIndex(
                method=MINILM_MODEL_NAME,
                texts=texts,
                problem_ids=problem_ids,
                embeddings=embeddings,
                model=model,
            )
            try:
                save_semantic_index(index)
            except OSError:
                pass
            return index
        except (ImportError, RuntimeError, OSError):
            pass

    vectorizer = _build_vectorizer(texts, max_features=6000)
    embeddings = _vectorize_texts(texts, vectorizer)
    index = SemanticIndex(
        method="tfidf-fallback",
        texts=texts,
        problem_ids=problem_ids,
        embeddings=embeddings,
        vectorizer=vectorizer,
    )
    try:
        save_semantic_index(index)
    except OSError:
        pass
    return index


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


def _batch_encode(model: Any, texts: list[str], batch_size: int = 64, normalize: bool = True) -> np.ndarray:
    """Encode texts using `model.encode` in batches and return stacked numpy array."""
    if not texts:
        return np.zeros((0, 0), dtype=float)
    embeddings_list = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            emb = model.encode(batch, normalize_embeddings=normalize, show_progress_bar=False)
        except TypeError:
            # fallback if model has different signature
            emb = model.encode(batch, show_progress_bar=False)
        embeddings_list.append(np.asarray(emb))
    return np.vstack(embeddings_list)


def save_semantic_index(index: SemanticIndex, dir_path: Path | str = DEFAULT_CACHE_DIR, dtype: Any = np.float32) -> Path:
    """Persist embeddings and metadata to disk. Returns the directory path used."""
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    vec_path = dir_path / "semantic_vectors.npy"
    meta_path = dir_path / "semantic_meta.pkl"
    np.save(vec_path, np.asarray(index.embeddings, dtype=dtype))
    meta = {
        "method": index.method,
        "texts": index.texts,
        "problem_ids": index.problem_ids,
        "vectorizer": index.vectorizer,
    }
    with open(meta_path, "wb") as fh:
        pickle.dump(meta, fh)

    # optional Annoy persistence
    if ANN_AVAILABLE and index.embeddings is not None and getattr(index, "embeddings", None) is not None:
        try:
            dim = int(np.asarray(index.embeddings).shape[1])
            annoy_index = AnnoyIndex(dim, "angular")
            for i, vec in enumerate(np.asarray(index.embeddings)):
                annoy_index.add_item(i, vec.tolist())
            annoy_index.build(10)
            annoy_index.save(str(dir_path / "semantic.ann"))
        except (OSError, ValueError):
            # best-effort: don't fail saving meta if Annoy fails
            pass
    return dir_path


def load_semantic_index(dir_path: Path | str = DEFAULT_CACHE_DIR) -> SemanticIndex | None:
    """Load a persisted semantic index from disk. Returns SemanticIndex or None if not found."""
    dir_path = Path(dir_path)
    vec_path = dir_path / "semantic_vectors.npy"
    meta_path = dir_path / "semantic_meta.pkl"
    if not vec_path.exists() or not meta_path.exists():
        return None
    embeddings = np.load(vec_path)
    with open(meta_path, "rb") as fh:
        meta = pickle.load(fh)
    annoy_obj = None
    ann_file = dir_path / "semantic.ann"
    if ANN_AVAILABLE and ann_file.exists():
        try:
            dim = int(embeddings.shape[1])
            annoy_index = AnnoyIndex(dim, "angular")
            annoy_index.load(str(ann_file))
            annoy_obj = annoy_index
        except (OSError, ValueError):
            annoy_obj = None
    return SemanticIndex(
        method=meta.get("method", "persisted"),
        texts=meta.get("texts", []),
        problem_ids=meta.get("problem_ids", []),
        embeddings=embeddings,
        vectorizer=meta.get("vectorizer"),
        model=None,
        annoy_index=annoy_obj,
    )


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
            cand_emb = _batch_encode(solved_index.model, texts, batch_size=DEFAULT_BATCH_SIZE, normalize=True)
        except (AttributeError, RuntimeError, TypeError):
            cand_emb = _vectorize_texts(texts, solved_index.vectorizer)
    else:
        cand_emb = _vectorize_texts(texts, solved_index.vectorizer)

    # If Annoy is available and an annoy index exists, use ANN search to find nearest solved embedding for each candidate.
    sims: list[float] = []
    if ANN_AVAILABLE and getattr(solved_index, "annoy_index", None) is not None:
        try:
            annoy_idx = solved_index.annoy_index
            for i in range(len(texts)):
                vec = np.asarray(cand_emb[i])
                # Annoy expects lists and angular metric; request a small set of neighbors to check
                neighbors = annoy_idx.get_nns_by_vector(vec.tolist(), 5, include_distances=False)
                if not neighbors:
                    sims.append(0.0)
                    continue
                neighbor_vecs = np.asarray(solved_index.embeddings)[neighbors]
                scores = _cosine_scores(vec, neighbor_vecs)
                max_score = float(np.max(scores)) if scores.size else 0.0
                sims.append(max_score)
        except (OSError, ValueError):
            sims = []

    # fallback: brute-force cosine over all solved embeddings (TF-IDF or dense vectors)
    if not sims:
        for i in range(len(texts)):
            scores = _cosine_scores(np.asarray(cand_emb[i]), np.asarray(solved_index.embeddings))
            max_score = float(np.max(scores)) if scores.size else 0.0
            sims.append(max_score)

    flagged: list[str] = []
    for (idx, row), score in zip(leetcode_candidates.iterrows(), sims):
        if score >= threshold:
            flagged.append(str(row.get("problem_id")))
    return flagged

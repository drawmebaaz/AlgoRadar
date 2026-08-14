from __future__ import annotations

import os
import time

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    start = time.time()
    print(f"Loading {MODEL_NAME}...")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        ["binary search on answer", "dynamic programming graph shortest path"],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    elapsed = time.time() - start
    norm = float((embeddings[0] ** 2).sum() ** 0.5)

    print(f"Loaded in {elapsed:.2f}s")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"First embedding norm: {norm:.4f}")
    print("MiniLM verification passed.")


if __name__ == "__main__":
    main()

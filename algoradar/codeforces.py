from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests

from .config import (
    API_TIMEOUT_SECONDS,
    CACHE_DIR,
    CODEFORCES_BASE_URL,
    DEFAULT_SUBMISSION_LIMIT,
)
from .sample_data import make_sample_bundle


class CodeforcesAPIError(RuntimeError):
    """Raised when the Codeforces API cannot return usable data."""


class CodeforcesClient:
    def __init__(self, cache_dir: Path = CACHE_DIR, timeout: int = API_TIMEOUT_SECONDS) -> None:
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def user_status(self, handle: str, count: int = DEFAULT_SUBMISSION_LIMIT, force_refresh: bool = False) -> list[dict[str, Any]]:
        safe_handle = _safe_cache_key(handle)
        path = self.cache_dir / f"user_status_{safe_handle}_{count}.json"
        return self._cached(path, lambda: self._get("user.status", {"handle": handle, "count": count}), force_refresh)

    def user_rating(self, handle: str, force_refresh: bool = False) -> list[dict[str, Any]]:
        safe_handle = _safe_cache_key(handle)
        path = self.cache_dir / f"user_rating_{safe_handle}.json"
        return self._cached(path, lambda: self._get("user.rating", {"handle": handle}), force_refresh)

    def problemset(self, force_refresh: bool = False) -> dict[str, list[dict[str, Any]]]:
        path = self.cache_dir / "problemset.json"
        return self._cached(path, lambda: self._get("problemset.problems", {}), force_refresh, max_age_seconds=24 * 3600)

    def bundle(
        self,
        handle: str,
        count: int = DEFAULT_SUBMISSION_LIMIT,
        force_refresh: bool = False,
        allow_sample_fallback: bool = True,
    ) -> dict[str, Any]:
        try:
            with ThreadPoolExecutor(max_workers=3) as executor:
                problemset_future = executor.submit(self.problemset, force_refresh=force_refresh)
                submissions_future = executor.submit(self.user_status, handle, count, force_refresh)
                ratings_future = executor.submit(self.user_rating, handle, force_refresh)
                problemset = problemset_future.result()
                submissions = submissions_future.result()
                ratings = ratings_future.result()
            return {
                "handle": handle,
                "source": "codeforces",
                "submissions": submissions,
                "ratings": ratings,
                "problems": problemset.get("problems", []),
                "problem_statistics": problemset.get("problemStatistics", []),
            }
        except Exception as exc:
            if allow_sample_fallback:
                bundle = make_sample_bundle(handle)
                bundle["source"] = f"sample fallback: {exc}"
                return bundle
            raise

    def _get(self, method: str, params: dict[str, Any]) -> Any:
        url = f"{CODEFORCES_BASE_URL}/{method}"
        response = requests.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") != "OK":
            comment = payload.get("comment", "Unknown API error")
            raise CodeforcesAPIError(f"{method} failed: {comment}")
        return payload.get("result")

    def _cached(
        self,
        path: Path,
        loader: Callable[[], Any],
        force_refresh: bool = False,
        max_age_seconds: int = 6 * 3600,
    ) -> Any:
        if not force_refresh and path.exists():
            age = time.time() - path.stat().st_mtime
            if age < max_age_seconds:
                try:
                    return json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    pass

        data = loader()
        path.write_text(json.dumps(data), encoding="utf-8")
        return data


def _safe_cache_key(value: str) -> str:
    normalized = value.strip().lower() or "anonymous"
    return re.sub(r"[^a-z0-9_.-]+", "_", normalized)

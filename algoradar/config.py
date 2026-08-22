from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
MODEL_DIR = DATA_DIR / "models"

CODEFORCES_BASE_URL = "https://codeforces.com/api"
DEFAULT_SUBMISSION_LIMIT = 2500
API_TIMEOUT_SECONDS = 20

for directory in (DATA_DIR, CACHE_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

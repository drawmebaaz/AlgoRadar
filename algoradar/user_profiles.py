from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .config import CACHE_DIR

PROFILE_STORE = CACHE_DIR / "personal_profiles.json"


def load_profiles(path: Path = PROFILE_STORE) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    profiles = raw.get("profiles", raw if isinstance(raw, dict) else {})
    cleaned: dict[str, dict[str, str]] = {}
    for name, handles in profiles.items():
        if not isinstance(handles, dict):
            continue
        cleaned[str(name)] = {
            "codeforces": str(handles.get("codeforces", "") or ""),
            "leetcode": str(handles.get("leetcode", "") or ""),
            "codechef": str(handles.get("codechef", "") or ""),
        }
    return cleaned


def save_profile(name: str, handles: dict[str, Any], path: Path = PROFILE_STORE) -> str:
    safe_name = _clean_name(name)
    profiles = load_profiles(path)
    profiles[safe_name] = {
        "codeforces": str(handles.get("codeforces", "") or "").strip(),
        "leetcode": str(handles.get("leetcode", "") or "").strip(),
        "codechef": str(handles.get("codechef", "") or "").strip(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"profiles": profiles}, indent=2), encoding="utf-8")
    return safe_name


def delete_profile(name: str, path: Path = PROFILE_STORE) -> None:
    profiles = load_profiles(path)
    profiles.pop(name, None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"profiles": profiles}, indent=2), encoding="utf-8")


def _clean_name(name: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(name or "").strip())
    return cleaned[:48] or "My profile"

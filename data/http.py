"""HTTP helper: timeout, retries with exponential backoff, no retry on 4xx."""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

import httpx


class HttpFailure(Exception):
    pass


def get_json(url: str, *, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None,
             timeout: float = 6.0, retries: int = 3, as_text: bool = False) -> Any:
    last: Optional[Exception] = None
    for attempt in range(max(1, retries)):
        try:
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                resp = client.get(url, params=params, headers=headers)
            if 400 <= resp.status_code < 500:
                raise HttpFailure(f"HTTP {resp.status_code} from {url}")  # no retry on 4xx
            resp.raise_for_status()
            return resp.text if as_text else resp.json()
        except HttpFailure:
            raise
        except Exception as exc:  # network error, timeout, 5xx, bad JSON
            last = exc
            if attempt < retries - 1:
                time.sleep(min(0.4 * (2 ** attempt), 2.0))
    raise HttpFailure(f"{url} failed after {retries} attempts: {last}")

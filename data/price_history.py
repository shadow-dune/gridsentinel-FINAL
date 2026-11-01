"""Day-ahead price history from Energy-Charts (Fraunhofer ISE; data Bundesnetzagentur | SMARD.de, CC BY 4.0).

Used to train forecast/predictor.py. Timestamps are written as timezone-aware UTC ISO strings.
Never writes synthetic data: if the download fails, the existing file is left untouched and the caller is told.
"""
from __future__ import annotations

import csv
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Tuple

from config.settings import Settings
from data import http as dhttp

Series = List[Tuple[datetime, float]]  # (UTC timestamp, spot price EUR/kWh)

CITATION = "Energy-Charts (Fraunhofer ISE), data: Bundesnetzagentur | SMARD.de, CC BY 4.0"
_refresh_lock = threading.Lock()
_last_attempt = 0.0


def fetch_day_ahead(s: Settings, start: datetime, end: datetime) -> Series:
    """Day-ahead spot prices for the bidding zone between start and end (inclusive), in EUR/kWh."""
    j = dhttp.get_json(f"{s.ENERGY_CHARTS_URL}/price",
                       params={"bzn": s.BIDDING_ZONE, "start": int(start.timestamp()), "end": int(end.timestamp())},
                       timeout=max(s.HTTP_TIMEOUT_SECONDS, 15.0), retries=s.HTTP_MAX_RETRIES)
    try:
        unix, prices, unit = j["unix_seconds"], j["price"], str(j.get("unit", ""))
    except (KeyError, TypeError) as exc:
        raise dhttp.HttpFailure(f"Energy-Charts price response malformed: {exc}") from exc
    if "MWh" not in unit:
        raise dhttp.HttpFailure(f"Energy-Charts price unit unexpected: {unit!r}")
    out: Series = []
    for u, p in zip(unix, prices):
        if p is None:
            continue
        out.append((datetime.fromtimestamp(int(u), tz=timezone.utc), float(p) / 1000.0))
    return out


def write_csv(path: str | Path, series: Series) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "spot_eur_per_kwh"])
        for ts, v in series:
            w.writerow([ts.isoformat(), f"{v:.6f}"])
    os.replace(tmp, p)


def refresh_history(s: Settings) -> int:
    """Download PRICE_HISTORY_DAYS of history (plus anything already published for tomorrow). Returns row count."""
    now = datetime.now(timezone.utc)
    series = fetch_day_ahead(s, now - timedelta(days=s.PRICE_HISTORY_DAYS), now + timedelta(days=2))
    if len(series) < 96:
        raise dhttp.HttpFailure(f"Energy-Charts returned only {len(series)} price points")
    write_csv(s.PRICE_HISTORY_PATH, series)
    return len(series)


def ensure_fresh(s: Settings, min_retry_s: float = 1800.0) -> None:
    """Refresh the history file if it is missing or older than PRICE_HISTORY_REFRESH_HOURS.

    Failures are swallowed (the predictor then reports PredictionUnavailable on its own); retries are rate-limited.
    """
    global _last_attempt
    p = Path(s.PRICE_HISTORY_PATH)
    if p.exists() and time.time() - p.stat().st_mtime < s.PRICE_HISTORY_REFRESH_HOURS * 3600:
        return
    with _refresh_lock:
        if time.monotonic() - _last_attempt < min_retry_s and _last_attempt:
            return
        _last_attempt = time.monotonic()
        try:
            refresh_history(s)
        except Exception:
            pass

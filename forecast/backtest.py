"""Backtest on REAL historical data (Open-Meteo archive; UNVERIFIED endpoint - test in hour 0).

Compares two simple 12-hour ambient-temperature predictors against persistence using hourly history.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List

import numpy as np

from config.settings import Settings
from data.http import HttpFailure, get_json

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
UNAVAILABLE = "Backtest unavailable: not enough historical data."


def mae(a: List[float], b: List[float]) -> float:
    return float(np.mean(np.abs(np.array(a) - np.array(b))))


def evaluate(series: List[float], horizon: int = 12, days_back: int = 3) -> Dict[str, float]:
    """Walk-forward on an hourly series. Origins every 6 h. Models: persistence, same-hour-yesterday, mean of previous 3 days."""
    errs: Dict[str, List[float]] = {"persistence": [], "same_hour_yesterday": [], "mean_prev_3_days": []}
    y = np.array(series, dtype=float)
    for o in range(24 * days_back, len(y) - horizon, 6):
        truth = y[o + 1:o + 1 + horizon]
        errs["persistence"].append(mae(truth, [y[o]] * horizon))
        errs["same_hour_yesterday"].append(mae(truth, y[o + 1 - 24:o + 1 - 24 + horizon]))
        errs["mean_prev_3_days"].append(mae(truth, np.mean([y[o + 1 - 24 * d:o + 1 - 24 * d + horizon] for d in (1, 2, 3)], axis=0)))
    if not errs["persistence"]:
        raise ValueError("not enough history")
    return {k: round(float(np.mean(v)), 3) for k, v in errs.items()}


def run_backtest(s: Settings, days: int = 21) -> Dict[str, object]:
    try:
        end = date.today() - timedelta(days=3)
        d = get_json(ARCHIVE_URL, params={"latitude": s.FACILITY_LAT, "longitude": s.FACILITY_LON, "hourly": "temperature_2m",
                                          "start_date": (end - timedelta(days=days)).isoformat(), "end_date": end.isoformat(),
                                          "timezone": "UTC"}, timeout=s.HTTP_TIMEOUT_SECONDS, retries=s.HTTP_MAX_RETRIES)
        series = [v for v in d["hourly"]["temperature_2m"]]
        if any(v is None for v in series) or len(series) < 24 * 8:
            raise ValueError("incomplete history")
        res = evaluate(series)
    except (HttpFailure, KeyError, ValueError, TypeError):
        return {"available": False, "message": UNAVAILABLE, "mae_c": None}
    best = min(res, key=res.get)
    return {"available": True, "message": None, "metric": "MAE (C) over 12 h horizon", "mae_c": res,
            "best_model": best, "beats_persistence": res[best] < res["persistence"], "history_hours": len(series),
            "note": "Control uses the provider forecast directly (L3); this backtest documents how simple models compare to persistence."}

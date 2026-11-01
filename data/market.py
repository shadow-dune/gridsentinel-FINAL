"""Market sources for Germany (bidding zone DE-LU): day-ahead spot prices and grid frequency from Energy-Charts.

Price per 15-min step = day-ahead spot (EPEX auction result, published ~12:45 CET for the next day) + a fixed
adder for grid fees, taxes and levies (constants: market.price_adder_eur_per_kwh). Steps whose auction result is
not yet published are filled by the price model (price_kind=forecast_ml) or, if the model is unavailable, by the
published price at the same time yesterday (price_kind=forecast_yesterday). Every point carries its price_kind.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config.settings import Constants, Settings
from data import http as dhttp
from data.price_history import CITATION, Series, ensure_fresh, fetch_day_ahead
from data.schemas import GridSignal, MarketData, MarketPoint, PriceKind, SignalKind
from forecast.grid import floor15, step_times

NOMINAL_HZ = 50.0
HISTORY_DAYS = 7

_cache_lock = threading.Lock()
_cache: Dict[str, Tuple[float, datetime, Series]] = {}  # zone -> (monotonic fetch time, horizon start, series)


def _cached_series(s: Settings, start: datetime) -> Series:
    """7 days of history + everything published up to the end of the horizon. Cached for PRICE_CACHE_TTL_SECONDS."""
    with _cache_lock:
        hit = _cache.get(s.BIDDING_ZONE)
        if hit and time.monotonic() - hit[0] < s.PRICE_CACHE_TTL_SECONDS and hit[1] == start:
            return hit[2]
    series = fetch_day_ahead(s, start - timedelta(days=HISTORY_DAYS), start + timedelta(hours=13))
    if not series:
        raise dhttp.HttpFailure("Energy-Charts returned no day-ahead prices")
    with _cache_lock:
        _cache[s.BIDDING_ZONE] = (time.monotonic(), start, series)
    return series


def _lookup(table: Dict[datetime, float], ts: datetime, hourly: bool) -> Optional[float]:
    """Exact 15-min price; for an hourly series (pre-Oct-2025 data, some zones) the hourly price covering ts."""
    if ts in table:
        return table[ts]
    return table.get(ts.replace(minute=0)) if hourly else None


def _is_hourly(series: Series) -> bool:
    return not any(ts.minute for ts, _ in series)


def build_market(s: Settings, K: Constants, start: datetime, series: Series, predictor: Any = None) -> MarketData:
    """Pure: turn a raw price series into the 48-step MarketData for the horizon starting at `start`."""
    grid = step_times(start)
    table = {ts: v for ts, v in series}
    hourly = _is_hourly(series)
    adder = float(K.v("market.price_adder_eur_per_kwh"))

    spots: List[Optional[float]] = [_lookup(table, t, hourly) for t in grid]
    kinds: List[PriceKind] = [PriceKind.day_ahead if v is not None else PriceKind.forecast_ml for v in spots]
    if all(v is None for v in spots):
        raise dhttp.HttpFailure("no published day-ahead price covers the current horizon")

    missing = [k for k, v in enumerate(spots) if v is None]
    if missing:
        ml: Optional[List[float]] = None
        if predictor is not None:
            try:
                ml = predictor.predict([grid[k] for k in missing]).prices
            except Exception:
                ml = None
        for j, k in enumerate(missing):
            if ml is not None:
                spots[k], kinds[k] = ml[j], PriceKind.forecast_ml
                continue
            y = _lookup(table, grid[k] - timedelta(days=1), hourly)
            if y is None:
                raise dhttp.HttpFailure(f"no price for {grid[k].isoformat()}: not yet published and no forecast available")
            spots[k], kinds[k] = y, PriceKind.forecast_yesterday

    hist = [v for ts, v in series if start - timedelta(days=HISTORY_DAYS) <= ts < start]
    ref = hist if len(hist) >= 96 else [v for v, k in zip(spots, kinds) if k == PriceKind.day_ahead]
    threshold = float(np.percentile(ref, K.v("stress.s1_price_percentile")))

    pts = [MarketPoint(ts=t, spot_eur_per_kwh=round(float(v), 6), price_eur_per_kwh=round(float(v) + adder, 6),
                       price_kind=kind, is_peak_window=float(v) >= threshold)
           for t, v, kind in zip(grid, spots, kinds)]
    return MarketData(points=pts, bidding_zone=s.BIDDING_ZONE, source_citation=CITATION,
                      source_url=f"{s.ENERGY_CHARTS_URL}/price?bzn={s.BIDDING_ZONE}",
                      adder_eur_per_kwh=adder, s1_threshold_eur_per_kwh=round(threshold, 6))


def source_label(m: MarketData) -> str:
    n_ml = sum(p.price_kind == PriceKind.forecast_ml for p in m.points)
    n_y = sum(p.price_kind == PriceKind.forecast_yesterday for p in m.points)
    extra = []
    if n_ml:
        extra.append(f"{n_ml} steps ML forecast")
    if n_y:
        extra.append(f"{n_y} steps same-time-yesterday")
    return f"EPEX day-ahead {m.bidding_zone} via Energy-Charts" + (f" ({', '.join(extra)}; not yet published)" if extra else "")


def fetch_market(s: Settings, K: Constants, predictor: Any = None,
                 now: Optional[datetime] = None) -> Tuple[MarketData, str, str, SignalKind]:
    start = floor15(now or datetime.now(timezone.utc))
    series = _cached_series(s, start)
    table, hourly = dict(series), _is_hourly(series)
    if predictor is not None and any(_lookup(table, t, hourly) is None for t in step_times(start)):
        ensure_fresh(s)
    m = build_market(s, K, start, series, predictor)
    return m, source_label(m), m.source_url, SignalKind.PUBLISHED_SCHEDULE


def fetch_grid_signal(s: Settings) -> Tuple[GridSignal, str, str, SignalKind]:
    """Continental-Europe grid frequency; reported as the deficit below 50 Hz in mHz (positive = generation short)."""
    now = datetime.now(timezone.utc)
    j = dhttp.get_json(f"{s.ENERGY_CHARTS_URL}/frequency",
                       params={"region": s.FREQUENCY_REGION, "start": int(now.timestamp()) - 300, "end": int(now.timestamp())},
                       timeout=s.HTTP_TIMEOUT_SECONDS, retries=s.HTTP_MAX_RETRIES)
    try:
        pairs = [(u, f) for u, f in zip(j["unix_seconds"], j["data"]) if f is not None]
    except (KeyError, TypeError) as exc:
        raise dhttp.HttpFailure(f"Energy-Charts frequency response malformed: {exc}") from exc
    if not pairs:
        raise dhttp.HttpFailure("Energy-Charts returned no frequency samples in the last 5 minutes")
    u, f = pairs[-1]
    f = float(f)
    if not 49.0 <= f <= 51.0:
        raise dhttp.HttpFailure(f"implausible frequency {f} Hz")
    sig = GridSignal(name="grid_frequency_deficit_mhz", value=round((NOMINAL_HZ - f) * 1000.0, 2), unit="mHz",
                     observed_at=datetime.fromtimestamp(int(u), tz=timezone.utc))
    return sig, f"Grid frequency {s.FREQUENCY_REGION} via Energy-Charts ({f:.3f} Hz)", f"{s.ENERGY_CHARTS_URL}/frequency", SignalKind.MEASURED

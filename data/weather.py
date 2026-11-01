"""Weather clients. Open-Meteo (primary, no key) and OpenWeatherMap (fallback, key).

UNVERIFIED (test in hour 0): the Open-Meteo `minutely_15` parameters; for regions without native
15-minute models Open-Meteo may itself interpolate hourly data. No radiation is invented for the fallback.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from config.settings import Settings
from data.http import HttpFailure, get_json
from data.schemas import SignalKind, WeatherForecast, WeatherPoint
from forecast.grid import floor15, interp_to_grid, step_times, wet_bulb_stull

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OWM_URL = "https://api.openweathermap.org/data/2.5/forecast"


def _parse_utc(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc) if "+" not in s and not s.endswith("Z") \
        else datetime.fromisoformat(s.replace("Z", "+00:00"))


def _build(grid: List[datetime], temps: List[float], rhs: List[float], rads: List[Any], interp: bool) -> WeatherForecast:
    pts = [WeatherPoint(ts=t, temp_dry_c=round(tc, 2), rh_pct=round(min(100.0, max(0.0, rh)), 2),
                        temp_wet_c=round(wet_bulb_stull(tc, rh), 2),
                        solar_w_m2=None if r is None else max(0.0, round(float(r), 2)), interpolated=interp)
           for t, tc, rh, r in zip(grid, temps, rhs, rads)]
    return WeatherForecast(points=pts)


def fetch_open_meteo(s: Settings, now: datetime | None = None) -> Tuple[WeatherForecast, str, str, SignalKind]:
    now = now or datetime.now(timezone.utc)
    start = floor15(now)
    grid = step_times(start)
    base = {"latitude": s.FACILITY_LAT, "longitude": s.FACILITY_LON, "timezone": "UTC"}
    cols = "temperature_2m,relative_humidity_2m,shortwave_radiation"
    # 1) native 15-minute series
    try:
        d = get_json(OPEN_METEO_URL, params={**base, "minutely_15": cols, "forecast_minutely_15": 96},
                     timeout=s.HTTP_TIMEOUT_SECONDS, retries=s.HTTP_MAX_RETRIES)["minutely_15"]
        times = [_parse_utc(t) for t in d["time"]]
        idx = {t: i for i, t in enumerate(times)}
        if all(t in idx for t in grid):
            sel = [idx[t] for t in grid]
            temps, rhs, rads = ([d[k][i] for i in sel] for k in ("temperature_2m", "relative_humidity_2m", "shortwave_radiation"))
            if all(v is not None for v in temps + rhs):
                return _build(grid, temps, rhs, rads, False), "Open-Meteo minutely_15", OPEN_METEO_URL, SignalKind.MEASURED
        raise ValueError("minutely_15 series incomplete for the 12-hour grid")
    except (HttpFailure, KeyError, ValueError, TypeError):
        pass
    # 2) hourly series interpolated to 15 minutes (flagged interpolated=True)
    d = get_json(OPEN_METEO_URL, params={**base, "hourly": cols, "forecast_days": 2},
                 timeout=s.HTTP_TIMEOUT_SECONDS, retries=s.HTTP_MAX_RETRIES)["hourly"]
    times = [_parse_utc(t) for t in d["time"]]
    if any(v is None for v in d["temperature_2m"] + d["relative_humidity_2m"]):
        raise ValueError("hourly series contains nulls")
    temps = interp_to_grid(times, d["temperature_2m"], grid)
    rhs = interp_to_grid(times, d["relative_humidity_2m"], grid)
    rads = interp_to_grid(times, [0.0 if v is None else v for v in d["shortwave_radiation"]], grid)
    return _build(grid, temps, rhs, rads, True), "Open-Meteo hourly (interpolated to 15 min)", OPEN_METEO_URL, SignalKind.MEASURED


def fetch_openweather(s: Settings, now: datetime | None = None) -> Tuple[WeatherForecast, str, str, SignalKind]:
    if not s.OPENWEATHER_API_KEY:
        raise HttpFailure("OPENWEATHER_API_KEY not configured")
    now = now or datetime.now(timezone.utc)
    grid = step_times(floor15(now))
    d: Dict[str, Any] = get_json(OWM_URL, params={"lat": s.FACILITY_LAT, "lon": s.FACILITY_LON,
                                                  "appid": s.OPENWEATHER_API_KEY, "units": "metric"},
                                 timeout=s.HTTP_TIMEOUT_SECONDS, retries=s.HTTP_MAX_RETRIES)
    items = d["list"]
    times = [datetime.fromtimestamp(int(i["dt"]), tz=timezone.utc) for i in items]
    temps = interp_to_grid(times, [float(i["main"]["temp"]) for i in items], grid)
    rhs = interp_to_grid(times, [float(i["main"]["humidity"]) for i in items], grid)
    return _build(grid, temps, rhs, [None] * len(grid), True), "OpenWeatherMap 3-hourly (interpolated)", OWM_URL, SignalKind.MEASURED

"""Test fixtures. Inputs built here are TEST-ONLY and never imported by runtime code."""
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def make_price_series(start=None, horizon_steps=48, peak=(20, 28)):
    """TEST-ONLY synthetic day-ahead series: 7 days of history with a daily wave, then a horizon with one price peak."""
    from datetime import timedelta
    from forecast.grid import floor15

    start = start or floor15(datetime.now(timezone.utc))
    out = []
    for i in range(7 * 96, 0, -1):
        t = start - timedelta(minutes=15 * i)
        out.append((t, 0.10 + 0.05 * float(np.sin(2 * np.pi * (t.hour * 4 + t.minute // 15) / 96))))
    for k in range(horizon_steps):
        out.append((start + timedelta(minutes=15 * k), 0.30 if peak[0] <= k < peak[1] else 0.08))
    return start, out


@pytest.fixture
def env(tmp_path, monkeypatch):
    for k, v in {
        "FACILITY_LAT": "50.11",
        "FACILITY_LON": "8.68",
        "OPERATOR_TOKEN": "test-token-123",
        "DB_PATH": str(tmp_path / "t.db"),
        "PRICE_HISTORY_PATH": str(tmp_path / "price_history.csv"),
        "ENERGY_CHARTS_URL": "https://energy-charts.invalid",
        "FACILITY_TIMEZONE": "Europe/Berlin",
        "AUTO_CYCLE": "false",
        "HTTP_MAX_RETRIES": "1",
        "HTTP_TIMEOUT_SECONDS": "1",
        "LLM_API_KEY": "",
        "LLM_MODEL": "",
        "OPENWEATHER_API_KEY": "",
        "ALERT_FEED_URL": "",
    }.items():
        monkeypatch.setenv(k, v)

    # Simulate the offline test sandbox expected by test_ac5_cold_start_and_e3
    # (blocks real outbound HTTP calls to Open-Meteo while leaving TestClient WSGI/ASGI untouched)
    import data.http as dhttp
    import data.weather as dweather

    def _offline(*_a, **_kw):
        raise dhttp.HttpFailure("sandbox: outbound network disabled in unit tests")

    for attr in ("get_json", "get_text", "get", "fetch"):
        if hasattr(dhttp, attr):
            monkeypatch.setattr(dhttp, attr, _offline)
        if hasattr(dweather, attr):
            monkeypatch.setattr(dweather, attr, _offline)

    monkeypatch.setattr(
        httpx,
        "get",
        lambda *_a, **_kw: (_ for _ in ()).throw(httpx.ConnectError("offline sandbox")),
        raising=False,
    )

    from config.settings import get_settings

    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


def make_weather(offset: float = 0.0):
    """TEST-ONLY synthetic forecast used to exercise the pipeline offline."""
    from data.schemas import WeatherForecast, WeatherPoint
    from forecast.grid import floor15, step_times, wet_bulb_stull

    pts = []
    for k, t in enumerate(step_times(floor15(datetime.now(timezone.utc)))):
        td = 31.0 + offset + 7.0 * np.sin(np.pi * k / 48)
        pts.append(
            WeatherPoint(
                ts=t,
                temp_dry_c=float(td),
                rh_pct=60.0,
                temp_wet_c=wet_bulb_stull(td, 60.0),
                solar_w_m2=None,
            )
        )
    return WeatherForecast(points=pts)


@pytest.fixture
def service(env):
    from api.main import build_service
    from data.schemas import SignalKind

    from data import market as market_mod

    svc = build_service()

    def _market():
        start, series = make_price_series()
        m = market_mod.build_market(svc.d.settings, svc.d.K, start, series, None)
        return m, "TEST day-ahead", "test://prices", SignalKind.PUBLISHED_SCHEDULE

    svc.d.ingestor.chains["market"] = [_market]
    svc.d.ingestor.chains["weather"] = [
        lambda: (make_weather(), "TEST weather", "test://weather", SignalKind.MEASURED)
    ]
    svc.d.ingestor.chains["alerts"] = [lambda: (_ for _ in ()).throw(RuntimeError("no feed in tests"))]
    svc.d.ingestor.chains["grid_signal"] = [lambda: (_ for _ in ()).throw(RuntimeError("not configured"))]
    return svc


@pytest.fixture
def client(service):
    from api.main import create_app
    from fastapi.testclient import TestClient

    return TestClient(create_app(service))


HDR = {"X-Operator-Token": "test-token-123"}
"""Runtime settings (environment) and the documented physical-constants loader."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple
from zoneinfo import ZoneInfo

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Environment-backed settings. Required: FACILITY_LAT/LON/TIMEZONE, OPERATOR_TOKEN."""

    FACILITY_LAT: float = Field(ge=-90, le=90)
    FACILITY_LON: float = Field(ge=-180, le=180)
    FACILITY_TIMEZONE: str
    FACILITY_REGION_NAME: str = ""
    OPERATOR_TOKEN: str = Field(min_length=8)

    OPENWEATHER_API_KEY: str = ""
    ALERT_FEED_URL: str = ""

    # Market: day-ahead prices + grid frequency from Energy-Charts (Fraunhofer ISE; no key needed)
    BIDDING_ZONE: str = "DE-LU"
    FREQUENCY_REGION: str = "DE-Freiburg"
    ENERGY_CHARTS_URL: str = "https://api.energy-charts.info"
    PRICE_CACHE_TTL_SECONDS: int = 900
    PRICE_HISTORY_DAYS: int = 35
    PRICE_HISTORY_REFRESH_HOURS: float = 6.0

    LLM_PROVIDER: str = "groq"
    LLM_API_KEY: str = ""
    LLM_MODEL: str = ""
    LLM_BASE_URL: str = ""

    HTTP_TIMEOUT_SECONDS: float = 6.0
    HTTP_MAX_RETRIES: int = 3
    CYCLE_INTERVAL_MINUTES: int = 15
    MAX_AGENT_ITERATIONS: int = 8
    STALE_MARGIN_C: float = 1.0
    MAX_AGE_WEATHER_MIN: int = 90
    MAX_AGE_MARKET_MIN: int = 180
    MAX_AGE_ALERTS_MIN: int = 360
    MAX_AGE_TELEMETRY_S: int = 60

    AUTO_CYCLE: bool = True
    DASHBOARD_ORIGIN: str = "http://localhost:8501"
    API_BASE_URL: str = "http://127.0.0.1:8000"
    DB_PATH: str = str(ROOT / ".data" / "gridsentinel.db")
    PRICE_HISTORY_PATH: str = str(ROOT / ".data" / "price_history.csv")
    CONSTANTS_PATH: str = str(ROOT / "config" / "constants.yaml")
    DASHBOARD_REFRESH_SECONDS: int = 300

    model_config = SettingsConfigDict(env_file=str(ROOT / ".env"), env_file_encoding="utf-8", extra="ignore")

    @field_validator("FACILITY_TIMEZONE")
    @classmethod
    def _tz(cls, v: str) -> str:
        ZoneInfo(v)  # raises if invalid
        return v

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.FACILITY_TIMEZONE)

    def max_age_seconds(self, source: str) -> int:
        return {
            "weather": self.MAX_AGE_WEATHER_MIN * 60,
            "market": self.MAX_AGE_MARKET_MIN * 60,
            "grid_signal": self.MAX_AGE_MARKET_MIN * 60,
            "alerts": self.MAX_AGE_ALERTS_MIN * 60,
            "telemetry": self.MAX_AGE_TELEMETRY_S,
        }[source]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]


@dataclass(frozen=True)
class Constant:
    name: str
    value: Any
    unit: str
    source: str
    status: str  # cited | derived | assumed


class Constants:
    """Physical/model constants. Every entry MUST carry value, unit, source, status (AC3)."""

    VALID = {"cited", "derived", "assumed"}

    def __init__(self, raw: Dict[str, Dict[str, Dict[str, Any]]]):
        self._items: Dict[str, Constant] = {}
        for group, entries in raw.items():
            for key, spec in entries.items():
                name = f"{group}.{key}"
                missing = [f for f in ("value", "unit", "source", "status") if f not in spec]
                if missing or not str(spec.get("source", "")).strip() or spec["status"] not in self.VALID:
                    raise ValueError(f"Constant {name} must define value, unit, source, status (cited|derived|assumed)")
                self._items[name] = Constant(name, spec["value"], spec["unit"], spec["source"], spec["status"])

    def v(self, name: str) -> Any:
        return self._items[name].value

    def items(self) -> Iterator[Tuple[str, Constant]]:
        return iter(self._items.items())

    def assumed(self) -> list[Constant]:
        return [c for c in self._items.values() if c.status == "assumed"]


def load_constants(path: str | Path | None = None) -> Constants:
    p = Path(path) if path else ROOT / "config" / "constants.yaml"
    with open(p, "r", encoding="utf-8") as f:
        return Constants(yaml.safe_load(f))
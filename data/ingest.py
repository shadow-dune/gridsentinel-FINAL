"""Ingestion with fallback chain, last-known-good (LKG) and max-age rules (spec L1, E1-E5)."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel, ValidationError

from config.settings import Constants, Settings
from data import alerts as alerts_mod
from data import market as market_mod
from data import telemetry as tel_mod
from data import weather as weather_mod
from forecast.predictor import PredictiveModel
from data.schemas import (AlertFeed, GridSignal, ItLoadSample, MarketData, Provenance, SignalKind, SourceState,
                          Status, WeatherForecast)

Fetch = Callable[[], Tuple[BaseModel, str, str, SignalKind]]

SOURCES = ["weather", "market", "grid_signal", "alerts", "telemetry"]
MODELS: Dict[str, Type[BaseModel]] = {"weather": WeatherForecast, "market": MarketData, "grid_signal": GridSignal,
                                      "alerts": AlertFeed, "telemetry": ItLoadSample}


class Inputs:
    """Snapshot of all sources for one cycle."""

    def __init__(self, states: Dict[str, SourceState], notices: List[str]) -> None:
        self.states, self.notices = states, notices
        self.fetched_at = datetime.now(timezone.utc)

    def get(self, name: str) -> Optional[BaseModel]:
        st = self.states[name]
        return MODELS[name].model_validate(st.data) if st.data is not None and st.provenance.status != Status.FAILED else None

    def status(self, name: str) -> Status:
        return self.states[name].provenance.status

    def any_stale(self) -> bool:
        return any(self.status(n) == Status.STALE for n in ("weather", "market", "telemetry"))

    def to_dict(self) -> Dict[str, Any]:
        return {n: st.model_dump(mode="json") for n, st in self.states.items()}

    def health(self, s: Settings) -> List[Dict[str, Any]]:
        now = datetime.now(timezone.utc)
        out = []
        for n, st in self.states.items():
            p = st.provenance
            out.append({"source": n, "source_name": p.source_name, "status": p.status.value, "signal_kind": p.signal_kind.value,
                        "latency_ms": p.latency_ms, "last_success": None if p.status == Status.FAILED and st.data is None else p.fetched_at.isoformat(),
                        "age_s": None if st.data is None else round((now - p.fetched_at).total_seconds(), 1),
                        "max_age_s": s.max_age_seconds(n), "stale_reason": p.stale_reason, "notice": st.notice})
        return out


class Ingestor:
    def __init__(self, settings: Settings, constants: Constants, store) -> None:
        self.s, self.K, self.store = settings, constants, store
        self.predictor = PredictiveModel(csv_path=settings.PRICE_HISTORY_PATH, tz=settings.FACILITY_TIMEZONE)
        # Overridable per source (used by tests); defaults are the live clients.
        self.chains: Dict[str, List[Fetch]] = {
            "weather": [lambda: weather_mod.fetch_open_meteo(self.s), lambda: weather_mod.fetch_openweather(self.s)],
            "market": [lambda: market_mod.fetch_market(self.s, self.K, self.predictor)],
            "grid_signal": [lambda: market_mod.fetch_grid_signal(self.s)],
            "alerts": [lambda: alerts_mod.fetch_alerts(self.s)],
            "telemetry": [lambda: tel_mod.sample_telemetry(self.K)],
        }

    def _resolve(self, name: str, notices: List[str]) -> SourceState:
        errors: List[str] = []
        for i, fn in enumerate(self.chains[name]):
            t0 = time.perf_counter()
            try:
                model, src_name, src_url, kind = fn()
                MODELS[name].model_validate(model.model_dump(mode="json"))  # schema gate (E4)
            except ValidationError as exc:
                errors.append(f"{src_name_safe(fn)}: incomplete data ({exc.error_count()} validation errors)")
                notices.append("Source returned incomplete data and was rejected.")
                continue
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                continue
            status = Status.LIVE if i == 0 else Status.FALLBACK_LIVE
            prov = Provenance(source_name=src_name, source_url=src_url, fetched_at=datetime.now(timezone.utc), status=status,
                              signal_kind=kind, latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                              stale_reason=(f"Primary failed: {errors[-1]}" if i and errors else None))
            data = model.model_dump(mode="json")
            self.store.save_lkg(name, data, prov.model_dump(mode="json"))
            if name == "weather" and i > 0:
                notices.append("Primary weather source unavailable. Using backup source.")  # E1
            if name == "grid_signal" and isinstance(model, GridSignal):
                self.store.add_grid_sample(model.name, model.observed_at, model.value)
            return SourceState(name=name, provenance=prov, data=data)
        return self._from_lkg(name, "; ".join(errors) or "no source available", notices)

    def _from_lkg(self, name: str, reason: str, notices: List[str]) -> SourceState:
        now = datetime.now(timezone.utc)
        lkg = self.store.load_lkg(name)
        max_age = self.s.max_age_seconds(name)
        if lkg:
            payload, prov_d = lkg
            try:
                MODELS[name].model_validate(payload)  # LKG written by an older schema is not usable
                prov = Provenance.model_validate(prov_d)
            except ValidationError:
                lkg = None
        if lkg:
            age = (now - prov.fetched_at).total_seconds()
            if age <= max_age:  # STALE: older than live but inside max age
                prov = prov.model_copy(update={"status": Status.STALE, "stale_reason": reason})
                if name in ("weather", "market", "telemetry"):
                    notices.append(f"Showing last live data from {prov.fetched_at.isoformat()}. "
                                   f"Safety margin widened by {self.s.STALE_MARGIN_C} °C.")  # E2
                return SourceState(name=name, provenance=prov, data=payload)
            reason = f"{reason}; last good response is {int(age)} s old (max {max_age} s)"  # AC6: FAILED
        prov = Provenance(source_name=f"{name} (no usable data)", source_url="n/a", fetched_at=now, status=Status.FAILED,
                          signal_kind=SignalKind.PUBLISHED_SCHEDULE if name == "market" else SignalKind.MEASURED,
                          latency_ms=0.0, stale_reason=reason)
        notice = None
        if name in ("weather", "market"):
            notice = f"No usable {name} data. Holding current safe state."  # E3
        if name == "telemetry":
            notice = "IT load telemetry unavailable. No constant was substituted."  # E5
        if notice:
            notices.append(notice)
        return SourceState(name=name, provenance=prov, data=None, notice=notice)

    def fetch_all(self) -> Inputs:
        notices: List[str] = []
        states = {n: self._resolve(n, notices) for n in SOURCES}
        return Inputs(states, list(dict.fromkeys(notices)))


def src_name_safe(fn: Callable) -> str:
    return getattr(fn, "__name__", "source")

"""Per-cycle context shared by agent tools, plus the dependency bundle."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set

from config.settings import Constants, Settings
from data.ingest import Ingestor, Inputs
from data.schemas import (Action, Citation, Constraint, CycleStatus, ItLoadSample, MarketData, Metrics, Mode, Plan,
                          Status, TraceStep, WeatherForecast)
from data.store import Store
from data.telemetry import sample_telemetry
from optimizer.mpc import Horizon
from twin.model import DigitalTwin, Plant


@dataclass
class Deps:
    settings: Settings
    K: Constants
    plant: Plant
    store: Store
    twin: DigitalTwin
    ingestor: Ingestor
    llm: Any
    sample_it_load: Optional[Callable[[], ItLoadSample]] = None  # overridable (tests); default = live psutil

    def it_sample(self) -> ItLoadSample:
        if self.sample_it_load:
            return self.sample_it_load()
        return sample_telemetry(self.K)[0]


@dataclass
class CycleContext:
    deps: Deps
    mode: Mode
    stress_offset_c: float = 0.0
    cycle_id: str = field(default_factory=lambda: "cyc_" + uuid.uuid4().hex[:10])
    started: float = field(default_factory=time.perf_counter)
    inputs: Optional[Inputs] = None
    weather: Optional[WeatherForecast] = None
    market: Optional[MarketData] = None
    telemetry: Optional[ItLoadSample] = None
    horizon: Optional[Horizon] = None
    constraints: List[Constraint] = field(default_factory=list)
    stress_windows: List[Dict[str, Any]] = field(default_factory=list)
    plan: Optional[Plan] = None
    baseline: Optional[Metrics] = None
    extras: Dict[str, Any] = field(default_factory=dict)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    trace: List[TraceStep] = field(default_factory=list)
    notices: List[str] = field(default_factory=list)
    deterministic: bool = False
    status: CycleStatus = CycleStatus.running
    done: Set[str] = field(default_factory=set)
    revision: int = 0
    iterations: int = 0
    capped: bool = False
    finished: bool = False

    @property
    def counterfactual(self) -> bool:
        return self.stress_offset_c > 0

    def notice(self, msg: str) -> None:
        if msg not in self.notices:
            self.notices.append(msg)

    def default_citations(self) -> List[Citation]:
        cites = []
        if self.inputs:
            for n, st in self.inputs.states.items():
                if st.provenance.status != Status.FAILED:
                    cites.append(Citation(source_name=st.provenance.source_name, ts=st.provenance.fetched_at))
        return cites or [Citation(source_name="none (no live data available)", ts=datetime.now(timezone.utc))]

    def add(self, tool: str, observed: str, decided: str, why: str, result: str,
            cites: Optional[List[Citation]] = None) -> None:
        self.trace.append(TraceStep(n=len(self.trace) + 1, observed=observed, decided=decided, why=why, tool=tool,
                                    tool_result_summary=result, citations=cites or self.default_citations()))

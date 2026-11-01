"""Pydantic schemas (spec Q5). All timestamps are timezone-aware UTC."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class Status(str, Enum):
    LIVE = "LIVE"
    FALLBACK_LIVE = "FALLBACK_LIVE"
    STALE = "STALE"
    FAILED = "FAILED"


class SignalKind(str, Enum):
    MEASURED = "MEASURED"
    PUBLISHED_SCHEDULE = "PUBLISHED_SCHEDULE"
    MODEL_DERIVED = "MODEL_DERIVED"


class Mode(str, Enum):
    autonomous = "autonomous"
    approval_required = "approval_required"


class CycleStatus(str, Enum):
    running = "running"
    completed = "completed"
    held_safe = "held_safe"
    failed = "failed"


class ActionStatus(str, Enum):
    proposed = "proposed"
    rejected = "rejected"
    awaiting_approval = "awaiting_approval"
    applied = "applied"
    rolled_back = "rolled_back"
    failed = "failed"


class AlertType(str, Enum):
    heatwave = "heatwave"
    power_cut = "power_cut"
    grid_advisory = "grid_advisory"
    water_restriction = "water_restriction"
    none = "none"


class Provenance(BaseModel):
    source_name: str
    source_url: str
    fetched_at: datetime
    status: Status
    signal_kind: SignalKind
    latency_ms: float = Field(ge=0)
    stale_reason: Optional[str] = None


class WeatherPoint(BaseModel):
    ts: datetime
    temp_dry_c: float = Field(ge=-40, le=65)
    rh_pct: float = Field(ge=0, le=100)
    temp_wet_c: float
    solar_w_m2: Optional[float] = Field(default=None, ge=0)  # null when the source has no radiation
    interpolated: bool = False


class WeatherForecast(BaseModel):
    points: List[WeatherPoint]
    derived_fields: List[str] = ["temp_wet_c"]  # MODEL_DERIVED

    @field_validator("points")
    @classmethod
    def _48(cls, v: List[WeatherPoint]) -> List[WeatherPoint]:
        if len(v) != 48:
            raise ValueError("weather forecast must contain exactly 48 steps")
        return v


class PriceKind(str, Enum):
    day_ahead = "day_ahead"                      # published exchange result (EPEX day-ahead auction)
    forecast_ml = "forecast_ml"                  # not yet published: price model (MODEL_DERIVED)
    forecast_yesterday = "forecast_yesterday"    # not yet published, model unavailable: same time yesterday


class MarketPoint(BaseModel):
    ts: datetime
    price_eur_per_kwh: float          # all-in = spot + adder; may be negative when the spot price is deeply negative
    spot_eur_per_kwh: float
    price_kind: PriceKind
    is_peak_window: bool              # S1: spot at/above the s1 percentile of the previous 7 days


class MarketData(BaseModel):
    points: List[MarketPoint]
    bidding_zone: str
    source_citation: str
    source_url: str
    adder_eur_per_kwh: float
    s1_threshold_eur_per_kwh: float

    @field_validator("points")
    @classmethod
    def _48(cls, v: List[MarketPoint]) -> List[MarketPoint]:
        if len(v) != 48:
            raise ValueError("market data must contain exactly 48 steps")
        return v


class GridSignal(BaseModel):
    name: str
    value: float
    unit: str
    observed_at: datetime


class ItLoadSample(BaseModel):
    ts: datetime
    cpu_pct: float = Field(ge=0, le=100)
    mem_pct: float = Field(ge=0, le=100)
    it_load_kw: float = Field(gt=0)
    scaling_formula: str
    host_label: str


class AlertItem(BaseModel):
    title: str
    summary: str
    published: str
    link: str
    source_feed: str


class AlertFeed(BaseModel):
    alerts: List[AlertItem]


class Constraint(BaseModel):
    type: AlertType
    severity: int = Field(ge=1, le=3)
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    region_match: bool
    evidence: str = Field(max_length=200)


class ConstraintList(BaseModel):
    constraints: List[Constraint]


class PlanStep(BaseModel):
    step: int = Field(ge=0, le=47)
    q_mech_kw: float
    q_evap_kw: float
    air_setpoint_c: float
    expected_power_kw: float
    expected_water_l: float
    cooling_mode: str = "idle"  # mechanical | evaporative | hybrid | idle


class Metrics(BaseModel):
    cost_eur: float
    peak_kw: float
    water_l: float
    max_temp_c: float
    energy_kwh: float


class Plan(BaseModel):
    plan_id: str
    steps: List[PlanStep]
    relaxed_constraints: List[str] = []
    metrics: Metrics


class Action(BaseModel):
    action_id: str
    cycle_id: str
    plan_id: str
    step: int = 0
    status: ActionStatus
    rejection_reasons: List[str] = []
    created_at: datetime
    applied_at: Optional[datetime] = None
    note: Optional[str] = None


class Citation(BaseModel):
    source_name: str
    ts: datetime


class TraceStep(BaseModel):
    n: int
    observed: str
    decided: str
    why: str
    tool: str
    tool_result_summary: str
    citations: List[Citation]


class SourceState(BaseModel):
    """One ingested source: provenance plus (when available) the validated payload."""

    name: str
    provenance: Provenance
    data: Optional[Dict[str, Any]] = None
    notice: Optional[str] = None

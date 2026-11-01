"""Deterministic safety validator (spec L11, V1-V6). Treats the LLM AND the optimizer as untrusted."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from config.settings import Constants
from data.schemas import Plan, Status
from optimizer.mpc import plan_hash
from twin.model import Plant, simulate


@dataclass
class Reason:
    code: str
    message: str

    def text(self) -> str:
        return f"{self.code}: {self.message}"


@dataclass
class ValidationResult:
    ok: bool
    reasons: List[Reason] = field(default_factory=list)

    def texts(self) -> List[str]:
        return [r.text() for r in self.reasons]


def validate_plan(K: Constants, plant: Plant, plan: Plan, record: Optional[Dict[str, Any]], statuses: Dict[str, Status],
                  stale_margin_c: float) -> ValidationResult:
    """record = the optimizer's registry entry for plan.plan_id (x0, q_it_kw, forecasts, plan_hash...)."""
    R: List[Reason] = []
    tol = K.v("safety.numeric_tolerance_c")
    # V1 - the plan must be exactly what the optimizer produced
    if record is None:
        R.append(Reason("V1", "plan_id is not in the optimizer registry"))
    if plan_hash(plan.steps) != plan.plan_id or (record is not None and record.get("plan_hash") != plan.plan_id):
        R.append(Reason("V1", "plan content does not match the optimizer's recorded hash (tampered or edited plan)"))
    # V6 - required data must not be FAILED
    for src in ("weather", "market", "telemetry"):
        if statuses.get(src) == Status.FAILED:
            R.append(Reason("V6", f"required input '{src}' is FAILED"))
    if record is None:
        return ValidationResult(False, R)

    tmin = K.v("safety.temp_min_c")
    usable_max = K.v("safety.temp_hard_max_c") - stale_margin_c - record.get("revision_margin_c", 0.0)
    t_out, t_wet = record["t_out"], record["t_wet"]
    prev_sp = record["x0"][0]
    prev_q: Optional[float] = record.get("prev_cooling_kw")
    slew = K.v("safety.max_setpoint_slew_c_per_step")
    for s in plan.steps:
        k = s.step
        # V2 bounds
        if s.q_mech_kw < -1e-9 or s.q_evap_kw < -1e-9:
            R.append(Reason("V2", f"step {k}: negative cooling"))
        if s.q_mech_kw + s.q_evap_kw > plant.cap_kw + 1e-3:
            R.append(Reason("V2", f"step {k}: cooling exceeds plant capacity {plant.cap_kw} kW"))
        if s.q_evap_kw > plant.evap_cap(t_out[k], t_wet[k]) + 1e-3:
            R.append(Reason("V2", f"step {k}: evaporative cooling exceeds evaporative capacity"))
        if not (tmin - tol <= s.air_setpoint_c <= usable_max + tol):
            R.append(Reason("V2", f"step {k}: air setpoint {s.air_setpoint_c:.2f} C outside [{tmin}, {usable_max:.2f}] C"))
        # V3 slew
        if abs(s.air_setpoint_c - prev_sp) > slew + 1e-6:
            R.append(Reason("V3", f"step {k}: setpoint change {abs(s.air_setpoint_c - prev_sp):.2f} C exceeds {slew} C/step"))
        prev_sp = s.air_setpoint_c
        # V4 ramp
        q = s.q_mech_kw + s.q_evap_kw
        if prev_q is not None and abs(q - prev_q) > plant.ramp + 1e-3:
            R.append(Reason("V4", f"step {k}: cooling ramp {abs(q - prev_q):.1f} kW exceeds {plant.ramp} kW/step"))
        prev_q = q
    # V5 dry run on the twin with the recorded forecast
    sim = simulate(plant, np.array(record["x0"]), [s.q_mech_kw for s in plan.steps], [s.q_evap_kw for s in plan.steps],
                   record["q_it_kw"], t_out, t_wet)
    if min(sim["t_air"]) < tmin - tol or max(sim["t_air"]) > usable_max + tol:
        R.append(Reason("V5", f"dry-run air temperature leaves [{tmin}, {usable_max:.2f}] C "
                              f"(min {min(sim['t_air']):.2f}, max {max(sim['t_air']):.2f})"))
    return ValidationResult(not R, R)

"""Stress-window detection (spec L6, S1-S4) and alert-constraint conflict handling (L5)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config.settings import Constants
from data.schemas import Constraint, MarketData, WeatherForecast
from forecast.grid import STEP_MIN


def resolve_conflicts(cons: List[Constraint]) -> Tuple[List[Constraint], bool]:
    """Overlapping constraints of one type with different severity => keep the most severe, flag conflict."""
    far = datetime.max.replace(tzinfo=timezone.utc)
    near = datetime.min.replace(tzinfo=timezone.utc)
    kept: List[Constraint] = []; conflict = False
    for c in sorted([c for c in cons if c.region_match and c.type.value != "none"], key=lambda c: -c.severity):
        clash = [k for k in kept if k.type == c.type and (c.start or near) <= (k.end or far) and (k.start or near) <= (c.end or far)]
        if clash:
            conflict = conflict or any(k.severity != c.severity for k in clash)
            continue
        kept.append(c)
    return kept, conflict


def detect_stress(K: Constants, weather: WeatherForecast, market: MarketData, constraints: List[Constraint], q_it_kw: float,
                  plant_cap_kw: float, r_env: float, grid_history: Optional[List[float]], grid_current: Optional[float]
                  ) -> Tuple[List[Dict[str, bool]], List[Dict[str, Any]]]:
    n = len(weather.points)
    flags: List[Dict[str, bool]] = [{"S1": False, "S2": False, "S3": False, "S4": False} for _ in range(n)]
    for k in range(n):
        flags[k]["S1"] = market.points[k].is_peak_window
        demand = q_it_kw + max(0.0, weather.points[k].temp_dry_c - K.v("safety.temp_recommended_max_c")) / r_env
        flags[k]["S4"] = demand > K.v("stress.s4_capacity_fraction") * plant_cap_kw
        for c in constraints:
            if c.severity >= K.v("stress.s3_min_severity") and c.region_match:
                ts = weather.points[k].ts
                if (c.start is None or ts >= c.start) and (c.end is None or ts <= c.end + timedelta(minutes=STEP_MIN)):
                    flags[k]["S3"] = True
    if grid_current is not None and grid_history and len(grid_history) >= K.v("stress.s2_min_samples"):
        if grid_current > float(np.percentile(grid_history, K.v("stress.s2_percentile"))):
            for k in range(min(n, int(K.v("stress.s2_persist_steps")))):
                flags[k]["S2"] = True
    windows: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None
    for k in range(n):
        ind = [i for i, v in flags[k].items() if v]
        if ind:
            if cur is None:
                cur = {"start": weather.points[k].ts, "end": weather.points[k].ts, "indicators": set(ind), "first_step": k}
            cur["end"] = weather.points[k].ts; cur["indicators"] |= set(ind)
        elif cur is not None:
            windows.append(cur); cur = None
    if cur is not None:
        windows.append(cur)
    for w in windows:
        w["indicators"] = sorted(w["indicators"]); w["end"] = w["end"] + timedelta(minutes=STEP_MIN)
    return flags, windows

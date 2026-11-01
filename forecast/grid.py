"""Time-grid helpers: 48 x 15-minute steps, interpolation, wet-bulb (Stull 2011)."""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import List, Sequence

import numpy as np

STEP_MIN = 15
N_STEPS = 48
DT_H = STEP_MIN / 60.0


def floor15(dt: datetime) -> datetime:
    dt = dt.astimezone(timezone.utc)
    return dt.replace(minute=(dt.minute // 15) * 15, second=0, microsecond=0)


def step_times(start: datetime, n: int = N_STEPS) -> List[datetime]:
    return [start + timedelta(minutes=STEP_MIN * i) for i in range(n)]


def interp_to_grid(src_times: Sequence[datetime], src_values: Sequence[float], grid: Sequence[datetime]) -> List[float]:
    """Linear interpolation onto the grid. Raises if the grid is not covered by the source."""
    xs = np.array([t.timestamp() for t in src_times])
    gx = np.array([t.timestamp() for t in grid])
    if gx[0] < xs[0] - 1 or gx[-1] > xs[-1] + 1:
        raise ValueError("source series does not cover the 12-hour grid")
    return [float(v) for v in np.interp(gx, xs, np.array(src_values, dtype=float))]


def wet_bulb_stull(temp_c: float, rh_pct: float) -> float:
    """Stull (2011) wet-bulb approximation; valid for RH 5-99 % and T -20..50 C. MODEL_DERIVED."""
    rh = max(5.0, min(99.0, rh_pct))
    t = temp_c
    tw = (t * math.atan(0.151977 * math.sqrt(rh + 8.313659)) + math.atan(t + rh) - math.atan(rh - 1.676331)
          + 0.00391838 * rh ** 1.5 * math.atan(0.023101 * rh) - 4.686035)
    return min(tw, t)

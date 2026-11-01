"""Real host telemetry (psutil) mapped to IT load. Host is labeled honestly as backend_host."""
from __future__ import annotations

import platform
from datetime import datetime, timezone
from typing import Tuple

import psutil

from config.settings import Constants
from data.schemas import ItLoadSample, SignalKind


def it_load_kw(cpu_pct: float, mem_pct: float, K: Constants) -> float:
    rated, idle = K.v("facility.rated_it_capacity_kw"), K.v("facility.idle_it_power_fraction")
    u = K.v("facility.cpu_weight") * cpu_pct / 100.0 + K.v("facility.mem_weight") * mem_pct / 100.0
    return rated * (idle + (1.0 - idle) * u)


def sample_telemetry(K: Constants) -> Tuple[ItLoadSample, str, str, SignalKind]:
    cpu = float(psutil.cpu_percent(interval=0.2))
    mem = float(psutil.virtual_memory().percent)
    rated, idle = K.v("facility.rated_it_capacity_kw"), K.v("facility.idle_it_power_fraction")
    formula = (f"P_IT = {rated} kW * ({idle} + {round(1 - idle, 4)} * ({K.v('facility.cpu_weight')}*CPU + "
               f"{K.v('facility.mem_weight')}*RAM))")
    s = ItLoadSample(ts=datetime.now(timezone.utc), cpu_pct=cpu, mem_pct=mem, it_load_kw=round(it_load_kw(cpu, mem, K), 3),
                     scaling_formula=formula, host_label=f"backend_host:{platform.node()}")
    return s, "Host telemetry (psutil)", "psutil://local", SignalKind.MEASURED

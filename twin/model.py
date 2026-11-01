"""Digital twin (SIMULATED ACTUATOR): two-node RC thermal model, plant, water. Spec L7/L8.

State x = [T_air, T_mass]; input u = [Q_cool, Q_IT, T_out] (kW, kW, C).
Discretised EXACTLY (zero-order hold via matrix exponential) => unconditionally stable for any dt.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.linalg import expm

from config.settings import Constants
from data.store import Store
from forecast.grid import DT_H


@dataclass(frozen=True)
class Plant:
    ca: float
    cm: float
    re: float
    rm: float
    cap_kw: float
    cop_nom: float
    cop_slope: float
    cop_quad: float
    cop_floor: float
    ce_nom: float
    ce_slope: float
    ce_floor: float
    evap_k: float
    evap_max: float
    h_fg: float
    ramp: float

    @classmethod
    def from_constants(cls, K: Constants) -> "Plant":
        g = K.v
        return cls(g("thermal.c_air_kwh_per_c"), g("thermal.c_mass_kwh_per_c"), g("thermal.r_env_c_per_kw"),
                   g("thermal.r_mass_c_per_kw"), g("plant.max_cooling_thermal_kw"), g("plant.cop_mech_nominal"),
                   g("plant.cop_mech_slope_per_c"), g("plant.cop_mech_quad_per_c2"), g("plant.cop_mech_floor"),
                   g("plant.cop_evap_nominal"), g("plant.cop_evap_slope_per_c_wet"), g("plant.cop_evap_floor"),
                   g("plant.evap_cap_kw_per_k_depression"), g("plant.max_evap_thermal_kw"),
                   g("plant.latent_heat_kj_per_kg"), g("plant.max_cooling_ramp_kw_per_step"))

    # --- plant characteristics ------------------------------------------------------
    def cop_mech(self, t_out: float) -> float:
        d = t_out - 25.0
        return max(self.cop_floor, self.cop_nom + self.cop_slope * d + self.cop_quad * max(0.0, d) ** 2)

    def cop_evap(self, t_wet: float) -> float:
        return max(self.ce_floor, self.ce_nom + self.ce_slope * (t_wet - 20.0))

    def evap_cap(self, t_out: float, t_wet: float) -> float:
        return float(max(0.0, min(self.evap_max, self.evap_k * max(0.0, t_out - t_wet))))

    def power_kw(self, p_it: float, qm: float, qe: float, t_out: float, t_wet: float) -> float:
        return p_it + qm / self.cop_mech(t_out) + qe / self.cop_evap(t_wet)

    def water_l(self, qe_kw: float, dt_h: float = DT_H) -> float:
        """water_l = Q_evap * dt * 3600 / h_fg (1 kg = 1 L)."""
        return qe_kw * dt_h * 3600.0 / self.h_fg

    # --- dynamics -------------------------------------------------------------------
    def continuous(self) -> Tuple[np.ndarray, np.ndarray]:
        A = np.array([[-(1 / self.rm + 1 / self.re) / self.ca, 1 / (self.rm * self.ca)],
                      [1 / (self.rm * self.cm), -1 / (self.rm * self.cm)]])
        B = np.array([[-1 / self.ca, 1 / self.ca, 1 / (self.re * self.ca)], [0.0, 0.0, 0.0]])
        return A, B

    def discrete(self, dt_h: float = DT_H) -> Tuple[np.ndarray, np.ndarray]:
        A, B = self.continuous()
        M = np.zeros((5, 5))
        M[:2, :2], M[:2, 2:] = A, B
        E = expm(M * dt_h)
        return E[:2, :2], E[:2, 2:]


def step_state(plant: Plant, x: np.ndarray, q_cool: float, q_it: float, t_out: float) -> np.ndarray:
    Ad, Bd = plant.discrete()
    return Ad @ x + Bd @ np.array([q_cool, q_it, t_out])


def simulate(plant: Plant, x0: np.ndarray, qm: List[float], qe: List[float], q_it: float,
             t_out: List[float], t_wet: List[float]) -> Dict[str, List[float]]:
    """Forward simulation of a cooling schedule on the twin. Returns trajectories (post-step values)."""
    Ad, Bd = plant.discrete()
    x = np.array(x0, dtype=float)
    ta, tm, power, water = [], [], [], []
    for k in range(len(qm)):
        x = Ad @ x + Bd @ np.array([qm[k] + qe[k], q_it, t_out[k]])
        ta.append(float(x[0])); tm.append(float(x[1]))
        power.append(plant.power_kw(q_it, qm[k], qe[k], t_out[k], t_wet[k]))
        water.append(plant.water_l(qe[k]))
    return {"t_air": ta, "t_mass": tm, "power_kw": power, "water_l": water}


class DigitalTwin:
    """Persistent twin state. Only validated actions ever change it. Never real hardware."""

    KEY = "twin_state"

    def __init__(self, plant: Plant, K: Constants, store: Store) -> None:
        self.plant, self.K, self.store = plant, K, store

    def state(self) -> Dict[str, float]:
        s = self.store.get_kv(self.KEY)
        if s is None:
            s = {"t_air": self.K.v("thermal.initial_air_temp_c"), "t_mass": self.K.v("thermal.initial_mass_temp_c"),
                 "q_mech_kw": None, "q_evap_kw": None, "updated_at": None, "label": "SIMULATED ACTUATOR - DIGITAL TWIN"}
            self.store.set_kv(self.KEY, s)
        return s

    def snapshot(self) -> Dict[str, float]:
        return dict(self.state())

    def restore(self, snap: Dict[str, float]) -> None:
        self.store.set_kv(self.KEY, dict(snap))  # exact restore (AC16)

    def x0(self) -> np.ndarray:
        s = self.state()
        return np.array([s["t_air"], s["t_mass"]], dtype=float)

    def apply(self, qm: float, qe: float, q_it: float, t_out: float, now_iso: str) -> Dict[str, float]:
        x = step_state(self.plant, self.x0(), qm + qe, q_it, t_out)
        new = {**self.state(), "t_air": float(x[0]), "t_mass": float(x[1]), "q_mech_kw": qm, "q_evap_kw": qe, "updated_at": now_iso}
        self.store.set_kv(self.KEY, new)
        return new

"""Receding-horizon optimizer (LP via HiGHS) and the reactive baseline. Spec L9/L10.

The LLM never computes setpoints: this module does. Linear program over 48 x 15-min steps.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from config.settings import Constants
from data.schemas import Metrics, Plan, PlanStep
from forecast.grid import DT_H, N_STEPS
from twin.model import Plant, simulate


@dataclass
class Horizon:
    """Everything the optimizer needs for one cycle (all derived from live inputs)."""
    x0: np.ndarray
    q_it_kw: float
    t_out: List[float]
    t_wet: List[float]
    price: List[float]          # all-in EUR/kWh per step (day-ahead spot + adder)
    stress: List[bool]
    prev_cooling_kw: Optional[float] = None  # None on first start: no ramp constraint at step 0
    stale_margin_c: float = 0.0
    revision_margin_c: float = 0.0


def plan_hash(steps: List[PlanStep]) -> str:
    canon = json.dumps([s.model_dump() for s in steps], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


def mode_of(qm: float, qe: float) -> str:
    if qm + qe < 1e-6:
        return "idle"
    if qe < 0.1 * (qm + qe):
        return "mechanical"
    if qm < 0.1 * (qm + qe):
        return "evaporative"
    return "hybrid"


def limits(K: Constants, h: Horizon) -> Tuple[float, float, float]:
    tmin = K.v("safety.temp_min_c")
    hard = K.v("safety.temp_hard_max_c") - h.stale_margin_c - h.revision_margin_c
    rec = min(K.v("safety.temp_recommended_max_c"), hard)
    return tmin, rec, hard


def metrics_from_sim(plant: Plant, h: Horizon, qm: List[float], qe: List[float]) -> Tuple[Metrics, Dict[str, List[float]]]:
    sim = simulate(plant, h.x0, qm, qe, h.q_it_kw, h.t_out, h.t_wet)
    p = sim["power_kw"]
    m = Metrics(cost_eur=float(sum(h.price[k] * p[k] * DT_H for k in range(N_STEPS))), peak_kw=float(max(p)),
                water_l=float(sum(sim["water_l"])), max_temp_c=float(max(sim["t_air"])), energy_kwh=float(sum(p) * DT_H))
    return m, sim


def _steps(plant: Plant, h: Horizon, qm: List[float], qe: List[float], sim: Dict[str, List[float]]) -> List[PlanStep]:
    return [PlanStep(step=k, q_mech_kw=round(qm[k], 4), q_evap_kw=round(qe[k], 4), air_setpoint_c=round(sim["t_air"][k], 4),
                     expected_power_kw=round(sim["power_kw"][k], 4), expected_water_l=round(sim["water_l"][k], 4),
                     cooling_mode=mode_of(qm[k], qe[k])) for k in range(N_STEPS)]


def reactive_baseline(K: Constants, plant: Plant, h: Horizon) -> Tuple[Metrics, Dict[str, List[float]], List[float], List[float]]:
    """No lookahead: each step apply the minimum cooling keeping next T_air <= recommended max."""
    _, rec, _ = limits(K, h)
    share = K.v("optimizer.baseline_evap_share")
    Ad, Bd = plant.discrete()
    x = np.array(h.x0, dtype=float)
    qm_l, qe_l = [], []
    for k in range(N_STEPS):
        free = Ad @ x + Bd @ np.array([0.0, h.q_it_kw, h.t_out[k]])
        g = -Bd[0, 0]  # degC of air cooling per kW of cooling in one step
        need = max(0.0, (free[0] - rec) / g)
        need = min(need, plant.cap_kw)
        qe = min(share * need, plant.evap_cap(h.t_out[k], h.t_wet[k]))
        qm = need - qe
        x = Ad @ x + Bd @ np.array([qm + qe, h.q_it_kw, h.t_out[k]])
        qm_l.append(qm); qe_l.append(qe)
    m, sim = metrics_from_sim(plant, h, qm_l, qe_l)
    return m, sim, qm_l, qe_l


class Infeasible(Exception):
    pass


def _solve(K: Constants, plant: Plant, h: Horizon, free: set, baseline_tm_end: Optional[float]):
    N = N_STEPS
    tmin, rec, hard = limits(K, h)
    Ad, Bd = plant.discrete()
    iq, ie, isl, ita, itm = 0, N, 2 * N, 3 * N, 4 * N
    ipk, isp, isw = 5 * N, 5 * N + 1, 5 * N + 2
    nv = 5 * N + 3
    cm = [plant.cop_mech(t) for t in h.t_out]
    ce = [plant.cop_evap(w) for w in h.t_wet]
    lam = K.v("optimizer.stress_price_premium_lambda")
    # premium scales |price| so a negative spot price on a stress step is not made more attractive
    price = [h.price[k] + (lam * abs(h.price[k]) if h.stress[k] else 0.0) for k in range(N)]
    kw = 3600.0 * DT_H / plant.h_fg
    peak_cap = K.v("optimizer.peak_cap_kw")

    c = np.zeros(nv)
    for k in range(N):
        c[iq + k] = price[k] * DT_H / cm[k]
        c[ie + k] = price[k] * DT_H / ce[k] + K.v("optimizer.w_water_eur_per_l") * kw
        c[isl + k] = K.v("optimizer.w_violation_temp")
    c[ipk] = K.v("optimizer.w_peak_eur_per_kw")
    c[isp] = K.v("optimizer.w_violation_peak")
    c[isw] = K.v("optimizer.w_violation_water")

    # equality: exact discretised dynamics
    Aeq = lil_matrix((2 * N, nv)); beq = np.zeros(2 * N)
    for t in range(N):
        for r, base in ((0, ita), (1, itm)):
            row = 2 * t + r
            Aeq[row, base + t] = 1.0
            Aeq[row, iq + t] = -Bd[r, 0]; Aeq[row, ie + t] = -Bd[r, 0]
            beq[row] = Bd[r, 1] * h.q_it_kw + Bd[r, 2] * h.t_out[t]
            if t == 0:
                beq[row] += Ad[r, 0] * h.x0[0] + Ad[r, 1] * h.x0[1]
            else:
                Aeq[row, ita + t - 1] = -Ad[r, 0]; Aeq[row, itm + t - 1] = -Ad[r, 1]

    rows, rhs = [], []
    def add(coeffs: Dict[int, float], b: float):
        rows.append(coeffs); rhs.append(b)

    for t in range(N):
        add({iq + t: 1 / cm[t], ie + t: 1 / ce[t], ipk: -1.0}, -h.q_it_kw)                      # peak epigraph
        if peak_cap is not None:
            add({iq + t: 1 / cm[t], ie + t: 1 / ce[t], isp: -1.0}, peak_cap - h.q_it_kw)       # soft peak cap
        add({iq + t: 1.0, ie + t: 1.0}, plant.cap_kw)                                           # capacity
        # ramp (total cooling); at t=0 only when the previously applied cooling is known
        if t or h.prev_cooling_kw is not None:
            prev = {iq + t - 1: -1.0, ie + t - 1: -1.0} if t else {}
            p0 = h.prev_cooling_kw if (t == 0) else 0.0
            add({iq + t: 1.0, ie + t: 1.0, **prev}, plant.ramp + p0)
            neg = {iq + t: -1.0, ie + t: -1.0}
            if t:
                neg.update({iq + t - 1: 1.0, ie + t - 1: 1.0})
            add(neg, plant.ramp - p0)
        # setpoint slew
        sl = K.v("safety.max_setpoint_slew_c_per_step")
        add({ita + t: 1.0, **({ita + t - 1: -1.0} if t else {})}, sl + (h.x0[0] if t == 0 else 0.0))
        add({ita + t: -1.0, **({ita + t - 1: 1.0} if t else {})}, sl - (h.x0[0] if t == 0 else 0.0))
        add({ita + t: 1.0, isl + t: -1.0}, rec)                                                # recommended max (soft)
    add({ie + t: kw for t in range(N)} | {isw: -1.0}, K.v("optimizer.water_budget_l"))          # water budget
    if baseline_tm_end is not None:
        add({itm + N - 1: 1.0}, baseline_tm_end + K.v("optimizer.terminal_mass_tolerance_c"))   # fair comparison

    Aub = lil_matrix((len(rows), nv))
    for i, co in enumerate(rows):
        for j, v in co.items():
            Aub[i, j] = v
    bub = np.array(rhs, dtype=float)

    tol = 1e-4
    bounds = [(0, None)] * N + [(0, plant.evap_cap(h.t_out[k], h.t_wet[k])) for k in range(N)]
    bounds += [(0, (hard - tol - rec) if "recommended maximum" in free else 0.0)] * N
    bounds += [(tmin + tol, hard - tol)] * N + [(None, None)] * N
    bounds += [(0, None), (0, None if "peak cap" in free else 0.0), (0, None if "water budget" in free else 0.0)]
    res = linprog(c, A_ub=Aub.tocsr(), b_ub=bub, A_eq=Aeq.tocsr(), b_eq=beq, bounds=bounds, method="highs")
    if res.status != 0:
        raise Infeasible(res.message)
    x = res.x
    return [max(0.0, float(v)) for v in x[iq:iq + N]], [max(0.0, float(v)) for v in x[ie:ie + N]]


def optimize(K: Constants, plant: Plant, h: Horizon) -> Tuple[Optional[Plan], Optional[Metrics], Dict[str, Any]]:
    """Returns (plan | None, baseline_metrics, extras). None => infeasible even after relaxation (E9)."""
    base_m, base_sim, bqm, bqe = reactive_baseline(K, plant, h)
    extras: Dict[str, Any] = {"baseline_sim": base_sim, "baseline_qm": bqm, "baseline_qe": bqe}
    levels: List[Tuple[set, List[str]]] = [(set(), [])]
    free: set = set(); relaxed: List[str] = []
    if K.v("optimizer.peak_cap_kw") is not None:
        free = free | {"peak cap"}; relaxed = relaxed + ["peak cap"]; levels.append((set(free), list(relaxed)))
    free = free | {"water budget"}; relaxed = relaxed + ["water budget"]; levels.append((set(free), list(relaxed)))
    free = free | {"recommended maximum"}; relaxed = relaxed + ["recommended maximum"]; levels.append((set(free), list(relaxed)))
    for i, (fr, rel) in enumerate(levels):
        try:
            qm, qe = _solve(K, plant, h, fr, base_sim["t_mass"][-1] if i == 0 else None)
        except Infeasible:
            continue
        # clip numerical noise to physical caps, then verify with the exact simulator
        qe = [min(q, plant.evap_cap(h.t_out[k], h.t_wet[k])) for k, q in enumerate(qe)]
        m, sim = metrics_from_sim(plant, h, qm, qe)
        steps = _steps(plant, h, qm, qe, sim)
        plan = Plan(plan_id=plan_hash(steps), steps=steps, relaxed_constraints=rel, metrics=m)
        extras["plan_sim"] = sim
        return plan, base_m, extras
    return None, base_m, extras

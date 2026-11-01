"""UT1-UT9 and optimizer acceptance checks (AC8, AC9)."""
import numpy as np
import pytest

from config.settings import load_constants
from forecast.grid import wet_bulb_stull
from optimizer.mpc import Horizon, optimize, plan_hash
from safety.validator import validate_plan
from data.schemas import Plan, Status
from twin.model import Plant, simulate

K = load_constants(); P = Plant.from_constants(K); N = 48


def test_constants_have_sources():  # AC3
    assert len(list(K.items())) > 30
    assert all(c.source.strip() and c.status in {"cited", "derived", "assumed"} for _, c in K.items())


def test_ut1_zero_cooling_never_cools():
    s = simulate(P, np.array([24.0, 24.0]), [0.0] * N, [0.0] * N, 100.0, [24.0] * N, [18.0] * N)
    assert all(b >= a - 1e-9 for a, b in zip([24.0] + s["t_air"], s["t_air"]))


def test_ut2_steady_state_holds():
    qit, tout = 100.0, 30.0
    Ad, Bd = P.discrete()
    x = np.array([25.0, 25.0])
    # steady state heat to remove with air == mass == 25 C
    q = qit + (tout - 25.0) / P.re
    s = simulate(P, x, [q] * N, [0.0] * N, qit, [tout] * N, [20.0] * N)
    assert abs(s["t_air"][0] - 25.0) < 0.01 and abs(s["t_air"][-1] - s["t_air"][-2]) < 0.01


def test_ut3_stable_over_horizon():
    eig = np.linalg.eigvals(P.discrete()[0])
    assert all(abs(e) < 1 for e in eig)
    s = simulate(P, np.array([24.0, 24.0]), [300.0] * N, [0.0] * N, 300.0, [35.0] * N, [25.0] * N)
    assert np.all(np.isfinite(s["t_air"])) and max(abs(np.diff(s["t_air"]))) < 15


def test_ut4_cop_and_power():
    assert P.cop_mech(35) < P.cop_mech(25)
    assert P.power_kw(0, 100, 0, 35, 25) > P.power_kw(0, 100, 0, 25, 20)


def test_ut5_water():
    assert abs(P.water_l(100.0, dt_h=1.0) - 148.1) < 0.1


def test_ut6_wet_bulb_reference():
    assert abs(wet_bulb_stull(20.0, 50.0) - 13.7) < 0.5  # Stull (2011) published example


def _horizon(price_mult=2.0, t_peak=38.0, q_it=300.0):
    t_out = [30 + (t_peak - 30) * np.sin(np.pi * k / N) for k in range(N)]
    price = [0.2 * (price_mult if 24 <= k < 32 else 1.0) for k in range(N)]
    return Horizon(x0=np.array([24.0, 24.0]), q_it_kw=q_it, t_out=t_out, t_wet=[t - 6 for t in t_out], price=price,
                   stress=[24 <= k < 32 for k in range(N)])


def test_ut7_and_ac8_ac9_precooling():
    h = _horizon()
    plan, base, ex = optimize(K, P, h)
    assert plan is not None
    q = [s.q_mech_kw + s.q_evap_kw for s in plan.steps]
    b = [m + e for m, e in zip(ex["baseline_qm"], ex["baseline_qe"])]
    assert sum(q[24:32]) <= sum(b[24:32])                        # UT7
    assert sum(q[:24]) > sum(b[:24])                             # AC9 pre-cooling
    assert max(ex["plan_sim"]["power_kw"][24:32]) <= max(ex["baseline_sim"]["power_kw"][24:32]) + 1e-6
    assert all(18 - 1e-3 <= s.air_setpoint_c <= 27 + 1e-3 for s in plan.steps)   # AC8


def test_ut8_infeasible_holds_safe():
    h = _horizon(q_it=900.0, t_peak=45.0)  # heat load far above plant capacity
    plan, _, _ = optimize(K, P, h)
    assert plan is None


def _rec(plan, h):
    return {"plan_hash": plan.plan_id, "x0": h.x0.tolist(), "q_it_kw": h.q_it_kw, "t_out": h.t_out, "t_wet": h.t_wet,
            "prev_cooling_kw": None, "revision_margin_c": 0.0}


OK = {"weather": Status.LIVE, "market": Status.LIVE, "telemetry": Status.LIVE}


def test_valid_plan_passes_and_v1_tamper_rejected():
    h = _horizon(); plan, _, _ = optimize(K, P, h)
    assert validate_plan(K, P, plan, _rec(plan, h), OK, 0.0).ok
    bad = plan.model_copy(deep=True); bad.steps[3].q_mech_kw += 50          # AC13 / scenario 4
    res = validate_plan(K, P, bad, _rec(plan, h), OK, 0.0)
    assert not res.ok and any(r.code == "V1" for r in res.reasons)


def test_ut9_slew_bounds_ramp_rejected():
    h = _horizon(); plan, _, _ = optimize(K, P, h)
    def tampered(mut):
        p = plan.model_copy(deep=True); mut(p)
        p.plan_id = plan_hash(p.steps)          # attacker also fixes the hash
        rec = _rec(plan, h); rec["plan_hash"] = p.plan_id
        return validate_plan(K, P, p, rec, OK, 0.0)
    r = tampered(lambda p: setattr(p.steps[5], "air_setpoint_c", p.steps[4].air_setpoint_c + 1.6))
    assert any(x.code == "V3" for x in r.reasons) and all(x.message for x in r.reasons)
    r = tampered(lambda p: setattr(p.steps[7], "air_setpoint_c", 28.0))
    assert any(x.code == "V2" for x in r.reasons)
    r = tampered(lambda p: (setattr(p.steps[9], "q_mech_kw", p.steps[8].q_mech_kw + 400.0)))
    assert any(x.code == "V4" for x in r.reasons)


def test_v6_failed_input_rejected():
    h = _horizon(); plan, _, _ = optimize(K, P, h)
    res = validate_plan(K, P, plan, _rec(plan, h), {**OK, "weather": Status.FAILED}, 0.0)
    assert any(r.code == "V6" for r in res.reasons)

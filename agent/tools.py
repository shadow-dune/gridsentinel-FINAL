"""Allowlisted agent tools (spec L12). The LLM can call only these and can never set numbers."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, List

import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationError

from agent.context import CycleContext, Deps
from agent.llm import ConstraintParseError, LLMError
from data.schemas import Action, ActionStatus, CycleStatus, Mode, Status
from forecast.stress import detect_stress as _detect_stress, resolve_conflicts
from optimizer.mpc import Horizon, optimize, plan_hash
from safety.validator import validate_plan as _validate
from data.schemas import Plan


class _NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _PlanArg(_NoArgs):
    plan_id: str


class _ActionArg(_NoArgs):
    action_id: str


class _MsgArg(_NoArgs):
    message: str


ALLOWLIST: Dict[str, type] = {"get_inputs": _NoArgs, "parse_alerts": _NoArgs, "detect_stress": _NoArgs,
                              "optimize_plan": _NoArgs, "validate_plan": _PlanArg, "execute_step": _PlanArg,
                              "verify_outcome": _ActionArg, "rollback": _ActionArg, "notify_operator": _MsgArg}

DESCRIPTIONS = {
    "get_inputs": "Summarise the validated live inputs and their health.",
    "parse_alerts": "Parse alert bulletins into validated constraints (isolated LLM call).",
    "detect_stress": "Detect stress windows (S1-S4) over the 12-hour horizon.",
    "optimize_plan": "Run the deterministic optimizer and reactive baseline. Returns plan_id and metrics.",
    "validate_plan": "Run the deterministic safety validator on a plan_id.",
    "execute_step": "Validate again and apply step 0 of the plan to the digital twin (or queue for approval).",
    "verify_outcome": "Report the verification result of an applied action.",
    "rollback": "Roll back an applied action on the digital twin.",
    "notify_operator": "Write an in-app operator alert.",
}


def tool_schemas() -> List[Dict[str, Any]]:
    out = []
    for name, model in ALLOWLIST.items():
        schema = model.model_json_schema()
        out.append({"type": "function", "function": {"name": name, "description": DESCRIPTIONS[name],
                                                     "parameters": {"type": "object", "properties": schema.get("properties", {}),
                                                                    "required": schema.get("required", []),
                                                                    "additionalProperties": False}}})
    return out


# ------------------------------------------------------------------------------------------
def apply_action(d: Deps, action: Dict[str, Any]) -> Dict[str, Any]:
    """Apply step 0 to the twin, verify, roll back on deviation (L13). Used by autonomous mode and the approve endpoint."""
    rec = d.store.get_plan(action["plan_id"])
    now = datetime.now(timezone.utc)
    if rec is None:
        action.update(status=ActionStatus.failed.value, note="Plan not found in registry")
        return action
    plan = Plan.model_validate(rec["plan"])
    step0 = plan.steps[0]
    try:
        q_it = d.it_sample().it_load_kw  # fresh live telemetry at application time
    except Exception:
        action.update(status=ActionStatus.failed.value, note="IT load telemetry unavailable. No constant was substituted.")
        return action
    snapshot = d.twin.snapshot()
    new = d.twin.apply(step0.q_mech_kw, step0.q_evap_kw, q_it, rec["t_out"][0], now.isoformat())
    dev = abs(new["t_air"] - step0.air_setpoint_c)
    verification = {"predicted_t_air_c": round(step0.air_setpoint_c, 3), "actual_t_air_c": round(new["t_air"], 3),
                    "abs_error_c": round(dev, 3), "tolerance_c": d.K.v("safety.verify_tolerance_c")}
    action["_snapshot"] = snapshot
    if dev > d.K.v("safety.verify_tolerance_c"):
        d.twin.restore(snapshot)
        action.update(status=ActionStatus.rolled_back.value,
                      note="Twin response deviated from prediction; action rolled back.", verification=verification)
    else:
        action.update(status=ActionStatus.applied.value, applied_at=now.isoformat(), verification=verification)
    return action


def rollback_action(d: Deps, action: Dict[str, Any]) -> Dict[str, Any]:
    snap = action.get("_snapshot")
    if snap is not None:
        d.twin.restore(snap)
    action.update(status=ActionStatus.rolled_back.value, note="Rolled back by operator/agent")
    return action


class Tools:
    def __init__(self, ctx: CycleContext) -> None:
        self.ctx, self.d = ctx, ctx.deps

    # -- dispatch (allowlist + schema validation) --------------------------------------------
    def dispatch(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        c = self.ctx
        if name not in ALLOWLIST:
            c.add(name, "LLM requested a tool", "refused", "Tool is not on the allowlist", "refused: tool not allowed")
            return {"error": f"tool '{name}' is not allowed"}
        try:
            parsed = ALLOWLIST[name].model_validate(args or {})
        except ValidationError as exc:
            c.add(name, "LLM supplied arguments", "refused", "Arguments failed schema validation", str(exc)[:200])
            return {"error": f"invalid arguments: {exc.errors()[0]['msg']}"}
        try:
            return getattr(self, name)(**parsed.model_dump())
        except Exception as exc:  # per-tool error handling: never crash the loop
            c.add(name, "tool raised an error", "continue", "Per-tool error handling", f"{type(exc).__name__}: {exc}"[:240])
            return {"error": f"{type(exc).__name__}: {exc}"}

    # -- tools ------------------------------------------------------------------------------------
    def get_inputs(self) -> Dict[str, Any]:
        c = self.ctx
        summary = {n: st.provenance.status.value for n, st in c.inputs.states.items()}
        c.add("get_inputs", "Fetched weather, market, grid signal, alerts, telemetry",
              "Use validated inputs only", "Gaps are never filled with invented values",
              ", ".join(f"{k}={v}" for k, v in summary.items()))
        c.done.add("get_inputs")
        return {"statuses": summary, "notices": c.notices}

    def parse_alerts(self) -> Dict[str, Any]:
        c = self.ctx
        feed = c.inputs.get("alerts") if c.inputs else None
        if feed is None or not feed.alerts:
            c.add("parse_alerts", "No alert bulletins available", "skip", "Nothing to parse", "0 constraints")
            c.done.add("parse_alerts"); return {"constraints": 0}
        llm = self.d.llm
        if not llm.configured:
            c.deterministic = True; c.notice("Deterministic mode (no LLM).")
            c.add("parse_alerts", f"{len(feed.alerts)} bulletins, LLM not configured", "skip alert parsing",
                  "Deterministic mode (no LLM)", "0 constraints")
            c.done.add("parse_alerts"); return {"constraints": 0}
        try:
            cons = llm.parse_alerts(feed.alerts, datetime.now(timezone.utc))
        except ConstraintParseError as exc:
            c.notice("Alert parsing failed; continuing without alert-derived constraints.")
            c.add("parse_alerts", "LLM output invalid after 2 retries", "continue without alert constraints",
                  "Schema validation failed", str(exc)[:160]); c.done.add("parse_alerts"); return {"constraints": 0}
        except LLMError as exc:
            c.deterministic = True; c.notice("Deterministic mode (no LLM).")
            c.add("parse_alerts", "LLM unreachable", "continue in deterministic mode", str(exc)[:120], "0 constraints")
            c.done.add("parse_alerts"); return {"constraints": 0}
        kept, conflict = resolve_conflicts(cons)
        if conflict:
            c.notice("Conflicting alerts detected; using the more cautious interpretation.")
        c.constraints = kept
        c.add("parse_alerts", f"{len(feed.alerts)} bulletins parsed by LLM into {len(cons)} constraints",
              f"keep {len(kept)} region-matched constraints" + (" (conflict -> most severe kept)" if conflict else ""),
              "Constraints may only make the system more conservative", f"{len(kept)} constraints")
        c.done.add("parse_alerts")
        return {"constraints": len(kept), "conflict": conflict}

    def detect_stress(self) -> Dict[str, Any]:
        c = self.ctx
        grid = c.inputs.get("grid_signal")
        hist = None
        cur = grid.value if grid is not None else None
        if grid is not None:
            from datetime import timedelta
            hist = self.d.store.grid_history(grid.name, datetime.now(timezone.utc) - timedelta(days=7))
        flags, windows = _detect_stress(self.d.K, c.weather, c.market, c.constraints, c.horizon.q_it_kw, self.d.plant.cap_kw,
                                        self.d.plant.re, hist, cur)
        c.horizon.stress = [any(f.values()) for f in flags]
        c.stress_windows = windows
        desc = "; ".join(f"{w['start'].isoformat()}→{w['end'].isoformat()} [{','.join(w['indicators'])}]" for w in windows) or "none"
        if grid is None:
            c.notice("Live grid signal unavailable: stress indicator S2 not evaluated.")
        c.add("detect_stress", "Combined S1 (day-ahead price in the top band of the last 7 days), S2 (grid under-frequency), S3 (alerts), S4 (projected plant load)",
              f"{len(windows)} stress window(s)", "Preemptive action needs the window before it starts", desc)
        c.done.add("detect_stress")
        return {"windows": [{"start": w["start"].isoformat(), "end": w["end"].isoformat(), "indicators": w["indicators"]} for w in windows]}

    def optimize_plan(self) -> Dict[str, Any]:
        c = self.ctx
        if "detect_stress" not in c.done:
            self.detect_stress()
        h: Horizon = c.horizon
        h.revision_margin_c = 0.5 * c.revision
        plan, base, extras = optimize(self.d.K, self.d.plant, h)
        c.baseline = base
        c.extras.update(extras)  # keep ml_* / llm_risk set during _prepare
        if plan is None:
            c.status = CycleStatus.held_safe
            msg = "No safe plan found. Holding current state and alerting the operator."
            c.notice(msg); self.d.store.add_alert(f"{c.cycle_id}: {msg}")
            c.add("optimize_plan", "Optimizer infeasible after relaxing peak cap, water budget, recommended maximum",
                  "hold safe state", "Hard limits are never relaxed", msg)
            return {"error": msg}
        c.plan = plan
        self.d.store.save_plan(plan.plan_id, {"plan": plan.model_dump(mode="json"), "plan_hash": plan_hash(plan.steps),
                                              "x0": h.x0.tolist(), "q_it_kw": h.q_it_kw, "t_out": h.t_out, "t_wet": h.t_wet,
                                              "prev_cooling_kw": h.prev_cooling_kw, "revision_margin_c": h.revision_margin_c,
                                              "cycle_id": c.cycle_id})
        if plan.relaxed_constraints:
            c.notice(f"Constraints relaxed: {', '.join(plan.relaxed_constraints)}.")
        m, b = plan.metrics, base
        c.add("optimize_plan", f"Horizon of 48 steps, IT load {h.q_it_kw:.1f} kW, {len(c.stress_windows)} stress window(s)",
              f"plan {plan.plan_id}", "Receding-horizon LP minimises cost, peak draw, water and violations",
              f"plan cost {m.cost_eur:.2f} EUR, peak {m.peak_kw:.1f} kW, water {m.water_l:.0f} L | "
              f"reactive cost {b.cost_eur:.2f} EUR, peak {b.peak_kw:.1f} kW, water {b.water_l:.0f} L")
        c.done.add("optimize_plan")
        return {"plan_id": plan.plan_id, "metrics": m.model_dump(), "baseline": b.model_dump(), "relaxed": plan.relaxed_constraints}

    def _validate(self, plan_id: str):
        rec = self.d.store.get_plan(plan_id)
        plan = Plan.model_validate(rec["plan"]) if rec else None
        if plan is None:
            return None, None
        margin = self.d.settings.STALE_MARGIN_C if self.ctx.inputs.any_stale() else 0.0
        statuses = {n: self.ctx.inputs.status(n) for n in ("weather", "market", "telemetry")}
        return plan, _validate(self.d.K, self.d.plant, plan, rec, statuses, margin)

    def validate_plan(self, plan_id: str) -> Dict[str, Any]:
        c = self.ctx
        plan, res = self._validate(plan_id)
        if res is None:
            c.add("validate_plan", f"plan_id {plan_id}", "reject", "Unknown plan_id", "V1: plan_id is not in the optimizer registry")
            return {"ok": False, "reasons": ["V1: plan_id is not in the optimizer registry"]}
        c.add("validate_plan", f"Plan {plan_id}: bounds, slew, ramp, dry run", "accept" if res.ok else "reject",
              "Deterministic validator is the only gate to the actuator", "OK" if res.ok else "; ".join(res.texts())[:300])
        c.done.add("validate_plan")
        return {"ok": res.ok, "reasons": res.texts()}

    def execute_step(self, plan_id: str) -> Dict[str, Any]:
        c = self.ctx
        plan, res = self._validate(plan_id)  # validator is mandatory before any action
        now = datetime.now(timezone.utc)
        for old in self.d.store.list_actions():  # supersede stale pending actions
            if old["status"] == ActionStatus.awaiting_approval.value:
                old.update(status=ActionStatus.rejected.value, rejection_reasons=[f"Superseded by cycle {c.cycle_id}"])
                self.d.store.put_action(old["action_id"], old)
        act = {"action_id": "act_" + c.cycle_id[4:] + f"_{len(c.actions)}", "cycle_id": c.cycle_id, "plan_id": plan_id, "step": 0,
               "status": ActionStatus.proposed.value, "rejection_reasons": [], "created_at": now.isoformat(), "applied_at": None, "note": None}
        if res is None or not res.ok:
            reasons = res.texts() if res else ["V1: plan_id is not in the optimizer registry"]
            act.update(status=ActionStatus.rejected.value, rejection_reasons=reasons)
            c.notice(f"Action rejected: {'; '.join(reasons)[:200]}.")
            c.add("execute_step", f"Validator rejected plan {plan_id}", "do not apply", "Safety validator must approve first", "; ".join(reasons)[:300])
        elif c.mode == Mode.approval_required:
            act.update(status=ActionStatus.awaiting_approval.value)
            c.add("execute_step", "approval_required mode", "queue action for operator approval", "Operator approval needed before the twin changes",
                  f"{act['action_id']} awaiting_approval")
        else:
            act = apply_action(self.d, act)
            if act["status"] == ActionStatus.rolled_back.value:
                c.notice("Twin response deviated from prediction; action rolled back.")
            c.add("execute_step", "autonomous mode; validator approved", f"apply step 0 -> {act['status']}",
                  "Only the first plan step is applied (receding horizon)", f"{act['action_id']} {act['status']}; {act.get('verification', {})}")
        c.actions.append(act); self.d.store.put_action(act["action_id"], act)
        c.done.add("execute_step")
        return {"action_id": act["action_id"], "status": act["status"], "reasons": act["rejection_reasons"]}

    def verify_outcome(self, action_id: str) -> Dict[str, Any]:
        c = self.ctx
        act = self.d.store.get_action(action_id)
        if act is None:
            return {"error": "Not found."}
        v = act.get("verification")
        c.add("verify_outcome", f"Action {action_id} is {act['status']}",
              "within tolerance" if act["status"] == "applied" else act["status"], "Compare twin response with prediction",
              str(v) if v else "nothing to verify yet")
        c.done.add("verify_outcome")
        return {"status": act["status"], "verification": v}

    def rollback(self, action_id: str) -> Dict[str, Any]:
        c = self.ctx
        act = self.d.store.get_action(action_id)
        if act is None:
            return {"error": "Not found."}
        if act["status"] != ActionStatus.applied.value:
            return {"error": "This action is no longer awaiting approval."}
        act = rollback_action(self.d, act); self.d.store.put_action(action_id, act)
        c.add("rollback", f"Action {action_id}", "rolled back", "Restore previous twin state", "rolled_back")
        return {"status": act["status"]}

    def notify_operator(self, message: str) -> Dict[str, Any]:
        self.d.store.add_alert(f"{self.ctx.cycle_id}: {message[:300]}")
        self.ctx.add("notify_operator", "Operator attention requested", "alert written", "In-app alert log", message[:200])
        return {"ok": True}

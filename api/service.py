"""Control service: runs cycles, approvals, rollbacks, inputs cache, performance."""
from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import psutil

from agent.context import CycleContext, Deps
from agent.loop import run_agent
from agent.tools import apply_action, rollback_action
from data.ingest import Inputs
from data.schemas import ActionStatus, CycleStatus, Mode, Status, WeatherForecast, WeatherPoint
from forecast.grid import wet_bulb_stull
from data.price_history import ensure_fresh
from optimizer.mpc import Horizon


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        self.status, self.code, self.message = status, code, message


class ControlService:
    def __init__(self, deps: Deps) -> None:
        self.d = deps
        self._run_lock = threading.Lock()
        self._contexts: Dict[str, CycleContext] = {}
        self._inputs: Optional[Inputs] = None
        self._proc = psutil.Process()
        self._scheduler: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def mode(self) -> Mode:
        return Mode(self.d.store.get_kv("mode", Mode.approval_required.value))

    def get_settings(self) -> Dict[str, Any]:
        return {
            "mode": self.mode().value,
            "cycle_interval_minutes": self.d.store.get_kv("interval", self.d.settings.CYCLE_INTERVAL_MINUTES),
        }

    def put_settings(self, mode: Optional[str], interval: Optional[int]) -> Dict[str, Any]:
        if mode is not None:
            self.d.store.set_kv("mode", Mode(mode).value)
        if interval is not None:
            if not 1 <= interval <= 240:
                raise ApiError(422, "validation_error", "cycle_interval_minutes must be between 1 and 240.")
            self.d.store.set_kv("interval", interval)
        return self.get_settings()

    def inputs(self, max_age_s: float = 20.0) -> Inputs:
        if self._inputs is None or (datetime.now(timezone.utc) - self._inputs.fetched_at).total_seconds() > max_age_s:
            self._inputs = self.d.ingestor.fetch_all()
        return self._inputs

    def inputs_payload(self) -> Dict[str, Any]:
        inp = self.inputs()
        all_failed = all(st.provenance.status == Status.FAILED for st in inp.states.values())
        return {
            "sources": inp.to_dict(),
            "notices": inp.notices,
            "waiting_for_live_data": all_failed,
            "message": "Waiting for live data…" if all_failed else None,
            "llm_configured": bool(getattr(self.d.llm, "configured", False)),
            "assumed_constants": [
                {"name": c.name, "value": c.value, "unit": c.unit, "source": c.source} for c in self.d.K.assumed()
            ],
        }

    def data_health(self) -> List[Dict[str, Any]]:
        return self.inputs().health(self.d.settings)

    def start_cycle(self, mode: Optional[str] = None, stress_offset_c: float = 0.0, background: bool = True) -> str:
        if not 0 <= stress_offset_c <= 8:
            raise ApiError(422, "validation_error", "Stress offset must be between 0 and 8 °C.")
        if not self._run_lock.acquire(blocking=False):
            raise ApiError(409, "conflict", "A control cycle is already running.")
        ctx = CycleContext(
            deps=self.d,
            mode=Mode(mode) if mode else self.mode(),
            stress_offset_c=float(stress_offset_c),
        )
        self._contexts[ctx.cycle_id] = ctx
        self.d.store.put_cycle(ctx.cycle_id, self._result(ctx))
        if background:
            threading.Thread(target=self._run, args=(ctx,), daemon=True).start()
        else:
            self._run(ctx)
        return ctx.cycle_id

    def _prepare(self, ctx: CycleContext) -> bool:
        inp = self.d.ingestor.fetch_all()
        self._inputs = inp
        ctx.inputs = inp
        for n in inp.notices:
            ctx.notice(n)
        w, m, t = inp.get("weather"), inp.get("market"), inp.get("telemetry")
        if w is None or m is None or t is None:
            ctx.status = CycleStatus.held_safe
            for n, v in (("weather", w), ("market", m), ("telemetry", t)):
                if v is None:
                    self.d.store.add_alert(f"{ctx.cycle_id}: {n} unavailable; holding safe state")
            if m is None and inp.states["market"].provenance.stale_reason:
                ctx.notice(f"Market prices: {inp.states['market'].provenance.stale_reason}")
            ctx.add(
                "prepare",
                "A required input (weather, market prices or IT telemetry) is FAILED",
                "hold safe state",
                "No value is ever fabricated; retry next cycle",
                "held_safe",
            )
            return False

        if ctx.counterfactual:
            pts = [
                WeatherPoint(
                    **{
                        **p.model_dump(),
                        "temp_dry_c": p.temp_dry_c + ctx.stress_offset_c,
                        "temp_wet_c": wet_bulb_stull(p.temp_dry_c + ctx.stress_offset_c, p.rh_pct),
                    }
                )
                for p in w.points
            ]
            w = WeatherForecast(points=pts)
            ctx.notice(f"Counterfactual: +{ctx.stress_offset_c} °C applied on top of live data")

        ctx.weather, ctx.market, ctx.telemetry = w, m, t

        # 1. Spot-price model over the whole horizon, shown next to the published day-ahead prices so it can be judged.
        #    Control only uses it for steps whose auction result is not yet published (see data/market.py).
        try:
            ensure_fresh(self.d.settings)
            pf = self.d.ingestor.predictor.predict([p.ts for p in w.points])
            ctx.extras["ml_predicted_prices"] = pf.prices
            ctx.extras["ml_lower_bound"] = pf.lower_bound
            ctx.extras["ml_upper_bound"] = pf.upper_bound
            ctx.extras["ml_model_name"] = pf.model
            ctx.extras["ml_mae"] = pf.mae
        except Exception as exc:
            ctx.extras["ml_predicted_prices"] = None
            ctx.extras["ml_unavailable_reason"] = f"{type(exc).__name__}: {exc}"[:200]

        # 2. Optional GenAI Risk Classifier (only when a real grid_signal is present and LLM supports it)
        grid_sig = inp.get("grid_signal")
        ctx.extras["llm_risk"] = None
        if (
            grid_sig is not None
            and getattr(self.d.llm, "configured", False)
            and hasattr(self.d.llm, "classify_grid_risk")
        ):
            try:
                features = {"current_frequency_hz": round(50.0 - grid_sig.value / 1000.0, 4)}
                risk_out = self.d.llm.classify_grid_risk(features, w.points[0].temp_dry_c)
                if isinstance(risk_out, dict) and risk_out.get("risk_level") in ("LOW", "MEDIUM", "HIGH"):
                    ctx.extras["llm_risk"] = risk_out["risk_level"]
            except Exception:
                ctx.extras["llm_risk"] = None

        st = self.d.twin.state()
        prev = None if st.get("q_mech_kw") is None else st["q_mech_kw"] + st["q_evap_kw"]

        ctx.horizon = Horizon(
            x0=self.d.twin.x0(),
            q_it_kw=t.it_load_kw,
            t_out=[p.temp_dry_c for p in w.points],
            t_wet=[p.temp_wet_c for p in w.points],
            price=[p.price_eur_per_kwh for p in m.points],
            stress=[False] * 48,
            prev_cooling_kw=prev,
            stale_margin_c=self.d.settings.STALE_MARGIN_C if inp.any_stale() else 0.0,
        )
        return True

    def _run(self, ctx: CycleContext) -> None:
        try:
            if self._prepare(ctx):
                run_agent(ctx)
            if ctx.status == CycleStatus.running:
                ctx.status = CycleStatus.completed
        except Exception as exc:
            ctx.status = CycleStatus.failed
            ctx.notice(f"Cycle failed: {type(exc).__name__}")
            ctx.add(
                "cycle",
                "unexpected error",
                "stop cycle",
                "Errors are logged without secrets",
                f"{type(exc).__name__}: {exc}"[:200],
            )
        finally:
            self.d.store.put_cycle(ctx.cycle_id, self._result(ctx, final=True))
            self._run_lock.release()

    def _result(self, ctx: CycleContext, final: bool = False) -> Dict[str, Any]:
        runtime = round((time.perf_counter() - ctx.started) * 1000, 1)
        mem = round(self._proc.memory_info().rss / 1048576, 1)
        comparison = None
        if ctx.plan and ctx.baseline:
            comparison = {}
            for k in ("cost_eur", "peak_kw", "water_l", "max_temp_c", "energy_kwh"):
                p, b = getattr(ctx.plan.metrics, k), getattr(ctx.baseline, k)
                comparison[k] = {
                    "plan": p,
                    "baseline": b,
                    "delta": p - b,
                    "savings_pct": None if b == 0 else round((b - p) / b * 100, 1),
                }
        traj = None
        if ctx.extras.get("plan_sim"):
            traj = {"plan": ctx.extras["plan_sim"], "baseline": ctx.extras["baseline_sim"]}

        res = {
            "cycle_id": ctx.cycle_id,
            "status": ctx.status.value,
            "mode": ctx.mode.value,
            "counterfactual": ctx.counterfactual,
            "stress_offset_c": ctx.stress_offset_c,
            "deterministic_mode": ctx.deterministic,
            "stress_windows": [
                {"start": w["start"].isoformat(), "end": w["end"].isoformat(), "indicators": w["indicators"]}
                for w in ctx.stress_windows
            ],
            "plan": ctx.plan.model_dump(mode="json") if ctx.plan else None,
            "baseline": ctx.baseline.model_dump() if ctx.baseline else None,
            "comparison": comparison,
            "trajectories": traj,
            "trace": [t.model_dump(mode="json") for t in ctx.trace],
            "notices": ctx.notices,
            "actions": [{k: v for k, v in a.items() if not k.startswith("_")} for a in ctx.actions],
            "data_health": ctx.inputs.health(self.d.settings) if ctx.inputs else [],
            "forecast": self._forecast(ctx),
            "runtime_ms": runtime,
            "memory_mb": mem,
            "message": (
                "No stress window detected."
                if final and ctx.status == CycleStatus.completed and not ctx.stress_windows
                else None
            ),
            "llm_risk": ctx.extras.get("llm_risk"),
            "ml_predicted_prices": ctx.extras.get("ml_predicted_prices"),
            "ml_lower_bound": ctx.extras.get("ml_lower_bound"),
            "ml_upper_bound": ctx.extras.get("ml_upper_bound"),
            "ml_model_name": ctx.extras.get("ml_model_name"),
            "ml_mae": ctx.extras.get("ml_mae"),
            "ml_unavailable_reason": ctx.extras.get("ml_unavailable_reason"),
        }
        if final:
            self.d.store.set_kv("perf", {"cycle_id": ctx.cycle_id, "runtime_ms": runtime, "memory_mb": mem})
        return res

    @staticmethod
    def _forecast(ctx: CycleContext) -> Optional[Dict[str, Any]]:
        if not ctx.weather or not ctx.market:
            return None
        return {
            "ts": [p.ts.isoformat() for p in ctx.weather.points],
            "temp_dry_c": [p.temp_dry_c for p in ctx.weather.points],
            "temp_wet_c": [p.temp_wet_c for p in ctx.weather.points],
            "price_eur_per_kwh": [p.price_eur_per_kwh for p in ctx.market.points],
            "spot_eur_per_kwh": [p.spot_eur_per_kwh for p in ctx.market.points],
            "price_kind": [p.price_kind.value for p in ctx.market.points],
            "s1_threshold_eur_per_kwh": ctx.market.s1_threshold_eur_per_kwh,
            "price_adder_eur_per_kwh": ctx.market.adder_eur_per_kwh,
        }

    def get_cycle(self, cycle_id: str) -> Dict[str, Any]:
        ctx = self._contexts.get(cycle_id)
        if ctx is not None and ctx.status == CycleStatus.running:
            return self._result(ctx)
        c = self.d.store.get_cycle(cycle_id)
        if c is None:
            raise ApiError(404, "not_found", "Not found.")
        return c

    def latest_cycle(self) -> Dict[str, Any]:
        c = self.d.store.latest_cycle()
        if c is None:
            raise ApiError(404, "not_found", "Not found.")
        return self.get_cycle(c["cycle_id"])

    def context(self, cycle_id: str) -> Optional[CycleContext]:
        return self._contexts.get(cycle_id)

    def _action(self, action_id: str) -> Dict[str, Any]:
        a = self.d.store.get_action(action_id)
        if a is None:
            raise ApiError(404, "not_found", "Not found.")
        return a

    def list_actions(self) -> Dict[str, Any]:
        acts = [{k: v for k, v in a.items() if not k.startswith("_")} for a in self.d.store.list_actions()]
        return {"actions": acts, "operator_alerts": self.d.store.list_alerts()}

    def approve(self, action_id: str) -> Dict[str, Any]:
        a = self._action(action_id)
        if a["status"] != ActionStatus.awaiting_approval.value:
            raise ApiError(409, "conflict", "This action is no longer awaiting approval.")
        a = apply_action(self.d, a)
        self.d.store.put_action(action_id, a)
        return {k: v for k, v in a.items() if not k.startswith("_")}

    def reject(self, action_id: str) -> Dict[str, Any]:
        a = self._action(action_id)
        if a["status"] != ActionStatus.awaiting_approval.value:
            raise ApiError(409, "conflict", "This action is no longer awaiting approval.")
        a.update(status=ActionStatus.rejected.value, rejection_reasons=["Rejected by operator"])
        self.d.store.put_action(action_id, a)
        return {k: v for k, v in a.items() if not k.startswith("_")}

    def rollback(self, action_id: str) -> Dict[str, Any]:
        a = self._action(action_id)
        if a["status"] != ActionStatus.applied.value:
            raise ApiError(409, "conflict", "This action is no longer awaiting approval.")
        a = rollback_action(self.d, a)
        self.d.store.put_action(action_id, a)
        return {k: v for k, v in a.items() if not k.startswith("_")}

    def twin_state(self) -> Dict[str, Any]:
        return {**self.d.twin.state(), "label": "SIMULATED ACTUATOR - DIGITAL TWIN"}

    def performance(self) -> Dict[str, Any]:
        return self.d.store.get_kv(
            "perf",
            {
                "cycle_id": None,
                "runtime_ms": None,
                "memory_mb": round(self._proc.memory_info().rss / 1048576, 1),
            },
        )

    def backtest(self) -> Dict[str, Any]:
        from forecast.backtest import run_backtest

        return run_backtest(self.d.settings)

    def start_scheduler(self) -> None:
        if self._scheduler or not self.d.settings.AUTO_CYCLE:
            return

        def loop() -> None:
            while not self._stop.wait(float(self.get_settings()["cycle_interval_minutes"]) * 60):
                try:
                    self.start_cycle()
                except ApiError:
                    pass

        self._scheduler = threading.Thread(target=loop, daemon=True)
        self._scheduler.start()

    def stop(self) -> None:
        self._stop.set()
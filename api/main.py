"""FastAPI backend (spec Q4 endpoints, Q8 error shapes, Q9 security)."""
from __future__ import annotations

import asyncio
import hmac
import json
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from agent.context import Deps
from agent.llm import LLMClient
from api.service import ApiError, ControlService
from config.settings import get_settings, load_constants
from data.ingest import Ingestor
from data.schemas import CycleStatus
from data.store import Store
from twin.model import DigitalTwin, Plant

MAX_BODY = 64 * 1024


def build_service() -> ControlService:
    s = get_settings()
    K = load_constants(s.CONSTANTS_PATH)
    store = Store(s.DB_PATH)
    plant = Plant.from_constants(K)
    deps = Deps(
        settings=s,
        K=K,
        plant=plant,
        store=store,
        twin=DigitalTwin(plant, K, store),
        ingestor=Ingestor(s, K, store),
        llm=LLMClient(s),
    )
    return ControlService(deps)


class CycleBody(BaseModel):
    mode: Optional[str] = None
    stress_offset_c: float = 0.0


class SettingsBody(BaseModel):
    mode: Optional[str] = None
    cycle_interval_minutes: Optional[int] = None


def create_app(service: Optional[ControlService] = None) -> FastAPI:
    svc = service or build_service()
    s = svc.d.settings

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        svc.start_scheduler()
        yield
        svc.stop()

    app = FastAPI(title="GridSentinel API", lifespan=lifespan)
    app.state.service = svc
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[s.DASHBOARD_ORIGIN],
        allow_methods=["GET", "POST", "PUT"],
        allow_headers=["Content-Type", "X-Operator-Token"],
    )

    @app.middleware("http")
    async def limit_body(request: Request, call_next):
        if int(request.headers.get("content-length", "0") or 0) > MAX_BODY:
            return JSONResponse(
                {"error": {"code": "payload_too_large", "message": "Request body too large."}},
                status_code=413,
            )
        return await call_next(request)

    @app.exception_handler(ApiError)
    async def api_error(_: Request, e: ApiError):
        return JSONResponse({"error": {"code": e.code, "message": e.message}}, status_code=e.status)

    @app.exception_handler(RequestValidationError)
    async def val_error(_: Request, e: RequestValidationError):
        return JSONResponse({"error": {"code": "validation_error", "message": "Invalid request body."}}, status_code=422)

    def auth(x_operator_token: Optional[str] = Header(default=None)) -> None:
        if not x_operator_token or not hmac.compare_digest(x_operator_token, s.OPERATOR_TOKEN):
            raise ApiError(401, "unauthorized", "Operator token required.")

    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        # AC1 strictly requires {"status": "ok"}
        return {"status": "ok"}

    @app.get("/api/inputs")
    def inputs() -> Dict[str, Any]:
        return svc.inputs_payload()

    @app.get("/api/data-health")
    def data_health() -> Dict[str, Any]:
        return {"sources": svc.data_health()}

    @app.post("/api/cycles", status_code=202, dependencies=[Depends(auth)])
    def start(body: CycleBody) -> Dict[str, str]:
        return {"cycle_id": svc.start_cycle(body.mode, body.stress_offset_c)}

    @app.get("/api/cycles/latest")
    def latest() -> Dict[str, Any]:
        return svc.latest_cycle()

    @app.get("/api/cycles/{cycle_id}")
    def cycle(cycle_id: str) -> Dict[str, Any]:
        return svc.get_cycle(cycle_id)

    @app.get("/api/cycles/{cycle_id}/events")
    async def events(cycle_id: str):
        svc.get_cycle(cycle_id)  # 404 if unknown

        async def gen():
            sent = 0
            while True:
                ctx = svc.context(cycle_id)
                trace = [t.model_dump(mode="json") for t in ctx.trace] if ctx else svc.get_cycle(cycle_id)["trace"]
                done = ctx is None or ctx.status != CycleStatus.running
                for t in trace[sent:]:
                    yield f"data: {json.dumps(t)}\n\n"
                sent = len(trace)
                if done:
                    yield "event: end\ndata: {}\n\n"
                    return
                await asyncio.sleep(0.3)

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/api/twin/state")
    def twin() -> Dict[str, Any]:
        return svc.twin_state()

    @app.get("/api/actions")
    def actions() -> Dict[str, Any]:
        return svc.list_actions()

    @app.post("/api/actions/{action_id}/approve", dependencies=[Depends(auth)])
    def approve(action_id: str) -> Dict[str, Any]:
        return svc.approve(action_id)

    @app.post("/api/actions/{action_id}/reject", dependencies=[Depends(auth)])
    def reject(action_id: str) -> Dict[str, Any]:
        return svc.reject(action_id)

    @app.post("/api/actions/{action_id}/rollback", dependencies=[Depends(auth)])
    def rollback(action_id: str) -> Dict[str, Any]:
        return svc.rollback(action_id)

    @app.put("/api/settings", dependencies=[Depends(auth)])
    def put_settings(body: SettingsBody) -> Dict[str, Any]:
        if body.mode not in (None, "autonomous", "approval_required"):
            raise ApiError(422, "validation_error", "mode must be autonomous or approval_required.")
        return svc.put_settings(body.mode, body.cycle_interval_minutes)

    @app.get("/api/forecast/backtest")
    def backtest() -> Dict[str, Any]:
        return svc.backtest()

    @app.get("/api/performance")
    def perf() -> Dict[str, Any]:
        return svc.performance()

    return app


def app_factory() -> FastAPI:  # uvicorn api.main:app_factory --factory
    return create_app()


app = create_app()
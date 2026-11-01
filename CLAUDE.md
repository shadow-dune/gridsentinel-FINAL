# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

GridSentinel: a preemptive cooling controller for a **simulated** data center in Frankfurt am Main, Germany (bidding zone DE-LU, prices in EUR) (a physics-based digital twin — no real hardware). Each control cycle ingests live inputs, forecasts 12 h (48 × 15-min steps), detects stress windows, solves an LP plan, gates it through a deterministic safety validator, and applies only step 0 to the twin (receding horizon). An LLM may orchestrate the tool calls but **never sets numeric setpoints** — the optimizer does.

Python 3.12, FastAPI backend + Streamlit dashboard, SQLite persistence (`.data/gridsentinel.db`). Not a git repo.

## Commands

```bash
# `.venv/` (Windows layout) and `venv/` (python3.10) were copied from other machines and do not work here;
# use the pyenv Python 3.12 directly or create a fresh venv.
pip install -r requirements.txt
cp .env.example .env                 # Frankfurt defaults filled in; set OPERATOR_TOKEN (>= 8 chars)
python fetch_history.py              # real DE-LU day-ahead history -> .data/price_history.csv (backend also auto-refreshes)

set -a; source .env; set +a
uvicorn api.main:app_factory --factory --port 8000   # backend
streamlit run ui/app.py                              # dashboard; reads API_BASE_URL + OPERATOR_TOKEN from env
./start.sh                                           # both (what the Docker / HF Spaces image runs, UI on $PORT=7860)

python -m pytest tests -q                            # all tests
python -m pytest tests/test_core.py::test_ut3_stable_over_horizon -q   # single test
```

Tests are fully offline: `tests/conftest.py` sets env vars via `monkeypatch`, blocks outbound HTTP (patches `data.http.get_json` — call it as `dhttp.get_json` at call time so the patch applies), clears the `get_settings()` lru_cache, and replaces `svc.d.ingestor.chains[...]` with synthetic sources. The market chain runs the real `data.market.build_market` on a TEST-ONLY series from `make_price_series()`. Use the `service` / `client` fixtures and the `HDR` operator-token header for API tests. `test_core.py` = model/optimizer/validator unit tests (UT*, AC8/9); `test_system.py` = API, agent, ingestion and E-case scenarios.


## Architecture (the cycle pipeline)

Wiring lives in `api/main.py:build_service()` → a `Deps` bundle (`agent/context.py`: settings, constants `K`, `Plant`, `Store`, `DigitalTwin`, `Ingestor`, `LLMClient`) handed to `api/service.py:ControlService`. `ControlService.start_cycle` runs a cycle (in a background thread; also a scheduler loop when `AUTO_CYCLE=true`), with state in a `CycleContext`.

1. **Ingest** (`data/ingest.py`): per source (`weather`, `market`, `grid_signal`, `alerts`, `telemetry`) try a fallback chain of fetchers → schema-validate (`data/schemas.py`) → status `LIVE` / `FALLBACK_LIVE`; on failure fall back to the last-known-good in SQLite as `STALE` if within `MAX_AGE_*`, else `FAILED`. STALE widens the safety margin by `STALE_MARGIN_C`; FAILED weather/market ⇒ cycle holds the safe state. Every value carries `Provenance` with a `SignalKind` (MEASURED / PUBLISHED_SCHEDULE / MODEL_DERIVED).
   - Price (`data/market.py`): EPEX day-ahead spot for `BIDDING_ZONE` from Energy-Charts (no key) + `market.price_adder_eur_per_kwh` (assumed). Steps not yet published (next day appears ~12:45 CET) are filled by the price model, else same-time-yesterday; every `MarketPoint` carries `price_kind`. S1 (`is_peak_window`) = spot ≥ 75th percentile of the previous 7 days. Spot prices can be negative; don't add `> 0` constraints.
   - Grid signal: Energy-Charts frequency, stored as `grid_frequency_deficit_mhz` (positive = under 50 Hz), so S2's "above 90th percentile" means under-frequency stress.
   - Price model (`forecast/predictor.py`, owned by `Ingestor.predictor`): trained on `PRICE_HISTORY_PATH` (tz-aware UTC timestamps; naive ones are rejected), refreshed by `data/price_history.ensure_fresh`. Candidates use ≥1-day lags; winner picked by 7-day walk-forward MAE (on real data `same_time_yesterday` currently wins).
   - Telemetry is psutil on the backend host, mapped to IT kW.
2. **Forecast / stress** (`forecast/`): `grid.py` (15-min step times, Stull wet-bulb), `predictor.py`, `stress.py` (S1–S4 windows), `backtest.py` (baseline MAE for `/api/forecast/backtest`).
3. **Agent** (`agent/loop.py:run_agent`): if `LLM_MODEL` etc. are configured, an OpenAI-compatible tool loop (`agent/llm.py` — the only place LLM access happens) capped at `MAX_AGENT_ITERATIONS`; otherwise or on `LLMError`, a deterministic sequence. Even after the LLM finishes, the deterministic path runs any required steps it skipped. Tools are allowlisted in `agent/tools.py` (`get_inputs, parse_alerts, detect_stress, optimize_plan, validate_plan, execute_step, verify_outcome, rollback, notify_operator`). Alert text is treated as untrusted data.
4. **Optimizer** (`optimizer/mpc.py`, `Horizon.price` = all-in EUR/kWh): LP via SciPy HiGHS over two continuous cooling variables (mechanical, evaporative); mode derived via `mode_of`. Also computes a `reactive_baseline` for comparison; the plan's final mass temperature must not be warmer than the baseline's. Relaxation order on infeasibility: peak cap → water budget → recommended max; hard limits (18–27 °C) never relax. Plans are registered with a `plan_hash`.
5. **Safety** (`safety/validator.py:validate_plan`, rules V1–V6): the sole gate to the twin. `execute_step` always re-validates; V1 rejects plans whose hash doesn't match the registry (edited plans).
6. **Twin** (`twin/model.py`): 2-node RC thermal model (air + mass) + plant + water, discretised exactly with a matrix exponential (explicit Euler would be unstable here). State changes only when an action is applied. After applying, telemetry is re-sampled; >1.0 °C deviation from prediction triggers automatic rollback.

Modes (`data/schemas.py:Mode`): `autonomous` applies immediately; `approval_required` queues an `awaiting_approval` action for `/api/actions/{id}/approve|reject`. A new cycle supersedes older pending actions.

## API / UI

`api/main.py` routes are all under `/api/*`; mutating endpoints (`POST /api/cycles`, action approve/reject/rollback, `PUT /api/settings`) require header `X-Operator-Token`. Errors use the shape `{"error": {"message": ...}}` via `ApiError`; tests assert exact message strings. `GET /api/cycles/{id}/events` streams the cycle's decision log. `ui/app.py` is a single large Streamlit file that only talks to the backend over HTTP.

## Project conventions

- **No invented/dummy data in runtime code.** Missing inputs must surface as FAILED / held-safe with a notice, never a substituted constant. Synthetic data is allowed only under `tests/`.
- Every entry in `config/constants.yaml` must have `value, unit, source, status` (`cited|derived|assumed`); `load_constants` raises otherwise and a test checks it. `assumed` ones are shown in the UI.
- Code comments reference spec IDs (AC#, E#, UT#, V#, S#, Q#); test names follow the same IDs — keep them aligned when changing behavior.
- `ctx.extras` is shared between `_prepare` (ml_*, llm_risk) and `optimize_plan` (sims): merge into it, never reassign.
- `.data/india-backup/` holds the pre-migration India files (old DB, synthetic IEX CSV, TNERC tariff); nothing reads them.
- `ui/app.py:ensure_backend` starts the backend as a subprocess when `API_BASE_URL` is local and unhealthy (needed for Streamlit Community Cloud); its startup log goes to `.data/backend.log`. `load_dotenv` is pinned to the project's `.env`, because a stale India-era `.env` exists in the parent folder.
- Settings are read through cached `config.settings.get_settings()`; call `get_settings.cache_clear()` after changing env in tests.
- README lists external endpoints (Open-Meteo minutely_15/archive, OpenWeatherMap, Electricity Maps, LLM tool calling) as UNVERIFIED against live services.

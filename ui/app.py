"""GridSentinel dashboard (Streamlit). Talks to the backend over HTTP only; the operator token stays server-side."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, List, Optional, Tuple
from zoneinfo import ZoneInfo

import httpx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from dotenv import load_dotenv

# Streamlit Community Cloud: copy root-level secrets into the environment explicitly, so this file and the backend
# child process (which inherits os.environ) see them no matter when Streamlit loads secrets. Real env vars win.
try:
    for _key, _value in st.secrets.items():
        if isinstance(_value, (str, int, float, bool)):
            os.environ.setdefault(_key, str(_value))
except Exception:
    pass  # no secrets.toml (local runs use .env)

load_dotenv(Path(__file__).resolve().parent.parent / ".env")  # only this project's .env, never a parent folder's

API = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
TOKEN = os.environ.get("OPERATOR_TOKEN", "")
HEAD = {"X-Operator-Token": TOKEN}
REFRESH_RATE = int(os.environ.get("DASHBOARD_REFRESH_SECONDS", "300"))
TZ_NAME = os.environ.get("FACILITY_TIMEZONE", "Europe/Berlin")
try:
    TZ = ZoneInfo(TZ_NAME)
except Exception:
    TZ, TZ_NAME = ZoneInfo("Europe/Berlin"), "Europe/Berlin"

CHIP = {
    "LIVE": "#15803d",
    "FALLBACK_LIVE": "#1d4ed8",
    "STALE": "#b45309",
    "FAILED": "#b91c1c",
    "NOT CONFIGURED": "#64748b",
    "PUBLISHED": "#0f766e",
    "CACHED": "#1d4ed8",
    "FALLBACK": "#b45309",
    "applied": "#15803d",
    "rejected": "#b91c1c",
    "awaiting_approval": "#b45309",
    "rolled_back": "#475569",
    "proposed": "#475569",
    "failed": "#b91c1c",
}
OPTIONAL_SOURCES = {"alerts", "grid_signal"}  # weather, market and telemetry are required: never relabelled
TAG_RE = re.compile(r"^\[(LIVE|CACHED|FALLBACK)\]\s*")

st.set_page_config(page_title="GridSentinel Controller", layout="wide")
st.markdown(
    """<style>
[data-testid="stHeader"] {display: none;}
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
.stApp {background:#F8FAFC;color:#0F172A;font-family:Inter,-apple-system,"Segoe UI",sans-serif}
.card {background:#FFFFFF;border:1px solid #E2E8F0;border-radius:12px;padding:16px;margin-bottom:12px;box-shadow:0 1px 3px rgba(15,23,42,0.06)}
.card h4 {margin:0 0 8px 0;font-size:0.92rem;font-weight:700;color:#334155;text-transform:uppercase;letter-spacing:0.03em}
.kpi-card {background:#FFFFFF;border:1px solid #E2E8F0;border-radius:12px;padding:18px 14px;text-align:center;box-shadow:0 1px 3px rgba(15,23,42,0.06)}
.kpi-label {color:#475569;font-size:0.78rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em}
.kpi-value {font-size:1.75rem;font-weight:800;color:#0F172A;margin:6px 0 4px 0;font-variant-numeric:tabular-nums}
.kpi-sub {color:#64748B;font-size:0.82rem;margin-bottom:6px}
.kpi-delta {display:inline-block;padding:3px 10px;border-radius:999px;font-size:0.78rem;font-weight:700}
.muted {color:#64748B;font-size:0.82rem;line-height:1.35}
.chip {display:inline-block;color:#fff;border-radius:999px;padding:2px 10px;font-size:0.70rem;font-weight:700;margin-right:6px;letter-spacing:0.02em}
.tag {display:inline-block;background:#F1F5F9;border:1px solid #CBD5E1;border-radius:6px;padding:1px 7px;font-size:0.68rem;font-weight:600;color:#475569}
.badge {border:2px solid #DC2626;background:#FEF2F2;color:#DC2626;border-radius:8px;padding:6px 12px;font-weight:800;font-size:0.8rem;display:inline-block}
.banner {border-radius:8px;padding:10px 14px;margin:8px 0;font-size:0.88rem;font-weight:500}
.warn {background:#FFFBEB;border:1px solid #FCD34D;color:#92400E}
.err {background:#FEF2F2;border:1px solid #FCA5A5;color:#991B1B}
.info {background:#EFF6FF;border:1px solid #BFDBFE;color:#1E40AF}
.trace {border-left:4px solid #2563EB;background:#FFFFFF;border-radius:8px;padding:12px 14px;margin-bottom:10px;font-size:0.88rem;box-shadow:0 1px 2px rgba(15,23,42,0.05)}
.stRadio label p, .stSlider label p, .stMarkdown p {color:#0F172A !important;}
button[kind="primary"] {background-color:#DC2626 !important;color:white !important;border:none !important;border-radius:8px !important;font-weight:700 !important;}
button[kind="primary"]:hover {background-color:#B91C1C !important;color:white !important;}
button[kind="secondary"] {background-color:#F1F5F9 !important;color:#0F172A !important;border:1px solid #CBD5E1 !important;border-radius:8px !important;}
button[kind="secondary"]:hover {background-color:#E2E8F0 !important;color:#0F172A !important;}
[data-testid="stDataFrame"] > div {background-color:white !important;color:#0F172A !important;}
</style>""",
    unsafe_allow_html=True,
)

if not TOKEN:
    st.error("⚠️ OPERATOR_TOKEN is missing from your .env file.")


# ------------------------------------------------------------------------------------------ backend autostart
# Single-process hosts (Streamlit Community Cloud) only run this file. If API_BASE_URL points at this machine and
# nothing answers there, start the FastAPI backend as a child process. With start.sh / Docker / Codespaces the
# backend is already up, the health check passes and nothing is started. Set BACKEND_AUTOSTART=false to disable.
ROOT = Path(__file__).resolve().parent.parent


def _backend_healthy() -> bool:
    try:
        return httpx.get(API + "/api/health", timeout=1.0).status_code == 200
    except Exception:
        return False


def _pump_output(proc: subprocess.Popen, log_path: Path) -> None:
    """Copy backend output to .data/backend.log and to this process's stdout (visible in Streamlit Cloud logs)."""
    with open(log_path, "ab") as log:
        for line in iter(proc.stdout.readline, b""):
            log.write(line)
            log.flush()
            sys.stdout.write("[backend] " + line.decode(errors="replace"))
            sys.stdout.flush()


@st.cache_resource(show_spinner="Starting the controller backend…")
def ensure_backend() -> Optional[subprocess.Popen]:
    """Runs once per Streamlit server process. Raises (and is retried on the next page load) if the start fails.

    Returns the child process (None when the backend is external or already running)."""
    if _backend_healthy():
        return None
    target = urlparse(API)
    if os.environ.get("BACKEND_AUTOSTART", "true").lower() == "false" or target.hostname not in ("127.0.0.1", "localhost"):
        return None
    log_path = ROOT / ".data" / "backend.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[frontend] starting backend on {target.hostname}:{target.port or 8000}", flush=True)
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app_factory", "--factory",
         "--host", "127.0.0.1", "--port", str(target.port or 8000)],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    threading.Thread(target=_pump_output, args=(proc, log_path), daemon=True).start()
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            time.sleep(0.3)  # let the pump thread flush the last lines
            tail = log_path.read_text(errors="replace")[-1500:]
            # never show received settings on the page: they can contain OPERATOR_TOKEN / LLM_API_KEY
            tail = re.sub(r"input_value=.*?(, input_type=|\]$)", r"input_value=<hidden>\1", tail, flags=re.M)
            raise RuntimeError(f"The backend stopped during startup (exit code {proc.returncode}).\n\n{tail}")
        if _backend_healthy():
            print("[frontend] backend is up", flush=True)
            return proc
        time.sleep(0.5)
    proc.terminate()
    raise RuntimeError("The backend did not answer within 90 seconds. See the app logs.")


try:
    _proc = ensure_backend()
    if _proc is not None and _proc.poll() is not None:  # backend crashed after starting: start it again
        print(f"[frontend] backend exited with code {_proc.returncode}; restarting", flush=True)
        ensure_backend.clear()
        ensure_backend()
except RuntimeError as exc:
    st.error("Could not start the controller backend. Check the secrets (OPERATOR_TOKEN and FACILITY_*), then reload.")
    st.code(str(exc), language="text")
    st.stop()


# ------------------------------------------------------------------------------------------ helpers
def call(method: str, path: str, **kw: Any) -> Tuple[bool, Any]:
    try:
        r = httpx.request(method, API + path, headers=HEAD if method != "GET" else None, timeout=15, **kw)
        return r.status_code < 400, r.json()
    except Exception as e:
        return False, {
            "error": {
                "code": "unreachable",
                "message": f"Cannot reach the controller: {type(e).__name__}. Retrying on next refresh.",
            }
        }


@st.cache_data(ttl=30)
def get_health() -> Tuple[bool, Any]:
    return call("GET", "/api/health")


@st.cache_data(ttl=5)
def latest_cycle() -> Tuple[bool, Any]:
    return call("GET", "/api/cycles/latest")


def chip(text: str) -> str:
    return f'<span class="chip" style="background:{CHIP.get(text, "#475569")}">{text}</span>'


def banner(kind: str, text: str) -> None:
    st.markdown(f'<div class="banner {kind}">{text}</div>', unsafe_allow_html=True)


def parse_ts(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def fmt(ts: Optional[str]) -> str:
    dt = parse_ts(ts)
    return "—" if dt is None else dt.astimezone(TZ).strftime("%H:%M:%S") + f" ({TZ_NAME})"


def local_iso(ts: str) -> str:
    """Facility-local wall time, naive ISO (Plotly draws it as given)."""
    dt = parse_ts(ts)
    return ts if dt is None else dt.astimezone(TZ).replace(tzinfo=None).isoformat()


def hhmm(ts: str) -> str:
    dt = parse_ts(ts)
    return ts if dt is None else dt.astimezone(TZ).strftime("%H:%M")


def padded_range(values: List[float], min_span: float, floor_val: Optional[float] = None) -> List[float]:
    """Compute a clean Y-axis range with a minimum span so tiny ripples never fill the entire plot height."""
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return [0.0, min_span]
    lo, hi = min(clean), max(clean)
    mid = 0.5 * (lo + hi)
    span = max(hi - lo, min_span)
    low_bound = mid - 0.6 * span
    high_bound = mid + 0.6 * span
    if floor_val is not None and low_bound < floor_val:
        low_bound = floor_val
        high_bound = max(high_bound, low_bound + min_span)
    return [round(low_bound, 2), round(high_bound, 2)]


def split_tag(name: str) -> Tuple[Optional[str], str]:
    m = TAG_RE.match(name or "")
    return (m.group(1) if m else None), TAG_RE.sub("", name or "")


def uniq(items: List[str]) -> List[str]:
    return list(dict.fromkeys(items))


# ------------------------------------------------------------------------------------------ header
c1, c2, c3, c4, c5 = st.columns([2.8, 2.7, 2.3, 2.2, 1.4])
with c1:
    st.markdown("### GridSentinel Controller")
with c2:
    st.markdown('<div class="badge">SIMULATED ACTUATOR — DIGITAL TWIN</div>', unsafe_allow_html=True)
with c3:
    ok_inp_hdr, inp_hdr = call("GET", "/api/inputs")
    llm_ok = bool(inp_hdr.get("llm_configured", False)) if ok_inp_hdr and isinstance(inp_hdr, dict) else False
    ok_l, latest_cyc = latest_cycle()
    risk = latest_cyc.get("llm_risk") if ok_l and isinstance(latest_cyc, dict) else None
    risk_level = risk if risk in ("LOW", "MEDIUM", "HIGH") else "UNKNOWN"
    risk_color = {"HIGH": "#DC2626", "MEDIUM": "#D97706", "LOW": "#15803D"}.get(risk_level, "#64748B")
    llm_html = (
        '<span style="color:#15803D;font-weight:700;">🟢 LLM Online</span>'
        if llm_ok
        else '<span style="color:#DC2626;font-weight:700;">🔴 Deterministic</span>'
    )
    st.markdown(
        f'<div style="margin-top:6px;font-size:0.9rem;">{llm_html}  |  '
        f'<span style="color:{risk_color};font-weight:700;" '
        f'title="Set by the LLM classifier; UNKNOWN = no classifier result">⚠️ RISK: {risk_level}</span></div>',
        unsafe_allow_html=True,
    )
with c4:
    mode = st.radio(
        "Mode",
        ["approval_required", "autonomous"],
        horizontal=True,
        index=0,
        key="mode",
        label_visibility="collapsed",
    )
with c5:
    run = st.button("Run cycle now", type="primary", width="stretch")

offset = st.slider("Counterfactual Heatwave Test: Add °C to live forecast", 0.0, 8.0, 0.0, 0.5, key="offset")
if offset > 0:
    banner("info", f"Counterfactual: +{offset:.1f} °C applied on top of live data")

if "last_mode" not in st.session_state:
    st.session_state.last_mode = mode
if st.session_state.last_mode != mode:
    call("PUT", "/api/settings", json={"mode": mode})
    st.session_state.last_mode = mode

flash = st.session_state.pop("flash", None)
if flash:
    banner("err", flash)

# ------------------------------------------------------------------------------------------ streaming trace
if run:
    ok, resp = call("POST", "/api/cycles", json={"mode": mode, "stress_offset_c": offset})
    if not ok:
        banner("err", resp.get("error", {}).get("message", "Request failed."))
    else:
        cid = resp["cycle_id"]
        st.subheader("Live Execution Stream")
        box = st.container()
        streamed = False
        try:
            with httpx.stream("GET", f"{API}/api/cycles/{cid}/events", timeout=120) as r:
                for line in r.iter_lines():
                    if line.startswith("data: {") and '"n"' in line:
                        t = json.loads(line[6:])
                        box.markdown(
                            f'<div class="trace"><b>Step {t["n"]}: {t["tool"].replace("_", " ").title()}</b><br>'
                            f'<span style="color:#334155;"><b>Observed:</b> {t["observed"]}<br>'
                            f'<b>Decided:</b> {t["decided"]}<br>'
                            f'<b>Why:</b> {t["why"]}<br>'
                            f'<b>Result:</b> {t["tool_result_summary"]}</span></div>',
                            unsafe_allow_html=True,
                        )
                streamed = True
        except Exception:
            banner("err", "Cannot reach the controller.")
        if streamed:
            latest_cycle.clear()
            st.rerun()


# ------------------------------------------------------------------------------------------ dashboard
@st.fragment(run_every=REFRESH_RATE or None)
def dashboard() -> None:
    ok_i, inputs = call("GET", "/api/inputs")
    if ok_i:
        st.session_state.last_inputs = (inputs, datetime.now(TZ).strftime("%H:%M:%S"))
    else:
        banner("err", inputs["error"]["message"])
        saved = st.session_state.get("last_inputs")
        if not saved:
            return
        inputs = saved[0]
        banner("warn", f"Showing last known data from {saved[1]} ({TZ_NAME}); it may be out of date.")

    ok_c, cyc = latest_cycle()
    if ok_c and isinstance(cyc, dict) and cyc.get("cycle_id"):
        st.session_state.last_cycle = cyc
    elif not ok_c and cyc.get("error", {}).get("code") == "unreachable":
        cyc = st.session_state.get("last_cycle")
    else:
        cyc = None

    src = inputs["sources"]
    for n in uniq(list(inputs.get("notices", [])) + (list(cyc.get("notices", [])) if cyc else [])):
        banner("info" if n.startswith(("Deterministic", "Counterfactual")) else "warn", n)

    # 1. Live Inputs Row -------------------------------------------------------------------------
 # 1. Live Inputs Row -------------------------------------------------------------------------
    st.markdown("#### Live Inputs")
    cols = st.columns(5)

    def render_input_card(col: Any, title: str, key: str, body: str, tag: str = "", sub_body: str = "") -> None:
        s = src[key]
        p = s["provenance"]
        src_tag, clean = split_tag(p["source_name"])
        reason = p.get("stale_reason") or ""
        # Optional sources the operator chose not to set up are not failures (required ones stay FAILED).
        not_configured = key in OPTIONAL_SOURCES and p["status"] == "FAILED" and "not configured" in reason
        if not_configured:
            status = "NOT CONFIGURED"
        elif p["status"] in ("FAILED", "STALE"):
            status = p["status"]  # Never mask FAILED or STALE behind PUBLISHED
        elif src_tag in ("FALLBACK", "CACHED"):
            status = src_tag
        elif p["signal_kind"] == "PUBLISHED_SCHEDULE":
            status = "PUBLISHED"
        else:
            status = p["status"]
        if not_configured:
            stale = '<br><span style="font-size:0.75rem;">Optional; set ALERT_FEED_URL to enable.</span>' if key == "alerts" else ""
        elif reason:
            stale = f'<br><span style="color:#DC2626;font-size:0.75rem;word-break:break-word;">{reason}</span>'
        else:
            stale = ""
        sub_html = f'<div style="font-size:0.78rem;font-weight:600;color:#475569;margin-bottom:4px;">{sub_body}</div>' if sub_body else ""
        col.markdown(
            f'<div class="card"><h4>{title}</h4>{chip(status)}<span class="tag">{tag or p["signal_kind"]}</span><br>'
            f'<div style="font-size:1.25rem;font-weight:800;margin:10px 0 4px 0;color:#0F172A;">{body}</div>'
            f'{sub_html}'
            f'<div class="muted" style="word-break:break-word;">{clean}<br>'
            f'{fmt(p["fetched_at"]) if s["data"] else ("off" if not_configured else "no data")}{stale}</div></div>',
            unsafe_allow_html=True,
        )

    w, m, g, a, t = (src[k]["data"] for k in ("weather", "market", "grid_signal", "alerts", "telemetry"))
    render_input_card(
        cols[0],
        "Weather",
        "weather",
        f'{w["points"][0]["temp_dry_c"]:.1f} °C · {w["points"][0]["rh_pct"]:.0f}% RH' if w else "—",
    )
    render_input_card(
        cols[1],
        "Electricity Price",
        "market",
        f'€{m["points"][0]["price_eur_per_kwh"]:.3f}/kWh' if m else "—",
        "day-ahead spot + fixed adder",
        sub_body=(f'Spot €{m["points"][0]["spot_eur_per_kwh"]:.3f} + adder €{m["adder_eur_per_kwh"]:.3f} · {m["bidding_zone"]}'
                  if m else ""),
    )
    render_input_card(
        cols[2],
        "Grid Frequency",
        "grid_signal",
        f'{50.0 - g["value"] / 1000.0:.3f} Hz' if g else "—",
        sub_body=(f'{g["value"]:+.1f} mHz below 50 Hz' if g else ""),
    )
    render_input_card(cols[3], "Alerts", "alerts", f'{len(a["alerts"])} bulletins' if a else "—")
    render_input_card(
        cols[4],
        "IT Load",
        "telemetry",
        f'{t["it_load_kw"]:.0f} kW' if t else "—",
        "MEASURED→MODEL_DERIVED",
        sub_body=f'Host: {t["host_label"]}' if t else "",
    )

    if not cyc:
        st.info("No control cycle has run yet. Press **Run cycle now**.")
        return

    fc = cyc.get("forecast")

    # 2. 12-Hour Forecast & Stress Windows (2-Row Synchronized Subplots) -------------------------
    st.markdown(f"#### 12-Hour Forecast and Stress Windows ({TZ_NAME})")
    if fc:
        xs = [local_iso(x) for x in fc["ts"]]
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.10,
            row_heights=[0.56, 0.44],
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
            subplot_titles=(
                "Thermal Environment (°C) & Planned Facility Power (kW)",
                "Electricity Price (€/kWh): Day-Ahead Spot + Adder, and the Spot Price Model",
            ),
        )

        # --- Row 1 Left Axis: Ambient Dry-Bulb & Wet-Bulb Temperature ---
        temps_dry = [float(v) for v in fc["temp_dry_c"]]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=temps_dry,
                name="Ambient Temp (°C)",
                mode="lines",
                line=dict(color="#2563EB", width=3),
                hovertemplate="%{y:.1f} °C<extra>Ambient Temp</extra>",
            ),
            row=1,
            col=1,
            secondary_y=False,
        )
        if fc.get("temp_wet_c"):
            temps_wet = [float(v) for v in fc["temp_wet_c"]]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=temps_wet,
                    name="Wet-Bulb Temp (°C)",
                    mode="lines",
                    line=dict(color="#06B6D4", width=1.8, dash="dot"),
                    hovertemplate="%{y:.1f} °C<extra>Wet-Bulb Temp</extra>",
                ),
                row=1,
                col=1,
                secondary_y=False,
            )
            temp_range = padded_range(temps_dry + temps_wet, min_span=6.0)
        else:
            temp_range = padded_range(temps_dry, min_span=6.0)

        # --- Row 1 Right Axis: Planned Facility Power (stabilized scale, no 448.852 kW zoom cliff) ---
        if cyc.get("plan") and cyc["plan"].get("steps"):
            steps = cyc["plan"]["steps"]
            p_xs = [xs[s["step"]] for s in steps]
            p_ys = [float(s["expected_power_kw"]) for s in steps]
            fig.add_trace(
                go.Scatter(
                    x=p_xs,
                    y=p_ys,
                    name="Planned Facility Power incl. IT (kW)",
                    mode="lines+markers",
                    line=dict(color="#10B981", width=2),
                    marker=dict(size=6, color="#059669", symbol="diamond"),
                    hovertemplate="%{y:.1f} kW<extra>Planned Power</extra>",
                ),
                row=1,
                col=1,
                secondary_y=True,
            )
            power_range = padded_range(p_ys, min_span=max(25.0, 0.06 * (sum(p_ys) / len(p_ys))), floor_val=0.0)
        else:
            power_range = [0.0, 500.0]

        # --- Row 2: Price actually used by the optimizer, split by where each step's price came from ---
        prices = [float(v) for v in fc["price_eur_per_kwh"]]
        spots = [float(v) for v in fc.get("spot_eur_per_kwh", prices)]
        kinds = fc.get("price_kind", ["day_ahead"] * len(prices))
        kind_style = {
            "day_ahead": ("All-in price, published day-ahead (PUBLISHED_SCHEDULE)", "#0F766E", "solid"),
            "forecast_ml": ("All-in price, not yet published: price model (MODEL_DERIVED)", "#7C3AED", "dash"),
            "forecast_yesterday": ("All-in price, not yet published: same time yesterday (MODEL_DERIVED)", "#B45309", "dot"),
        }
        for kind, (label, color, dash) in kind_style.items():
            if kind not in kinds:
                continue
            ys = [p if k == kind else None for p, k in zip(prices, kinds)]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    name=label,
                    mode="lines",
                    connectgaps=False,
                    line=dict(color=color, width=2.6, dash=dash, shape="hv"),
                    hovertemplate="€%{y:.3f}/kWh<extra>" + kind + "</extra>",
                ),
                row=2,
                col=1,
            )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=spots,
                name="Spot price only",
                mode="lines",
                line=dict(color="#94A3B8", width=1.4, shape="hv"),
                hovertemplate="€%{y:.3f}/kWh<extra>Spot</extra>",
            ),
            row=2,
            col=1,
        )
        all_price_vals = prices + spots
        thr = fc.get("s1_threshold_eur_per_kwh")
        if thr is not None:
            all_price_vals.append(float(thr))
            fig.add_hline(
                y=float(thr),
                line=dict(color="#DC2626", width=1, dash="dot"),
                annotation_text="S1 threshold (spot, 75th pct of last 7 days)",
                annotation_font=dict(size=10, color="#991B1B"),
                row=2,
                col=1,
            )

        # Spot-price model over the whole horizon, so it can be compared with the published prices
        ml_prices = cyc.get("ml_predicted_prices")
        if isinstance(ml_prices, list) and len(ml_prices) == len(xs):
            ml_lower, ml_upper = cyc.get("ml_lower_bound") or [], cyc.get("ml_upper_bound") or []
            model_name = cyc.get("ml_model_name") or "model"
            mae_val = (cyc.get("ml_mae") or {}).get(model_name)
            ml_label = f"Spot price model: {model_name}" + (f" (held-out MAE €{mae_val:.3f})" if mae_val is not None else "")
            ml_clean = [float(x) for x in ml_prices]
            all_price_vals.extend(ml_clean)
            if len(ml_lower) == len(xs) and len(ml_upper) == len(xs):
                fig.add_trace(
                    go.Scatter(
                        x=xs + xs[::-1],
                        y=[float(u) for u in ml_upper] + [float(l) for l in ml_lower[::-1]],
                        fill="toself",
                        fillcolor="rgba(139, 92, 246, 0.12)",
                        line=dict(color="rgba(255,255,255,0)"),
                        hoverinfo="skip",
                        name="Model 15th–85th pct band",
                        showlegend=False,
                    ),
                    row=2,
                    col=1,
                )
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ml_clean,
                    name=ml_label,
                    mode="lines",
                    line=dict(color="#8B5CF6", width=1.8, dash="dash"),
                    hovertemplate="€%{y:.3f}/kWh<extra>Model (spot)</extra>",
                ),
                row=2,
                col=1,
            )
        elif cyc.get("ml_unavailable_reason"):
            st.caption(f"Price model unavailable: {cyc['ml_unavailable_reason']}")

        # Shade stress windows across both subplots
        for w_ in cyc.get("stress_windows", []):
            x0_iso, x1_iso = local_iso(w_["start"]), local_iso(w_["end"])
            label_txt = "Grid Stress (" + "/".join(w_["indicators"]) + ")"
            fig.add_vrect(
                x0=x0_iso,
                x1=x1_iso,
                fillcolor="#EF4444",
                opacity=0.14,
                line_width=1,
                line_color="rgba(239, 68, 68, 0.4)",
                annotation_text=label_txt,
                annotation_position="top left",
                annotation_font=dict(size=11, color="#991B1B"),
                row=1,
                col=1,
            )
            fig.add_vrect(
                x0=x0_iso,
                x1=x1_iso,
                fillcolor="#EF4444",
                opacity=0.14,
                line_width=1,
                line_color="rgba(239, 68, 68, 0.4)",
                row=2,
                col=1,
            )

        lo_price = min(0.0, min(all_price_vals)) if all_price_vals else 0.0
        max_price = max(all_price_vals) if all_price_vals else 0.4
        fig.update_yaxes(
            title_text="Temp (°C)",
            range=temp_range,
            tickformat=".1f",
            gridcolor="#E2E8F0",
            row=1,
            col=1,
            secondary_y=False,
        )
        fig.update_yaxes(
            title_text="Power (kW)",
            range=power_range,
            tickformat=".1f",
            showgrid=False,
            row=1,
            col=1,
            secondary_y=True,
        )
        fig.update_yaxes(
            title_text="Price (€/kWh)",
            range=[round(lo_price * 1.15, 3), round(max_price * 1.18, 3)],
            tickprefix="€",
            tickformat=".2f",
            gridcolor="#E2E8F0",
            row=2,
            col=1,
        )
        fig.update_xaxes(gridcolor="#F1F5F9", row=1, col=1)
        fig.update_xaxes(title_text=f"Time ({TZ_NAME})", gridcolor="#E2E8F0", row=2, col=1)
        fig.update_layout(
            template="plotly_white",
            height=540,
            margin=dict(l=15, r=15, t=65, b=15),
            font=dict(color="#0F172A", size=12),
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.08,
                xanchor="left",
                x=0,
                bgcolor="rgba(255,255,255,0.92)",
                bordercolor="#CBD5E1",
                borderwidth=1,
            ),
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Top: Ambient temperature and planned total facility power (IT load + cooling). "
            "Bottom: the per-step price the optimizer uses = EPEX day-ahead spot (DE-LU, via Energy-Charts) + a fixed "
            "adder for grid fees, taxes and levies (assumed, see constants). Steps whose auction result is not yet "
            "published (~12:45 CET for the next day) are filled by the price model and drawn dashed. The purple line is "
            "the model's spot prediction for the whole horizon, shown so it can be checked against the published prices."
        )

    # 3. Cooling Plan Table & Agent Decision Trace -----------------------------------------------
    left, right = st.columns([1.25, 1])
    with left:
        st.markdown("#### Optimal Cooling Plan (First 12 Steps)")
        if cyc.get("plan") and fc:
            rows = [
                {
                    "Time": hhmm(fc["ts"][s["step"]]),
                    "Mode": s["cooling_mode"].title(),
                    "Cooling (kW)": round(float(s["q_mech_kw"] + s["q_evap_kw"]), 1),
                    "Setpoint (°C)": round(float(s["air_setpoint_c"]), 2),
                    "Power (kW)": round(float(s["expected_power_kw"]), 1),
                    "Water (L)": round(float(s["expected_water_l"]), 1),
                }
                for s in cyc["plan"]["steps"][:12]
            ]
            st.dataframe(rows, width="stretch", hide_index=True)
            if cyc["plan"].get("relaxed_constraints"):
                st.caption("Relaxed: " + ", ".join(cyc["plan"]["relaxed_constraints"]))
        else:
            st.caption("No plan generated (safe state).")

    with right:
        st.markdown("#### Latest Agent Decision")
        if cyc.get("trace"):
            ls = cyc["trace"][-1]
            st.markdown(
                f'<div class="trace" style="border-left-color:#7C3AED;">'
                f'<b>Final Action: {ls["tool"].replace("_", " ").title()}</b><br>'
                f'<span style="color:#334155;"><b>Observed:</b> {ls["observed"]}<br>'
                f'<b>Decided:</b> {ls["decided"]}<br>'
                f'<b>Why:</b> {ls["why"]}<br>'
                f'<b>Result:</b> {ls["tool_result_summary"]}</span></div>',
                unsafe_allow_html=True,
            )
            if cyc.get("message"):
                st.info(cyc["message"])
        else:
            st.caption("No reasoning trace available.")

    if cyc.get("trace"):
        with st.expander("Full reasoning trace (with sources and timestamps)"):
            for t_ in cyc["trace"]:
                cites = " · ".join(f'{c["source_name"]} @ {fmt(c["ts"])}' for c in t_["citations"][:4])
                st.markdown(
                    f'<div class="trace"><b>{t_["n"]}. {t_["tool"]}</b><br>'
                    f'observed: {t_["observed"]}<br>decided: {t_["decided"]}<br>'
                    f'why: {t_["why"]}<br>result: {t_["tool_result_summary"]}<br>'
                    f'<span class="muted">{cites}</span></div>',
                    unsafe_allow_html=True,
                )

    # 4. Reactive vs Pre-emptive Comparison (KPI Cards + 2-Row Synchronized Chart) ---------------
    st.markdown("#### Reactive vs Pre-emptive Impact (simulated)")
    st.caption(
        "Reactive = cools only as needed to hold 25 °C, no lookahead. Pre-emptive = optimizer plan that cools early when it pays. "
        "All figures are simulated on the digital twin with assumed model constants; they are not measurements."
    )
    if cyc.get("comparison"):
        kpi_specs = [
            ("cost_eur", "Cost (EUR)", "€", "", 2),
            ("peak_kw", "Peak Draw (kW)", "", " kW", 1),
            ("water_l", "Water Used (L)", "", " L", 0),
            ("max_temp_c", "Max Temp (°C)", "", " °C", 2),
        ]
        for col, (k, label, prefix, suffix, dp) in zip(st.columns(4), kpi_specs):
            v = cyc["comparison"][k]
            plan_val, base_val = float(v["plan"]), float(v["baseline"])
            if k == "water_l":
                delta_l = float(v["delta"])
                if abs(delta_l) < 0.5:
                    badge_txt, bg, fg = "0 L vs reactive", "#F1F5F9", "#475569"
                elif delta_l > 0:
                    badge_txt, bg, fg = f"{delta_l:+.0f} L vs reactive (evap tradeoff)", "#FEF3C7", "#92400E"
                else:
                    badge_txt, bg, fg = f"{delta_l:+.0f} L saved", "#DCFCE7", "#166534"
            elif k == "max_temp_c":
                ok_band = 18.0 - 1e-6 <= plan_val <= 27.0 + 1e-6
                badge_txt = "within 18–27 °C safe band" if ok_band else "OUTSIDE 18–27 °C safe band"
                bg, fg = ("#DCFCE7", "#166534") if ok_band else ("#FEE2E2", "#991B1B")
            else:
                pct = v.get("savings_pct")
                if pct is None or abs(float(pct)) < 0.05:
                    badge_txt, bg, fg = "0.0% vs baseline", "#F1F5F9", "#475569"
                elif float(pct) > 0:
                    badge_txt, bg, fg = f"{float(pct):+.1f}% saved", "#DCFCE7", "#166534"
                else:
                    badge_txt, bg, fg = f"{float(pct):+.1f}% more", "#FEE2E2", "#991B1B"

            col.markdown(
                f'<div class="kpi-card">'
                f'<div class="kpi-label">{label}</div>'
                f'<div class="kpi-value">{prefix}{plan_val:.{dp}f}{suffix}</div>'
                f'<div class="kpi-sub">vs {prefix}{base_val:.{dp}f}{suffix} baseline</div>'
                f'<span class="kpi-delta" style="background:{bg};color:{fg};">{badge_txt}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )

        tr = cyc.get("trajectories")
        if tr:
            x_axis = [local_iso(x) for x in fc["ts"]] if fc else list(range(len(tr["baseline"]["t_air"])))
            t_base = [float(x) for x in tr["baseline"]["t_air"]]
            t_plan = [float(x) for x in tr["plan"]["t_air"]]
            p_base = [float(x) for x in tr["baseline"]["power_kw"]]
            p_plan = [float(x) for x in tr["plan"]["power_kw"]]

            f2 = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.12,
                row_heights=[0.50, 0.50],
                subplot_titles=(
                    "Inlet Air Temperature Trajectory (°C) — Pre-cooling vs. Reactive",
                    "Facility Electrical Power Draw (kW) — Peak Shaving & Load Shifting",
                ),
            )

            f2.add_hline(
                y=25.0,
                line_dash="dot",
                line_color="#64748B",
                annotation_text="Recommended 25.0 °C",
                annotation_position="top left",
                row=1,
                col=1,
            )
            f2.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=t_base,
                    name="Reactive Temp (°C)",
                    mode="lines",
                    line=dict(color="#EF4444", width=2.2, dash="dash"),
                    hovertemplate="%{y:.2f} °C<extra>Reactive Temp</extra>",
                ),
                row=1,
                col=1,
            )
            f2.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=t_plan,
                    name="Pre-emptive Temp (°C)",
                    mode="lines",
                    line=dict(color="#2563EB", width=3),
                    fill="tonexty",
                    fillcolor="rgba(37, 99, 235, 0.10)",
                    hovertemplate="%{y:.2f} °C<extra>Pre-emptive Temp</extra>",
                ),
                row=1,
                col=1,
            )

            f2.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=p_base,
                    name="Reactive Power (kW)",
                    mode="lines",
                    line=dict(color="#F87171", width=2.2, dash="dot"),
                    hovertemplate="%{y:.1f} kW<extra>Reactive Power</extra>",
                ),
                row=2,
                col=1,
            )
            f2.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=p_plan,
                    name="Pre-emptive Power (kW)",
                    mode="lines",
                    line=dict(color="#059669", width=2.8),
                    fill="tonexty",
                    fillcolor="rgba(16, 185, 129, 0.12)",
                    hovertemplate="%{y:.1f} kW<extra>Pre-emptive Power</extra>",
                ),
                row=2,
                col=1,
            )

            t_lo = min(t_base + t_plan)
            t_hi = max(t_base + t_plan)
            f2.update_yaxes(
                title_text="Temp (°C)",
                range=[round(min(23.5, t_lo - 0.35), 2), round(max(25.5, t_hi + 0.35), 2)],
                tickformat=".2f",
                gridcolor="#E2E8F0",
                row=1,
                col=1,
            )
            f2.update_yaxes(
                title_text="Power (kW)",
                range=padded_range(p_base + p_plan, min_span=25.0, floor_val=0.0),
                tickformat=".1f",
                gridcolor="#E2E8F0",
                row=2,
                col=1,
            )
            f2.update_xaxes(gridcolor="#F1F5F9", row=1, col=1)
            f2.update_xaxes(title_text=f"Time ({TZ_NAME})", gridcolor="#E2E8F0", row=2, col=1)
            f2.update_layout(
                template="plotly_white",
                height=460,
                margin=dict(l=15, r=15, t=60, b=15),
                font=dict(color="#0F172A", size=12),
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.08,
                    xanchor="left",
                    x=0,
                    bgcolor="rgba(255,255,255,0.92)",
                    bordercolor="#CBD5E1",
                    borderwidth=1,
                ),
            )
            st.plotly_chart(f2, width="stretch")
            gap = max(abs(a_ - b_) for a_, b_ in zip(t_base, t_plan))
            st.caption(
                f"Largest temperature difference between the two plans: {gap:.2f} °C (within the 18–27 °C safe band). "
                "The savings come from shifting power use in time and switching to evaporative cooling ahead of stress windows."
            )
    else:
        st.caption("Comparison appears after a cycle produces a plan.")

    # 5. Action Log & Approvals ------------------------------------------------------------------
    st.markdown("#### Action Log & Approvals")
    ok_a, acts = call("GET", "/api/actions")
    if ok_a:
        pending = [x for x in acts["actions"][:5] if x["status"] == "awaiting_approval"]
        if pending and cyc and cyc.get("counterfactual"):
            banner(
                "warn",
                "The latest plan was built on counterfactual weather. Approving it applies that step to the twin's saved state; roll it back afterwards.",
            )
        for ac in acts["actions"][:5]:
            c1_, c2_, c3_ = st.columns([6, 1, 1])
            c1_.markdown(
                f'{chip(ac["status"])} <b>Action {ac["action_id"][-8:]}</b> '
                f'<span class="muted">({fmt(ac["created_at"])})</span><br>'
                f'<span class="muted">{"; ".join(ac["rejection_reasons"])[:200] or ac.get("note") or ""}</span>',
                unsafe_allow_html=True,
            )
            act_path = None
            if ac["status"] == "awaiting_approval":
                if c2_.button("Approve", key="ap" + ac["action_id"], width="stretch"):
                    act_path = "approve"
                if c3_.button("Reject", key="rj" + ac["action_id"], width="stretch"):
                    act_path = "reject"
            elif ac["status"] == "applied" and c2_.button("Rollback", key="rb" + ac["action_id"], width="stretch"):
                act_path = "rollback"
            if act_path:
                ok_x, resp_x = call("POST", f"/api/actions/{ac['action_id']}/{act_path}")
                if not ok_x:
                    st.session_state.flash = resp_x.get("error", {}).get("message", "Action failed.")
                st.rerun()
        if not acts["actions"]:
            st.caption("No actions yet.")
        if acts.get("operator_alerts"):
            with st.expander("Operator alerts (in-app)"):
                for al in acts["operator_alerts"][:10]:
                    st.write(f'{fmt(al["ts"])} — {al["message"]}')

    # 6. System Data Health ----------------------------------------------------------------------
    st.markdown("#### System Data Health")
    ok_h, h = call("GET", "/api/data-health")
    if ok_h:
        st.dataframe(
            [
                {
                    "Source": x["source"].title(),
                    "Status": x["status"],
                    "Kind": x["signal_kind"],
                    "Feed": split_tag(x.get("source_name", ""))[1],
                    "Feed state": split_tag(x.get("source_name", ""))[0] or "—",
                    "Latency (ms)": round(x["latency_ms"]),
                    "Last Success": fmt(x["last_success"]),
                    "Age (s)": x["age_s"],
                    "Error": x["stale_reason"] or "",
                }
                for x in h["sources"]
            ],
            width="stretch",
            hide_index=True,
        )
    ok_p, perf = call("GET", "/api/performance")
    if ok_p:
        st.markdown(
            f'<div class="muted">Last cycle: {perf.get("runtime_ms")} ms · {perf.get("memory_mb")} MB · '
            f'The facility is a physics-based digital twin (SIMULATED ACTUATOR); it never controls real hardware.</div>',
            unsafe_allow_html=True,
        )
    consts = inputs.get("assumed_constants", [])
    if consts:
        with st.expander("Assumed model constants (must be calibrated)"):
            st.dataframe([{"name": c["name"], "value": c["value"], "unit": c["unit"]} for c in consts], hide_index=True)


dashboard()
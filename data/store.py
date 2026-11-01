"""SQLite persistence: last-known-good cache, plans, actions, cycles, twin state, grid history, perf."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class Store:
    def __init__(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS lkg(name TEXT PRIMARY KEY, payload TEXT, provenance TEXT);
                CREATE TABLE IF NOT EXISTS grid_history(ts TEXT, name TEXT, value REAL);
                CREATE TABLE IF NOT EXISTS plans(plan_id TEXT PRIMARY KEY, record TEXT);
                CREATE TABLE IF NOT EXISTS actions(action_id TEXT PRIMARY KEY, seq INTEGER, body TEXT);
                CREATE TABLE IF NOT EXISTS cycles(cycle_id TEXT PRIMARY KEY, seq INTEGER, body TEXT);
                CREATE TABLE IF NOT EXISTS kv(key TEXT PRIMARY KEY, value TEXT);
                CREATE TABLE IF NOT EXISTS alerts_log(seq INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, message TEXT);
                """
            )
            self._conn.commit()

    # -- generic -----------------------------------------------------------------
    def _exec(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def _all(self, sql: str, params: tuple = ()) -> List[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    # -- last known good ---------------------------------------------------------
    def save_lkg(self, name: str, payload: Dict[str, Any], provenance: Dict[str, Any]) -> None:
        self._exec(
            "INSERT OR REPLACE INTO lkg(name,payload,provenance) VALUES(?,?,?)",
            (name, json.dumps(payload, default=str), json.dumps(provenance, default=str)),
        )

    def load_lkg(self, name: str) -> Optional[tuple[Dict[str, Any], Dict[str, Any]]]:
        rows = self._all("SELECT payload, provenance FROM lkg WHERE name=?", (name,))
        if not rows:
            return None
        try:
            return json.loads(rows[0]["payload"]), json.loads(rows[0]["provenance"])
        except Exception:
            return None

    # -- live-signal history (real observations only) ------------------------------
    def add_grid_sample(self, name: str, ts: datetime, value: float) -> None:
        self._exec("INSERT INTO grid_history(ts,name,value) VALUES(?,?,?)", (ts.isoformat(), name, value))

    def grid_history(self, name: str, since: datetime) -> List[float]:
        rows = self._all("SELECT value FROM grid_history WHERE name=? AND ts>=?", (name, since.isoformat()))
        return [float(r["value"]) for r in rows]

    # -- plans ---------------------------------------------------------------------
    def save_plan(self, plan_id: str, record: Dict[str, Any]) -> None:
        self._exec("INSERT OR REPLACE INTO plans(plan_id,record) VALUES(?,?)", (plan_id, json.dumps(record, default=str)))

    def get_plan(self, plan_id: str) -> Optional[Dict[str, Any]]:
        rows = self._all("SELECT record FROM plans WHERE plan_id=?", (plan_id,))
        return json.loads(rows[0]["record"]) if rows else None

    # -- actions / cycles (JSON bodies) -----------------------------------------------
    def put_action(self, action_id: str, body: Dict[str, Any]) -> None:
        with self._lock:
            row = self._conn.execute("SELECT seq FROM actions WHERE action_id=?", (action_id,)).fetchone()
            if row:
                seq = row["seq"]
            else:
                seq = (self._conn.execute("SELECT COALESCE(MAX(seq),0)+1 FROM actions").fetchone()[0])
            self._conn.execute(
                "INSERT OR REPLACE INTO actions(action_id,seq,body) VALUES(?,?,?)",
                (action_id, seq, json.dumps(body, default=str)),
            )
            self._conn.commit()

    def get_action(self, action_id: str) -> Optional[Dict[str, Any]]:
        rows = self._all("SELECT body FROM actions WHERE action_id=?", (action_id,))
        return json.loads(rows[0]["body"]) if rows else None

    def list_actions(self, limit: int = 200) -> List[Dict[str, Any]]:
        rows = self._all("SELECT body FROM actions ORDER BY seq DESC LIMIT ?", (limit,))
        return [json.loads(r["body"]) for r in rows]

    def put_cycle(self, cycle_id: str, body: Dict[str, Any]) -> None:
        with self._lock:
            row = self._conn.execute("SELECT seq FROM cycles WHERE cycle_id=?", (cycle_id,)).fetchone()
            seq = row["seq"] if row else self._conn.execute("SELECT COALESCE(MAX(seq),0)+1 FROM cycles").fetchone()[0]
            self._conn.execute(
                "INSERT OR REPLACE INTO cycles(cycle_id,seq,body) VALUES(?,?,?)",
                (cycle_id, seq, json.dumps(body, default=str)),
            )
            self._conn.commit()

    def get_cycle(self, cycle_id: str) -> Optional[Dict[str, Any]]:
        rows = self._all("SELECT body FROM cycles WHERE cycle_id=?", (cycle_id,))
        return json.loads(rows[0]["body"]) if rows else None

    def latest_cycle(self) -> Optional[Dict[str, Any]]:
        rows = self._all("SELECT body FROM cycles ORDER BY seq DESC LIMIT 1")
        return json.loads(rows[0]["body"]) if rows else None

    # -- key/value (twin state, settings, perf) --------------------------------------
    def set_kv(self, key: str, value: Any) -> None:
        self._exec("INSERT OR REPLACE INTO kv(key,value) VALUES(?,?)", (key, json.dumps(value, default=str)))

    def get_kv(self, key: str, default: Any = None) -> Any:
        rows = self._all("SELECT value FROM kv WHERE key=?", (key,))
        return json.loads(rows[0]["value"]) if rows else default

    # -- operator alerts (in-app log) ---------------------------------------------------
    def add_alert(self, message: str) -> None:
        self._exec("INSERT INTO alerts_log(ts,message) VALUES(?,?)", (datetime.now(timezone.utc).isoformat(), message))

    def list_alerts(self, limit: int = 50) -> List[Dict[str, str]]:
        rows = self._all("SELECT ts,message FROM alerts_log ORDER BY seq DESC LIMIT ?", (limit,))
        return [{"ts": r["ts"], "message": r["message"]} for r in rows]

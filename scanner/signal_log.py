"""SQLite signal journal: every signal (including ones that failed), alert de-duplication and outcome tracking.

Standard library only. Times are stored as ISO-8601 UTC strings. No secrets are ever written here.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from scanner.config import ScannerConfig
from scanner.states import RANK

log = logging.getLogger(__name__)
DB_FILE = Path("signals.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL, signal_time TEXT NOT NULL, bar_time TEXT NOT NULL,
    state TEXT NOT NULL, score REAL, coverage REAL, ref_price REAL, level REAL, invalidation REAL,
    reasons TEXT, features TEXT, provider TEXT, data_age_s REAL, delayed INTEGER DEFAULT 1,
    alerted INTEGER DEFAULT 0, UNIQUE(ticker, bar_time, state)
);
CREATE INDEX IF NOT EXISTS ix_sig_ticker_time ON signals(ticker, signal_time);
CREATE TABLE IF NOT EXISTS outcomes (
    signal_id INTEGER NOT NULL, target_pct REAL NOT NULL, horizon_min INTEGER NOT NULL,
    status TEXT NOT NULL,            -- hit | miss | ambiguous | pending | no_data
    max_up_pct REAL, max_down_pct REAL, minutes_to_target REAL, evaluated_at TEXT,
    PRIMARY KEY (signal_id, target_pct, horizon_min)
);
"""


def _iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).isoformat()


@dataclass
class AlertDecision:
    send: bool
    reason: str


class SignalLog:
    def __init__(self, path: Path | str = DB_FILE):
        self.path = str(path)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        finally:
            c.close()

    # ------------------------------------------------------------------ write
    def record(self, *, ticker: str, signal_time: datetime, bar_time: datetime, state: str, score: float,
               coverage: float, ref_price: float, level: float | None, invalidation: float | None,
               reasons: list[dict], features: dict, provider: str, data_age_s: float,
               delayed: bool = True) -> int | None:
        """Insert (idempotent per ticker+bar+state). Returns the row id, or None if it already existed."""
        feats = {}
        for k, v in features.items():
            try:
                feats[k] = None if pd.isna(v) else float(v)
            except (TypeError, ValueError):
                continue  # non-numeric (e.g. session date) is not stored
        with self._conn() as c:
            cur = c.execute(
                "INSERT OR IGNORE INTO signals (ticker,signal_time,bar_time,state,score,coverage,ref_price,level,"
                "invalidation,reasons,features,provider,data_age_s,delayed) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (ticker, _iso(signal_time), _iso(bar_time), state, score, coverage, ref_price, level, invalidation,
                 json.dumps(reasons, ensure_ascii=False), json.dumps(feats), provider, data_age_s, int(delayed)))
            return cur.lastrowid if cur.rowcount else None

    def mark_alerted(self, signal_id: int) -> None:
        with self._conn() as c:
            c.execute("UPDATE signals SET alerted=1 WHERE id=?", (signal_id,))

    # ------------------------------------------------------------------ read
    def latest_for(self, ticker: str) -> dict | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM signals WHERE ticker=? AND state != 'none' ORDER BY signal_time DESC, id DESC "
                          "LIMIT 1", (ticker,)).fetchone()
        if not r:
            return None
        return {"id": r["id"], "state": r["state"], "level": r["level"], "score": r["score"],
                "time": datetime.fromisoformat(r["signal_time"]), "alerted": bool(r["alerted"])}

    def last_row(self, ticker: str) -> dict | None:
        """Most recent journal row for the ticker in ANY state (used to log transitions, not every scan)."""
        with self._conn() as c:
            r = c.execute("SELECT * FROM signals WHERE ticker=? ORDER BY signal_time DESC, id DESC LIMIT 1",
                          (ticker,)).fetchone()
        return None if not r else {"state": r["state"], "score": r["score"], "time": datetime.fromisoformat(r["signal_time"])}

    def last_alerted(self, ticker: str) -> dict | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM signals WHERE ticker=? AND alerted=1 ORDER BY signal_time DESC LIMIT 1",
                          (ticker,)).fetchone()
        if not r:
            return None
        return {"state": r["state"], "score": r["score"], "level": r["level"],
                "time": datetime.fromisoformat(r["signal_time"])}

    def signals(self, limit: int = 500, states: tuple[str, ...] | None = None) -> pd.DataFrame:
        q, args = "SELECT * FROM signals", []
        if states:
            q += f" WHERE state IN ({','.join('?' * len(states))})"
            args = list(states)
        q += " ORDER BY signal_time DESC, id DESC LIMIT ?"
        with self._conn() as c:
            return pd.read_sql_query(q, c, params=[*args, limit])

    def outcomes(self) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql_query("SELECT * FROM outcomes", c)

    # ------------------------------------------------------------------ alert de-duplication
    def should_alert(self, ticker: str, state: str, score: float, level: float | None, now: datetime,
                     cfg: ScannerConfig) -> AlertDecision:
        """Alert on a new event, not on every scan.

        New: first alert for the ticker; a higher-ranked state; a failure after a confirmed alert; a new breakout
        level; or a score jump >= cfg.material_score_jump. The same state is never repeated without such an event.
        """
        if state in ("none", "watch", "expired"):
            return AlertDecision(False, "state not alertable")
        last = self.last_alerted(ticker)
        if last is None:
            return AlertDecision(True, "first alert for ticker")
        age = now - last["time"]
        if state == "failed":
            return AlertDecision(last["state"] == "confirmed" and age <= timedelta(hours=2), "failure after confirmed alert")
        if RANK[state] > RANK.get(last["state"], 0):
            return AlertDecision(True, "state upgraded")
        if level and last["level"] and level > last["level"] * 1.002:
            return AlertDecision(True, "new, higher breakout level")
        if score - (last["score"] or 0) >= cfg.material_score_jump:
            return AlertDecision(True, "score jumped materially")
        if age >= timedelta(minutes=cfg.cooldown_minutes) and self._reset_since(ticker, last["time"]):
            return AlertDecision(True, "new episode after the setup had reset")
        return AlertDecision(False, "duplicate: same state, no new event")

    def _reset_since(self, ticker: str, since: datetime) -> bool:
        """True if the ticker dropped back to none/watch/failed/expired after `since` (so a repeat is a new episode)."""
        with self._conn() as c:
            r = c.execute("SELECT 1 FROM signals WHERE ticker=? AND signal_time>? AND state IN "
                          "('none','watch','failed','expired') LIMIT 1", (ticker, _iso(since))).fetchone()
        return r is not None

    # ------------------------------------------------------------------ outcomes
    def evaluate_outcomes(self, bars_by_ticker: dict[str, pd.DataFrame], cfg: ScannerConfig,
                          now: datetime | None = None) -> int:
        """Fill outcomes for signals using 1-minute bars that came AFTER the signal bar.

        Entry reference = reference price at signal time (a hypothetical fill; no slippage is assumed here).
        If the target and the hypothetical stop fall in the same 1-minute bar the order is unknown and the
        result is stored as 'ambiguous' (never counted as a win).
        """
        now = now or datetime.now(timezone.utc)
        n = 0
        sig = self.signals(limit=5000, states=("early", "near_breakout", "confirmed"))
        done = {(r.signal_id, r.target_pct, r.horizon_min): r.status for r in self.outcomes().itertuples()}
        for r in sig.itertuples():
            bars = bars_by_ticker.get(r.ticker)
            if bars is None or bars.empty:
                continue
            t0 = datetime.fromisoformat(r.bar_time)
            for tgt, horizon in cfg.targets:
                if done.get((r.id, tgt, horizon)) in ("hit", "miss", "ambiguous"):
                    continue
                res = outcome_for(bars, t0, r.ref_price, tgt, horizon, cfg.stop_pct, now)
                with self._conn() as c:
                    c.execute("INSERT OR REPLACE INTO outcomes VALUES (?,?,?,?,?,?,?,?)",
                              (r.id, tgt, horizon, res["status"], res["max_up_pct"], res["max_down_pct"],
                               res["minutes_to_target"], _iso(now)))
                n += 1
        return n


def outcome_for(bars: pd.DataFrame, bar_time: datetime, ref_price: float, target_pct: float, horizon_min: int,
                stop_pct: float, now: datetime) -> dict:
    """Outcome of one hypothetical trade using only bars strictly after the signal bar."""
    idx = pd.DatetimeIndex(bars.index)
    t0 = pd.Timestamp(bar_time)
    t0 = t0.tz_localize("UTC") if t0.tzinfo is None else t0
    end = t0 + pd.Timedelta(minutes=horizon_min + 1)
    after = bars[(idx > t0) & (idx <= end)]
    # the horizon is only decided once its last minute has completed
    complete = pd.Timestamp(now) >= end + pd.Timedelta(minutes=1)
    if after.empty:
        return {"status": "pending" if not complete else "no_data", "max_up_pct": None, "max_down_pct": None,
                "minutes_to_target": None}
    tgt_px, stop_px = ref_price * (1 + target_pct / 100), ref_price * (1 - stop_pct / 100)
    max_up = (after["High"].max() / ref_price - 1) * 100
    max_dn = (after["Low"].min() / ref_price - 1) * 100
    for ts, b in after.iterrows():
        hit_t, hit_s = b["High"] >= tgt_px, b["Low"] <= stop_px
        if hit_t and hit_s:
            return {"status": "ambiguous", "max_up_pct": max_up, "max_down_pct": max_dn, "minutes_to_target": None}
        if hit_s:
            return {"status": "miss", "max_up_pct": max_up, "max_down_pct": max_dn, "minutes_to_target": None}
        if hit_t:
            return {"status": "hit", "max_up_pct": max_up, "max_down_pct": max_dn,
                    "minutes_to_target": (ts - t0).total_seconds() / 60}
    return {"status": "miss" if complete else "pending", "max_up_pct": max_up, "max_down_pct": max_dn,
            "minutes_to_target": None}

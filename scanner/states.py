"""Signal lifecycle classification (rules only).

States: none < watch < early < near_breakout < confirmed; plus failed / expired for signals that had triggered.
All states are computed from COMPLETED 1-minute bars. With delayed data (free providers) an 'early' state is an
early *indicator*, not a live prediction, and the UI labels it so.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from scanner.config import ScannerConfig
from scanner.scoring import ScoreResult, _ok

STATES = ("none", "watch", "early", "near_breakout", "confirmed", "failed", "expired")
RANK = {s: i for i, s in enumerate(STATES)}
LABELS = {
    "none": ("No setup", "لا توجد فرصة"),
    "watch": ("Watching", "تحت المراقبة"),
    "early": ("Early indicators", "مؤشرات مبكرة"),
    "near_breakout": ("Approaching breakout", "يقترب من الاختراق"),
    "confirmed": ("Confirmed breakout", "اختراق مؤكد"),
    "failed": ("Failed / lost momentum", "اختراق فاشل"),
    "expired": ("Expired", "انتهت الصلاحية"),
}


@dataclass
class Classification:
    state: str
    actionable: bool          # False whenever data/market conditions make it unsuitable to act on
    blocked: str | None       # why it is not actionable (data_stale, market_closed, insufficient_data, extended)
    extended: bool
    level: float | None       # breakout level the signal refers to
    invalidation: float | None
    expires_at: datetime | None


def classify(row: pd.Series, sc: ScoreResult, cfg: ScannerConfig, *, market_open: bool, data_age_s: float,
             signal_time: datetime | None = None, prior: dict | None = None) -> Classification:
    """prior: {'state','level','time'(aware datetime)} of the latest logged signal for this ticker, if any."""
    level = float(row["level"]) if _ok(row.get("level")) else None
    close = float(row["Close"])
    extended = bool(sc.extended)
    rvol = row.get("rvol")
    ret5 = row.get("ret5")
    dist = row.get("dist_pct")
    accel = row.get("accel")
    expires = signal_time + timedelta(minutes=cfg.expiry_minutes) if signal_time else None

    blocked = None
    if sc.coverage < cfg.min_coverage:
        blocked = "insufficient_data"
    elif not market_open:
        blocked = "market_closed"
    elif data_age_s > cfg.stale_after_seconds:
        blocked = "data_stale"

    def out(state: str, lvl=level, inval=None, ext=extended) -> Classification:
        actionable = state in ("early", "near_breakout", "confirmed") and blocked is None and not ext
        b = blocked or ("extended" if ext and state in ("early", "near_breakout", "confirmed") else None)
        return Classification(state, actionable, b, ext, lvl, inval, expires)

    if blocked == "insufficient_data":
        return out("none")

    # previously triggered signal: did it fail or expire?
    if prior and prior.get("state") in ("early", "near_breakout", "confirmed"):
        plevel = prior.get("level")
        ptime = prior.get("time")
        if prior["state"] == "confirmed" and plevel is not None and close < plevel:
            return out("failed", lvl=plevel)
        if ptime is not None and signal_time is not None and signal_time - ptime > timedelta(minutes=cfg.expiry_minutes):
            fresh = _ok(dist) and dist > 0 and _ok(rvol) and rvol >= cfg.confirm_min_rvol
            if not fresh:
                return out("expired", lvl=plevel)
    if bool(row.get("failed_breakout", False)):
        return out("failed")

    breakout = _ok(dist) and dist > 0
    if (breakout and _ok(rvol) and rvol >= cfg.confirm_min_rvol and _ok(ret5) and ret5 >= cfg.confirm_min_ret5_pct
            and sc.score >= cfg.confirm_min_score):
        return out("confirmed", inval=level)
    if (_ok(dist) and -cfg.near_breakout_pct <= dist <= 0 and _ok(rvol) and rvol >= 1.5
            and sc.score >= cfg.early_min_score):
        return out("near_breakout", inval=close * (1 - cfg.stop_pct / 100))
    if (_ok(rvol) and rvol >= cfg.early_min_rvol and _ok(accel) and accel > 0 and _ok(ret5) and ret5 > 0
            and sc.score >= cfg.early_min_score):
        return out("early", inval=close * (1 - cfg.stop_pct / 100))
    if sc.score >= cfg.watch_min_score:
        return out("watch")
    return out("none")

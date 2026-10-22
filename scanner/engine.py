"""Scanner orchestration: fetch -> pre-filter -> features -> score -> state -> journal -> (optional) alert."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from data.market_data import DataError
from data.us_calendar import market_status
from scanner.config import ScannerConfig
from scanner.features import FeatureError, build_features, data_age_seconds, latest
from scanner.providers import MinuteProvider, YFinanceMinuteProvider
from scanner.scoring import Reason, ScoreResult, score_row
from scanner.signal_log import SignalLog
from scanner.states import LABELS, RANK, Classification, classify

log = logging.getLogger(__name__)


@dataclass
class Candidate:
    ticker: str
    state: str
    score: float
    coverage: float
    ref_price: float
    bar_time: datetime
    data_age_s: float
    actionable: bool
    blocked: str | None
    extended: bool
    level: float | None
    invalidation: float | None
    expires_at: datetime | None
    components: dict
    reasons: list[Reason]
    features: dict
    provider: str
    delayed: bool
    alert_sent: bool = False
    alert_note: str = ""
    company: str | None = None
    catalysts: list = field(default_factory=list)

    @property
    def label_ar(self) -> str:
        return LABELS[self.state][1]

    @property
    def label_en(self) -> str:
        return LABELS[self.state][0]


@dataclass
class ScanReport:
    candidates: list[Candidate] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)   # ticker -> reason (filtered out or data problem)
    market_status: str = "closed"
    provider: str = ""
    delayed: bool = True
    scanned_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    warnings: list[str] = field(default_factory=list)

    def table(self) -> pd.DataFrame:
        rows = [{"ticker": c.ticker, "state": c.state, "score": c.score, "coverage": c.coverage,
                 "price": c.ref_price, "rvol": c.features.get("rvol"), "ret5_%": c.features.get("ret5"),
                 "dist_to_high_%": c.features.get("dist_pct"), "actionable": c.actionable,
                 "blocked": c.blocked, "bar_time": c.bar_time} for c in self.candidates]
        return pd.DataFrame(rows)


def prefilter(row: pd.Series, cfg: ScannerConfig) -> str | None:
    """Hard filters. Returns the reason a ticker is excluded, or None when it passes."""
    px = row.get("Close")
    if pd.isna(px) or not (cfg.min_price <= px <= cfg.max_price):
        return f"price {px:.2f} outside {cfg.min_price}-{cfg.max_price}" if pd.notna(px) else "no price"
    dv = row.get("avg_dollar_vol")
    if pd.isna(dv):
        return f"fewer than {cfg.min_prior_sessions} prior sessions: liquidity unknown"
    if dv < cfg.min_avg_dollar_volume:
        return f"avg dollar volume ${dv:,.0f} below ${cfg.min_avg_dollar_volume:,.0f}"
    sp = row.get("spread_pct")
    if pd.notna(sp) and sp > cfg.max_spread_pct:
        return f"spread {sp:.2f}% above {cfg.max_spread_pct}%"
    return None


def evaluate_ticker(ticker: str, bars: pd.DataFrame, bench: pd.DataFrame | None, cfg: ScannerConfig, *,
                    now: datetime, provider: str = "", delayed: bool = True, prior: dict | None = None,
                    quote: dict | None = None, apply_prefilter: bool = True) -> tuple[Candidate | None, str | None]:
    """Pure evaluation of one ticker. Returns (candidate, skip_reason)."""
    try:
        feats = build_features(bars, bench, cfg, now=now)
    except FeatureError as exc:
        return None, f"data: {exc}"
    row = latest(feats, quote)
    if apply_prefilter:
        why = prefilter(row, cfg)
        if why:
            return None, why
    sc: ScoreResult = score_row(row, cfg)
    age = data_age_seconds(feats, now)
    open_now = market_status(now) == "open"
    cl: Classification = classify(row, sc, cfg, market_open=open_now, data_age_s=age,
                                  signal_time=now, prior=prior)
    keep = {k: row.get(k) for k in ("rvol", "rvol_day", "ret1", "ret5", "ret15", "accel", "vwap", "ext_vwap_pct",
                                    "level", "dist_pct", "rsi14", "ema20", "ema50", "rs15", "avg_dollar_vol",
                                    "spread_pct", "hold_ratio", "bars_since_breakout", "atr14_pct", "vol30")}
    cand = Candidate(ticker, cl.state, sc.score, sc.coverage, float(row["Close"]), feats.index[-1].to_pydatetime(),
                     age, cl.actionable, cl.blocked, cl.extended, cl.level, cl.invalidation, cl.expires_at,
                     sc.components, sc.reasons, keep, provider, delayed)
    return cand, None


class Scanner:
    def __init__(self, provider: MinuteProvider | None = None, cfg: ScannerConfig | None = None,
                 journal: SignalLog | None = None, notifier=None, names: dict[str, str] | None = None):
        self.names = names or {}
        self.provider = provider or YFinanceMinuteProvider()
        self.cfg = (cfg or ScannerConfig()).validate()
        self.journal = journal
        self.notifier = notifier   # object with .enabled and .send(text); see scanner.notify

    def scan(self, tickers: list[str], now: datetime | None = None, send_alerts: bool = False,
             progress=None) -> ScanReport:
        """progress: optional callable(fraction: float, text: str) for UI progress bars."""
        now = now or datetime.now(timezone.utc)
        cfg = self.cfg
        rep = ScanReport(market_status=market_status(now), provider=self.provider.name,
                         delayed=not self.provider.realtime, scanned_at=now)
        if rep.market_status != "open":
            rep.warnings.append("market_closed: showing the last completed session for review only; nothing is actionable")
        if not self.provider.realtime:
            rep.warnings.append("delayed_data: this source is delayed/unofficial; never treat prices as real-time")
        tickers = list(dict.fromkeys(t.strip().upper() for t in tickers if t.strip()))
        data: dict = {}
        want = [cfg.benchmark] + [t for t in tickers if t != cfg.benchmark]
        step = 40
        try:
            for i in range(0, len(want), step):
                data.update(self.provider.bars(want[i:i + step], cfg.history_days))
                if progress:
                    progress(min(0.6, 0.6 * (i + step) / max(1, len(want))), f"تحميل البيانات {min(i + step, len(want))}/{len(want)}")
        except DataError as exc:
            rep.warnings.append(f"data_error: {exc}")
            return rep
        bench = data.get(cfg.benchmark)
        if isinstance(bench, Exception):
            rep.warnings.append(f"benchmark {cfg.benchmark} unavailable: relative strength cannot be scored")
            bench = None
        for k, t in enumerate(tickers):
            if progress:
                progress(0.6 + 0.4 * (k + 1) / max(1, len(tickers)), f"تحليل {t}")
            bars = data.get(t)
            if bars is None or isinstance(bars, Exception):
                rep.skipped[t] = f"data: {bars}" if bars is not None else "data: missing"
                continue
            prior = self.journal.latest_for(t) if self.journal else None
            quote = None
            try:
                quote = self.provider.quote(t)
            except Exception as exc:  # a quote failure must never kill the scan
                log.warning("quote failed for %s: %s", t, type(exc).__name__)
            cand, why = evaluate_ticker(t, bars, bench, cfg, now=now, provider=self.provider.name,
                                        delayed=not self.provider.realtime, prior=prior, quote=quote)
            if cand is None:
                rep.skipped[t] = why or "skipped"
                continue
            cand.company = self.names.get(t)
            self._journal_and_alert(cand, now, send_alerts)
            rep.candidates.append(cand)
        rep.candidates.sort(key=lambda c: (-RANK.get(c.state, 0), -c.score))
        return rep

    def _journal_and_alert(self, c: Candidate, now: datetime, send_alerts: bool) -> None:
        if not self.journal:
            return
        cfg = self.cfg
        can_send = bool(send_alerts and self.notifier and getattr(self.notifier, "enabled", False))
        alertable = c.actionable or (c.state == "failed" and c.blocked is None)
        decision = self.journal.should_alert(c.ticker, c.state, c.score, c.level, now, cfg) if (can_send and alertable) else None
        last = self.journal.last_row(c.ticker)
        changed = (last is None and c.state != "none") or (
            last is not None and (c.state != last["state"] or abs(c.score - (last["score"] or 0)) >= cfg.material_score_jump))
        if not (changed or (decision and decision.send)):
            return
        sid = self.journal.record(
            ticker=c.ticker, signal_time=now, bar_time=c.bar_time, state=c.state, score=c.score,
            coverage=c.coverage, ref_price=c.ref_price, level=c.level, invalidation=c.invalidation,
            reasons=[{"code": r.code, "kind": r.kind, "en": r.en, "ar": r.ar} for r in c.reasons],
            features=c.features, provider=c.provider, data_age_s=c.data_age_s, delayed=c.delayed)
        if decision is None:
            return
        c.alert_note = decision.reason
        if decision.send and sid:   # sid is None when this bar+state was already journaled (already handled)
            from scanner.notify import format_alert
            ok = self.notifier.send(format_alert(c, now))
            c.alert_sent = bool(ok)
            if ok:
                self.journal.mark_alerted(sid)

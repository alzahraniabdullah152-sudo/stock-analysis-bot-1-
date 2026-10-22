"""Evidence-based classification of price events. Never invents causes, headlines, quotes or dates.

For each event it lists official SEC filings near the move (with exact acceptance timestamps and links),
states whether each filing was knowable BEFORE the move, compares the move with the market, and assigns one of:

  direct_evidence : a high-weight filing was accepted before the first move day's open, AND the move is abnormal and
                    not explained by the market. The tool matches by filing type and timing only. It does not read the
                    filing text, so this is the strongest label it can give, not proof of causation.
  probable        : at least two independent clues, or a high-weight filing whose timing relative to the move is unclear.
  coincident      : only weak/late clues or the market itself moved the stock. No causal claim.
  unresolved      : no usable evidence from the available sources.

Free sources have no historical news archive, so no news is attributed to historical events.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from data.us_calendar import is_trading_day, next_trading_day, previous_trading_day
from history.sec import ITEMS, Filing

NY = ZoneInfo("America/New_York")
CLASSES = ("direct_evidence", "probable", "coincident", "unresolved")
LABELS_AR = {"direct_evidence": "سبب مدعوم بأدلة مباشرة", "probable": "تفسير محتمل مدعوم بأكثر من قرينة",
             "coincident": "حدث متزامن دون دليل سببي كافٍ", "unresolved": "سبب غير محسوم بسبب نقص المعلومات"}


@dataclass
class FilingEvidence:
    form: str
    items: list[str]
    meaning: list[str]
    accepted_utc: datetime | None
    filed: str
    timing: str            # before_open | during_session | after_close | unknown   (relative to the first move day)
    knowable_before_move: bool | None
    weight: str
    url: str


@dataclass
class Explanation:
    classification: str
    confidence: str                      # high | medium | low
    summary: str
    evidence: list[FilingEvidence] = field(default_factory=list)
    clues: list[str] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)


def timing_vs_day(accepted_utc: datetime | None, move_day: pd.Timestamp) -> tuple[str, bool | None]:
    """Where a filing falls relative to the first trading day of the move (NY clock)."""
    if accepted_utc is None:
        return "unknown", None
    t = accepted_utc.astimezone(NY)
    d = move_day.date()
    open_dt, close_dt = datetime.combine(d, time(9, 30), NY), datetime.combine(d, time(16, 0), NY)
    if t < open_dt:
        return "before_open", True
    if t < close_dt:
        return "during_session", None      # order within the day cannot be established from daily bars
    return "after_close", False


def nearby_filings(filings: list[Filing], start: pd.Timestamp, end: pd.Timestamp, before_days: int = 3,
                   after_days: int = 1) -> list[Filing]:
    """Filings accepted from `before_days` trading days before the move through `after_days` after its last day."""
    d0 = start.date()
    for _ in range(before_days):
        d0 = previous_trading_day(d0)
    d1 = end.date()
    for _ in range(after_days):
        d1 = next_trading_day(d1)
    lo = datetime.combine(d0, time(0, 0), NY)
    hi = datetime.combine(d1, time(23, 59), NY)
    out = [f for f in filings if f.accepted_utc and lo <= f.accepted_utc.astimezone(NY) <= hi]
    return out


def explain_event(event: pd.Series, filings: list[Filing] | None, *, sec_available: bool) -> Explanation:
    kind = event["type"]
    start, end = pd.Timestamp(event["start"]), pd.Timestamp(event["end"])
    move_day = start if is_trading_day(start.date()) else next_trading_day(start.date())
    move_day = pd.Timestamp(move_day)
    if event.get("split_artifact"):
        return Explanation("coincident", "high",
                           "Coincides with a stock split: the apparent move is a corporate-action artifact, not a price event.",
                           caveats=["Verify that the data provider adjusted this split."])
    alts: list[str] = []
    if event.get("market_driven"):
        alts.append(f"The market benchmark moved {event['market_ret_pct']:+.1f}% over the same window, "
                    f"which explains most of the {event['change_pct']:+.1f}% move.")
    exc = event.get("excess_pct")
    short_event = kind in ("surge", "plunge", "gap_up", "gap_down", "ath_break", "reversal_up")
    vm = event.get("volume_peak_mult") if short_event and pd.notna(event.get("volume_peak_mult", np.nan)) else event.get("volume_mult")
    vp = event.get("volume_peak_mult", vm)           # busiest day in the window vs the prior 50-day average
    clues: list[str] = []
    if pd.notna(vp) and vp >= 3:
        clues.append(f"Peak-day volume {vp:.1f}x the prior 50-day average (unusual participation).")
    if pd.notna(exc) and abs(exc) >= 0.5 * abs(event["change_pct"]) and abs(event["change_pct"]) > 0 and not event.get("market_driven"):
        clues.append(f"Move beyond the market: {exc:+.1f}% relative to the benchmark.")
    caveats = ["Free data has no historical news archive; no headlines were used.",
               "Only filing type and timing were matched; the filing text was not read."]
    if not sec_available:
        return Explanation("unresolved" if not event.get("market_driven") else "coincident", "low",
                           "SEC filings were not available (SEC_USER_AGENT not set or request failed); no causal evidence checked.",
                           clues=clues, alternatives=alts, caveats=caveats)
    near = nearby_filings(filings or [], start, end)
    ev: list[FilingEvidence] = []
    for f in near:
        tim, known = timing_vs_day(f.accepted_utc, move_day)
        if f.accepted_utc and f.accepted_utc.astimezone(NY).date() < move_day.date():
            tim, known = "before_open", True        # an earlier day is always knowable beforehand
        ev.append(FilingEvidence(f.form, list(f.items), [ITEMS.get(i, (i, ""))[0] for i in f.items], f.accepted_utc,
                                 f.filed, tim, known, f.weight(), f.url))
    high_before = [e for e in ev if e.weight == "high" and e.knowable_before_move is True]
    high_unclear = [e for e in ev if e.weight == "high" and e.knowable_before_move is None]
    any_before = [e for e in ev if e.weight in ("high", "medium") and e.knowable_before_move is True]
    abnormal = (pd.notna(vp) and vp >= 2) or (pd.notna(exc) and abs(exc) >= 0.5 * abs(event["change_pct"]))
    for e in ev:
        if e.knowable_before_move is False:
            alts.append(f"The {e.form} was accepted after the move began ({e.accepted_utc:%Y-%m-%d %H:%M} UTC); "
                        "it may be a consequence or coincidence, not a cause.")
    if kind in ("stagnation", "high_volatility", "ath_break") and not ev:
        return Explanation("unresolved", "low", "No filing-based evidence for this kind of period.", clues=clues,
                           alternatives=alts, caveats=caveats)

    if high_before and abnormal and not event.get("market_driven"):
        e = high_before[0]
        return Explanation("direct_evidence", "medium",
                           f"A {e.form} ({'; '.join(e.meaning) or 'see filing'}) was accepted before the move began, and the move "
                           "is abnormal relative to the market.", ev, clues, alts + ["Other simultaneous news may also be involved."], caveats)
    n_clues = len(clues) + (1 if high_before or high_unclear or any_before else 0)
    if n_clues >= 2 and not event.get("market_driven"):
        return Explanation("probable", "low", "Several clues point to a company-specific cause, but timing or content is not established.",
                           ev, clues, alts, caveats)
    if ev or event.get("market_driven"):
        return Explanation("coincident", "low", "Filings or market moves overlap this period, but there is not enough to attribute the move.",
                           ev, clues, alts, caveats)
    return Explanation("unresolved", "low", "No usable evidence found for this move in the available sources.", [], clues, alts, caveats)

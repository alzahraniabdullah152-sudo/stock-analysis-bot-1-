"""Small HTML building blocks. All dynamic text is escaped."""
from __future__ import annotations

import html
from datetime import datetime

import streamlit as st

from scanner.states import LABELS
from ui.i18n import tr
from ui.theme import PALETTE, STATE_COLORS

e = html.escape


def html_block(s: str) -> None:
    st.markdown(s, unsafe_allow_html=True)


def status_strip(market: str, provider: str, delayed: bool, last_bar: datetime | None, lang: str = "ar",
                 extra: str = "") -> None:
    """The honesty strip: session state, delay status, source, last completed bar. Shown on every page."""
    dot = {"open": PALETTE["up"], "pre/post": PALETTE["amber"], "closed": "#8A98A4"}.get(market, "#8A98A4")
    mk = {"open": "market_open", "pre/post": "market_pre", "closed": "market_closed"}.get(market, "market_closed")
    tag_bg, tag_fg = (PALETTE["amber_soft"], "#5A3E00") if delayed else (PALETTE["teal_soft"], PALETTE["teal"])
    lb = f"{last_bar:%Y-%m-%d %H:%M}" if last_bar else "-"
    html_block(
        f"<div class='strip {'rtl' if lang == 'ar' else ''}'>"
        f"<span><span class='dot' style='background:{dot}'></span>{e(tr(mk, lang))}</span><span class='sep'>|</span>"
        f"<span class='tag' style='background:{tag_bg};color:{tag_fg}'>{e(tr('delayed' if delayed else 'realtime', lang))}</span>"
        f"<span class='sep'>|</span><span>{e(tr('source', lang))}: <b>{e(provider)}</b></span>"
        f"<span class='sep'>|</span><span>{e(tr('last_bar', lang))}: <span class='num'>{e(lb)} ET</span></span>{extra}</div>")


def state_badge(state: str, lang: str = "ar") -> str:
    fg, bg = STATE_COLORS.get(state, STATE_COLORS["none"])
    label = LABELS.get(state, LABELS["none"])[1 if lang == "ar" else 0]
    return f"<span class='badge' style='color:{fg};background:{bg}'>{e(label)}</span>"


def score_meter(score: float, coverage: float, lang: str = "ar") -> str:
    color = PALETTE["teal"] if score >= 60 else PALETTE["blue"] if score >= 40 else "#8A98A4"
    return (f"<div class='meter'><div class='track'><div class='fill' style='width:{max(0, min(100, score)):.0f}%;background:{color}'></div></div>"
            f"<div class='cap'><span class='num'><b style='color:#14202B;font-size:.95rem'>{score:.0f}</b>/100</span>"
            f"<span>{e(tr('score_note', lang))} | {e(tr('coverage', lang))} <span class='num'>{coverage:.0%}</span></span></div></div>")


def reasons_html(reasons, lang: str = "ar") -> str:
    mark = {"+": ("✓", PALETTE["up"]), "-": ("−", PALETTE["muted"]), "!": ("!", PALETTE["amber"])}
    rows = []
    for r in reasons:
        m, c = mark.get(r.kind, ("•", PALETTE["muted"]))
        txt = r.ar if lang == "ar" else r.en
        rows.append(f"<div class='reason'><span class='m' style='color:{c}'>{m}</span><span>{e(txt)}</span></div>")
    return "".join(rows)


def empty_state(title: str, hint: str = "") -> None:
    html_block(f"<div class='empty'><b>{e(title)}</b>{e(hint)}</div>")


def note(text: str) -> None:
    html_block(f"<div class='note'>{e(text)}</div>")


def candidate_card(c, lang: str = "ar") -> None:
    sub = f"{c.company} | " if getattr(c, "company", None) else ""
    blocked = f" | {e(tr('blocked_' + c.blocked, lang))}" if c.blocked else ""
    html_block(
        f"<div class='cand {'rtl' if lang == 'ar' else ''}'><div class='top'><span><span class='tk num'>{e(c.ticker)}</span> "
        f"<span class='sub'>{e(sub)}<span class='num'>{c.ref_price:.2f}</span></span></span>"
        f"<span>{state_badge(c.state, lang)}</span></div>{score_meter(c.score, c.coverage, lang)}"
        f"<div class='sub'>{e(tr('actionable' if c.actionable else 'not_actionable', lang))}{blocked}</div></div>")

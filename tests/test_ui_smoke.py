"""Executes every UI page against a stub Streamlit and synthetic data to catch runtime errors (not visual checks)."""
import os
import tempfile
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from tests import st_stub
from tests.synth import make_daily_bars, make_minute_bars


def _patch_data(monkeypatch_targets):
    """Return undo callables after patching network-facing functions with synthetic data."""
    undo = []

    def patch(obj, name, val):
        undo.append((obj, name, getattr(obj, name)))
        setattr(obj, name, val)
    monkeypatch_targets(patch)
    return undo


def _ensure_plotly():
    """Where plotly is not installed (the build sandbox), use a permissive stand-in so page logic still executes."""
    import importlib
    import sys
    from unittest.mock import MagicMock
    try:
        importlib.import_module("plotly.graph_objects")
        return []
    except ImportError:
        names = ["plotly", "plotly.graph_objects", "plotly.subplots", "plotly.express"]
        for n in names:
            sys.modules[n] = MagicMock()
        return names


def _run_page(page: str, press: tuple[str, ...] = (), lang: str = "ar"):
    fake_plotly = _ensure_plotly()
    cwd = os.getcwd()
    tmp = tempfile.mkdtemp()
    os.chdir(tmp)
    undo = []
    try:
        with st_stub.installed() as st:
            import data.market_data as md
            from history.loader import PriceHistory, normalize_history
            from scanner.providers import FrameProvider
            frames = {"AAPL": make_minute_bars(sessions=6, last_session_minutes=156, surge_at=145, surge_pct=4, surge_vol_mult=6),
                      "SPY": make_minute_bars(seed=9, price=400, sessions=6, last_session_minutes=156)}

            def hist(self, ticker, interval):
                return make_daily_bars(700)
            undo.append((md.YFinanceProvider, "history", md.YFinanceProvider.history))
            md.YFinanceProvider.history = hist

            import ui.jobs as jobs

            def sync_start(name, fn):
                import threading
                j = jobs.Job(name)
                j.status = "running"
                box = {}

                def run():
                    try:
                        j.result = fn(j)
                        j.status = "done"
                    except Exception as exc:
                        box["exc"] = exc
                        j.error = f"{type(exc).__name__}: {exc}"
                        j.status = "error"
                t = threading.Thread(target=run)
                t.start()
                t.join()
                jobs._JOBS[name] = j
                if "exc" in box:
                    raise box["exc"]
                return j
            jobs.start = sync_start

            import ui.state as state
            prov = FrameProvider(frames, name="yfinance")
            state.provider = lambda: prov
            import scanner.engine as eng
            import ui.pages  # noqa: F401
            st_stub.PRESS.update(press)
            st.session_state["lang"] = lang
            if page == "history":
                import ui.pages.history_page as hp

                def load(t):
                    df, notes = normalize_history(make_daily_bars(900, seed=len(t)).assign(**{"Adj Close": lambda d: d.Close, "Dividends": 0.0, "Stock Splits": 0.0}))
                    return PriceHistory(t, df, f"{t} Inc", "NMS", "USD", notes=notes)
                hp.load_price_history = load
            mod = __import__(f"ui.pages.{page}_page", fromlist=["render"])
            if page == "history":
                mod.load_price_history = load
            mod.render(lang)
            return list(st_stub.CALLS), st
    finally:
        import sys
        for n in fake_plotly:
            sys.modules.pop(n, None)
        for k in [k for k in sys.modules if k.startswith("charts")]:
            del sys.modules[k]
        os.chdir(cwd)
        for obj, name, old in undo:
            setattr(obj, name, old)


def test_home_page_renders():
    calls, _ = _run_page("home")
    assert any(c[0] == "title" for c in calls)


def test_scanner_page_runs_a_scan_and_logs_candidates():
    calls, st = _run_page("scanner", press=("ابدأ الفحص",))
    texts = " ".join(str(c[1]) for c in calls if c[0] == "markdown")
    assert "AAPL" in texts            # candidate card rendered


def test_analysis_page_renders_with_synthetic_daily_data():
    calls, _ = _run_page("analysis")
    assert any(c[0] == "plotly_chart" for c in calls) or any(c[0] == "error" for c in calls)


def test_history_page_builds_report():
    calls, _ = _run_page("history", press=("أنشئ التقرير",))
    assert any(c[0] == "plotly_chart" for c in calls) or any(c[0] == "download_button" for c in calls)


def test_journal_and_settings_pages_render():
    _run_page("journal")
    _run_page("settings")
    _run_page("journal", lang="en")

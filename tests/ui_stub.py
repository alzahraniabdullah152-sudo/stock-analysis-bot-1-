"""Minimal fake `streamlit` and `plotly` so page code can be EXECUTED where those libraries are not installed.

It validates Python-level correctness (imports, names, types, control flow). It does not render anything and cannot
validate Plotly figure arguments or visual layout. Where real Streamlit exists, tests use streamlit.testing instead.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock


class Ctx:
    """Context manager / layout container: widget calls on it behave like the same call on `st`."""

    def __init__(self, st=None):
        self._st = st

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if self._st is not None and hasattr(self._st, name):
            return getattr(self._st, name)
        return lambda *a, **k: Ctx(self._st)

    def __iter__(self):
        return iter(())


class State(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as exc:
            raise AttributeError(k) from exc

    def __setattr__(self, k, v):
        self[k] = v


def _rec(name):
    def f(self, *a, **k):
        self.log.append((name, str(a[0])[:200] if a else ""))
    return f


class FakeSt(types.ModuleType):
    def __init__(self, press: set[str] | None = None):
        super().__init__("streamlit")
        self.session_state = State()
        self.sidebar = Ctx(self)
        self.press = press or set()
        self.log: list[tuple[str, str]] = []
        self.html: list[str] = []

    # layout -----------------------------------------------------------------------------------
    def columns(self, spec, **k):
        return [Ctx(self) for _ in range(spec if isinstance(spec, int) else len(spec))]

    def tabs(self, labels):
        return [Ctx(self) for _ in labels]

    def expander(self, *a, **k):
        return Ctx(self)

    spinner = container = form = expander

    # inputs -----------------------------------------------------------------------------------
    def text_input(self, label, value="", **k):
        return value

    text_area = text_input

    def selectbox(self, label, options, index=0, **k):
        return list(options)[index]

    def radio(self, label, options, index=0, key=None, **k):
        if key and key in self.session_state:
            return self.session_state[key]
        return list(options)[index]

    def multiselect(self, label, options, default=None, **k):
        d = list(default or [])
        bad = [x for x in d if x not in list(options)]
        assert not bad, f"multiselect default {bad} not in options for {label!r}"   # StreamlitDefaultNotInOptionsError
        return d

    def checkbox(self, label, value=False, **k):
        return value

    def number_input(self, label, *a, **k):
        value = k.get("value", a[2] if len(a) > 2 else (a[0] if a else 0))
        return value

    def slider(self, label, min_value=0, max_value=1, value=None, *a, **k):
        return value if value is not None else min_value

    def date_input(self, label, value=None, **k):
        return value

    def button(self, label, *a, **k):
        return label in self.press

    def download_button(self, *a, **k):
        return False

    def progress(self, *a, **k):
        return Ctx(self)

    # output -----------------------------------------------------------------------------------
    def markdown(self, body, *a, **k):
        if k.get("unsafe_allow_html"):
            self.html.append(body)
        self.log.append(("markdown", str(body)[:80]))

    warning = _rec("warning")
    error = _rec("error")
    info = _rec("info")
    success = _rec("success")
    caption = _rec("caption")
    write = _rec("write")
    subheader = _rec("subheader")
    header = _rec("header")
    json = _rec("json")
    code = _rec("code")
    metric = _rec("metric")
    dataframe = line_chart = plotly_chart = set_page_config = rerun = stop = _rec("noop")

    # decorators -------------------------------------------------------------------------------
    def cache_data(self, *a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f

    def fragment(self, *a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f


def install(press: set[str] | None = None) -> FakeSt:
    st = FakeSt(press)
    sys.modules["streamlit"] = st
    for name in ("plotly", "plotly.graph_objects", "plotly.subplots", "plotly.express"):
        sys.modules[name] = MagicMock()
    return st


def uninstall() -> None:
    for name in ("streamlit", "plotly", "plotly.graph_objects", "plotly.subplots", "plotly.express"):
        sys.modules.pop(name, None)
    for m in [m for m in sys.modules if m == "app" or m.startswith("ui.") or m == "ui"]:
        sys.modules.pop(m, None)

"""Minimal stand-in for the `streamlit` module so UI code paths can be executed in tests (no rendering)."""
from __future__ import annotations

import sys
import types
from contextlib import contextmanager


import threading


def _main_only():
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("st.session_state used from a background thread (not allowed in real Streamlit)")


class _State(dict):
    def __getitem__(self, k):
        _main_only()
        return super().__getitem__(k)

    def __setitem__(self, k, v):
        _main_only()
        super().__setitem__(k, v)

    def get(self, k, d=None):
        _main_only()
        return super().get(k, d)

    def __contains__(self, k):
        _main_only()
        return super().__contains__(k)

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v


CALLS: list[tuple[str, tuple]] = []
PRESS: set[str] = set()          # button labels that should return True


class _Ctx:
    def __init__(self, mod):
        self._mod = mod

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __getattr__(self, name):
        return getattr(self._mod, name)


def build() -> types.ModuleType:
    m = types.ModuleType("streamlit")
    m.session_state = _State()
    ctx = _Ctx(m)
    m.sidebar = ctx

    def rec(name):
        def f(*a, **k):
            CALLS.append((name, a))
            return None
        return f

    for n in ["title", "header", "subheader", "markdown", "write", "caption", "json", "dataframe", "plotly_chart", "line_chart",
              "info", "warning", "error", "success", "set_page_config", "progress", "code", "download_button", "rerun", "stop",
              "table", "metric", "text", "divider", "bar_chart"]:
        setattr(m, n, rec(n))

    def text_input(label, value="", *a, key=None, **k):
        return m.session_state.get(key, value) if key else value
    m.text_input = text_input
    m.text_area = lambda label, value="", *a, **k: value
    m.checkbox = lambda label, value=False, *a, **k: value

    def number_input(label, min_value=None, max_value=None, value=None, step=None, *a, **k):
        return value if value is not None else min_value
    m.number_input = number_input

    def selectbox(label, options, index=0, *a, key=None, **k):
        if key and key in m.session_state:
            return m.session_state[key]
        options = list(options)
        v = options[index] if options else None
        if key:
            m.session_state[key] = v
        return v
    m.selectbox = selectbox

    def radio(label, options, index=0, *a, key=None, **k):
        return selectbox(label, options, index, key=key)
    m.radio = radio
    m.multiselect = lambda label, options, default=None, *a, **k: list(default or [])
    m.slider = lambda label, mn=0, mx=1, value=None, *a, **k: value if value is not None else mn
    m.date_input = lambda label, value=None, *a, **k: value
    m.button = lambda label, *a, **k: label in PRESS
    m.form_submit_button = lambda label, *a, **k: label in PRESS
    m.file_uploader = lambda *a, **k: None

    def columns(spec, *a, **k):
        n = spec if isinstance(spec, int) else len(spec)
        return [ctx for _ in range(n)]
    m.columns = columns
    m.tabs = lambda labels, *a, **k: [ctx for _ in labels]
    m.expander = lambda *a, **k: ctx
    m.spinner = lambda *a, **k: ctx
    m.form = lambda *a, **k: ctx
    m.container = lambda *a, **k: ctx

    def deco(*dargs, **dkw):
        if len(dargs) == 1 and callable(dargs[0]) and not dkw:
            return dargs[0]
        return lambda f: f
    m.cache_data = deco
    m.cache_resource = deco
    m.fragment = deco
    m.column_config = types.SimpleNamespace(LinkColumn=lambda *a, **k: None)
    return m


@contextmanager
def installed():
    """Install the stub as `streamlit` and drop cached `ui.*` / `app` imports so they bind to it."""
    saved = {k: v for k, v in sys.modules.items() if k == "streamlit" or k == "app" or k.startswith("ui")}
    for k in list(saved):
        del sys.modules[k]
    m = build()
    CALLS.clear()
    PRESS.clear()
    sys.modules["streamlit"] = m
    try:
        yield m
    finally:
        for k in [k for k in sys.modules if k == "streamlit" or k == "app" or k.startswith("ui")]:
            del sys.modules[k]
        sys.modules.update(saved)

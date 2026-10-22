"""Streamlit entry point. Run:  streamlit run app.py   (educational research tool, not financial advice)."""
from __future__ import annotations

import streamlit as st

from config import setup_logging
from ui import logbuffer
from ui.components import html_block
from ui.i18n import tr
from ui.theme import CSS

setup_logging()
logbuffer.install()
st.set_page_config(page_title="US Stock Analyst", page_icon="📈", layout="wide")
html_block(CSS)

PAGES = ["home", "scanner", "analysis", "history", "journal", "settings"]
NAV_KEY = {"home": "nav_home", "scanner": "nav_scanner", "analysis": "nav_analysis", "history": "nav_history",
           "journal": "nav_journal", "settings": "nav_settings"}

if "lang" not in st.session_state:
    st.session_state["lang"] = "ar"
if "nav" not in st.session_state:
    st.session_state["nav"] = "home"

with st.sidebar:
    lang = st.session_state["lang"]
    st.markdown(f"### {tr('app_title', lang)}")
    st.radio("nav", PAGES, key="nav", label_visibility="collapsed", format_func=lambda p: tr(NAV_KEY[p], st.session_state["lang"]))
    st.selectbox(tr("language", lang), ["ar", "en"], key="lang", format_func=lambda x: {"ar": "العربية", "en": "English"}[x])
    st.caption(tr("disclaimer", lang))

lang = st.session_state["lang"]
if lang == "ar":
    html_block("<style>.block-container{direction:rtl} .block-container table,.block-container [data-testid='stDataFrame'],"
               ".block-container .js-plotly-plot,.block-container code{direction:ltr}</style>")

page = st.session_state["nav"]
if page == "home":
    from ui.pages import home_page as mod
elif page == "scanner":
    from ui.pages import scanner_page as mod
elif page == "analysis":
    from ui.pages import analysis_page as mod
elif page == "history":
    from ui.pages import history_page as mod
elif page == "journal":
    from ui.pages import journal_page as mod
else:
    from ui.pages import settings_page as mod
mod.render(lang)

"""Visual identity: calm petrol-teal on cool paper, honest-data status strip as the one memorable element."""
from __future__ import annotations

PALETTE = {
    "ink": "#14202B", "paper": "#F5F7F8", "panel": "#FFFFFF", "line": "#DDE3E8", "muted": "#5B6B78",
    "teal": "#0F6B6B", "teal_soft": "#D8ECEA", "up": "#1A8F5A", "down": "#C8453B", "amber": "#B7791F",
    "amber_soft": "#FBEFD5", "blue": "#2F5D8C",
}
STATE_COLORS = {  # state -> (text, background)
    "none": ("#5B6B78", "#EEF1F3"), "watch": ("#2F5D8C", "#E2ECF6"), "early": ("#0F6B6B", "#D8ECEA"),
    "near_breakout": ("#8A5A00", "#FBEFD5"), "confirmed": ("#14653F", "#D6F0E2"),
    "failed": ("#9B2C23", "#F8DEDB"), "expired": ("#5B6B78", "#EEF1F3"),
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;500;600&display=swap');
html, body, .stApp, p, h1, h2, h3, h4, h5, h6, label, li, td, th, button, input, textarea, [data-testid="stMarkdownContainer"] {
  font-family: 'IBM Plex Sans Arabic','Segoe UI',Tahoma,sans-serif;
}
[data-testid="stIconMaterial"], [data-testid="stIconMaterial"] *, span[class*="material"], .material-icons, .material-symbols-rounded {
  font-family: 'Material Symbols Rounded','Material Symbols Outlined','Material Icons' !important;
}
.block-container { padding-top: 1.4rem; max-width: 1180px; }
h1, h2, h3 { letter-spacing: 0; font-weight: 600; color: #14202B; }
h1 { font-size: 1.7rem; } h2 { font-size: 1.25rem; } h3 { font-size: 1.05rem; }
[data-testid="stSidebar"] { background: #E9EEF1; border-inline-end: 1px solid #DDE3E8; }
[data-testid="stMetric"] { background:#fff; border:1px solid #DDE3E8; border-radius:10px; padding:12px 14px; }
[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; font-weight:600; }
[data-testid="stExpander"] { border:1px solid #DDE3E8; border-radius:10px; background:#fff; }
.stTabs [data-baseweb="tab"] { font-weight:500; }
.stDataFrame, .js-plotly-plot, code, pre { direction:ltr; }
.num { font-variant-numeric: tabular-nums; direction:ltr; unicode-bidi: isolate; }

.strip { display:flex; flex-wrap:wrap; gap:8px 14px; align-items:center; padding:9px 14px; margin-bottom:14px;
  border-radius:10px; border:1px solid #DDE3E8; background:#fff; font-size:.88rem; }
.strip .dot { width:9px; height:9px; border-radius:50%; display:inline-block; margin-inline-end:6px; }
.strip .tag { padding:2px 9px; border-radius:999px; font-weight:500; font-size:.8rem; }
.strip .sep { color:#B6C1CA; }
.badge { display:inline-block; padding:2px 10px; border-radius:999px; font-size:.82rem; font-weight:600; }
.meter { margin:4px 0 2px; }
.meter .track { height:9px; border-radius:6px; background:#E3E9ED; overflow:hidden; direction:ltr; }
.meter .fill { height:100%; border-radius:6px; }
.meter .cap { display:flex; justify-content:space-between; font-size:.78rem; color:#5B6B78; margin-top:3px; }
.reason { display:flex; gap:8px; padding:3px 0; font-size:.93rem; line-height:1.55; }
.reason .m { width:1.1em; text-align:center; font-weight:700; flex:none; }
.cand { background:#fff; border:1px solid #DDE3E8; border-radius:10px; padding:12px 14px; margin-bottom:8px; }
.cand .top { display:flex; justify-content:space-between; align-items:baseline; gap:10px; flex-wrap:wrap; }
.cand .tk { font-size:1.15rem; font-weight:600; }
.cand .sub { color:#5B6B78; font-size:.84rem; }
.empty { border:1px dashed #B6C1CA; border-radius:10px; padding:18px 16px; background:#fff; color:#33414D; }
.empty b { display:block; margin-bottom:3px; }
.note { background:#FBEFD5; border:1px solid #F0D9A3; border-radius:10px; padding:9px 13px; font-size:.88rem; color:#5A3E00; }
.rtl, .rtl * { direction: rtl; }
.rtl .num, .rtl table, .rtl .stDataFrame { direction: ltr; }
.rtl .meter .track { direction: rtl; }
</style>
"""


def plotly_layout(height: int = 420) -> dict:
    """Shared Plotly look for any chart that wants it."""
    return dict(height=height, template="plotly_white", font=dict(family="IBM Plex Sans Arabic, Segoe UI, Tahoma"),
                margin=dict(l=10, r=10, t=36, b=10), colorway=[PALETTE["teal"], PALETTE["blue"], PALETTE["amber"], PALETTE["down"]])

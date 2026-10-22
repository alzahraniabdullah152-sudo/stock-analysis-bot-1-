# Changelog

## Upgrade (scanner, company history, new interface)

Added: `scanner/*`, `history/*`, `ui/*`, `data/us_calendar.py`, `.streamlit/config.toml`, 58 new tests.
Changed: `app.py` is now a router; the previous page moved unchanged (logic) to `ui/pages/analysis_page.py`;
`data.market_data.market_session` now uses the holiday / early-close calendar (same signature);
`.env.example`, `.gitignore`, `README.md`.
Unchanged: `analysis/`, `indicators/`, `backtest/`, `charts/`, `alerts/` and their tests (34 original tests still pass).
Backup of the pre-upgrade app.py and README: `backup_before_upgrade/`.

Bugs found and fixed while building:
- pandas 3 stores datetimes with microsecond resolution; integer-nanosecond arithmetic made a 15-minute outcome window 1000x too long
  (a random baseline appeared to win 70% of the time). Replaced by float minutes; cross-checked against an independent calculation.
- A cumulative-volume RVOL reacted too slowly to bursts; a same-window RVOL against earlier sessions is now the main measure.
- A mean volume multiple hid single-day volume spikes; short events now use the peak day.

# محلل الأسهم الأمريكية / US Stock Analyst

**ملخص بالعربي:** منصة تعليمية بحثية بـ Python و Streamlit فيها: (1) ماسح أسهم بالدقيقة يرتّب الفرص بدرجة من 100 مع أسباب مكتوبة،
(2) محلل تاريخي للشركات يفسّر الصعود والهبوط بإفصاحات SEC الرسمية وتوقيتها، (3) سجل إشارات ونتائجها، (4) اختبار تاريخي صارم،
(5) تنبيهات Telegram، (6) أدوات التحليل الفني القديمة (الرسم، حاسبة المخاطر، الاختبار، المتابعة).
**ليست نصيحة مالية. الدرجة ترتيب توجيهي وليست احتمال نجاح. لا ينفّذ أي صفقة تلقائياً.**
البيانات المجانية (Yahoo) **متأخرة وغير لحظية** ولا تغطي كل شيء؛ الواجهة تقول ذلك في كل صفحة.

Rule-based research tool. **Educational only. Not financial advice, not a profit predictor, no automatic order execution.**

## Pages

| Page | What it does |
|---|---|
| Home | Session status (holiday/early-close aware), latest scan, latest logged signals |
| Live scanner | Scans your list, the watchlist, or all NASDAQ/NYSE (daily pre-screen first). Background run, optional auto-repeat, Telegram alerts, per-candidate reasons, catalysts/news, historical context |
| Stock analysis | The original technical-analysis page: chart, trend/structure, 11 setups, risk calculator, backtest, watchlist alerts |
| Company history | Full price history with actual coverage, performance vs benchmark, price events, **sourced** explanations, SEC financials, risks, HTML export |
| Journal & performance | Every logged signal, outcomes (hit / miss / ambiguous / pending), calibration, rule replay with baselines |
| Settings & status | Thresholds and weights, Telegram test, connection test, recent log |

## Install and run

Python 3.11+.

Windows (PowerShell):
```powershell
py -3.11 -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
streamlit run app.py
```
macOS / Linux:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```
Tests: `python -m pytest -q`

## Configuration (`.env`, never committed)

| Variable | Needed for |
|---|---|
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | Alerts. Create a bot with @BotFather, message it, then read your chat id from `https://api.telegram.org/bot<token>/getUpdates`. Use Settings → "Send test message" |
| `SEC_USER_AGENT="Your Name you@email.com"` | SEC filings and financial statements (SEC requires clients to identify themselves). Without it those parts are off, never guessed |
| `ALPHAVANTAGE_API_KEY`, `DATA_PROVIDER` | Optional provider for the single-stock page (free tier is rate-limited) |

Scanner thresholds and score weights are edited in the app (Settings) and saved to `scanner_settings.json`.

## How the scanner works

1. **Data**: 1-minute bars (yfinance, delayed, ~7 days). Only completed bars; the forming bar is dropped.
2. **Pre-filter** (editable): price $2-50, average dollar volume ≥ $5M/session (from earlier sessions), spread ≤ 0.5% *when a real quote exists*.
3. **Features**: RVOL (last 5 minutes vs the same window in earlier sessions, not vs a quiet midday minute), VWAP, 1/5/15-minute returns and acceleration, break of the prior 15/30-minute high, EMA20/50, RSI, strength vs SPY, hold after breakout, distance from VWAP/level.
4. **Score 0-100** (`scanner/scoring.py`): RVOL 25, momentum 20, breakout 20, liquidity 15, relative strength 10, continuation 10. Each part is normalised to 0-1. **Missing data counts as 0 points but its weight stays in the denominator**, so missing data can only lower a score; coverage below 60% is never actionable. The weights are a starting point, not a proven model.
5. **States**: none → watch → early indicators → approaching breakout → confirmed breakout; plus failed and expired. Signals are never actionable when the market is closed, the data is stale, coverage is low, or the move is extended (late-entry risk).
6. **Alerts**: one per new event (first alert, upgraded state, new higher level, material score jump, failure after a confirmed alert, or a new episode after a reset). Everything is journaled in `signals.db`, including signals that failed.

With delayed data an "early" state is an early *indicator* from completed bars, not a live prediction.

## Company history and the evidence rules

Events (surge, plunge, gaps, drawdowns with recovery, sharp reversals, all-time-high breaks, long range-bound periods, volatility regimes) use editable thresholds. Each is compared with the market benchmark, and splits are checked so they are not reported as crashes.

Each explained event gets one class: **direct evidence**, **probable**, **coincident only**, or **unresolved**. "Direct" requires a high-weight SEC filing (for example Form 8-K Item 2.02) accepted *before the first move day's open*, an abnormal move, and a move not explained by the market. The tool matches filing type and timing only; **it does not read filing text**, so even "direct" is not proof. A filing accepted after the move is never called a cause. There is **no historical news** in free sources, so none is attributed to past events.

## Strict rules in the tests and backtest

- No look-ahead: tests prove feature values and signals are unchanged when future bars are added (and a deliberately injected leak makes that test fail).
- Signal on a completed bar → fill at the **next bar's open** plus slippage and commission (assumed, not measured: free data has no bid/ask).
- Target and stop inside the same minute → **ambiguous**, counted as not a win. Gaps through the stop fill at the open.
- Chronological development / hold-out split by whole sessions; nothing is tuned inside the replay.
- Baselines with the same filters and costs: random eligible bars and a simple 15-minute-momentum rule. Wilson 95% intervals; statistics are flagged when fewer than 30 signals.
- Probability: shown only if a model fitted on earlier sessions beats a constant base rate on later sessions with enough positives and negatives (Brier score). Otherwise the app says why it shows none.

## Verification status (what was actually executed)

| Area | Status |
|---|---|
| Unit and integration tests (92) incl. calendar, features, scoring, states, journal, outcomes, replay, calibration, SEC parsing, events, evidence rules, HTML, stream wrapper | **Executed** in the build sandbox with a stand-in test runner (pytest could not be installed there). Run `python -m pytest -q` on your machine |
| Every UI page executed end to end against a stub Streamlit and synthetic data | **Executed** (catches runtime errors; not a visual check). Component styling was rendered in Chromium and inspected |
| Live Yahoo / SEC EDGAR / Nasdaq Trader / Telegram requests | **Not executed**: the build sandbox could not reach them. Parsers are tested on fixtures shaped like the documented responses. First live run: Settings → "Test connections" |
| Streamlit pages in a real browser | **Not executed** by the author on a real Streamlit server |
| Real WebSocket market feed | **Not built**: no free real-time source exists. `scanner/stream.py` is a tested reconnect/de-dup wrapper to plug a paid vendor into |
| Backtest on real 1-minute data | **Not executed** (no data access). Free 1-minute history is ~7 days, so any result is a smoke test, not evidence |
| PDF export | Not built: export HTML and use the browser's "Save as PDF" |

## Data limitations

- yfinance is unofficial and delayed; 1-minute history ≈ 7 days; it may rate-limit. Prices are never labelled real-time.
- The universe comes from the currently listed symbols (Nasdaq Trader files): delisted names are missing (survivorship bias). Only plain tickers (1-5 letters) are scanned.
- Spread is unavailable from free bar data; it is marked as missing (and lowers coverage) rather than assumed.
- The holiday / early-close calendar is rule-based; one-off closures are unknown.
- Company founding date is not provided by the sources and is different from the listing date; the first-trade date comes from the provider and the actual data start is always shown.
- Financials come from SEC XBRL "as reported" (USD, labelled by fiscal year end); tags vary between companies, so some fields can be empty.
- yfinance `Close` is split-adjusted and `AdjClose` also dividend-adjusted; raw as-traded prices are not available.

## Troubleshooting

- *"Provider returned no data"*: wrong ticker, market holiday, or Yahoo rate limiting. Wait a minute or reduce the list.
- *Scan shows nothing*: normal most of the time. Relax thresholds in Settings, or note the market may be closed (results are then review-only).
- *SEC features off*: set `SEC_USER_AGENT` in `.env` and restart.
- *Telegram not configured*: set both variables and restart; use the test button.
- *Import errors*: run from the project folder with the virtual environment active.
- Auto-repeat scans and the older scheduler live inside the app process: they stop when the app stops.

## Project tree

```
app.py                      Router: sidebar navigation, theme, language
ui/                         theme.py (palette/CSS), components.py, i18n.py, jobs.py (background jobs), state.py, logbuffer.py
ui/pages/                   home, scanner, analysis (original app), history, journal, settings
scanner/                    config, providers, features, scoring, states, engine, signal_log, notify, loop, prescreen,
                            universe, catalysts, backtest, calibration, stream
history/                    loader, events, explain, sec, financials, report, link (scanner <-> history bridge)
data/                       market_data (original), us_calendar (holidays / early closes)
analysis/ indicators/ backtest/ charts/ alerts/   original modules (unchanged API)
tests/                      92 tests + synth.py / sec_fixtures.py / st_stub.py helpers (synthetic data is for tests only)
```

## Future work

A paid real-time provider (WebSocket) behind `MinuteProvider`; bid/ask quotes; sector-relative strength; delisted-symbol data; longer 1-minute history for a meaningful hold-out; PDF export; tuning only on development sessions with the hold-out untouched.

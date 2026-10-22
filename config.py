"""Central configuration: environment variables with safe defaults."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv is optional at import time
    pass


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return float(default)


def _i(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return int(default)


@dataclass(frozen=True)
class Settings:
    """Runtime settings. Secrets come only from environment variables."""

    data_provider: str = os.getenv("DATA_PROVIDER", "yfinance").lower()
    alphavantage_key: str = os.getenv("ALPHAVANTAGE_API_KEY", "")
    telegram_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()
    opening_range_minutes: int = _i("OPENING_RANGE_MINUTES", 30)
    alert_cooldown_minutes: int = _i("ALERT_COOLDOWN_MINUTES", 60)
    scan_interval_minutes: int = _i("SCAN_INTERVAL_MINUTES", 15)
    default_equity: float = _f("DEFAULT_ACCOUNT_EQUITY", 10000)
    default_risk_pct: float = _f("DEFAULT_RISK_PCT", 1.0)
    commission_per_share: float = _f("COMMISSION_PER_SHARE", 0.0)
    slippage_pct: float = _f("SLIPPAGE_PCT", 0.05)
    # Signal rules (configurable)
    breakout_rel_volume: float = 1.5
    adx_trend_min: float = 20.0
    atr_stop_mult: float = 1.5
    min_rr: float = 1.5
    use_completed_candles: bool = True
    sr_pivot_window: int = 5
    sr_zone_atr_mult: float = 0.5
    # Confluence score weights (sum = 100)
    score_weights: dict = field(
        default_factory=lambda: {
            "trend": 30,
            "momentum": 20,
            "volume": 15,
            "structure": 15,
            "mtf": 20,
        }
    )


SETTINGS = Settings()

#: interval -> (yfinance interval, max period yfinance allows, resample rule or None)
INTERVALS: dict[str, tuple[str, str, str | None]] = {
    "5m": ("5m", "60d", None),
    "15m": ("15m", "60d", None),
    "1h": ("60m", "730d", None),
    "4h": ("60m", "730d", "4h"),
    "1d": ("1d", "10y", None),
    "1wk": ("1wk", "20y", None),
}
INTRADAY = {"5m", "15m", "1h", "4h"}


LOG_FILE = "logs/app.log"


def setup_logging() -> None:
    """Console + rotating file log (best effort). Secrets are never logged by this project."""
    if logging.getLogger().handlers:
        return
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    try:
        from logging.handlers import RotatingFileHandler

        os.makedirs("logs", exist_ok=True)
        handlers.append(RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=3, encoding="utf-8"))
    except OSError:
        pass
    logging.basicConfig(
        level=getattr(logging, SETTINGS.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )

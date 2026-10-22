"""Scanner settings. Every threshold and weight is editable; none is a scientifically validated optimum."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

log = logging.getLogger(__name__)
SETTINGS_FILE = Path("scanner_settings.json")

DEFAULT_WEIGHTS = {
    "rvol": 25,          # relative volume (time-of-day aware)
    "momentum": 20,      # 5-minute return and its acceleration
    "breakout": 20,      # proximity to / confirmation of a short-term high
    "liquidity": 15,     # dollar volume and spread
    "relstrength": 10,   # strength vs the market benchmark
    "continuation": 10,  # demand persistence / hold above breakout, VWAP and EMAs
}


@dataclass
class ScannerConfig:
    # --- pre-filters ---------------------------------------------------------------------------
    min_price: float = 2.0
    max_price: float = 50.0
    min_avg_dollar_volume: float = 5_000_000.0   # mean of prior sessions (regular hours)
    max_spread_pct: float = 0.5                  # only enforced when a real bid/ask is available
    min_prior_sessions: int = 3                  # sessions needed for RVOL / liquidity baselines
    # --- features ------------------------------------------------------------------------------
    breakout_lookback: int = 15                  # minutes (15 or 30 recommended)
    skip_open_minutes: int = 3                   # RVOL on the first minutes is unreliable
    rvol_window: int = 5                         # minutes compared with the same window of earlier sessions
    include_extended_hours: bool = False
    benchmark: str = "SPY"
    history_days: int = 5                        # 1-minute history requested (provider limit ~7)
    # --- states --------------------------------------------------------------------------------
    watch_min_score: float = 35.0
    early_min_score: float = 50.0
    early_min_rvol: float = 2.0
    near_breakout_pct: float = 0.30              # within this % below the level
    confirm_min_score: float = 60.0
    confirm_min_rvol: float = 2.0
    confirm_min_ret5_pct: float = 0.5
    max_extension_pct: float = 4.0               # beyond this % above the level the move is "extended"
    max_vwap_extension_pct: float = 6.0
    min_coverage: float = 0.60                   # share of score weight backed by real data
    expiry_minutes: int = 30
    stale_after_seconds: int = 20 * 60           # last completed bar older than this (market open) => stale
    # --- alerts --------------------------------------------------------------------------------
    cooldown_minutes: int = 30
    material_score_jump: float = 15.0
    # --- score ---------------------------------------------------------------------------------
    weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    # --- outcome targets (pct, minutes) --------------------------------------------------------
    targets: list = field(default_factory=lambda: [[2.0, 15], [3.0, 30], [5.0, 60]])
    stop_pct: float = 1.5                        # hypothetical stop used only for outcome bookkeeping

    def validate(self) -> "ScannerConfig":
        if self.min_price <= 0 or self.max_price <= self.min_price:
            raise ValueError("price range invalid")
        if self.breakout_lookback < 5:
            raise ValueError("breakout_lookback must be >= 5 minutes")
        if any(w < 0 for w in self.weights.values()) or sum(self.weights.values()) <= 0:
            raise ValueError("weights must be non-negative with a positive sum")
        unknown = set(self.weights) - set(DEFAULT_WEIGHTS)
        if unknown:
            raise ValueError(f"unknown weight keys: {sorted(unknown)}")
        for k in DEFAULT_WEIGHTS:
            self.weights.setdefault(k, 0)
        return self

    # persistence -----------------------------------------------------------------------------
    def save(self, path: Path = SETTINGS_FILE) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path = SETTINGS_FILE) -> "ScannerConfig":
        if path.exists():
            try:
                raw = json.loads(path.read_text())
                names = {f.name for f in fields(cls)}
                return cls(**{k: v for k, v in raw.items() if k in names}).validate()
            except (json.JSONDecodeError, TypeError, ValueError, OSError) as exc:
                log.warning("scanner settings unreadable (%s); using defaults", exc)
        return cls()

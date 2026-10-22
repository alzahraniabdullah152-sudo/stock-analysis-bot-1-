"""Pure pandas/NumPy indicator implementations (no look-ahead; NaN during warm-up)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def wilder(s: pd.Series, n: int) -> pd.Series:
    """Wilder's smoothing (alpha = 1/n), seeded after n valid values."""
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_g, avg_l = wilder(gain, n), wilder(loss, n)
    rs = avg_g / avg_l.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    out = out.where(~((avg_l == 0) & (avg_g > 0)), 100.0)
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    return pd.DataFrame({"bb_mid": mid, "bb_upper": mid + k * sd, "bb_lower": mid - k * sd})


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["Close"].shift(1)
    return pd.concat(
        [df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1
    ).max(axis=1, skipna=False)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return wilder(true_range(df), n)


def adx(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    up = df["High"].diff()
    down = -df["Low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr_s = wilder(true_range(df), n)
    plus_di = 100 * wilder(plus_dm, n) / tr_s
    minus_di = 100 * wilder(minus_dm, n) / tr_s
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return pd.DataFrame({"adx": wilder(dx.dropna(), n).reindex(df.index), "plus_di": plus_di, "minus_di": minus_di})


def vwap_session(df: pd.DataFrame) -> pd.Series:
    """Session VWAP, resetting each calendar day (intraday data only)."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    day = df.index.normalize() if df.index.tz is None else df.index.tz_convert("America/New_York").normalize()
    pv = (tp * df["Volume"]).groupby(day).cumsum()
    vv = df["Volume"].groupby(day).cumsum()
    return pv / vv.replace(0, np.nan)


def relative_volume(volume: pd.Series, n: int = 20) -> pd.Series:
    """Volume / average of the PRIOR n bars (excludes current bar)."""
    return volume / volume.rolling(n, min_periods=n).mean().shift(1)


def fibonacci_levels(high: float, low: float, uptrend: bool = True) -> dict[str, float]:
    """Retracement levels of the swing high-low range."""
    rng = high - low
    ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    if uptrend:  # retracement down from the high
        return {f"{r:.3f}": high - rng * r for r in ratios}
    return {f"{r:.3f}": low + rng * r for r in ratios}


def prev_period_levels(df: pd.DataFrame, rule: str = "D") -> pd.DataFrame:
    """Previous day/week high & low aligned to each bar, using only completed prior periods."""
    idx = df.index.tz_convert("America/New_York") if df.index.tz is not None else df.index
    key = pd.Series(idx.tz_localize(None).to_period(rule), index=df.index)
    hi = df["High"].groupby(key).max()
    lo = df["Low"].groupby(key).min()
    out = pd.DataFrame(index=df.index)
    out["prev_high"] = key.map(hi.shift(1))
    out["prev_low"] = key.map(lo.shift(1))
    return out


def detect_gaps(df: pd.DataFrame, min_pct: float = 0.5) -> pd.DataFrame:
    """Gap up/down vs previous close (open vs prior close), in percent."""
    gap = (df["Open"] / df["Close"].shift(1) - 1) * 100
    return pd.DataFrame({
        "gap_pct": gap,
        "gap_up": gap >= min_pct,
        "gap_down": gap <= -min_pct,
    })


def add_all(df: pd.DataFrame, intraday: bool = False) -> pd.DataFrame:
    """Return a copy of df with every indicator column attached."""
    out = df.copy()
    c = out["Close"]
    for n in (20, 50, 100, 200):
        out[f"sma{n}"] = sma(c, n)
    for n in (9, 21, 50):
        out[f"ema{n}"] = ema(c, n)
    out["rsi14"] = rsi(c, 14)
    out = out.join(macd(c)).join(bollinger(c)).join(adx(out))
    out["atr14"] = atr(out, 14)
    out["vol_ma20"] = sma(out["Volume"], 20)
    out["rel_vol"] = relative_volume(out["Volume"], 20)
    if intraday:
        out["vwap"] = vwap_session(out)
    out = out.join(detect_gaps(out))
    return out

"""Educational position-sizing calculator. Hypothetical only; not advice."""
from __future__ import annotations

import math
from dataclasses import dataclass, field


class RiskInputError(ValueError):
    """Invalid risk-calculator input."""


@dataclass
class PositionPlan:
    shares: int
    entry: float
    stop: float
    side: str
    risk_per_share: float
    dollar_risk: float          # includes costs on a stop-out
    nominal_risk: float         # shares * risk_per_share
    position_value: float
    fees_roundtrip: float
    targets: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def position_size(equity: float, risk_pct: float, entry: float, stop: float,
                  targets: list[float] | None = None, commission_per_share: float = 0.0,
                  slippage_pct: float = 0.0, allow_fractional: bool = False,
                  max_position_pct: float = 100.0) -> PositionPlan:
    """Size a position so that a stop-out loses at most risk_pct of equity (incl. costs).

    Slippage is applied adversely on entry and stop. Whole shares by default.
    """
    for name, v in (("equity", equity), ("entry", entry), ("stop", stop)):
        if not math.isfinite(v) or v <= 0:
            raise RiskInputError(f"{name} must be a positive number")
    if not (0 < risk_pct <= 5):
        raise RiskInputError("risk_pct must be in (0, 5]; larger risk is not supported")
    if stop == entry:
        raise RiskInputError("stop must differ from entry")
    if commission_per_share < 0 or slippage_pct < 0:
        raise RiskInputError("costs cannot be negative")
    side = "long" if stop < entry else "short"
    slip = slippage_pct / 100
    fill = entry * (1 + slip) if side == "long" else entry * (1 - slip)
    stop_fill = stop * (1 - slip) if side == "long" else stop * (1 + slip)
    per_share = abs(fill - stop_fill) + 2 * commission_per_share
    if per_share <= 0:
        raise RiskInputError("risk per share is zero after costs")
    budget = equity * risk_pct / 100
    raw = budget / per_share
    cap = equity * max_position_pct / 100 / fill
    qty = min(raw, cap)
    shares = round(qty, 4) if allow_fractional else int(math.floor(qty))
    warns: list[str] = []
    if shares < 1 and not allow_fractional:
        warns.append("Risk budget too small for one whole share at this stop distance.")
    if raw > cap:
        warns.append("Position capped by max position size.")
    fees = 2 * commission_per_share * shares
    plan = PositionPlan(
        shares=shares, entry=fill, stop=stop_fill, side=side, risk_per_share=per_share,
        dollar_risk=per_share * shares, nominal_risk=abs(fill - stop_fill) * shares,
        position_value=fill * shares, fees_roundtrip=fees, warnings=warns,
    )
    for t in targets or []:
        valid = (t > entry) if side == "long" else (t < entry)
        if not valid:
            plan.targets.append({"target": t, "valid": False})
            continue
        exit_fill = t * (1 - slip) if side == "long" else t * (1 + slip)
        gross = (exit_fill - fill) if side == "long" else (fill - exit_fill)
        reward = (gross - 2 * commission_per_share) * shares
        plan.targets.append({
            "target": t, "valid": True, "reward": reward,
            "rr": reward / plan.dollar_risk if plan.dollar_risk > 0 else None,
        })
    return plan


def risk_reward(entry: float, stop: float, target: float) -> float | None:
    """R:R before costs; None if levels are invalid for the implied direction."""
    if not all(math.isfinite(x) for x in (entry, stop, target)) or entry == stop:
        return None
    if stop < entry and target > entry:
        return (target - entry) / (entry - stop)
    if stop > entry and target < entry:
        return (entry - target) / (stop - entry)
    return None

import math

import pytest

from analysis.risk_management import RiskInputError, position_size, risk_reward


def test_long_whole_shares_and_risk_budget():
    p = position_size(10_000, 1.0, entry=100, stop=98)
    assert p.side == "long"
    assert p.shares == 50  # $100 budget / $2 per share
    assert p.dollar_risk == pytest.approx(100)


def test_short_side_detected():
    p = position_size(10_000, 1.0, entry=100, stop=102)
    assert p.side == "short" and p.shares == 50


def test_costs_reduce_size():
    base = position_size(10_000, 1.0, 100, 98)
    costly = position_size(10_000, 1.0, 100, 98, commission_per_share=0.05, slippage_pct=0.1)
    assert costly.shares < base.shares
    assert costly.dollar_risk <= 100 + 1e-9  # never exceeds the risk budget


def test_target_reward_and_invalid_target():
    p = position_size(10_000, 1.0, 100, 98, targets=[104, 99])
    assert p.targets[0]["valid"] and p.targets[0]["reward"] == pytest.approx(200)
    assert p.targets[1]["valid"] is False


def test_position_cap_warns():
    p = position_size(10_000, 1.0, 100, 99.9, max_position_pct=50)
    assert p.position_value <= 5_000 + 1e-6
    assert any("capped" in w for w in p.warnings)


def test_tiny_budget_gives_zero_shares_with_warning():
    p = position_size(1_000, 0.1, 500, 480)
    assert p.shares == 0 and p.warnings


@pytest.mark.parametrize("kwargs", [
    dict(equity=0, risk_pct=1, entry=100, stop=98),
    dict(equity=1000, risk_pct=0, entry=100, stop=98),
    dict(equity=1000, risk_pct=50, entry=100, stop=98),
    dict(equity=1000, risk_pct=1, entry=100, stop=100),
    dict(equity=1000, risk_pct=1, entry=float("nan"), stop=98),
    dict(equity=1000, risk_pct=1, entry=100, stop=98, commission_per_share=-1),
])
def test_invalid_inputs_raise(kwargs):
    with pytest.raises(RiskInputError):
        position_size(**kwargs)


def test_risk_reward_validity():
    assert risk_reward(100, 98, 106) == pytest.approx(3)
    assert risk_reward(100, 102, 94) == pytest.approx(3)
    assert risk_reward(100, 98, 99) is None  # target on wrong side
    assert risk_reward(100, 100, 110) is None
    assert risk_reward(100, 98, math.nan) is None

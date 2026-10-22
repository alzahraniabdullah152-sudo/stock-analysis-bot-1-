"""Annual financial summary from SEC XBRL company facts (as reported, USD).

Fiscal periods are labelled by their period END date (companies' fiscal years differ from calendar years).
Values the filings do not report stay empty; nothing is estimated. Free cash flow = operating cash flow - capex,
shown only when both are reported.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

DURATION_TAGS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "op_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "shares_wtd": ["WeightedAverageNumberOfSharesOutstandingBasic"],
    "stock_issued": ["StockIssuedDuringPeriodValueNewIssues"],
}
INSTANT_TAGS = {
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "debt": ["LongTermDebt", "LongTermDebtNoncurrent"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "equity": ["StockholdersEquity"],
    "total_assets": ["Assets"],
}
ANNUAL_FORMS = {"10-K", "10-K/A", "10-KT", "20-F", "40-F"}


def _unit_rows(facts: dict, tag: str) -> list[dict]:
    node = facts.get("facts", {}).get("us-gaap", {}).get(tag)
    if not node:
        return []
    units = node.get("units", {})
    key = "USD" if "USD" in units else ("shares" if "shares" in units else None)
    return units.get(key, []) if key else []


def _days(a: str, b: str) -> int:
    return (date.fromisoformat(b) - date.fromisoformat(a)).days


def _series(facts: dict, tags: list[str], instant: bool) -> dict[str, tuple[float, str, str]]:
    """period_end -> (value, accession, filed). First tag with data wins per period; latest filing wins on restatement."""
    out: dict[str, tuple[float, str, str]] = {}
    for tag in tags:
        best: dict[str, dict] = {}
        for r in _unit_rows(facts, tag):
            if r.get("form") not in ANNUAL_FORMS or r.get("fp") != "FY":
                continue
            if not instant:
                if "start" not in r or not (340 <= _days(r["start"], r["end"]) <= 380):
                    continue
            cur = best.get(r["end"])
            if cur is None or r.get("filed", "") > cur.get("filed", ""):
                best[r["end"]] = r
        for end, r in best.items():
            out.setdefault(end, (float(r["val"]), r.get("accn", ""), r.get("filed", "")))
    return out


def annual_table(facts: dict) -> pd.DataFrame:
    """One row per fiscal year end, newest last."""
    cols: dict[str, dict[str, tuple]] = {}
    for name, tags in DURATION_TAGS.items():
        cols[name] = _series(facts, tags, instant=False)
    for name, tags in INSTANT_TAGS.items():
        cols[name] = _series(facts, tags, instant=True)
    ends = sorted({e for s in cols.values() for e in s})
    if not ends:
        return pd.DataFrame()
    df = pd.DataFrame(index=pd.to_datetime(ends))
    df.index.name = "fiscal_year_end"
    for name, s in cols.items():
        df[name] = [s[e][0] if e in s else np.nan for e in ends]
    df["gross_margin_%"] = df["gross_profit"] / df["revenue"] * 100
    df["operating_margin_%"] = df["operating_income"] / df["revenue"] * 100
    df["net_margin_%"] = df["net_income"] / df["revenue"] * 100
    df["free_cash_flow"] = df["op_cash_flow"] - df["capex"]
    df["revenue_growth_%"] = df["revenue"].pct_change(fill_method=None) * 100
    df["net_income_growth_%"] = np.where(df["net_income"].shift(1) > 0, df["net_income"].pct_change(fill_method=None) * 100, np.nan)
    df["current_ratio"] = df["current_assets"] / df["current_liabilities"]
    df["debt_to_equity"] = np.where(df["equity"] > 0, df["debt"] / df["equity"], np.nan)
    df["share_count_change_%"] = df["shares_wtd"].pct_change(fill_method=None) * 100
    return df.replace([np.inf, -np.inf], np.nan)


def flags(df: pd.DataFrame) -> list[str]:
    """Plain descriptive flags from the latest year(s). Informational, not a rating."""
    out: list[str] = []
    if df.empty:
        return ["No annual XBRL data found for this company."]
    last = df.iloc[-1]
    if pd.notna(last.get("net_income")) and last["net_income"] < 0:
        out.append(f"Net loss in the fiscal year ending {df.index[-1]:%Y-%m-%d}.")
    if pd.notna(last.get("free_cash_flow")) and last["free_cash_flow"] < 0:
        out.append("Negative free cash flow in the latest fiscal year.")
    if pd.notna(last.get("share_count_change_%")) and last["share_count_change_%"] > 10:
        out.append(f"Weighted share count rose {last['share_count_change_%']:.0f}% year over year (possible dilution).")
    if pd.notna(last.get("current_ratio")) and last["current_ratio"] < 1:
        out.append(f"Current ratio {last['current_ratio']:.2f} (< 1): short-term liabilities exceed current assets.")
    if pd.notna(last.get("revenue_growth_%")) and last["revenue_growth_%"] < 0:
        out.append(f"Revenue declined {abs(last['revenue_growth_%']):.0f}% year over year.")
    return out or ["No automatic flags raised from the available annual data."]

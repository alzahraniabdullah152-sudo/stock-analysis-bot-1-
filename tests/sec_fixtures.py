"""Fixtures with the documented shape of SEC EDGAR JSON (invented values; for tests only)."""
COMPANY_TICKERS = {"0": {"cik_str": 1234567, "ticker": "TEST", "title": "Test Corp"},
                   "1": {"cik_str": 7654321, "ticker": "BRK-B", "title": "Other Inc"}}


def submissions(accepted_8k: str = "2017-10-02T11:30:12.000Z"):
    """recent filings, column-oriented like the real API."""
    return {"filings": {"recent": {
        "accessionNumber": ["0001234567-17-000010", "0001234567-17-000004", "0001234567-18-000001"],
        "filingDate": ["2017-10-02", "2017-08-03", "2018-02-20"],
        "acceptanceDateTime": [accepted_8k, "2017-08-03T20:05:00.000Z", "2018-02-20T21:10:00.000Z"],
        "form": ["8-K", "10-Q", "10-K"],
        "items": ["2.02,9.01", "", ""],
        "primaryDocument": ["ex99.htm", "q.htm", "k.htm"],
        "primaryDocDescription": ["8-K", "10-Q", "10-K"]}, "files": []}}


def _fact(end, val, start=None, form="10-K", fp="FY", filed=None):
    d = {"end": end, "val": val, "accn": "0001234567-18-000001", "fy": int(end[:4]), "fp": fp, "form": form,
         "filed": filed or f"{int(end[:4]) + 1}-02-20"}
    if start:
        d["start"] = start
    return d


def company_facts():
    def dur(vals):
        return {"units": {"USD": [_fact(f"{y}-12-31", v, f"{y}-01-01") for y, v in vals.items()]}}

    def inst(vals):
        return {"units": {"USD": [_fact(f"{y}-12-31", v) for y, v in vals.items()]}}
    return {"facts": {"us-gaap": {
        "Revenues": dur({2019: 1000.0, 2020: 1200.0, 2021: 900.0}),
        "GrossProfit": dur({2019: 400.0, 2020: 500.0, 2021: 300.0}),
        "NetIncomeLoss": dur({2019: 100.0, 2020: 150.0, 2021: -50.0}),
        "NetCashProvidedByUsedInOperatingActivities": dur({2019: 120.0, 2020: 160.0, 2021: -10.0}),
        "PaymentsToAcquirePropertyPlantAndEquipment": dur({2019: 30.0, 2020: 40.0, 2021: 20.0}),
        "WeightedAverageNumberOfSharesOutstandingBasic": {"units": {"shares": [
            _fact("2019-12-31", 100.0, "2019-01-01"), _fact("2020-12-31", 101.0, "2020-01-01"),
            _fact("2021-12-31", 130.0, "2021-01-01")]}},
        "AssetsCurrent": inst({2019: 500.0, 2020: 600.0, 2021: 300.0}),
        "LiabilitiesCurrent": inst({2019: 300.0, 2020: 350.0, 2021: 400.0}),
        "LongTermDebt": inst({2019: 200.0, 2020: 220.0, 2021: 400.0}),
        "StockholdersEquity": inst({2019: 400.0, 2020: 500.0, 2021: 350.0}),
    }}}

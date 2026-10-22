"""Score -> probability calibration, with hard gates.

A probability is shown ONLY if a model fitted on earlier sessions beats a constant base rate on later sessions
(Brier score, by a margin that survives bootstrap resampling) and the samples are large enough. Otherwise the report says why no probability is available.
Fitted with a 1-feature logistic regression (Platt-style) implemented in NumPy; nothing is tuned on the test part.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1e-3, iters: int = 50) -> tuple[float, float]:
    w = np.zeros(2)
    X = np.column_stack([np.ones_like(x), x])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) + l2 * np.r_[0, w[1]]
        H = X.T @ (X * (p * (1 - p))[:, None]) + l2 * np.diag([0, 1]) + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return float(w[0]), float(w[1])


def _predict(w: tuple[float, float], score: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-(w[0] + w[1] * (score - 50) / 25)))


def calibration_report(outcomes: pd.DataFrame, target_pct: float, horizon_min: int, *, train_frac: float = 0.6,
                       min_total: int = 200, min_pos: int = 30, min_neg: int = 30, bins: int = 5) -> dict:
    g = outcomes[(outcomes["target_pct"] == target_pct) & (outcomes["horizon_min"] == horizon_min)
                 & (outcomes["status"] != "no_data") & outcomes["score"].notna()].copy()
    out = {"ok": False, "reason": "", "n": len(g), "target": f"+{target_pct}% / {horizon_min}min"}
    if len(g) < min_total:
        out["reason"] = f"only {len(g)} resolved signals; at least {min_total} are needed before any probability is shown"
        return out
    g["y"] = (g["status"] == "hit").astype(float)           # ambiguous counts as not-hit
    sessions = sorted(g["session"].unique())
    if len(sessions) < 4:
        out["reason"] = "fewer than 4 sessions: a chronological train/test split is not meaningful"
        return out
    cut = max(1, int(len(sessions) * train_frac))
    tr, te = g[g["session"].isin(sessions[:cut])], g[g["session"].isin(sessions[cut:])]  # whole sessions: no overlap
    for name, part in (("train", tr), ("test", te)):
        if part["y"].sum() < min_pos or (len(part) - part["y"].sum()) < min_neg:
            out["reason"] = f"{name} part has too few positives/negatives (need >= {min_pos}/{min_neg})"
            return out
    w = _fit_logistic((tr["score"].to_numpy() - 50) / 25, tr["y"].to_numpy())
    p_te = _predict(w, te["score"].to_numpy())
    base = float(tr["y"].mean())
    brier_model = float(np.mean((p_te - te["y"]) ** 2))
    brier_base = float(np.mean((base - te["y"]) ** 2))
    edges = np.quantile(te["score"], np.linspace(0, 1, bins + 1))
    te = te.assign(bin=pd.cut(te["score"], np.unique(edges), include_lowest=True), p=p_te)
    table = te.groupby("bin", observed=True).agg(n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean")).reset_index()
    table["bin"] = table["bin"].astype(str)
    out.update({"brier_model": brier_model, "brier_base_rate": brier_base, "train_n": len(tr), "test_n": len(te),
                "table": table, "params": w, "base_rate_train": base})
    diff = (base - te["y"].to_numpy()) ** 2 - (p_te - te["y"].to_numpy()) ** 2   # >0 where the model is better
    rng = np.random.default_rng(0)
    boots = rng.choice(diff, size=(500, len(diff)), replace=True).mean(axis=1)
    out["improvement_%"] = float((brier_base - brier_model) / brier_base * 100)
    out["bootstrap_share_better"] = float((boots > 0).mean())
    if w[1] <= 0:
        out["reason"] = "the score is not positively related to success on earlier sessions: no probability is shown"
        return out
    if out["improvement_%"] < 2.0 or out["bootstrap_share_better"] < 0.95:
        out["reason"] = ("the model does not clearly beat a constant base rate on later sessions "
                         "(needs >= 2% Brier improvement, confirmed in >= 95% of bootstrap resamples): no probability is shown")
        return out
    out["ok"] = True
    out["reason"] = "calibrated on earlier sessions, checked on later sessions"
    return out

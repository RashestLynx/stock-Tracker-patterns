"""
How selling covered calls would have done on each stock over its price history, compared with just holding it.

Every 21 trading days (about a month) the test "sells" one call on 100 shares and holds it to expiry:
  - the strike is set so the call's delta is 0.15 ("safer": shares get called away less often) or
    0.25 ("balanced"), the levels the app suggests;
  - the premium is the Black-Scholes price using the stock's own volatility over the past 60 trading days.
    Real option prices usually carry a bit more volatility than that (traders pay for protection), but you
    also sell at the bid, below the fair price, so the two roughly cancel. Treat the result as an estimate;
  - at expiry the stock is worth min(price, strike) to you plus the premium kept. If the shares are called away,
    the test buys them back at the market price, so you stay invested (what most people do).
Dividends are left out of both sides, so holding vs. covered calls is a fair comparison.
"""
import math
import warnings

import numpy as np

STEP = 21          # trading days per call
LOOK = 60          # days of history for volatility
RATE = 0.04        # interest rate for option pricing
Z = {"safer": -1.0364, "balanced": -0.6745}   # d1 that gives delta 0.15 and 0.25
COLS = ["years", "hold", "safer", "balanced", "called_safer", "called_balanced", "premium_balanced", "drop_hold", "drop_balanced"]

_erf = np.frompyfunc(math.erf, 1, 1)


def ncdf(x):
    return 0.5 * (1 + _erf(np.asarray(x, dtype=float) / math.sqrt(2)).astype(float))


def call_price(S, K, sigma, T, r=RATE):
    sq = sigma * math.sqrt(T)
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / sq
    return S * ncdf(d1) - K * math.exp(-r * T) * ncdf(d1 - sq)


def simulate(C, years=10):
    """C: tickers x days of closing prices (NaN where missing). Returns {column: array per ticker}."""
    N, n = C.shape
    start = max(LOOK + 1, n - int(years * 252) - STEP)
    T = STEP / 252
    logs = np.log(C)
    rets = {k: [] for k in ("hold", *Z)}
    called = {k: [] for k in Z}
    prem = []
    for t in range(start, n - STEP, STEP):
        S, S2 = C[:, t], C[:, t + STEP]
        lr = np.diff(logs[:, t - LOOK:t + 1], axis=1)
        good = np.isfinite(lr).sum(axis=1) >= 40
        sigma = np.nanstd(lr, axis=1) * math.sqrt(252)
        ok = good & np.isfinite(S) & np.isfinite(S2) & (S > 0) & np.isfinite(sigma) & (sigma > 0.02)
        sig = np.where(ok, np.clip(sigma, 0.05, 3.0), 0.3)
        Sx = np.where(ok, S, 1.0)
        S2x = np.where(ok, S2, 1.0)
        rets["hold"].append(np.where(ok, S2x / Sx - 1, np.nan))
        for k, z in Z.items():
            # strike for this delta: ln(K/S) = (r + sigma^2/2) T - z sigma sqrt(T)
            K = Sx * np.exp((RATE + sig ** 2 / 2) * T - z * sig * math.sqrt(T))
            p = call_price(Sx, K, sig, T)
            r = (np.minimum(S2x, K) + p) / Sx - 1
            rets[k].append(np.where(ok, r, np.nan))
            called[k].append(np.where(ok, (S2x > K).astype(float), np.nan))
            if k == "balanced":
                prem.append(np.where(ok, p / Sx, np.nan))
    if not rets["hold"]:
        return {}
    R = {k: np.array(v) for k, v in rets.items()}           # periods x tickers
    count = np.isfinite(R["hold"]).sum(axis=0)
    per_year = 252 / STEP

    def ann(x):
        g = np.nansum(np.log1p(np.clip(x, -0.99, None)), axis=0)
        return np.where(count > 0, np.expm1(g / np.maximum(count, 1) * per_year), np.nan)

    def drop(x):
        curve = np.nancumsum(np.log1p(np.clip(np.nan_to_num(x), -0.99, None)), axis=0)
        peak = np.maximum.accumulate(curve, axis=0)
        return np.expm1((curve - peak).min(axis=0))

    out = {"years": count / per_year, "hold": ann(R["hold"]), "safer": ann(R["safer"]), "balanced": ann(R["balanced"]),
           "called_safer": np.nanmean(np.array(called["safer"]), axis=0),
           "called_balanced": np.nanmean(np.array(called["balanced"]), axis=0),
           "premium_balanced": np.nanmean(np.array(prem), axis=0) * per_year,
           "drop_hold": drop(R["hold"]), "drop_balanced": drop(R["balanced"])}
    return out


def run(tickers, C, min_years=3):
    """Per-ticker results (rounded) plus a summary across stocks, for the app."""
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sim = simulate(C)
    if not sim:
        return None
    rows = {}
    for i, t in enumerate(tickers):
        if not sim["years"][i] >= min_years:
            continue
        vals = [sim[c][i] for c in COLS]
        if not all(np.isfinite(v) for v in vals):
            continue
        rows[t] = [round(float(v), 3 if c != "years" else 1) for c, v in zip(COLS, vals)]
    if not rows:
        return None
    a = np.array(list(rows.values()))
    j = {c: k for k, c in enumerate(COLS)}
    summary = {
        "stocks": len(rows),
        "balanced_beat_hold": round(float(np.mean(a[:, j["balanced"]] > a[:, j["hold"]])), 3),
        "safer_beat_hold": round(float(np.mean(a[:, j["safer"]] > a[:, j["hold"]])), 3),
        "median_hold": round(float(np.median(a[:, j["hold"]])), 3),
        "median_balanced": round(float(np.median(a[:, j["balanced"]])), 3),
        "median_called_balanced": round(float(np.median(a[:, j["called_balanced"]])), 3),
        "median_premium_balanced": round(float(np.median(a[:, j["premium_balanced"]])), 3),
        "smaller_drop_balanced": round(float(np.mean(a[:, j["drop_balanced"]] > a[:, j["drop_hold"]])), 3),
    }
    # did it beat holding more often on calm, slow stocks than on fast growers?
    fast = a[:, j["hold"]] >= 0.15
    slow = a[:, j["hold"]] < 0.05
    if fast.sum() >= 20:
        summary["beat_hold_fast_growers"] = round(float(np.mean(a[fast, j["balanced"]] > a[fast, j["hold"]])), 3)
    if slow.sum() >= 20:
        summary["beat_hold_slow_stocks"] = round(float(np.mean(a[slow, j["balanced"]] > a[slow, j["hold"]])), 3)
    return {"cols": COLS, "rows": rows, "summary": summary,
            "method": "Monthly calls at delta 0.15 (safer) or 0.25 (balanced), priced from the stock's own 60-day "
                      "volatility, held to expiry; shares bought back if called away; dividends left out of both sides."}

#!/usr/bin/env python3
"""
Nightly pattern scan of every U.S. stock.

1. Daily prices for the whole market from Massive's "grouped daily" endpoint (one request per day; 2 years kept in a
   cache so each night only adds the newest day). Free plan: 5 requests a minute.
2. Every price pattern in patterns.py, run over each stock's full history.
3. Track record: for each pattern, across every stock and every day it fired (except the last 20 trading days), the
   return over the next 20 trading days compared with the S&P 500 (SPY) over the same days.
4. Smart money: insider purchases (SEC Form 4) and Congress purchases.
5. Writes out/patterns.json.gz: today's signals per stock, plus each pattern's market-wide track record.

Environment: MASSIVE_KEY (required), BARGO_KEY and SEC_USER_AGENT (optional), CACHE_DIR (default ./cache).
"""
import gzip
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import date, timedelta

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patterns import PATTERNS, detect  # noqa: E402
import smart_money  # noqa: E402

KEY = os.environ.get("MASSIVE_KEY", "").strip()
CACHE = os.environ.get("CACHE_DIR", "cache")
OUT = os.environ.get("OUT_DIR", "out")
AHEAD = 20
RECENT = 5          # signals from the last 5 trading days count as "now"
MIN_PRICE = 3.0
MIN_DOLLAR_VOL = 2_000_000
TICKER = re.compile(r"^[A-Z]{1,5}(\.[A-Z])?$")
_last_call = [0.0]


def massive_day(d):
    """One trading day for every U.S. stock: {ticker: (o, h, l, c, v)}, or None on a market holiday."""
    path = os.path.join(CACHE, "days", d.isoformat() + ".json.gz")
    if os.path.exists(path):
        return json.load(gzip.open(path, "rt"))
    for attempt in range(6):
        wait = 12.5 - (time.time() - _last_call[0])  # 5 requests a minute
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()
        url = (f"https://api.massive.com/v2/aggs/grouped/locale/us/market/stocks/{d.isoformat()}"
               f"?adjusted=true&apiKey={KEY}")
        try:
            body = json.loads(urllib.request.urlopen(url, timeout=60).read())
            break
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(30)
                continue
            if e.code in (401, 403):
                sys.exit(f"Massive rejected the key (HTTP {e.code}). Check the MASSIVE_KEY secret.")
            raise
    else:
        raise RuntimeError(f"Massive kept refusing {d}")
    rows = {r["T"]: (r["o"], r["h"], r["l"], r["c"], r["v"]) for r in body.get("results") or []
            if all(k in r for k in ("T", "o", "h", "l", "c", "v"))}
    if not rows and d >= date.today() - timedelta(days=1):
        return None  # today's data isn't out yet: don't cache an empty day
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(rows, gzip.open(path, "wt"))
    return rows


def load_market(years=2):
    days, data = [], []
    d = date.today() - timedelta(days=int(365 * years) - 5)
    while d <= date.today():
        if d.weekday() < 5:
            rows = massive_day(d)
            if rows:
                days.append(d)
                data.append(rows)
        d += timedelta(days=1)
    print(f"{len(days)} trading days, latest {days[-1] if days else 'none'}")
    return days, data


SPLITS = [2, 3, 4, 5, 8, 10, 15, 20, 1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 8, 1 / 10, 1 / 15, 1 / 20, 3 / 2, 2 / 3]


def fix_splits(o, h, l, c, v):
    """Cached days stay as they were downloaded, so a later stock split shows up as a fake jump; undo it."""
    for i in range(1, len(c)):
        ratio = c[i - 1] / c[i]
        for s in SPLITS:
            if abs(ratio / s - 1) < 0.03 and abs(o[i] * s / c[i - 1] - 1) < 0.08:
                o[:i] /= s; h[:i] /= s; l[:i] /= s; c[:i] /= s; v[:i] *= s
                break
    return o, h, l, c, v


def series(days, data):
    """{ticker: (o, h, l, c, v, day_indexes)} for every ticker in the latest day's data."""
    tickers = [t for t in data[-1] if TICKER.match(t)]
    out = {}
    for t in tickers:
        idx = [i for i, d in enumerate(data) if t in d]
        if len(idx) < 60 or idx[-1] != len(data) - 1:
            continue
        arr = np.array([data[i][t] for i in idx], float)
        o, h, l, c, v = (arr[:, k].copy() for k in range(5))
        if (c <= 0).any():
            continue
        out[t] = (*fix_splits(o, h, l, c, v), np.array(idx))
    return out


def main():
    if not KEY:
        sys.exit("MASSIVE_KEY isn't set. Add it under Settings → Secrets and variables → Actions.")
    days, data = load_market()
    if len(days) < 260:
        print("Warning: less than a year of history so far; long-term patterns will be thin.")
    all_series = series(days, data)
    spy = all_series.get("SPY")
    if spy is None:
        sys.exit("SPY missing from the data")
    spy_c = np.full(len(days), np.nan)
    spy_c[spy[5]] = spy[3]

    # Universe: real, liquid stocks only.
    universe = {}
    for t, s in all_series.items():
        o, h, l, c, v, idx = s
        if c[-1] < MIN_PRICE or np.mean(c[-20:] * v[-20:]) < MIN_DOLLAR_VOL:
            continue
        universe[t] = s
    print(f"{len(all_series)} tickers with data, {len(universe)} pass the price and volume filters")

    stats = {p: [] for p in PATTERNS}          # market-wide: excess returns after each pattern
    hits = {}
    per_stock_hist = {}
    t0 = time.time()
    for n_done, (t, (o, h, l, c, v, idx)) in enumerate(universe.items()):
        try:
            sig = detect(o, h, l, c, v)
        except Exception as e:
            print(f"{t}: detector error {e}")
            continue
        last = len(c) - 1
        fwd_ok = idx + AHEAD < len(days)
        for p, mask in sig.items():
            fired = np.where(mask)[0]
            # Track record: positions with 20 days after them, in this stock and in SPY.
            ex = []
            for i in fired:
                j = i + AHEAD
                if j <= last and fwd_ok[i] and idx[j] == idx[i] + AHEAD:
                    s0, s1 = spy_c[idx[i]], spy_c[idx[j]]
                    if s0 > 0 and s1 > 0:
                        ex.append((c[j] / c[i] - 1) - (s1 / s0 - 1))
            stats[p] += ex
            recent = fired[fired >= last - (RECENT - 1)]
            if len(recent):
                ago = int(last - recent[-1])
                hits.setdefault(t, []).append([p, ago])
                per_stock_hist[(t, p)] = ex
        if n_done % 500 == 0:
            print(f"  {n_done}/{len(universe)} stocks, {time.time() - t0:.0f}s")

    # Smart money.
    sm = {}
    try:
        sm.update(smart_money.insider_signals(os.path.join(CACHE, "sec")))
    except Exception as e:
        print(f"insider signals failed: {e}")
    for t, lst in smart_money.congress_signals().items():
        sm.setdefault(t, []).extend(lst)
    for t, lst in sm.items():
        t2 = t.replace("-", ".")
        if t2 not in universe:
            continue
        for p, note in lst:
            hits.setdefault(t2, []).append([p, 0, note])

    def summary(ex):
        ex = np.array(ex)
        if len(ex) < 1:
            return {"n": 0}
        return {"n": int(len(ex)), "beat": round(float((ex > 0).mean()), 3), "avg": round(float(ex.mean()), 4),
                "med": round(float(np.median(ex)), 4)}

    patterns_out = {}
    for p, (name, kind, mood, desc) in PATTERNS.items():
        patterns_out[p] = {"name": name, "kind": kind, "mood": mood, "desc": desc, **summary(stats[p])}

    stocks_out = {}
    # Every scanned stock gets its price levels, so the app can alert live when one breaks (even with no pattern).
    for t, (o, h, l, c, v, idx) in universe.items():
        def r(x):
            return round(float(x), 4)
        stocks_out[t] = {"price": r(c[-1]), "chg": r(c[-1] / c[-2] - 1) if len(c) > 1 else 0,
                         "hi20": r(h[-21:-1].max()), "lo20": r(l[-21:-1].min()),
                         "hi52": r(h[-253:-1].max()), "lo52": r(l[-253:-1].min()), "hits": []}
    for t, lst in hits.items():
        rows = []
        for item in lst:
            p, ago = item[0], item[1]
            row = {"p": p, "ago": ago}
            if len(item) > 2:
                row["note"] = item[2]
            own = per_stock_hist.get((t, p))
            if own is not None:
                # This stock's own earlier times (excluding the current one, which has no future yet).
                s = summary(own)
                if s["n"]:
                    row["own"] = s
            rows.append(row)
        stocks_out[t]["hits"] = rows

    result = {
        "date": days[-1].isoformat(),
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "days_of_history": len(days),
        "stocks_scanned": len(universe),
        "ahead_days": AHEAD,
        "patterns": patterns_out,
        "stocks": stocks_out,
    }
    os.makedirs(OUT, exist_ok=True)
    with gzip.open(os.path.join(OUT, "patterns.json.gz"), "wt") as f:
        json.dump(result, f, separators=(",", ":"))
    n_hits = sum(len(s["hits"]) for s in stocks_out.values())
    with_hits = sum(1 for s in stocks_out.values() if s["hits"])
    print(f"Done: {len(universe)} stocks scanned, {with_hits} with signals, {n_hits} signals, "
          f"{os.path.getsize(os.path.join(OUT, 'patterns.json.gz')) / 1024:.0f} KB")
    best = sorted(((p, s) for p, s in patterns_out.items() if s.get("n", 0) >= 200),
                  key=lambda kv: -kv[1]["avg"] * kv[1]["mood"] if kv[1]["mood"] else 0)
    for p, s in best[:10]:
        print(f"  {s['name']}: {s['n']} times, beat SPY {s['beat']:.0%}, avg {s['avg']:+.2%} over 20 days")


if __name__ == "__main__":
    main()

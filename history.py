"""
Ten years of daily prices from Nasdaq's free quote API (split-adjusted, no key), so the grading model can learn from
and be tested on far more than the 2 years the free Massive plan gives.

Each stock's history is downloaded once and kept in the cache; Massive's nightly days are added on top. The first
download of ~5,000 stocks takes a few nights (a time budget per night); the model switches to the long history once
most stocks have it.

Limit: Nasdaq only has stocks that still trade today, so in the years before the Massive data, companies that later
failed are missing. That flatters older results somewhat; the last 2 years (from Massive) include them.
"""
import gzip
import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

import numpy as np

HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
           "Accept": "application/json, text/plain, */*", "Accept-Language": "en-US,en;q=0.9",
           "Origin": "https://www.nasdaq.com", "Referer": "https://www.nasdaq.com/"}
ETFS = {"SPY"}
_slow = threading.Event()


def _num(s):
    s = (s or "").replace("$", "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return float("nan")


def fetch(ticker, years=10):
    """{'d': [iso dates oldest first], 'c': [...], 'v': [...]}, or {} if Nasdaq has nothing. Raises on errors worth retrying."""
    today = date.today()
    start = today - timedelta(days=int(365.25 * years) + 5)
    sym = ticker.replace(".", "-")
    klass = "etf" if ticker in ETFS else "stocks"
    url = (f"https://api.nasdaq.com/api/quote/{sym}/historical?assetclass={klass}&fromdate={start.isoformat()}"
           f"&limit=9999&todate={today.isoformat()}")
    for attempt in range(3):
        try:
            body = urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60).read()
            break
        except urllib.error.HTTPError as e:
            if e.code in (429, 403, 503):
                _slow.set()
                time.sleep(60 * (attempt + 1))
                continue
            raise
        except Exception:
            time.sleep(5)
    else:
        raise RuntimeError("Nasdaq kept refusing")
    j = json.loads(body)
    rows = (((j.get("data") or {}).get("tradesTable") or {}).get("rows")) or []
    out = {"d": [], "c": [], "v": []}
    for r in reversed(rows):
        try:
            d = datetime.strptime(r["date"], "%m/%d/%Y").date().isoformat()
        except (KeyError, ValueError):
            continue
        c = _num(r.get("close"))
        if c > 0:
            out["d"].append(d)
            out["c"].append(c)
            out["v"].append(_num(r.get("volume")) if np.isfinite(_num(r.get("volume"))) else 0.0)
    return out if out["d"] else {}


def path(cache_dir, t):
    return os.path.join(cache_dir, "hist", t.replace("/", "_") + ".json.gz")


def backfill(tickers, cache_dir, budget_s=4200, threads=1, pause=1.0):
    """Downloads histories not yet cached (SPY first), within a time budget. Returns how many are cached now."""
    os.makedirs(os.path.join(cache_dir, "hist"), exist_ok=True)
    todo = [t for t in ["SPY"] + sorted(set(tickers) - {"SPY"}) if not os.path.exists(path(cache_dir, t))]
    t0 = time.time()
    done = [0]

    def one(t):
        if time.time() - t0 > budget_s:
            return
        time.sleep(pause * (5 if _slow.is_set() else 1))   # Nasdaq blocks rapid or parallel requests
        try:
            h = fetch(t)
        except Exception as e:
            print(f"  history for {t} failed: {e}", flush=True)
            return
        with gzip.open(path(cache_dir, t), "wt") as f:
            json.dump(h, f)
        done[0] += 1
        if done[0] % 250 == 0:
            print(f"  {done[0]} histories downloaded, {time.time() - t0:.0f}s", flush=True)

    if todo:
        print(f"History: {len(todo)} stocks to download (budget {budget_s // 60} minutes)")
        with ThreadPoolExecutor(threads) as ex:
            list(ex.map(one, todo))
    have = sum(1 for t in tickers if os.path.exists(path(cache_dir, t)))
    print(f"History: {have} of {len(set(tickers))} stocks have 10 years cached")
    return have


def load(cache_dir, t):
    try:
        with gzip.open(path(cache_dir, t), "rt") as f:
            return json.load(f)
    except Exception:
        return {}


def merged_panel(cache_dir, days, data, tickers):
    """Calendar of Nasdaq history (before Massive's first day) + Massive's days, and close/volume matrices.
    Returns (dates, C, V) with NaN where a stock didn't trade (rows in the order of `tickers`)."""
    spy = load(cache_dir, "SPY")
    first = days[0].isoformat()
    old = [d for d in spy.get("d", []) if d < first]
    dates = [date.fromisoformat(d) for d in old] + list(days)
    pos = {d.isoformat(): i for i, d in enumerate(dates)}
    n, N = len(dates), len(tickers)
    ix = {t: i for i, t in enumerate(tickers)}
    C = np.full((N, n), np.nan)
    V = np.zeros((N, n), np.float32)
    k0 = len(old)
    for t, i in ix.items():
        h = load(cache_dir, t)
        for d, c, v in zip(h.get("d", ()), h.get("c", ()), h.get("v", ())):
            if d < first:
                j = pos.get(d)
                if j is not None:
                    C[i, j] = c
                    V[i, j] = v
    for j, row in enumerate(data):
        for t, r in row.items():
            i = ix.get(t)
            if i is not None and r[3] > 0:
                C[i, k0 + j] = r[3]
                V[i, k0 + j] = r[4]
    return dates, C, V


if __name__ == "__main__":
    # Backfill: python history.py <minutes> [slice] [slices]
    # The stock list comes from the latest published scan, biggest companies first; with slices, each job (its own
    # server address, which Nasdaq limits separately) takes every n-th stock.
    import sys
    base = "https://github.com/RashestLynx/stock-Tracker-patterns/releases/download/latest/"
    pats = json.load(gzip.open(urllib.request.urlopen(base + "patterns.json.gz"), "rt"))
    try:
        caps = {t: e.get("cap", 0) for t, e in json.load(gzip.open(urllib.request.urlopen(base + "grades.json.gz"), "rt"))["grades"].items()}
    except Exception:
        caps = {}
    order = sorted(set(pats["stocks"]) | {"SPY"}, key=lambda t: (t != "SPY", -caps.get(t, 0), t))
    mins = int(sys.argv[1]) if len(sys.argv) > 1 else 280
    k, n = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 1)
    mine = order[k::n] + (["SPY"] if k else [])
    backfill(mine, os.environ.get("CACHE_DIR", "cache"), budget_s=mins * 60)

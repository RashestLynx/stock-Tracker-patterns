import time, sys, collections, urllib.request, urllib.error
sys.path.insert(0, ".")
import history
from datetime import date, timedelta
t0 = time.time(); c = collections.Counter()
for i, t in enumerate(["SPY", "AAPL", "MSFT", "KO", "JNJ", "XOM", "PG", "F", "T", "AMD", "INTC", "WMT", "NKE", "DIS", "BA", "PFE",
                       "MRK", "CVX", "HD", "LOW", "TGT", "SBUX", "MCD", "IBM", "ORCL", "CSCO", "QCOM", "TXN", "AVGO", "ADBE"]):
    url = (f"https://api.nasdaq.com/api/quote/{t}/historical?assetclass={'etf' if t == 'SPY' else 'stocks'}"
           f"&fromdate={(date.today() - timedelta(days=3660)).isoformat()}&limit=9999&todate={date.today().isoformat()}")
    s = time.time()
    try:
        r = urllib.request.urlopen(urllib.request.Request(url, headers=history.HEADERS), timeout=60); b = r.read(); code = r.status
    except urllib.error.HTTPError as e:
        code = e.code; b = e.read()[:150]
    except Exception as e:
        code = type(e).__name__; b = b""
    c[code] += 1
    print(i, t, code, f"{time.time() - s:.1f}s", len(b), b[:80] if code != 200 else ""); sys.stdout.flush()
    time.sleep(1.0)
print("done", dict(c), f"{time.time() - t0:.0f}s")

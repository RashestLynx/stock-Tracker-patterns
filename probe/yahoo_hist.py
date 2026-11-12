import json, time, urllib.request, urllib.error, collections, sys
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
      "Accept": "application/json, text/plain, */*", "Accept-Language": "en-US,en;q=0.9"}
TICK = ["AAPL","MSFT","NVDA","KO","JNJ","PLTR","AMD","F","T","XOM","PG","O","WMT","COST","INTC","BA","DIS","NKE","PFE","MRK"]
def get(url, h=UA, t=15):
    try:
        r = urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=t); return r.status, r.read()
    except urllib.error.HTTPError as e: return e.code, e.read()[:200]
    except Exception as e: return type(e).__name__, b""
def run(name, fn, n=20, pause=0.5):
    c = collections.Counter(); t0 = time.time(); info = ""
    for t in TICK[:n]:
        code, body = fn(t); c[code] += 1
        if code == 200 and not info: info = body[:250]
        time.sleep(pause)
    print(f"{name}: {dict(c)} in {time.time()-t0:.0f}s | {info!r}"[:600]); sys.stdout.flush()
run("yahoo q1 chart 10y", lambda t: get(f"https://query1.finance.yahoo.com/v8/finance/chart/{t}?range=10y&interval=1d"))
run("yahoo q2 chart 10y", lambda t: get(f"https://query2.finance.yahoo.com/v8/finance/chart/{t}?range=10y&interval=1d"))
run("yahoo q1 chart period", lambda t: get(f"https://query1.finance.yahoo.com/v8/finance/chart/{t}?period1=1262304000&period2=1762992000&interval=1d"))
NAS = {**UA, "Origin": "https://www.nasdaq.com", "Referer": "https://www.nasdaq.com/"}
run("nasdaq historical 10y", lambda t: get(f"https://api.nasdaq.com/api/quote/{t}/historical?assetclass=stocks&fromdate=2016-01-01&limit=9999&todate=2026-11-11", NAS))
code, body = get("https://api.nasdaq.com/api/quote/AAPL/historical?assetclass=stocks&fromdate=2020-08-20&limit=20&todate=2020-09-10", NAS)
print("nasdaq AAPL around 2020 split:", code, body[:900])
run("stooq csv", lambda t: get(f"https://stooq.com/q/d/l/?s={t.lower()}.us&i=d"), n=3)
run("tiingo nokey", lambda t: get(f"https://api.tiingo.com/tiingo/daily/{t}/prices?startDate=2016-01-01"), n=2)

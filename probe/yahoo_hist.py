import json, time, urllib.request, urllib.error, gzip, collections
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"}
r = json.load(gzip.open(urllib.request.urlopen("https://github.com/RashestLynx/stock-Tracker-patterns/releases/download/latest/patterns.json.gz"), "rt"))
tickers = sorted(r["stocks"])[:400]
codes = collections.Counter(); t0 = time.time(); lens = []
for i, t in enumerate(tickers):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{t.replace('.', '-')}?range=10y&interval=1d&events=split,div"
    try:
        b = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=20).read())
        res = b["chart"]["result"][0]; lens.append(len(res.get("timestamp") or [])); codes[200] += 1
        if i == 0: print(t, list(res["indicators"].keys()), list(res.get("events", {}).keys()), lens[-1])
    except urllib.error.HTTPError as e:
        codes[e.code] += 1
        if e.code == 429: time.sleep(5)
    except Exception as e:
        codes[type(e).__name__] += 1
    time.sleep(0.25)
    if i % 100 == 99: print(i + 1, dict(codes), f"{time.time() - t0:.0f}s")
print("done", dict(codes), "median days", sorted(lens)[len(lens) // 2] if lens else 0)

import json, urllib.request
NAS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
       "Accept": "application/json, text/plain, */*", "Origin": "https://www.nasdaq.com", "Referer": "https://www.nasdaq.com/"}
for t, a, b in [("AAPL", "2016-11-01", "2026-11-11"), ("NVDA", "2016-11-01", "2026-11-11"), ("AAPL", "2020-08-20", "2020-09-10"),
                ("NVDA", "2024-06-03", "2024-06-14"), ("BRK-B", "2025-01-01", "2026-11-11"), ("XOM", "2016-11-01", "2026-11-11")]:
    url = f"https://api.nasdaq.com/api/quote/{t}/historical?assetclass=stocks&fromdate={a}&limit=9999&todate={b}"
    j = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=NAS), timeout=60).read())
    rows = ((j.get("data") or {}).get("tradesTable") or {}).get("rows") or []
    print(t, a, b, "records", (j.get("data") or {}).get("totalRecords"), "first", rows[-1] if rows else None, "last", rows[0] if rows else None)
    if t == "NVDA" and a.startswith("2024"):
        print("  around split:", [(r["date"], r["close"]) for r in rows[::-1]])

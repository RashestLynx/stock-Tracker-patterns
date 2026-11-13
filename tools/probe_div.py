"""One-off: does Massive's dividends endpoint answer on this key? Prints no key."""
import json, os, urllib.request
k = os.environ.get("K", "")
for t in ("KO", "JNJ", "NVDA"):
    u = f"https://api.massive.com/v3/reference/dividends?ticker={t}&order=desc&sort=ex_dividend_date&limit=6&apiKey={k}"
    try:
        j = json.loads(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "StockTracker"}), timeout=30).read())
        r = j.get("results") or []
        print("OK", t, j.get("status"), len(r), [(x.get("ex_dividend_date"), x.get("cash_amount")) for x in r[:4]])
    except Exception as e:
        print("FAIL", t, str(e).replace(k, "***"))

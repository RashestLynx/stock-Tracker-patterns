"""One-off: which free options-chain sources answer, and what fields they give."""
import json, urllib.request, gzip
UA = {"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)", "Accept": "application/json"}
def get(u):
    try:
        r = urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=30)
        b = r.read()
        if b[:2] == b"\x1f\x8b": b = gzip.decompress(b)
        print("OK", r.status, u, len(b)); return json.loads(b)
    except Exception as e:
        print("FAIL", u, e); return None
c = get("https://cdn.cboe.com/api/global/delayed_quotes/options/KO.json")
if c:
    d = c.get("data", {}); print("CBOE KEYS", list(d)[:30]); o = d.get("options", [])
    print("N", len(o)); print("SAMPLE", json.dumps(o[:2])[:900]); print("UNDERLYING", {k: d.get(k) for k in ("current_price", "iv30", "close", "symbol")})
    print("TIMESTAMP", c.get("timestamp"))
n = get("https://api.nasdaq.com/api/quote/KO/option-chain?assetclass=stocks&limit=60&fromdate=all&todate=undefined&excode=oprac&callput=callput&money=all&type=all")
if n: print("NASDAQ", json.dumps(n)[:1200])
y = get("https://query2.finance.yahoo.com/v7/finance/options/KO")
if y: print("YAHOO", json.dumps(y)[:300])
e = get("https://api.nasdaq.com/api/analyst/KO/earnings-date")
if e: print("EARN", json.dumps(e)[:500])
dv = get("https://api.nasdaq.com/api/quote/KO/dividends?assetclass=stocks")
if dv: print("DIV", json.dumps(dv)[:600])

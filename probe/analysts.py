import http.cookiejar, json, urllib.request
jar = http.cookiejar.CookieJar()
op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
UA = {"User-Agent": "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Mobile Safari/537.36"}
def get(u):
    try:
        r = op.open(urllib.request.Request(u, headers=UA), timeout=20); return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:200]
print("fc", get("https://fc.yahoo.com")[0], [c.name for c in jar])
st, crumb = get("https://query2.finance.yahoo.com/v1/test/getcrumb"); print("crumb", st, crumb[:20])
for t in ["NVDA", "AAPL", "KO"]:
    st, b = get(f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{t}?modules=financialData,recommendationTrend&crumb={crumb}")
    print(t, st)
    if st == 200:
        r = json.loads(b)["quoteSummary"]["result"][0]
        fd = r["financialData"]
        print({k: fd.get(k, {}).get("raw") if isinstance(fd.get(k), dict) else fd.get(k) for k in
               ["currentPrice", "targetLowPrice", "targetMeanPrice", "targetHighPrice", "numberOfAnalystOpinions", "recommendationKey"]})
        print(r["recommendationTrend"]["trend"][0])
st, b = get("https://query1.finance.yahoo.com/v7/finance/quote?symbols=NVDA,AAPL&fields=targetMeanPrice&crumb=" + crumb)
print("v7", st, b[:300])

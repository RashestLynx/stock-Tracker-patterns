import json, urllib.request, urllib.error
UA = {"User-Agent": "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Mobile Safari/537.36",
      "Accept": "application/json, text/plain, */*", "Accept-Language": "en-US,en;q=0.9", "Origin": "https://www.nasdaq.com", "Referer": "https://www.nasdaq.com/"}
def get(u, h=UA):
    try:
        r = urllib.request.urlopen(urllib.request.Request(u, headers=h), timeout=20); return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:300]
    except Exception as e:
        return 0, str(e)
for t in ["NVDA", "KO", "BRK.B"]:
    for path in ["targetprice", "ratings"]:
        st, b = get(f"https://api.nasdaq.com/api/analyst/{t}/{path}")
        print(t, path, st, b[:900].replace("\n", " "))
        print()
st, b = get("https://finnhub.io/api/v1/stock/recommendation?symbol=NVDA", {"User-Agent": "x"})
print("finnhub nokey", st, b[:120])

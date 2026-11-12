import json, os, urllib.request, collections
k = os.environ["MASSIVE_KEY"]
pubs = collections.Counter()
for t in ["COST", "NVDA", "AAPL", "PLTR", "KO"]:
    j = json.loads(urllib.request.urlopen(f"https://api.massive.com/v2/reference/news?ticker={t}&limit=20&order=desc&sort=published_utc&apiKey={k}", timeout=30).read())
    res = j.get("results", [])
    for r in res: pubs[(r.get("publisher") or {}).get("name")] += 1
    if t == "COST" and res:
        r = res[0]
        print("keys:", sorted(r.keys()))
        print("sample:", r.get("title"), "|", (r.get("publisher") or {}).get("name"), "|", r.get("published_utc"), "|", (r.get("insights") or [])[:1])
    import time; time.sleep(13)
print("publishers:", dict(pubs))

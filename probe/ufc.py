import json, urllib.request, urllib.parse
H = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/124"}
def get(u, raw=False):
    try:
        b = urllib.request.urlopen(urllib.request.Request(u, headers=H), timeout=30).read()
        return b if raw else json.loads(b)
    except Exception as e: return {"ERR": str(e)}
E = "https://site.api.espn.com/apis/site/v2/sports/mma/ufc"
j = get(f"{E}/scoreboard?dates=20251101-20251130&limit=1000")
ev = j.get("events", [])
print("events", len(ev))
if ev:
    c = ev[0]["competitions"][0]
    print("competition keys", sorted(c.keys()))
    print("type", c.get("type"), "| status", (c.get("status") or {}).get("type", {}).get("detail"), "|", (c.get("status") or {}).get("result"))
    x = c["competitors"][0]; print("competitor keys", sorted(x.keys())); print("athlete keys", sorted((x.get("athlete") or {}).keys()))
    print("athlete", {k: x["athlete"].get(k) for k in ("id", "displayName", "age", "height", "weight", "reach", "dateOfBirth")}, "records", x.get("records"))
    print("notes", c.get("notes"), "| details", str(c.get("details"))[:200])
    aid = x["athlete"]["id"]
    a = get(f"https://site.web.api.espn.com/apis/common/v3/sports/mma/athletes/{aid}")
    print("athlete v3 keys", sorted((a.get("athlete") or {}).keys())[:40], str(a)[:300] if "ERR" in a else "")
    at = a.get("athlete", {}); print({k: at.get(k) for k in ("age", "displayHeight", "displayWeight", "displayReach", "weightClass", "stance", "dateOfBirth")})
    s = get(f"https://site.web.api.espn.com/apis/common/v3/sports/mma/athletes/{aid}/fighthistory")
    print("fighthistory", str(s)[:600])
r = get("https://news.google.com/rss/search?q=" + urllib.parse.quote('"Paddy Pimblett" injury OR "missed weight" OR "weight cut"') + "&hl=en-US&gl=US&ceid=US:en", raw=True)
print("google news rss", r[:500] if isinstance(r, bytes) else r)

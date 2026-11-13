"""One-off: prints which ESPN stat groups exist per fighter."""
import json, urllib.request
def get(u):
    try:
        return json.loads(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=30).read())
    except Exception as e:
        print("FAIL", u, e); return {}
for fid in ("4024488", "3088812", "2335639"):
    s = get(f"https://site.web.api.espn.com/apis/common/v3/sports/mma/athletes/{fid}/stats")
    print("TOP KEYS", list(s))
    for c in s.get("categories", []):
        print("CAT", c.get("name"), c.get("displayName"), c.get("labels"), "n=", len(c.get("statistics", [])))
        if c.get("statistics"): print("  SAMPLE", c["statistics"][0])
    for k in s:
        if k != "categories": print("EXTRA", k, json.dumps(s[k])[:600])

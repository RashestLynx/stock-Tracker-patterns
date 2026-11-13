"""One-off: prints which ESPN / UFC fields exist, to decide which fight factors can be measured."""
import json, urllib.request
def get(u):
    try:
        return json.loads(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=30).read())
    except Exception as e:
        print("FAIL", u, e); return {}
j = get("https://site.api.espn.com/apis/site/v2/sports/mma/ufc/scoreboard?dates=20251001-20251031&limit=1000")
ev = j["events"][0]; c = ev["competitions"][0]
print("EVENT KEYS", list(ev)); print("COMP KEYS", list(c))
print("VENUE", json.dumps(c.get("venue") or ev.get("venues"))[:400])
print("FORMAT", c.get("format")); print("ODDS", json.dumps(c.get("odds"))[:500])
print("STATUS", json.dumps(c.get("status"))[:400])
x = c["competitors"][0]; print("COMPETITOR KEYS", list(x)); print("ATHLETE", json.dumps(x.get("athlete"))[:600])
print("RECORDS", x.get("records")); print("LINESCORES", json.dumps(x.get("linescores"))[:300])
fid = x.get("id")
a = get(f"https://site.web.api.espn.com/apis/common/v3/sports/mma/athletes/{fid}").get("athlete", {})
print("ATHLETE V3 KEYS", list(a)); print({k: a.get(k) for k in ("citizenship", "citizenshipCountry", "birthPlace", "association", "flag", "age", "stance", "displayWeight")})
for path in (f"athletes/{fid}/stats", f"athletes/{fid}/splits", f"athletes/{fid}/gamelog"):
    s = get(f"https://site.web.api.espn.com/apis/common/v3/sports/mma/{path}")
    print(path.upper(), json.dumps(s)[:1500])
sm = get(f"https://site.api.espn.com/apis/site/v2/sports/mma/ufc/summary?event={ev['id']}")
print("SUMMARY KEYS", list(sm)); print(json.dumps(sm)[:1500])
# upcoming card odds
up = get("https://site.api.espn.com/apis/site/v2/sports/mma/ufc/scoreboard")
for e in up.get("events", [])[:1]:
    for cc in e["competitions"][:3]:
        print("UPCOMING ODDS", json.dumps(cc.get("odds"))[:400])

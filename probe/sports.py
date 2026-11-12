import json, urllib.request, urllib.error, sys
H = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36", "Accept": "*/*"}
def get(u):
    try:
        r = urllib.request.urlopen(urllib.request.Request(u, headers=H), timeout=30); return r.status, r.read()
    except urllib.error.HTTPError as e: return e.code, e.read()[:200]
    except Exception as e: return type(e).__name__, b""
E = "https://site.api.espn.com/apis/site/v2/sports"
for name, path, rng in [("NBA", "basketball/nba", "20250101-20250131"), ("NHL", "hockey/nhl", "20250101-20250131"),
                        ("MLB", "baseball/mlb", "20250601-20250630"), ("CFB", "football/college-football", "20241101-20241130"),
                        ("CBB", "basketball/mens-college-basketball", "20250201-20250205"), ("NFL", "football/nfl", "20241101-20241130"),
                        ("EPL", "soccer/eng.1", "20250101-20250131"), ("MLS", "soccer/usa.1", "20250601-20250630"),
                        ("UCL", "soccer/uefa.champions", "20250101-20250228"), ("UFC", "mma/ufc", "20250101-20250331")]:
    extra = "&groups=50" if name in ("CBB",) else "&groups=80" if name == "CFB" else ""
    code, b = get(f"{E}/{path}/scoreboard?dates={rng}&limit=1000{extra}")
    n, sample = 0, ""
    if code == 200:
        j = json.loads(b); ev = j.get("events", []); n = len(ev)
        if ev:
            c = ev[0]["competitions"][0]
            sample = " vs ".join(f"{x.get('team', x.get('athlete', {})).get('displayName', x.get('athlete', {}).get('displayName', '?'))} {x.get('score', '')} home={x.get('homeAway', '')} winner={x.get('winner')}" for x in c.get("competitors", []))
    print(name, code, "events", n, "|", sample[:200]); sys.stdout.flush()
for name, u in [("EPL csv", "https://www.football-data.co.uk/mmz4281/2425/E0.csv"), ("MLS csv", "https://www.football-data.co.uk/new/USA.csv"),
                ("nflverse", "https://github.com/nflverse/nfldata/raw/master/data/games.csv")]:
    code, b = get(u); print(name, code, len(b), b[:160].replace(b"\n", b" | ") if code == 200 else b"")
# ESPN team stats endpoint (side-by-side numbers)
code, b = get(f"{E}/basketball/nba/teams/2/statistics"); print("team stats", code, b[:200])

import json, urllib.request, urllib.error, sys
H = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36", "Accept": "*/*"}
def get(u):
    try:
        r = urllib.request.urlopen(urllib.request.Request(u, headers=H), timeout=30); return r.status, r.read()
    except urllib.error.HTTPError as e: return e.code, e.read()[:300]
    except Exception as e: return type(e).__name__, b""
E = "https://site.api.espn.com/apis/site/v2/sports"
tests = [("NBA day", f"{E}/basketball/nba/scoreboard?dates=20250115"),
         ("NBA range noLimit", f"{E}/basketball/nba/scoreboard?dates=20250101-20250107"),
         ("NBA season", f"{E}/basketball/nba/scoreboard?dates=2025&seasontype=2"),
         ("NBA team schedule", f"{E}/basketball/nba/teams/2/schedule?season=2025"),
         ("NHL day", f"{E}/hockey/nhl/scoreboard?dates=20250115"),
         ("MLB day", f"{E}/baseball/mlb/scoreboard?dates=20250615"),
         ("CFB week", f"{E}/football/college-football/scoreboard?dates=20241102&groups=80&limit=300"),
         ("CBB day", f"{E}/basketball/mens-college-basketball/scoreboard?dates=20250201&groups=50&limit=400"),
         ("EPL day", f"{E}/soccer/eng.1/scoreboard?dates=20250104"),
         ("UCL day", f"{E}/soccer/uefa.champions/scoreboard?dates=20250121"),
         ("core events", "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/seasons/2025/types/2/events?limit=5"),
         ("UFC athlete", f"{E}/mma/ufc/scoreboard?dates=20250301"),
         ("kalshi settled", "https://api.elections.kalshi.com/trade-api/v2/markets?series_ticker=KXNBAGAME&status=settled&limit=5")]
for name, u in tests:
    code, b = get(u)
    info = ""
    if code == 200:
        try:
            j = json.loads(b)
            if "events" in j and isinstance(j["events"], list):
                ev = j["events"]; info = f"events {len(ev)}"
                if ev and "competitions" in ev[0]:
                    c = ev[0]["competitions"][0]; info += " | " + " vs ".join(f"{x.get('team',{}).get('abbreviation', x.get('athlete',{}).get('displayName','?'))} {x.get('score','')} {x.get('homeAway','')}" for x in c.get("competitors", []))
                    info += f" | odds keys {list((c.get('odds') or [{}])[0].keys())[:8]}"
            elif "markets" in j:
                m = j["markets"]; info = f"markets {len(m)} | " + str({k: m[0].get(k) for k in ('ticker','result','last_price_dollars','previous_price_dollars','close_time')} if m else "")
            else:
                info = str(list(j.keys()))[:200] + " " + str(j.get("count", ""))
        except Exception as e:
            info = f"parse {e}"
    else:
        info = b[:150]
    print(name, code, info[:300]); sys.stdout.flush()

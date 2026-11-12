import json, urllib.request
H = {"User-Agent": "Mozilla/5.0"}
def get(u):
    try: return json.loads(urllib.request.urlopen(urllib.request.Request(u, headers=H), timeout=30).read())
    except Exception as e: return {"ERR": str(e)}
E = "https://site.api.espn.com/apis/site/v2/sports"
for p, tid in [("football/nfl", 12), ("baseball/mlb", 10), ("football/college-football", 333), ("soccer/usa.1", 9720), ("soccer/eng.1", 359)]:
    j = get(f"{E}/{p}/teams/{tid}"); t = j.get("team", {})
    v = (t.get("franchise") or {}).get("venue") or t.get("venue") or {}
    print(p, t.get("displayName"), "venue:", v.get("fullName"), v.get("address"), "indoor", v.get("indoor"), "grass", v.get("grass"))
for p in ["football/nfl", "baseball/mlb", "football/college-football", "soccer/eng.1"]:
    j = get(f"{E}/{p}/scoreboard"); ev = j.get("events", [])
    if ev:
        c = ev[0]["competitions"][0]; print(p, "upcoming:", ev[0].get("name"), ev[0].get("date"), c.get("venue", {}).get("address"), c.get("venue", {}).get("indoor"), "weather:", ev[0].get("weather"))
    else: print(p, "no events", str(j)[:100])
g = get("https://geocoding-api.open-meteo.com/v1/search?name=Kansas%20City&count=5&country_code=US"); print("geo", [(r["name"], r.get("admin1"), r["latitude"], r["longitude"]) for r in g.get("results", [])][:3])
a = get("https://archive-api.open-meteo.com/v1/archive?latitude=39.05&longitude=-94.48&start_date=2019-01-01&end_date=2026-11-10&daily=temperature_2m_max,precipitation_sum,wind_speed_10m_max&temperature_unit=fahrenheit&wind_speed_unit=mph&timezone=America/Chicago")
print("archive days", len((a.get("daily") or {}).get("time", [])), str(a)[:150] if "ERR" in a else "")
f = get("https://api.open-meteo.com/v1/forecast?latitude=39.05&longitude=-94.48&hourly=temperature_2m,precipitation,wind_speed_10m&temperature_unit=fahrenheit&wind_speed_unit=mph&timezone=UTC&forecast_days=8")
print("forecast hours", len((f.get("hourly") or {}).get("time", [])))

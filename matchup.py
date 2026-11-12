"""
Matchup and weather layer on top of the Elo ratings.

For every game, using only what was known before it:
  - the Elo chance (as log-odds)
  - rest: days since each team's last game
  - form: average margin over each team's last 5 games
  - season: average margin so far this season (3+ games)
  - head to head: the home team's margin against this opponent over the last 3 years (shrunk toward 0)
  - weather at outdoor games (NFL: nflverse's game-time readings; MLB: that day at the home park, from Open-Meteo's
    historical archive): wind, cold, rain, and each one times the Elo chance (bad weather makes games more random)

A logistic regression learns how much each matters. It's tested exactly like the ratings: for each of the last 3
seasons it's fit only on earlier seasons, then scored on that season. The app uses it only if it beat plain Elo on
those unseen seasons. Weather is checked the same way on its own: kept only if it helped.

Upcoming games for the next week get both chances, with the forecast for outdoor games.
"""
import json
import math
import os
import urllib.parse
from datetime import date, datetime, timedelta

import sys

# When sports.py runs as a script it's the "__main__" module; use that one (it holds this run's results).
_main = sys.modules.get("__main__")
if _main is not None and hasattr(_main, "learn_league") and hasattr(_main, "LEAGUES"):
    sports = _main
else:
    import sports

NAMES = ["elo", "rest", "form", "season", "h2h", "wind", "cold", "rain", "wind_x", "cold_x", "rain_x"]
WEATHER = {"wind", "cold", "rain", "wind_x", "cold_x", "rain_x"}
WEATHER_LEAGUES = {"nfl", "mlb"}


# ------------------------------------------------------------ places and weather
def geocode(addr):
    """(lat, lon) for an ESPN venue address, via Open-Meteo's free geocoder (zip code first, then city)."""
    def look(q, country=None):
        u = "https://geocoding-api.open-meteo.com/v1/search?count=5&name=" + urllib.parse.quote(q) + (f"&country_code={country}" if country else "")
        r = (sports.get(u) or {}).get("results") or []
        return r
    zipc, city, state = (addr or {}).get("zipCode"), (addr or {}).get("city"), (addr or {}).get("state") or ""
    for q, c in ((zipc, "US"), (city, "US"), (city, "CA"), (city, None)):
        if not q:
            continue
        res = look(q, c)
        if state:
            pick = [x for x in res if state.lower() in (x.get("admin1") or "").lower() or
                    (len(state) == 2 and (x.get("admin1_code") or "").upper() == state.upper())]
            res = pick or res
        if res:
            return res[0]["latitude"], res[0]["longitude"]
    return None


def team_venues(key, path):
    """{team name: {"lat", "lon", "indoor"}} from each team's home venue on ESPN (cached a month)."""
    def fetch():
        out = {}
        teams = (sports.get(f"{sports.ESPN}/{path}/teams?limit=1000") or {}).get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", [])
        for t in teams:
            tid, name = t["team"]["id"], t["team"].get("displayName")
            j = (sports.get(f"{sports.ESPN}/{path}/teams/{tid}") or {}).get("team", {})
            v = (j.get("franchise") or {}).get("venue") or j.get("venue") or {}
            ll = geocode(v.get("address")) if v.get("address") else None
            if ll:
                out[name] = {"lat": ll[0], "lon": ll[1], "indoor": bool(v.get("indoor")), "venue": v.get("fullName")}
        return out
    return sports.cached(f"{key}/venues.json.gz", fetch, keep_days=30) or {}


def daily_history(lat, lon, start, end):
    """{iso date: (max temp F, max wind mph, rain mm)} from Open-Meteo's archive."""
    u = (f"https://archive-api.open-meteo.com/v1/archive?latitude={lat:.3f}&longitude={lon:.3f}&start_date={start}&end_date={end}"
         "&daily=temperature_2m_max,precipitation_sum,wind_speed_10m_max&temperature_unit=fahrenheit&wind_speed_unit=mph&timezone=auto")
    d = (sports.get(u) or {}).get("daily") or {}
    return {t: (a, b, c) for t, a, c, b in zip(d.get("time", []), d.get("temperature_2m_max", []),
                                                d.get("precipitation_sum", []), d.get("wind_speed_10m_max", []))}


def forecast(lat, lon, when):
    """(temp F, wind mph, rain mm) at the hour of `when` (UTC datetime), or None."""
    u = (f"https://api.open-meteo.com/v1/forecast?latitude={lat:.3f}&longitude={lon:.3f}&hourly=temperature_2m,precipitation,"
         "wind_speed_10m&temperature_unit=fahrenheit&wind_speed_unit=mph&timezone=UTC&forecast_days=10")
    h = (sports.get(u) or {}).get("hourly") or {}
    key = when.strftime("%Y-%m-%dT%H:00")
    if key in h.get("time", []):
        i = h["time"].index(key)
        return h["temperature_2m"][i], h["wind_speed_10m"][i], h["precipitation"][i]
    return None


def league_weather(key, path, games):
    """{(date, home): (temp, wind, rain, outdoors)} for past games."""
    if key == "nfl":
        return dict(sports.NFL_WX)
    if key != "mlb":
        return {}
    venues = team_venues(key, path)
    out = {}
    start, end = games[0][0], (date.today() - timedelta(days=2)).isoformat()
    for team, v in venues.items():
        if v["indoor"]:
            continue
        hist = sports.cached(f"{key}/wx/{team.replace(' ', '_')}-{start}.json.gz",
                             lambda v=v: daily_history(v["lat"], v["lon"], start, end), keep_days=1)
        for d, w in (hist or {}).items():
            out[(d, team)] = (w[0], w[1], w[2], True)
    return out


def wx_features(w, elo):
    """Weather numbers on a common scale; zero indoors or when unknown."""
    if not w or not w[3]:
        return [0.0] * 6
    temp, wind, rain = w[0], w[1], w[2]
    wind = max(0.0, (wind or 0) - 8) / 10          # above a light breeze
    cold = max(0.0, 45 - temp) / 20 if temp is not None else 0.0
    rain = min(rain or 0, 20) / 10
    return [wind, cold, rain, wind * elo, cold * elo, rain * elo]


# ------------------------------------------------------------ features
class History:
    """Each team's past games, for features known before the next game."""

    def __init__(self, scale):
        self.games = {}   # team -> list of (date, opp, margin)
        self.scale = scale

    def feats(self, d, h, a):
        def last(t):
            g = self.games.get(t, [])
            return g
        gh, ga = last(h), last(a)
        dd = date.fromisoformat(d)
        def rest(g):
            return min(10, (dd - date.fromisoformat(g[-1][0])).days) if g else 7
        def form(g):
            m = [x[2] for x in g[-5:]]
            return sum(m) / len(m) if m else 0.0
        def season(g):
            m = []
            for x in reversed(g):
                if (dd - date.fromisoformat(x[0])).days > 200:
                    break
                m.append(x[2])
            return sum(m) / len(m) if len(m) >= 3 else 0.0
        cut = (dd - timedelta(days=3 * 365)).isoformat()
        h2h = [x[2] for x in gh if x[1] == a and x[0] >= cut]
        sc = self.scale
        return [(rest(gh) - rest(ga)) / 7, (form(gh) - form(ga)) / sc, (season(gh) - season(ga)) / sc,
                sum(h2h) / (len(h2h) + 3) / sc]

    def add(self, d, h, a, hs, as_):
        self.games.setdefault(h, []).append((d, a, hs - as_))
        self.games.setdefault(a, []).append((d, h, as_ - hs))


def logit(p):
    p = min(1 - 1e-4, max(1e-4, p))
    return math.log(p / (1 - p))


def build(games, preds, weather, scale):
    X, hist = [], History(scale)
    for i, g in enumerate(games):
        d, h, a, hs, as_ = g[:5]
        e = logit(preds[i])
        X.append([e] + hist.feats(d, h, a) + wx_features(weather.get((d, h)), e))
        hist.add(d, h, a, hs, as_)
    return X, hist


# ------------------------------------------------------------ logistic regression
def fit(X, y, cols, lam=2.0, iters=30):
    """L2-regularized logistic regression, pulled toward 'just use Elo' (weight 1 on Elo, 0 on the rest)."""
    import numpy as np
    A = np.array([[1.0] + [row[c] for c in cols] for row in X])
    yv = np.array(y)
    prior = np.zeros(A.shape[1])
    if cols and cols[0] == 0:
        prior[1] = 1.0
    reg = lam * np.eye(A.shape[1])
    reg[0, 0] = 0
    w = prior.copy()
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(A @ w, -30, 30)))
        g = A.T @ (p - yv) + reg @ (w - prior)
        Hm = (A * (p * (1 - p))[:, None]).T @ A + reg
        step = np.linalg.solve(Hm, g)
        w -= step
        if np.abs(step).max() < 1e-7:
            break
    return w


def predict(w, row, cols):
    z = w[0] + sum(w[k + 1] * row[c] for k, c in enumerate(cols))
    return 1 / (1 + math.exp(-max(-30, min(30, z))))


def evaluate(key, games, scale):
    """Walk-forward test of the matchup layer, with and without weather. Returns (record, fitted model or None)."""
    L = sports.learn_league.last
    name, _, path, draws = sports.LEAGUES[key]
    weather = league_weather(key, path, games) if key in WEATHER_LEAGUES else {}
    allc = list(range(len(NAMES)))
    noweather = [i for i, n in enumerate(NAMES) if n not in WEATHER]
    variants = {"elo": [0], "matchup": noweather, "matchup+weather": allc} if weather else {"elo": [0], "matchup": noweather}
    seasons = L["seasons"]
    y = [1.0 if g[3] > g[4] else 0.0 if g[3] < g[4] else 0.5 for g in games]
    scored = {v: [0.5] * len(games) for v in variants}
    test_idx = []
    for s in L["tests"]:
        preds = L["runs"][L["chosen"][s]][0]
        X, _ = build(games, preds, weather, scale)
        tr = [i for i, x in enumerate(seasons) if 1 <= x < s]
        te = [i for i, x in enumerate(seasons) if x == s]
        test_idx += te
        for v, cols in variants.items():
            w = fit([X[i] for i in tr], [y[i] for i in tr], cols)
            for i in te:
                scored[v][i] = predict(w, X[i], cols)
    d_prior = sum(L["d_test"].values()) / max(1, len(L["d_test"]))
    res = {v: sports.score_games(games, scored[v], test_idx, draws, d_prior / 0.95) for v in variants}
    MARGIN = 0.0005   # must beat the simpler model by a real amount, not a rounding error
    used = None
    if res["matchup"]["logloss"] < res["elo"]["logloss"] - MARGIN:
        used = "matchup"
    helped = "matchup+weather" in res and res["matchup+weather"]["logloss"] < min(res["matchup"]["logloss"], res["elo"]["logloss"]) - MARGIN
    if helped:
        used = "matchup+weather"
    rec = {"tested": res, "used": used or "elo", "weather_helped": helped}
    # today's model, fit on every season (if it earned a place)
    model = None
    preds, _, ratings = L["runs"][L["best"]]
    X, hist = build(games, preds, weather, scale)
    if used:
        cols = variants[used]
        tr = [i for i, x in enumerate(seasons) if x >= 1]
        w = fit([X[i] for i in tr], [y[i] for i in tr], cols)
        model = {"w": [round(float(x), 4) for x in w], "cols": cols}
        rec["weights"] = {NAMES[c]: round(float(w[k + 1]), 3) for k, c in enumerate(cols)}
    return rec, model, hist, ratings


def upcoming(key, path, games, model, hist, ratings, home_edge, scale, draws, d0):
    """Next 7 days: both chances for each game, with the forecast at outdoor games."""
    if key == "ufc":
        return []
    today = date.today()
    extra = "&groups=80" if key == "cfb" else "&groups=50" if key == "cbb" else ""
    j = sports.get(f"{sports.ESPN}/{path}/scoreboard?dates={today:%Y%m%d}-{today + timedelta(days=7):%Y%m%d}&limit=1000{extra}") or {}
    out = []
    for ev in j.get("events", []):
        c = (ev.get("competitions") or [{}])[0]
        if ((c.get("status") or {}).get("type") or {}).get("completed"):
            continue
        comp = c.get("competitors") or []
        h = next((x for x in comp if x.get("homeAway") == "home"), None)
        a = next((x for x in comp if x.get("homeAway") == "away"), None)
        if not h or not a:
            continue
        hn, an = h["team"].get("displayName"), a["team"].get("displayName")
        if hn not in ratings or an not in ratings:
            continue
        neutral = bool(c.get("neutralSite"))
        diff = ratings[hn] - ratings[an] + (0 if neutral else home_edge)
        e = 1 / (1 + 10 ** (-diff / 400))
        when = datetime.fromisoformat((ev.get("date") or "").replace("Z", "+00:00")).replace(tzinfo=None)
        v = c.get("venue") or {}
        wx = None
        if key in WEATHER_LEAGUES or key in ("cfb", "mls", "epl"):
            if not v.get("indoor") and v.get("address"):
                ll = geocode(v["address"])
                f = forecast(*ll, when) if ll else None
                if f:
                    wx = (f[0], f[1], f[2], True)
        row = {"date": when.date().isoformat(), "time": ev.get("date"), "home": hn, "away": an, "venue": v.get("fullName"),
               "indoor": bool(v.get("indoor")), "elo": round(e, 4)}
        if wx:
            row["wx"] = {"temp": round(wx[0]), "wind": round(wx[1]), "rain": round(wx[2], 1)}
        if model:
            f = [logit(e)] + hist.feats(when.date().isoformat(), hn, an) + wx_features(wx if key in WEATHER_LEAGUES else None, logit(e))
            row["full"] = round(predict(model["w"], f, model["cols"]), 4)
            feats = dict(zip(NAMES, f))
            row["why"] = {k: round(feats[k], 3) for k in ("rest", "form", "season", "h2h")}
        out.append(row)
    return out

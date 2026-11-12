#!/usr/bin/env python3
"""
Sports ratings for every league the app covers, learned from years of results and tested on seasons they never saw.

For each league:
  1. Results: ESPN team schedules (NBA, NHL, MLB, college football and basketball), nflverse (NFL, with sportsbook
     lines), football-data.co.uk (Premier League and MLS, with closing bookmaker odds), ESPN fight cards (UFC).
  2. Elo ratings with a margin-of-victory multiplier and the autocorrelation fix (FiveThirtyEight's method), home
     advantage, and part of each rating pulled back toward average between seasons.
  3. The settings (how fast ratings move, home edge, how much to pull back) are picked walk-forward: for each test
     season, only the seasons before it are used to choose them. The test season's predictions are then scored:
     how often the favorite won, Brier score and log loss, and how well "70%" means 70% (calibration). Where
     sportsbook odds exist, the same games are scored for the books too, so you can see who was more accurate.
  4. Side by side for every team: rating and rank, record, points for and against, last 10, home and away, strength
     of schedule, streak, 30-day rating change, recent games (for head-to-head), and current injuries.
  5. Kalshi's own track record: settled game markets, price a day before the end vs. what happened, binned, so the
     app can correct the favorite-longshot bias (cheap contracts winning less often than their price says).

Writes out/sports.json.gz.
"""
import csv
import gzip
import io
import json
import math
import os
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta

CACHE = os.path.join(os.environ.get("CACHE_DIR", "cache"), "sports")
OUT = os.environ.get("OUT_DIR", "out")
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
      "Accept": "application/json, text/csv, */*"}
ESPN = "https://site.api.espn.com/apis/site/v2/sports"
SEASONS = 7
_last = [0.0]

LEAGUES = {   # app key -> (name, source, espn path, draws allowed)
    "nfl": ("NFL", "nflverse", "football/nfl", False),
    "nba": ("NBA", "espn", "basketball/nba", False),
    "nhl": ("NHL", "espn", "hockey/nhl", False),
    "mlb": ("MLB", "espn", "baseball/mlb", False),
    "cfb": ("College football", "espn", "football/college-football", False),
    "cbb": ("College basketball", "espn", "basketball/mens-college-basketball", False),
    "epl": ("Premier League", "fd", "soccer/eng.1", True),
    "mls": ("MLS", "fd_mls", "soccer/usa.1", True),
    "ufc": ("UFC", "ufc", "mma/ufc", False),
}
GRID = {   # (K values, home edges, pull-back shares) tried per league
    "nfl": ([15, 20, 25, 30], [30, 48, 65], [0.25, 0.33, 0.5]),
    "nba": ([15, 20, 25], [50, 75, 100], [0.25, 0.33, 0.5]),
    "nhl": ([4, 6, 8, 10], [20, 35, 50], [0.25, 0.33, 0.5]),
    "mlb": ([3, 4, 6], [15, 24, 35], [0.25, 0.33, 0.5]),
    "cfb": ([20, 25, 30, 40], [40, 55, 70], [0.2, 0.33, 0.5]),
    "cbb": ([15, 20, 30], [60, 80, 100], [0.2, 0.33, 0.5]),
    "epl": ([15, 20, 30], [40, 60, 80], [0.2, 0.33, 0.5]),
    "mls": ([15, 20, 30], [50, 75, 100], [0.2, 0.33, 0.5]),
    "ufc": ([40, 60, 90], [0], [0.0, 0.1]),
}


# ------------------------------------------------------------ fetching
def get(url, pace=0.25, tries=4, raw=False):
    for attempt in range(tries):
        wait = pace - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40) as r:
                b = r.read()
                if r.headers.get("Content-Encoding") == "gzip" or b[:2] == b"\x1f\x8b":
                    b = gzip.decompress(b)
                return b if raw else json.loads(b)
        except urllib.error.HTTPError as e:
            if e.code in (404, 400):
                return None
            time.sleep(5 * (attempt + 1))
        except Exception:
            time.sleep(3 * (attempt + 1))
    return None


def cached(name, fetch, keep_days=None):
    """Cache a JSON-able result on disk. keep_days None = forever."""
    path = os.path.join(CACHE, name)
    if os.path.exists(path) and (keep_days is None or time.time() - os.path.getmtime(path) < keep_days * 86400):
        with gzip.open(path, "rt") as f:
            return json.load(f)
    val = fetch()
    if val is not None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with gzip.open(path, "wt") as f:
            json.dump(val, f)
    return val


def _score(x):
    if isinstance(x, dict):
        x = x.get("value", x.get("displayValue"))
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def espn_games(key, path):
    """Finished games, one row per game: (iso date, home, away, home score, away score, neutral, home abbr, away abbr)."""
    teams = cached(f"{key}/teams.json.gz", lambda: [
        {"id": t["team"]["id"], "name": t["team"].get("displayName"), "abbr": t["team"].get("abbreviation")}
        for sp in (get(f"{ESPN}/{path}/teams?limit=1000") or {}).get("sports", [])
        for lg in sp.get("leagues", []) for t in lg.get("teams", [])], keep_days=30) or []
    today = date.today()
    this = today.year + (1 if key in ("nba", "nhl", "cbb") and today.month >= 9 else 0)
    games = {}
    for season in range(this - SEASONS + 1, this + 1):
        current = season == this or (season == this - 1 and today.month <= 3)
        for t in teams:
            def fetch(t=t, season=season):
                out = []
                for st in (2, 3):   # regular season, playoffs
                    j = get(f"{ESPN}/{path}/teams/{t['id']}/schedule?season={season}&seasontype={st}")
                    for ev in (j or {}).get("events", []):
                        c = (ev.get("competitions") or [{}])[0]
                        if not ((c.get("status") or {}).get("type") or {}).get("completed"):
                            continue
                        comp = c.get("competitors") or []
                        h = next((x for x in comp if x.get("homeAway") == "home"), None)
                        a = next((x for x in comp if x.get("homeAway") == "away"), None)
                        if not h or not a:
                            continue
                        hs, as_ = _score(h.get("score")), _score(a.get("score"))
                        if hs is None or as_ is None:
                            continue
                        out.append([ev["id"], (ev.get("date") or "")[:10], h["team"].get("displayName"), a["team"].get("displayName"),
                                    hs, as_, bool(c.get("neutralSite")), h["team"].get("abbreviation"), a["team"].get("abbreviation")])
                return out
            rows = cached(f"{key}/{season}/{t['id']}.json.gz", fetch, keep_days=0.5 if current else None) or []
            for r in rows:
                games[r[0]] = r[1:]
    return sorted(games.values())


NFL_WX = {}   # (date, home team) -> (temp F, wind mph, rain mm, outdoors) from nflverse
NFL_MOVED = {"OAK": "LV", "SD": "LAC", "STL": "LA"}   # relocated teams keep their rating
NFL_ESPN = {"LA": "LAR", "WAS": "WSH"}
EPL_NAMES = {"Man United": "Manchester United", "Man City": "Manchester City", "Nott'm Forest": "Nottingham Forest",
             "Wolves": "Wolverhampton Wanderers", "Sheffield United": "Sheffield United", "Spurs": "Tottenham Hotspur",
             "Tottenham": "Tottenham Hotspur", "West Ham": "West Ham United", "Newcastle": "Newcastle United",
             "Brighton": "Brighton & Hove Albion", "Leicester": "Leicester City", "Leeds": "Leeds United",
             "Norwich": "Norwich City", "Luton": "Luton Town", "Ipswich": "Ipswich Town", "Sheffield Weds": "Sheffield Wednesday"}


def nfl_names():
    teams = cached("nfl/teams.json.gz", lambda: [
        {"abbr": t["team"].get("abbreviation"), "name": t["team"].get("displayName")}
        for sp in (get(f"{ESPN}/football/nfl/teams?limit=100") or {}).get("sports", [])
        for lg in sp.get("leagues", []) for t in lg.get("teams", [])], keep_days=30) or []
    return {t["abbr"]: t["name"] for t in teams}


def nfl_games():
    def fetch():
        body = get("https://github.com/nflverse/nfldata/raw/master/data/games.csv", raw=True)
        return body.decode() if body else None
    text = cached("nfl/games.csv.json.gz", fetch, keep_days=0.5)
    rows = []
    names = nfl_names()
    first = date.today().year - SEASONS - 1
    for r in csv.DictReader(io.StringIO(text or "")):
        if not r.get("home_score") or int(r["season"]) < first:
            continue
        book = None
        try:
            mh, ma = float(r["home_moneyline"]), float(r["away_moneyline"])
            ph = 100 / (mh + 100) if mh > 0 else -mh / (-mh + 100)
            pa = 100 / (ma + 100) if ma > 0 else -ma / (-ma + 100)
            book = ph / (ph + pa)
        except (KeyError, ValueError, ZeroDivisionError):
            pass
        h, a = (NFL_MOVED.get(x, x) for x in (r["home_team"], r["away_team"]))
        try:
            outdoor = (r.get("roof") or "") in ("outdoors", "open")
            NFL_WX[(r["gameday"], names.get(NFL_ESPN.get(h, h), h))] = (
                float(r["temp"]) if r.get("temp") else None, float(r["wind"]) if r.get("wind") else None, None, outdoor,
                r.get("gametime") or "13:00", r.get("location") == "Neutral")
        except ValueError:
            pass
        rows.append((r["gameday"], names.get(NFL_ESPN.get(h, h), h), names.get(NFL_ESPN.get(a, a), a), float(r["home_score"]),
                     float(r["away_score"]), r.get("location") == "Neutral", NFL_ESPN.get(h, h), NFL_ESPN.get(a, a), book))
    return sorted(rows)


def fd_games(key):
    """Premier League / MLS from football-data.co.uk, with closing bookmaker odds (home, draw, away probabilities)."""
    urls = []
    if key == "epl":
        y = date.today().year if date.today().month >= 8 else date.today().year - 1
        for s in range(y - SEASONS + 1, y + 1):
            urls.append((s, f"https://www.football-data.co.uk/mmz4281/{s % 100:02d}{(s + 1) % 100:02d}/E0.csv", s == y))
    else:
        urls.append((0, "https://www.football-data.co.uk/new/USA.csv", True))
    rows = []
    for s, u, current in urls:
        def fetch(u=u):
            b = get(u, raw=True)
            return b.decode("utf-8-sig", "replace") if b else None
        text = cached(f"{key}/{s}.csv.json.gz", fetch, keep_days=0.5 if current else None) or ""
        for r in csv.DictReader(io.StringIO(text)):
            try:
                d = datetime.strptime(r["Date"], "%d/%m/%Y" if len(r["Date"]) == 10 else "%d/%m/%y").date()
                h, a = (r.get("HomeTeam") or r.get("Home")), (r.get("AwayTeam") or r.get("Away"))
                hg, ag = float(r.get("FTHG") or r.get("HG")), float(r.get("FTAG") or r.get("AG"))
            except (KeyError, ValueError, TypeError):
                continue
            if key == "mls" and d.year < date.today().year - SEASONS:
                continue
            book = None
            for cols in (("PSCH", "PSCD", "PSCA"), ("AvgCH", "AvgCD", "AvgCA"), ("B365H", "B365D", "B365A")):
                try:
                    inv = [1 / float(r[c]) for c in cols]
                    book = [x / sum(inv) for x in inv]
                    break
                except (KeyError, ValueError, ZeroDivisionError):
                    continue
            h, a = EPL_NAMES.get(h, h), EPL_NAMES.get(a, a)
            rows.append((d.isoformat(), h, a, hg, ag, False, "", "", book))
    return sorted(rows)


def ufc_fights():
    rows = []
    today = date.today()
    for y in range(today.year - SEASONS, today.year + 1):
        for q in range(4):
            a = date(y, 3 * q + 1, 1)
            if a > today:
                break
            b = (date(y + (q == 3), (3 * q + 3) % 12 + 1, 1) - timedelta(days=1))
            current = b >= today - timedelta(days=7)
            def fetch(a=a, b=b):
                j = get(f"{ESPN}/mma/ufc/scoreboard?dates={a:%Y%m%d}-{b:%Y%m%d}&limit=1000")
                out = []
                for ev in (j or {}).get("events", []):
                    for c in ev.get("competitions") or []:
                        comp = c.get("competitors") or []
                        if len(comp) != 2 or not ((c.get("status") or {}).get("type") or {}).get("completed"):
                            continue
                        w = [x for x in comp if x.get("winner")]
                        if len(w) != 1:
                            continue
                        n1, n2 = ((x.get("athlete") or {}).get("displayName") for x in comp)
                        out.append([(c.get("date") or ev.get("date") or "")[:10], n1, n2, 1.0 if comp[0].get("winner") else 0.0,
                                    0.0 if comp[0].get("winner") else 1.0])
                return out
            for r in cached(f"ufc/{a:%Y%m%d}.json.gz", fetch, keep_days=0.5 if current else None) or []:
                rows.append((r[0], r[1], r[2], r[3], r[4], True, r[1], r[2], None))
    return sorted(rows)


def injuries(path):
    j = get(f"{ESPN}/{path}/injuries")
    out = {}
    for team in (j or {}).get("injuries", []):
        name = team.get("displayName") or (team.get("team") or {}).get("displayName")
        lst = []
        for inj in team.get("injuries", [])[:12]:
            ath = inj.get("athlete") or {}
            lst.append({"name": ath.get("displayName"), "pos": (ath.get("position") or {}).get("abbreviation"),
                        "status": inj.get("status"), "note": (inj.get("shortComment") or "")[:140]})
        if name and lst:
            out[name] = lst
    return out


# ------------------------------------------------------------ Elo
def season_of(d, gap_starts):
    return sum(1 for g in gap_starts if d >= g)


def run_elo(games, K, hfa, revert, draws, mov=True, by_year=False):
    """Runs through games in order. Returns per-game pregame home-win probability (and draw prob) and final ratings."""
    R = {}
    preds = []
    last_date = None
    season = 0
    seasons = []
    for g in games:
        d, h, a, hs, as_, neutral = g[:6]
        dd = date.fromisoformat(d)
        # off-season (or, for sports with no off-season like UFC, a new calendar year):
        # pull every rating partway back to average
        if last_date and ((dd.year != last_date.year) if by_year else (dd - last_date).days > 60):
            season += 1
            for t in R:
                R[t] = 1500 + (R[t] - 1500) * (1 - revert)
        last_date = dd
        rh, ra = R.get(h, 1500.0), R.get(a, 1500.0)
        diff = rh - ra + (0 if neutral else hfa)
        e = 1 / (1 + 10 ** (-diff / 400))
        preds.append(e)
        seasons.append(season)
        result = 1.0 if hs > as_ else 0.0 if hs < as_ else 0.5
        if mov and hs != as_:
            margin = abs(hs - as_)
            wdiff = diff if hs > as_ else -diff
            mult = math.log(margin + 1) * 2.2 / (wdiff * 0.001 + 2.2)
        else:
            mult = 1.0
        shift = K * mult * (result - e)
        R[h], R[a] = rh + shift, ra - shift
    return preds, seasons, R


def draw_prob(e, d0):
    return d0 * 4 * e * (1 - e)


def score_games(games, preds, idx, draws, d0=0.0):
    """Accuracy of the favorite, Brier score and log loss over games in idx (two-way: home win vs not)."""
    n = hit = 0
    brier = ll = 0.0
    for i in idx:
        hs, as_ = games[i][3], games[i][4]
        e = preds[i]
        if draws:
            pd = draw_prob(e, d0)
            ph, pa = e - pd / 2, 1 - e - pd / 2
            probs = [ph, pd, pa]
            out = 0 if hs > as_ else 2 if hs < as_ else 1
            brier += sum((p - (k == out)) ** 2 for k, p in enumerate(probs))
            ll -= math.log(max(1e-6, probs[out]))
            hit += max(range(3), key=lambda k: probs[k]) == out
        else:
            if hs == as_:
                continue
            y = 1.0 if hs > as_ else 0.0
            brier += (e - y) ** 2
            ll -= math.log(max(1e-6, e if y else 1 - e))
            hit += (e >= 0.5) == (y == 1.0)
        n += 1
    return {"n": n, "acc": round(hit / n, 4) if n else None, "brier": round(brier / n, 4) if n else None,
            "logloss": round(ll / n, 4) if n else None}


def calibration(probs_outcomes):
    bins = []
    for lo in range(0, 100, 10):
        sel = [(p, y) for p, y in probs_outcomes if lo / 100 <= p < (lo + 10) / 100 or (lo == 90 and p == 1)]
        if sel:
            bins.append({"from": lo, "to": lo + 10, "n": len(sel), "said": round(sum(p for p, _ in sel) / len(sel), 3),
                         "won": round(sum(y for _, y in sel) / len(sel), 3)})
    return bins


def learn_league(key, games):
    name, _, _, draws = LEAGUES[key]
    Ks, Hs, Rs = GRID[key]
    runs = {}
    for K in Ks:
        for H in Hs:
            for Rv in Rs:
                runs[(K, H, Rv)] = run_elo(games, K, H, Rv, draws, mov=key != "ufc", by_year=key == "ufc")
    seasons = runs[next(iter(runs))][1]
    last = max(seasons) if seasons else 0
    tests = [s for s in range(2, last + 1)][-3:]   # score the last 3 seasons (each needs earlier seasons to tune on)
    def draw_rate(idx):
        return sum(1 for i in idx if games[i][3] == games[i][4]) / max(1, len(idx)) if draws else 0.0
    test_idx, chosen, d_test = [], {}, {}
    for s in tests:
        tune = [i for i, x in enumerate(seasons) if 1 <= x < s]
        dt = draw_rate(tune)
        best = min(runs, key=lambda p: score_games(games, runs[p][0], tune, draws, dt / 0.95)["logloss"] or 9)
        chosen[s] = best
        test_idx += [(i, best) for i, x in enumerate(seasons) if x == s]
        d_test[s] = dt
    # out-of-sample predictions: each test game predicted with settings chosen from earlier seasons only
    oos_pred = {i: runs[p][0][i] for i, p in test_idx}
    idx = sorted(oos_pred)
    preds_full = [oos_pred.get(i, 0.5) for i in range(len(games))]
    d_prior = sum(d_test.values()) / max(1, len(d_test))   # draw rate known before the test seasons
    rec = score_games(games, preds_full, idx, draws, d_prior / 0.95)
    pairs = [(preds_full[i], 1.0 if games[i][3] > games[i][4] else 0.0) for i in idx if games[i][3] != games[i][4]]
    rec["calibration"] = calibration(pairs)
    # season by season: solid ratings should hold up in every test season, not just on average
    rec["by_season"] = []
    for sn in tests:
        ix = [i for i in idx if seasons[i] == sn]
        r = score_games(games, preds_full, ix, draws, d_prior / 0.95)
        if r["n"]:
            rec["by_season"].append({"from": games[ix[0]][0], "to": games[ix[-1]][0], "n": r["n"], "acc": r["acc"],
                                     "brier": r["brier"], "base": round(sum(1 for i in ix if games[i][3] > games[i][4]) / len(ix), 3)})
    # the sportsbooks on the very same games, where odds exist
    with_book = [i for i in idx if len(games[i]) > 8 and games[i][8] is not None]
    if with_book:
        bk = [0.0] * len(games)
        for i in with_book:
            b = games[i][8]
            bk[i] = b[0] + (b[1] / 2 if draws else 0) if isinstance(b, list) else b
        if draws:
            n = len(with_book)
            hit = sum(1 for i in with_book if max(range(3), key=lambda k: games[i][8][k]) ==
                      (0 if games[i][3] > games[i][4] else 2 if games[i][3] < games[i][4] else 1))
            br = sum(sum((games[i][8][k] - (k == (0 if games[i][3] > games[i][4] else 2 if games[i][3] < games[i][4] else 1))) ** 2
                         for k in range(3)) for i in with_book) / n
            rec["books"] = {"n": n, "acc": round(hit / n, 4), "brier": round(br, 4)}
        else:
            rec["books"] = score_games(games, bk, with_book, False)
        rec["model_same_games"] = score_games(games, preds_full, with_book, draws, d_prior / 0.95)
    d0 = draw_rate(range(len(games)))   # for today's predictions, every game counts
    best_now = min(runs, key=lambda p: score_games(games, runs[p][0], [i for i, x in enumerate(seasons) if x >= 1], draws, d0 / 0.95)["logloss"] or 9)
    rec["settings"] = {"k": best_now[0], "home": best_now[1], "pullback": best_now[2]}
    rec["tested_seasons"] = len(tests)
    rec["from"] = games[idx[0]][0] if idx else ""
    rec["to"] = games[idx[-1]][0] if idx else ""
    rec["base"] = round(sum(1 for i in idx if games[i][3] > games[i][4]) / max(1, len(idx)), 3)   # how often home won
    preds, seasons_all, ratings = runs[best_now]
    # kept for the matchup-and-weather layer (matchup.py)
    learn_league.last = {"runs": runs, "seasons": seasons, "chosen": chosen, "tests": tests, "d_test": d_test,
                         "best": best_now, "d0": d0}
    return rec, ratings, best_now, d0


def team_table(games, ratings, key):
    """Side-by-side numbers for every team, from this season's games (and recent games for head to head)."""
    if not games:
        return {}
    last_d = date.fromisoformat(games[-1][0])
    start = last_d
    for g in reversed(games):   # this season starts after the last 60-day gap
        d = date.fromisoformat(g[0])
        if (start - d).days > 60:
            break
        start = d
    season_games = [g for g in games if date.fromisoformat(g[0]) >= start]
    stats = {}
    for g in season_games:
        d, h, a, hs, as_ = g[:5]
        for team, opp, f, ag, home in ((h, a, hs, as_, True), (a, h, as_, hs, False)):
            s = stats.setdefault(team, {"w": 0, "l": 0, "t": 0, "pf": 0.0, "pa": 0.0, "hw": 0, "hl": 0, "aw": 0, "al": 0,
                                        "opp": [], "res": []})
            r = "W" if f > ag else "L" if f < ag else "T"
            s["w" if r == "W" else "l" if r == "L" else "t"] += 1
            s["pf"] += f
            s["pa"] += ag
            if r != "T":
                s[("h" if home else "a") + ("w" if r == "W" else "l")] += 1
            s["opp"].append(ratings.get(opp, 1500))
            s["res"].append(r)
    ranked = sorted(ratings, key=lambda t: -ratings[t])
    rank = {t: i + 1 for i, t in enumerate(ranked)}
    abbr = {}
    for g in games[-4000:]:
        abbr[g[1]] = g[6]
        abbr[g[2]] = g[7]
    recent = {}
    for g in reversed(games):
        for team, opp, f, ag, home in ((g[1], g[2], g[3], g[4], True), (g[2], g[1], g[4], g[3], False)):
            lst = recent.setdefault(team, [])
            if len(lst) < 40:
                lst.append([g[0], opp, int(f), int(ag), 1 if home else 0])
    out = {}
    for t, s in stats.items():
        gp = s["w"] + s["l"] + s["t"]
        streak, last = 0, None
        for r in reversed(s["res"]):
            if last is None or r == last:
                streak, last = streak + 1, r
            else:
                break
        l10 = s["res"][-10:]
        out[t] = {"abbr": abbr.get(t, ""), "elo": round(ratings.get(t, 1500)), "rank": rank.get(t), "of": len(ranked),
                  "w": s["w"], "l": s["l"], "t": s["t"], "pf": round(s["pf"] / gp, 1), "pa": round(s["pa"] / gp, 1),
                  "home": f"{s['hw']}-{s['hl']}", "away": f"{s['aw']}-{s['al']}",
                  "l10": f"{l10.count('W')}-{l10.count('L')}" + (f"-{l10.count('T')}" if 'T' in l10 else ""),
                  "streak": f"{last}{streak}" if last else "", "sos": round(sum(s["opp"]) / len(s["opp"])),
                  "games": recent.get(t, [])}
    return out


def trend(games, K, H, Rv, draws, days=30):
    """Each team's rating change over the last `days` days."""
    if not games:
        return {}
    cut = (date.fromisoformat(games[-1][0]) - timedelta(days=days)).isoformat()
    _, _, before = run_elo([g for g in games if g[0] < cut], K, H, Rv, draws)
    _, _, now = run_elo(games, K, H, Rv, draws)
    return {t: round(now[t] - before.get(t, 1500)) for t in now}


# ------------------------------------------------------------ Kalshi's own calibration
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = ["KXNFLGAME", "KXNBAGAME", "KXNHLGAME", "KXMLBGAME", "KXNCAAFGAME", "KXNCAAMBGAME", "KXUFCFIGHT", "KXEPLGAME", "KXMLSGAME"]


def kalshi_calibration():
    """Each night, game markets that ended in the last 30 hours: their price 24 hours earlier (before the game) vs.
    the result. Kept and added to over time, so the picture of Kalshi's bias grows every day."""
    path = os.path.join(CACHE, "kalshi_pairs.json")
    pairs = json.load(open(path)) if os.path.exists(path) else {}
    since = datetime.utcnow() - timedelta(hours=30)
    for s in SERIES:
        cursor = ""
        for _ in range(10):
            j = get(f"{KALSHI}/markets?series_ticker={s}&status=settled&limit=1000&min_close_ts={int(since.timestamp())}"
                    + (f"&cursor={cursor}" if cursor else ""), pace=0.6)
            if not j:
                break
            for m in j.get("markets", []):
                try:
                    p = float(m.get("previous_price_dollars") or 0)
                    closed = datetime.fromisoformat((m.get("close_time") or "").replace("Z", "+00:00")).replace(tzinfo=None)
                except ValueError:
                    continue
                if closed >= since and 0.01 <= p <= 0.99 and m.get("result") in ("yes", "no"):
                    pairs[m["ticker"]] = [p, 1.0 if m["result"] == "yes" else 0.0, closed.date().isoformat()]
            cursor = j.get("cursor") or ""
            if not cursor:
                break
    json.dump(pairs, open(path, "w"))
    vals = list(pairs.values())
    bins = []
    for lo in range(0, 100, 10):
        sel = [(p, y) for p, y, _ in vals if lo / 100 <= p < (lo + 10) / 100]
        if len(sel) >= 20:
            bins.append({"from": lo, "to": lo + 10, "n": len(sel), "price": round(sum(p for p, _ in sel) / len(sel), 3),
                         "won": round(sum(y for _, y in sel) / len(sel), 3)})
    first = min((d for _, _, d in vals), default="")
    return {"n": len(vals), "bins": bins, "since": first,
            "note": "Kalshi sports game markets: the price about a day before each game vs. how often that side won."}


# ------------------------------------------------------------ main
def main():
    os.makedirs(CACHE, exist_ok=True)
    out = {"date": date.today().isoformat(), "leagues": {}}
    for key, (name, src, path, draws) in LEAGUES.items():
        t0 = time.time()
        try:
            games = (nfl_games() if src == "nflverse" else fd_games("epl") if src == "fd" else fd_games("mls") if src == "fd_mls"
                     else ufc_fights() if src == "ufc" else espn_games(key, path))
            games = [g for g in games if g[1] and g[2]]
            if len(games) < 200:
                print(f"{name}: only {len(games)} games found, skipping")
                continue
            rec, ratings, (K, H, Rv), d0 = learn_league(key, games)
            teams = team_table(games, ratings, key) if key != "ufc" else {
                t: {"elo": round(r), "rank": i + 1, "of": len(ratings)} for i, (t, r) in enumerate(sorted(ratings.items(), key=lambda kv: -kv[1]))}
            tr = trend(games, K, H, Rv, draws) if key != "ufc" else {}
            for t in teams:
                teams[t]["trend"] = tr.get(t, 0)
            inj = injuries(path) if key in ("nfl", "nba", "nhl", "mlb") else {}
            up = []
            try:
                import matchup
                margins = [abs(g[3] - g[4]) for g in games if g[3] != g[4]]
                scale = max(1.0, sum(margins) / max(1, len(margins)))
                mrec, mmodel, hist, rts = matchup.evaluate(key, games, scale)
                rec["matchup"] = mrec
                up = matchup.upcoming(key, path, games, mmodel, hist, rts, H, scale, draws, d0)
                if key in matchup.WEATHER_LEAGUES:
                    wx = matchup.league_weather(key, path, games)
                    preds_now = learn_league.last["runs"][learn_league.last["best"]][0]
                    splits = matchup.weather_splits(games, preds_now, wx)
                    for t, rows in splits.items():
                        if t in teams:
                            teams[t]["weather"] = rows
                    rec["weather_effect"] = matchup.league_weather_effect(games, preds_now, wx)
                    print(f"  {name} weather: {sum(1 for w in wx.values() if w[3])} outdoor games with weather; "
                          f"{json.dumps(rec['weather_effect'])}", flush=True)
                t = mrec["tested"]
                print(f"  {name} matchup layer: " + ", ".join(f"{v} logloss {r['logloss']} acc {r['acc']:.1%}" for v, r in t.items())
                      + f" -> using {mrec['used']}; weather helped: {mrec['weather_helped']}; {len(up)} upcoming games", flush=True)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"::warning title={name} matchup layer failed::{e}")
            out["leagues"][key] = {"name": name, "record": rec, "teams": teams, "injuries": inj, "upcoming": up,
                                   "home_edge": H, "draw_rate": round(d0, 3) if draws else None, "games": len(games)}
            b = rec.get("books")
            print(f"{name}: {len(games)} games; tested on {rec['tested_seasons']} seasons ({rec['n']} games): favorite won "
                  f"{(rec['acc'] or 0):.1%}, Brier {rec['brier']}" + (f" | books on same games {b['acc']:.1%}, Brier {b['brier']}; "
                  f"model {rec['model_same_games']['acc']:.1%}" if b else "") + f" | {time.time() - t0:.0f}s"
                  + " | by season: " + ", ".join(f"{x['from'][:4]}: {x['acc']:.1%}" for x in rec["by_season"]), flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"::warning title={name} ratings failed::{e}")
    try:
        out["kalshi"] = kalshi_calibration()
        print("Kalshi calibration:", json.dumps(out["kalshi"]["bins"])[:600])
    except Exception as e:
        print(f"::warning title=Kalshi calibration failed::{e}")
    os.makedirs(OUT, exist_ok=True)
    with gzip.open(os.path.join(OUT, "sports.json.gz"), "wt") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"Wrote sports.json.gz ({os.path.getsize(os.path.join(OUT, 'sports.json.gz')) // 1024} KB)")


if __name__ == "__main__":
    main()

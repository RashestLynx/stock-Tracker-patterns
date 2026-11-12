"""Matchup layer: finds real rest and weather effects, and doesn't invent them when there are none."""
import os, random, sys
from datetime import date, timedelta
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sports, matchup

def league(rest_effect, wind_effect, seed):
    rnd = random.Random(seed)
    strength = {f"T{i}": rnd.gauss(0, 150) for i in range(24)}
    games, wx, d, last = [], {}, date(2018, 9, 1), {}
    for s in range(7):
        for t in strength:
            strength[t] = 0.7 * strength[t] + rnd.gauss(0, 70)
        for wk in range(40):
            names = list(strength); rnd.shuffle(names)
            for h, a in zip(names[::2], names[1::2]):
                gd = d + timedelta(days=rnd.choice([0, 1, 2]))
                rh = min(10, (gd - last.get(h, gd - timedelta(days=7))).days); ra = min(10, (gd - last.get(a, gd - timedelta(days=7))).days)
                wind = rnd.choice([3, 5, 8, 12, 20, 28])
                diff = strength[h] - strength[a] + 40 + rest_effect * (rh - ra)
                diff *= 1 - wind_effect * max(0, wind - 8) / 20   # windy games are more random
                p = 1 / (1 + 10 ** (-diff / 400))
                win = rnd.random() < p
                m = rnd.randint(1, 14)
                games.append((gd.isoformat(), h, a, 20 + m if win else 20, 20 if win else 20 + m, False, h, a, None))
                wx[(gd.isoformat(), h)] = (60, wind, 0, True)
                last[h] = last[a] = gd
            d += timedelta(days=4)
        d += timedelta(days=120)
    return sorted(games), wx

sports.GRID["mlb"] = ([10, 20, 30], [20, 40, 60], [0.25, 0.5])
for label, rest, wind, expect_better in (("real rest and wind effects", 25, 0.9, True), ("no extra effects", 0, 0.0, False)):
    games, wx = league(rest, wind, seed=5 if expect_better else 6)
    sports.NFL_WX.clear()
    matchup.league_weather = lambda key, path, g, wx=wx: wx
    sports.learn_league("mlb", games)
    rec, model, hist, rts = matchup.evaluate("mlb", games, 7.0)
    t = rec["tested"]
    print(label, {v: r["logloss"] for v, r in t.items()}, "used", rec["used"], "weather helped", rec["weather_helped"], rec.get("weights"))
    if expect_better:
        assert rec["used"] != "elo" and t[rec["used"]]["logloss"] < t["elo"]["logloss"], rec
        assert rec["weather_helped"], rec
        assert rec["weights"]["rest"] > 0 and rec["weights"]["wind_x"] < 0, rec["weights"]
    else:
        gain = t["elo"]["logloss"] - min(r["logloss"] for r in t.values())
        assert gain < 0.004, gain   # no meaningful invented edge
print("matchup ok")

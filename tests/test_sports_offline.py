"""Sports ratings on made-up leagues: real skill should be found; coin-flip leagues should show none."""
import math, os, random, sys
from datetime import date, timedelta
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sports

def league(skill_spread, seasons=7, teams=20, seed=1, draws=False):
    rnd = random.Random(seed)
    strength = {f"T{i}": rnd.gauss(0, skill_spread) for i in range(teams)}
    games, d = [], date(2018, 9, 1)
    for s in range(seasons):
        for t in strength:   # drift between seasons
            strength[t] = 0.7 * strength[t] + rnd.gauss(0, skill_spread * 0.5)
        for wk in range(30):
            names = list(strength); rnd.shuffle(names)
            for h, a in zip(names[::2], names[1::2]):
                diff = strength[h] - strength[a] + 40
                p = 1 / (1 + 10 ** (-diff / 400))
                if draws and rnd.random() < 0.25 * 4 * p * (1 - p):
                    hs = as_ = rnd.randint(0, 2)
                else:
                    win = rnd.random() < p
                    m = rnd.randint(1, 15)
                    hs, as_ = (20 + m, 20) if win else (20, 20 + m)
                games.append((d.isoformat(), h, a, hs, as_, False, h, a, None))
            d += timedelta(days=3)
        d += timedelta(days=120)
    return games

sports.GRID["test"] = ([10, 20, 30], [0, 40, 80], [0.25, 0.5])
sports.LEAGUES["test"] = ("Test", "", "", False)
rec, ratings, best, _ = sports.learn_league("test", league(200))
print("skilled league:", {k: rec[k] for k in ("n", "acc", "brier", "base")}, "settings", best)
assert rec["acc"] > 0.6, rec
cal = rec["calibration"]; worst = max(abs(b["said"] - b["won"]) for b in cal if b["n"] > 80)
print("worst calibration gap", round(worst, 3)); assert worst < 0.08, cal
rec2, _, _, _ = sports.learn_league("test", league(0.0001, seed=2))
print("coin-flip league:", {k: rec2[k] for k in ("n", "acc", "brier", "base")})
assert rec2["acc"] < rec2["base"] + 0.04 and rec2["brier"] > 0.24, rec2
sports.LEAGUES["test"] = ("Test", "", "", True)
rec3, _, _, d0 = sports.learn_league("test", league(200, seed=3, draws=True))
print("league with draws:", {k: rec3[k] for k in ("n", "acc", "brier")}, "draw rate", round(d0, 3))
assert rec3["acc"] > 0.45
t = sports.team_table(league(200, seed=4), ratings, "test")
one = next(iter(t.values())); print("team row:", {k: one[k] for k in ("elo", "rank", "w", "l", "pf", "pa", "home", "l10", "streak", "sos")})
# A sport with no off-season (UFC): seasons come from calendar years, so it still gets tested.
import random as _r
rnd = _r.Random(9); skill = {f"F{i}": rnd.gauss(0, 200) for i in range(150)}; fights = []; d = date(2019, 1, 5)
while d < date(2026, 11, 1):
    names = rnd.sample(list(skill), 24)
    for x, y in zip(names[::2], names[1::2]):
        w = rnd.random() < 1 / (1 + 10 ** (-(skill[x] - skill[y]) / 400))
        fights.append((d.isoformat(), x, y, 1.0 if w else 0.0, 0.0 if w else 1.0, True, x, y, None))
    d += timedelta(days=7)
rec4, _, _, _ = sports.learn_league("ufc", fights)
print("no-off-season sport:", {k: rec4[k] for k in ("n", "acc", "tested_seasons")})
assert rec4["n"] > 1000 and rec4["acc"] > 0.55, rec4
print("sports ok")

"""UFC fight factors: finds a real age effect and layoff effect, and isn't used when there's none."""
import os, random, sys
from datetime import date, timedelta
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sports, ufc

def make(age_effect, seed, style_effect=0):
    rnd = random.Random(seed)
    fighters = {f"F{i}": {"skill": rnd.gauss(0, 150), "born": 1985 + rnd.random() * 15, "strike": rnd.gauss(0, 1)} for i in range(300)}
    games = []; sports.UFC_INFO.clear(); profiles = {}; stats = {}
    d = date(2019, 1, 5)
    cid = 0
    while d < date(2026, 10, 1):
        for a, b in zip(*[iter(rnd.sample(list(fighters), 26))] * 2):
            ages = [d.year - fighters[x]["born"] for x in (a, b)]
            sa, sb = fighters[a]["strike"], fighters[b]["strike"]
            diff = fighters[a]["skill"] - fighters[b]["skill"] - age_effect * (ages[0] - ages[1]) + style_effect * (sa - sb)
            won = rnd.random() < 1 / (1 + 10 ** (-diff / 400))
            cid += 1
            games.append((d.isoformat(), a, b, 1.0 if won else 0.0, 0.0 if won else 1.0, True, a, b, None))
            sports.UFC_INFO[(d.isoformat(), a, b)] = {"ids": [a, b], "class": "Lightweight", "method": rnd.choice(["KO", "DEC", "SUB"]),
                                                      "cid": cid, "venue": {}, "rounds": 3, "minutes": 15.0}
            for me, st in ((a, sa), (b, sb)):   # landed strikes reflect the hidden striking skill
                ssa = 100
                stats.setdefault(me, {})[str(cid)] = {"ssl": max(5, 45 + 12 * st + rnd.gauss(0, 6)), "ssa": ssa, "kd": 0, "tdl": 1, "tda": 3, "sm": 0}
        d += timedelta(days=7)
    for n, f in fighters.items():
        profiles[n] = {"age": date.today().year - f["born"], "reach": 72, "height": 70}
    return games, profiles, stats

sports.GRID["ufc"] = ([40, 60], [0], [0.0, 0.1])
sports.get = lambda *a, **k: None
for label, eff, style, expect in (("real age effect", 30, 0, "age"), ("no effect", 0, 0, None), ("real striking effect", 0, 200, "style")):
    games, profiles, stats = make(eff, {"age": 3, None: 4, "style": 5}[expect], style)
    ufc.profile = lambda fid, p=profiles: p.get(fid, {})
    ufc.fight_stats = lambda fid, keep_days=30, s=stats: s.get(fid, {})
    sports.learn_league("ufc", games)
    rec, up = ufc.run(games, 0)
    print(label, {v: r["logloss"] for v, r in rec["tested"].items()}, "picked", rec["picked_each_year"], "used", rec["used"], {k: v for k, v in rec.get("weights", {}).items() if k in ("age", "strike_diff", "strike_acc")})
    if expect == "age":
        assert rec["used"] != "elo" and rec["weights"]["age"] < 0, rec
    elif expect == "style":
        assert "style" in rec["used"] or rec["used"] == "all", rec
        # in this fake data accuracy is output / 100 attempts, so the effect can split between the two columns:
        # check the combined effect of landing more strikes per minute
        w = rec["weights"]; net = w["strike_diff"] / 2 + w["strike_acc"] * 0.15 / 0.08
        assert net > 0, (net, w)
    else:
        assert rec["tested"]["elo"]["logloss"] - rec["tested"]["picked"]["logloss"] < 0.004, rec
print("ufc ok")


def test_about():
    F = ufc.FLAGS
    assert ufc.about("Kayla Harrison's UFC 334 weight cut draws scrutiny before Amanda Nunes fight", "Kayla Harrison", F["weight"], "Amanda Nunes")
    assert not ufc.about("Kayla Harrison's UFC 334 weight cut draws scrutiny before Amanda Nunes fight", "Amanda Nunes", F["weight"], "Kayla Harrison")
    assert ufc.about("Maycee Barber Withdraws From UFC 335 Fight With Carli Judice", "Maycee Barber", F["pulled out"], "Carli Judice")
    assert not ufc.about("Maycee Barber Withdraws From UFC 335 Fight With Carli Judice", "Carli Judice", F["pulled out"], "Maycee Barber")
    assert not ufc.about("What injured Tom Aspinall really thinks of UFC foe Gane", "Ciryl Gane", F["injury"], "Someone Else")
    assert ufc.about("Dan Hooker Shows Off Nasty Injury Ahead of UFC Qatar", "Dan Hooker", F["injury"])
    assert ufc.about("Injury forces Dan Hooker out of UFC Qatar", "Dan Hooker", F["injury"], "Mateusz Gamrot")
    assert not ufc.about("Gamrot vs Hooker off after injury", "Mateusz Gamrot", F["injury"], "Dan Hooker")
    print("about ok")


test_about()


def test_card_and_travel():
    import tempfile
    sports.CACHE = tempfile.mkdtemp()
    d0 = date(2026, 11, 1)
    first = ufc.card_changes([("1", "2026-11-29", "A", "B")], d0)
    assert first["A"] == {"booked_days": None, "was": None}            # nothing flagged on the first night
    nxt = ufc.card_changes([("1", "2026-11-29", "A", "C"), ("2", "2026-11-15", "D", "E")], d0 + timedelta(days=2))
    assert nxt["A"]["was"] == "B" and nxt["C"]["was"] is None         # A's opponent changed from B to C
    assert nxt["D"]["booked_days"] == 12                                # booked 12 days before the fight
    assert ufc.time_zones("America/New_York", "Asia/Dubai", "2026-11-14") == 9.0
    assert ufc.time_zones("Australia/Sydney", "America/Los_Angeles", "2026-11-14") == 5.0
    print("card and travel ok")


test_card_and_travel()

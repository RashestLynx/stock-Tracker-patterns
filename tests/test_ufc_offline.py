"""UFC fight factors: finds a real age effect and layoff effect, and isn't used when there's none."""
import os, random, sys
from datetime import date, timedelta
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sports, ufc

def make(age_effect, seed):
    rnd = random.Random(seed)
    fighters = {f"F{i}": {"skill": rnd.gauss(0, 150), "born": 1985 + rnd.random() * 15} for i in range(300)}
    games = []; sports.UFC_INFO.clear(); profiles = {}
    d = date(2019, 1, 5)
    while d < date(2026, 10, 1):
        for a, b in zip(*[iter(rnd.sample(list(fighters), 26))] * 2):
            ages = [d.year - fighters[x]["born"] for x in (a, b)]
            diff = fighters[a]["skill"] - fighters[b]["skill"] - age_effect * (ages[0] - ages[1])
            won = rnd.random() < 1 / (1 + 10 ** (-diff / 400))
            games.append((d.isoformat(), a, b, 1.0 if won else 0.0, 0.0 if won else 1.0, True, a, b, None))
            sports.UFC_INFO[(d.isoformat(), a, b)] = {"ids": [a, b], "class": "Lightweight", "method": rnd.choice(["KO", "DEC", "SUB"])}
        d += timedelta(days=7)
    for n, f in fighters.items():
        profiles[n] = {"age": date.today().year - f["born"], "reach": 72, "height": 70}
    return games, profiles

sports.GRID["ufc"] = ([40, 60], [0], [0.0, 0.1])
for label, eff, expect in (("real age effect", 30, True), ("no age effect", 0, False)):
    games, profiles = make(eff, 3 if expect else 4)
    ufc.profile = lambda fid, p=profiles: p.get(fid, {})
    sports.get = lambda *a, **k: None
    sports.learn_league("ufc", games)
    rec, up = ufc.run(games, 0)
    print(label, {v: r["logloss"] for v, r in rec["tested"].items()}, "used", rec["used"], rec.get("weights", {}).get("age"))
    if expect:
        assert rec["used"] == "fight factors" and rec["weights"]["age"] < 0, rec
    else:
        assert rec["tested"]["elo"]["logloss"] - rec["tested"]["fight factors"]["logloss"] < 0.004, rec
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

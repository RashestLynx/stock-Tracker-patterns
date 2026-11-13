"""
UFC fight factors on top of the fighter ratings, tested on past fights the model never saw.

Measured for every past fight, using only what was known before it:
  - the rating chance (as log-odds)
  - age gap (older fighters decline), reach gap, height gap
  - layoff: days since each fighter's last fight (ring rust)
  - coming off a knockout loss (chin damage)
  - changing weight class (moving up or down) since the last fight
  - experience: number of UFC fights so far; current win or loss streak

A logistic regression learns how much each matters. For each test year it's fit only on earlier years and scored on
that year; the app uses it only if it beats the plain ratings on those unseen years.

Recent news can't be tested on the past (old news isn't searchable the same way), so it's shown separately as flags:
headlines from the last three weeks about injuries, pull-outs, missed weight or hard weight cuts, from Google News.
"""
import html
import math
import re
import urllib.parse
from datetime import date, datetime, timedelta

import sys

_main = sys.modules.get("__main__")
if _main is not None and hasattr(_main, "learn_league") and hasattr(_main, "UFC_INFO"):
    sports = _main
else:
    import sports

import matchup

NAMES = ["elo", "age", "reach", "height", "layoff_a", "layoff_b", "ko_loss", "class_change", "experience", "streak"]
CLASSES = ["W Strawweight", "W Flyweight", "W Bantamweight", "W Featherweight", "Flyweight", "Bantamweight", "Featherweight",
           "Lightweight", "Welterweight", "Middleweight", "Light Heavyweight", "Heavyweight"]
FLAGS = {
    "injury": r"injur|hurt|surgery|torn|broke|fracture|concussion|illness|sick|hospital",
    "pulled out": r"pull(ed|s)? out|withdr[ae]w|out of (the )?fight|replac(ed|ement)|short[- ]notice",
    "weight": r"miss(ed|es)? weight|weight[- ]cut|off weight|over the (weight )?limit|catchweight|failed to make weight|dehydrat",
}


def profile(fid):
    """Age, reach and height (inches) and stance for one fighter, from ESPN (cached a month)."""
    def fetch():
        j = sports.get(f"https://site.web.api.espn.com/apis/common/v3/sports/mma/athletes/{fid}") or {}
        a = j.get("athlete") or {}
        def inches(s):
            m = re.match(r"(\d+)' (\d+)", s or "")
            if m:
                return int(m.group(1)) * 12 + int(m.group(2))
            m = re.match(r"([\d.]+)", s or "")
            return float(m.group(1)) if m else None
        return {"age": a.get("age"), "reach": inches(a.get("displayReach")), "height": inches(a.get("displayHeight")),
                "stance": (a.get("stance") or {}).get("text"), "name": a.get("displayName"),
                "class": (a.get("weightClass") or {}).get("shortName")}
    return sports.cached(f"ufc2/fighters/{fid}.json.gz", fetch, keep_days=30) or {}


class Career:
    """Each fighter's past fights, for factors known before the next fight."""

    def __init__(self):
        self.f = {}   # name -> list of (date, won, method, class)

    def feats(self, d, a, b, pa, pb, wc, today_age):
        def side(n, prof):
            h = self.f.get(n, [])
            dd = date.fromisoformat(d)
            layoff = min(900, (dd - date.fromisoformat(h[-1][0])).days) if h else 365
            ko = 1.0 if h and not h[-1][1] and h[-1][2] == "KO" else 0.0
            changed = 0.0
            if h and h[-1][3] in CLASSES and wc in CLASSES:
                changed = float(CLASSES.index(wc) - CLASSES.index(h[-1][3]))   # + moving up, - moving down
            streak = 0
            for x in reversed(h):
                if streak == 0:
                    streak = 1 if x[1] else -1
                elif (streak > 0) == x[1]:
                    streak += 1 if x[1] else -1
                else:
                    break
            age = None
            if prof.get("age") is not None:
                age = prof["age"] - (date.today() - dd).days / 365.25   # age at the time of the fight
            return layoff, ko, changed, len(h), streak, age
        la, ka, ca, ea, sa, aa = side(a, pa)
        lb, kb, cb, eb, sb, ab = side(b, pb)
        def gap(x, y, scale):
            return (x - y) / scale if x is not None and y is not None else 0.0
        return [gap(aa, ab, 5), gap(pa.get("reach"), pb.get("reach"), 3), gap(pa.get("height"), pb.get("height"), 3),
                min(la, 900) / 365 - 1, min(lb, 900) / 365 - 1, ka - kb, abs(ca) - abs(cb),
                math.log1p(ea) - math.log1p(eb), (sa - sb) / 3]

    def add(self, d, a, b, a_won, method, wc):
        self.f.setdefault(a, []).append((d, a_won, method, wc))
        self.f.setdefault(b, []).append((d, not a_won, method, wc))


def build(games, preds, profiles):
    X, car = [], Career()
    for i, g in enumerate(games):
        d, a, b = g[0], g[1], g[2]
        info = sports.UFC_INFO.get((d, a, b), {})
        ids = info.get("ids") or [None, None]
        pa, pb = profiles.get(ids[0], {}), profiles.get(ids[1], {})
        X.append([matchup.logit(preds[i])] + car.feats(d, a, b, pa, pb, info.get("class", ""), None))
        car.add(d, a, b, g[3] > g[4], info.get("method", ""), info.get("class", ""))
    return X, car


def about(title, name, kind_pat, opponent=""):
    """True if the flag in a headline is about this fighter, not someone else named in it.
    The fighter must be named before the flag word with no other fighter in between
    ("Harrison's weight cut ... before Nunes fight" flags Harrison, not Nunes), or right after it
    when no opponent is named ("Injury forces Hooker out")."""
    t = title.lower()
    m = re.search(kind_pat, t, re.I)
    if not m:
        return False
    last = name.split()[-1].lower()
    pos = [x.start() for x in re.finditer(r"\b" + re.escape(last) + r"\b", t)]
    if not pos:
        return False
    opp = opponent.split()[-1].lower() if opponent else ""
    before = [p for p in pos if p < m.start()]
    if before:
        gap = t[before[-1] + len(last):m.start()]
        # another fighter or "foe/opponent/vs" between the name and the flag means the flag is someone else's
        return not ((opp and opp in gap) or re.search(r"\b(vs\.?|foe|opponent|rival|against)\b", gap))
    return not (opp and opp in t) and len(t[m.end():pos[0]].split()) <= 3


def news_flags(name, opponent=""):
    """Headlines from the last 3 weeks about injuries, pull-outs or weight trouble (Google News)."""
    q = f'"{name}" (injury OR injured OR "pulled out" OR withdraws OR "missed weight" OR "weight cut" OR hospital OR replacement)'
    raw = sports.get("https://news.google.com/rss/search?q=" + urllib.parse.quote(q) + "&hl=en-US&gl=US&ceid=US:en", raw=True)
    if not raw:
        return []
    text = raw.decode("utf-8", "replace")
    out = []
    cutoff = datetime.utcnow() - timedelta(days=21)
    last = name.split()[-1].lower()
    for item in re.findall(r"<item>(.*?)</item>", text, re.S)[:25]:
        title = html.unescape(re.sub(r"<.*?>", "", (re.search(r"<title>(.*?)</title>", item, re.S) or [None, ""])[1]))
        link = (re.search(r"<link>(.*?)</link>", item, re.S) or [None, ""])[1].strip()
        when = (re.search(r"<pubDate>(.*?)</pubDate>", item, re.S) or [None, ""])[1]
        try:
            dt = datetime.strptime(when.strip()[:25], "%a, %d %b %Y %H:%M:%S")
        except ValueError:
            continue
        if dt < cutoff or last not in title.lower():
            continue
        kinds = [k for k, pat in FLAGS.items() if about(title, name, pat, opponent)]
        if kinds:
            out.append({"title": title[:200], "link": link, "date": dt.date().isoformat(), "kinds": kinds})
    return out[:5]


def run(games, home_edge):
    L = sports.learn_league.last
    # fighter profiles (first run fetches each fighter once; cached a month)
    ids = {i for info in sports.UFC_INFO.values() for i in info.get("ids") or [] if i}
    profiles = {i: profile(i) for i in ids}
    seasons = L["seasons"]
    y = [1.0 if g[3] > g[4] else 0.0 for g in games]
    variants = {"elo": [0], "fight factors": list(range(len(NAMES)))}
    scored = {v: [0.5] * len(games) for v in variants}
    test_idx = []
    for s in L["tests"]:
        preds = L["runs"][L["chosen"][s]][0]
        X, _ = build(games, preds, profiles)
        tr = [i for i, x in enumerate(seasons) if 1 <= x < s]
        te = [i for i, x in enumerate(seasons) if x == s]
        test_idx += te
        for v, cols in variants.items():
            w = matchup.fit([X[i] for i in tr], [y[i] for i in tr], cols)
            for i in te:
                scored[v][i] = matchup.predict(w, X[i], cols)
    res = {v: sports.score_games(games, scored[v], test_idx, False) for v in variants}
    used = "fight factors" if res["fight factors"]["logloss"] < res["elo"]["logloss"] - 0.0005 else "elo"
    rec = {"tested": res, "used": used}
    preds, _, ratings = L["runs"][L["best"]]
    X, car = build(games, preds, profiles)
    tr = [i for i, x in enumerate(seasons) if x >= 1]
    model = None
    if used != "elo":
        cols = variants[used]
        w = matchup.fit([X[i] for i in tr], [y[i] for i in tr], cols)
        model = (w, cols)
        rec["weights"] = {NAMES[c]: round(float(w[k + 1]), 3) for k, c in enumerate(cols)}

    # upcoming fights in the next 5 weeks
    today = date.today()
    j = sports.get(f"{sports.ESPN}/mma/ufc/scoreboard?dates={today:%Y%m%d}-{today + timedelta(days=35):%Y%m%d}&limit=1000") or {}
    up = []
    for ev in j.get("events", []):
        for c in ev.get("competitions") or []:
            comp = c.get("competitors") or []
            if len(comp) != 2 or ((c.get("status") or {}).get("type") or {}).get("completed"):
                continue
            a, b = ((x.get("athlete") or {}).get("displayName") for x in comp)
            if not a or not b:
                continue
            fa, fb = comp[0].get("id"), comp[1].get("id")
            pa, pb = profile(fa) if fa else {}, profile(fb) if fb else {}
            ra, rb = ratings.get(a, 1500), ratings.get(b, 1500)
            e = 1 / (1 + 10 ** (-(ra - rb) / 400))
            d = (c.get("date") or ev.get("date") or "")[:10]
            wc = (c.get("type") or {}).get("abbreviation") or ""
            f = car.feats(d, a, b, pa, pb, wc, None)
            row = {"date": d, "home": a, "away": b, "venue": ((c.get("venue") or {}).get("fullName") or ""), "indoor": True,
                   "elo": round(e, 4), "class": wc, "fighters": {}}
            if model:
                row["full"] = round(matchup.predict(model[0], [matchup.logit(e)] + f, model[1]), 4)
            for n, prof in ((a, pa), (b, pb)):
                h = car.f.get(n, [])
                last = h[-1] if h else None
                row["fighters"][n] = {
                    "age": prof.get("age"), "reach": prof.get("reach"), "height": prof.get("height"), "stance": prof.get("stance"),
                    "ufc_fights": len(h), "record": f"{sum(1 for x in h if x[1])}-{sum(1 for x in h if not x[1])}",
                    "layoff_days": (date.fromisoformat(d) - date.fromisoformat(last[0])).days if last else None,
                    "last": (("Won" if last[1] else "Lost") + (f" by {last[2]}" if last[2] else "") + f" ({last[0]})") if last else "",
                    "class_change": (CLASSES.index(wc) - CLASSES.index(last[3])) if last and last[3] in CLASSES and wc in CLASSES else 0,
                    "news": news_flags(n, b if n == a else a),
                }
            up.append(row)
    return rec, up

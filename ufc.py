"""
UFC fight factors on top of the fighter ratings, tested on past fights the model never saw.

Measured for every past fight, using only what was known before it:
  base      - the rating chance; age, reach and height gaps; layoff (ring rust); coming off a knockout loss;
              moving weight class; UFC experience; win/loss streak
  style     - from each fighter's past fight stats on ESPN: striking differential (landed minus absorbed per minute),
              striking accuracy and defence, takedown threat against the opponent's takedown defence,
              submission attempts, knockdowns
  damage    - career knockout losses and knockdowns suffered (a worn-down chin)
  context   - fighting in your home country, time zones travelled, altitude (Mexico City, Salt Lake City...),
              5-round fights (cardio, age), the small UFC Apex cage, fights in the last two years
  physical  - southpaw vs orthodox, height relative to the weight class (a proxy for a hard weight cut)

A logistic regression learns how much each matters. Groups of factors are compared walk-forward: each test year
is fit only on earlier years, and the group used for a year is picked on the year before it. The app uses the
factors only if that honest, picked-in-advance version beats the plain ratings.

Things that can't be tested on past fights are shown as flags, not put in the percentage:
  - headlines from the last three weeks (Google News, plus Reddit's r/MMA where reachable) about injuries,
    illness, pull-outs, missed weight or hard cuts, visa trouble, suspensions, camp changes
  - short-notice bookings and opponent changes, from the fight card checked every night
"""
import gzip
import html
import json
import math
import os
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

CLASSES = ["W Strawweight", "W Flyweight", "W Bantamweight", "W Featherweight", "Flyweight", "Bantamweight", "Featherweight",
           "Lightweight", "Welterweight", "Middleweight", "Light Heavyweight", "Heavyweight"]

# (name, group, what it means in plain words, unit) -- column 0 is the rating
FACTORS = [
    ("elo", "base", "rating", ""),
    ("age", "base", "younger", "years"),
    ("reach", "base", "longer reach", "in"),
    ("height", "base", "taller", "in"),
    ("layoff_a", "base", "ring rust", ""),
    ("layoff_b", "base", "ring rust", ""),
    ("ko_loss", "base", "coming off a knockout loss", ""),
    ("class_change", "base", "changing weight class", ""),
    ("experience", "base", "more UFC experience", ""),
    ("streak", "base", "on a better streak", ""),
    ("strike_diff", "style", "out-strikes opponents (landed minus absorbed per minute)", ""),
    ("strike_acc", "style", "more accurate striking", ""),
    ("strike_def", "style", "harder to hit", ""),
    ("grapple", "style", "takedowns vs. the opponent's takedown defense", ""),
    ("subs", "style", "more submission attempts", ""),
    ("kd", "style", "more knockdowns", ""),
    ("ko_losses", "damage", "fewer career knockout losses", ""),
    ("kd_taken", "damage", "knocked down less often", ""),
    ("home", "context", "fighting in their home country", ""),
    ("travel", "context", "less travel across time zones", ""),
    ("altitude", "context", "home edge at altitude", ""),
    ("five_age", "context", "younger in a 5-round fight", ""),
    ("five_pace", "context", "out-strikes opponents in a 5-round fight", ""),
    ("apex", "context", "out-strikes opponents in the small Apex cage", ""),
    ("active", "context", "more active (fights in the last 2 years)", ""),
    ("southpaw", "physical", "southpaw against an orthodox fighter", ""),
    ("size", "physical", "bigger for the weight class", ""),
]
NAMES = [f[0] for f in FACTORS]
GROUPS = {g: [i for i, f in enumerate(FACTORS) if f[1] == g] for g in ("base", "style", "damage", "context", "physical")}
VARIANTS = {
    "elo": [0],
    "base": GROUPS["base"],
    "base+style": GROUPS["base"] + GROUPS["style"],
    "base+style+damage": GROUPS["base"] + GROUPS["style"] + GROUPS["damage"],
    "base+context": GROUPS["base"] + GROUPS["context"],
    "all": list(range(len(FACTORS))),
}

FLAGS = {
    "injury": r"injur|hurt|surgery|torn|broke|fracture|concussion|hospital",
    "illness": r"\bill\b|illness|sick|flu\b|staph|infection|covid",
    "pulled out": r"pull(ed|s)? out|withdr[ae]w|out of (the )?fight|replac(ed|ement)|short[- ]notice|scrapped|cancel",
    "weight": r"miss(ed|es)? weight|weight[- ]cut|off weight|over the (weight )?limit|catchweight|failed to make weight|dehydrat|"
              r"pounds? (over|heavy)|weigh[- ]ins?",
    "visa": r"\bvisa\b|travel (issue|problem)|immigration",
    "suspension": r"suspen|usada|anti-doping|drug test|failed (a )?test|banned",
    "camp": r"parts? ways|split(s)? (with|from)|leaves? .{0,20}(gym|team|camp)|new (gym|camp|coach)|switch(es|ed)? (gym|camp)s?",
}


# ------------------------------------------------------------ data
def profile(fid):
    """Age, reach and height (inches), stance, country and gym for one fighter, from ESPN (cached a month)."""
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
                "class": (a.get("weightClass") or {}).get("shortName"),
                "country": a.get("citizenship") or (a.get("flag") or {}).get("alt") or "",
                "gym": (a.get("association") or {}).get("name") or ""}
    return sports.cached(f"ufc3/fighters/{fid}.json.gz", fetch, keep_days=30) or {}


def _num(x):
    try:
        return float(str(x).split("/")[0].rstrip("%"))
    except (TypeError, ValueError):
        return 0.0


def fight_stats(fid, keep_days=30):
    """{fight id: {ssl, ssa, kd, tdl, tda, sm}} for every UFC fight of one fighter, from ESPN (cached)."""
    def fetch():
        j = sports.get(f"https://site.web.api.espn.com/apis/common/v3/sports/mma/athletes/{fid}/stats") or {}
        out = {}
        for cat in j.get("categories") or []:
            labels = [str(l).strip() for l in cat.get("labels") or []]
            for row in cat.get("statistics") or []:
                m = re.search(r"~c:(\d+)", row.get("uid") or "")
                if not m:
                    continue
                vals = dict(zip(labels, row.get("stats") or []))
                d = out.setdefault(m.group(1), {})
                for k, name in (("SSL", "ssl"), ("SSA", "ssa"), ("KD", "kd"), ("TDL", "tdl"), ("TDA", "tda"), ("SM", "sm")):
                    if k in vals:
                        d[name] = _num(vals[k])
        return out
    return sports.cached(f"ufc3/stats/{fid}.json.gz", fetch, keep_days=keep_days) or {}


_PLACES = {}


def place(q):
    """{lat, lon, elev (m), tz, cc} for a city or country name, via Open-Meteo's free geocoder (cached)."""
    if not q:
        return None
    path = os.path.join(sports.CACHE, "ufc3/places.json.gz")
    if not _PLACES and os.path.exists(path):
        with gzip.open(path, "rt") as f:
            _PLACES.update(json.load(f))
    if q in _PLACES:
        return _PLACES[q]
    r = ((sports.get("https://geocoding-api.open-meteo.com/v1/search?count=1&name=" + urllib.parse.quote(q)) or {})
         .get("results") or [None])[0]
    val = {"lat": r.get("latitude"), "lon": r.get("longitude"), "elev": r.get("elevation") or 0, "tz": r.get("timezone"),
           "cc": r.get("country_code")} if r else None
    _PLACES[q] = val
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt") as f:
        json.dump(_PLACES, f)
    return val


COUNTRY_ALIASES = {"USA": "United States", "US": "United States", "UK": "United Kingdom", "England": "United Kingdom",
                   "Scotland": "United Kingdom", "Wales": "United Kingdom", "Northern Ireland": "United Kingdom",
                   "UAE": "United Arab Emirates", "Korea": "South Korea", "Republic of Korea": "South Korea"}


def country_of(name):
    p = place(COUNTRY_ALIASES.get(name, name)) if name else None
    return (p or {}).get("cc")


def venue_place(v):
    """Where a fight was held: city first (state for the US), else the country."""
    if not v:
        return None
    for q in (f"{v.get('city')}" if v.get("city") else None, v.get("country")):
        p = place(q)
        if p:
            return p
    return None


def utc_offset(tz, d):
    try:
        from zoneinfo import ZoneInfo
        return datetime.fromisoformat(d + "T20:00:00").replace(tzinfo=ZoneInfo(tz)).utcoffset().total_seconds() / 3600
    except Exception:
        return None


def time_zones(home_tz, event_tz, d, home_lon=None, event_lon=None):
    """Hours of time difference between home and the fight. Uses time-zone names when both are known,
    else longitude (15 degrees = 1 hour), since the geocoder often has no time zone for a whole country."""
    a, b = utc_offset(home_tz, d) if home_tz else None, utc_offset(event_tz, d) if event_tz else None
    if (a is None or b is None) and home_lon is not None and event_lon is not None:
        x = abs(home_lon - event_lon) / 15 % 24
        return round(min(x, 24 - x), 1)
    if a is None or b is None:
        return 0.0
    x = abs(a - b) % 24
    return min(x, 24 - x)


# ------------------------------------------------------------ factors
class Career:
    """Each fighter's past fights, for factors known before the next fight."""

    def __init__(self):
        self.f = {}   # name -> list of (date, won, method, class)
        self.s = {}   # name -> running totals of fight stats

    def totals(self, n):
        return self.s.get(n) or {"min": 0.0, "ssl": 0.0, "ssa": 0.0, "ossl": 0.0, "ossa": 0.0, "kd": 0.0, "okd": 0.0,
                                 "tdl": 0.0, "tda": 0.0, "otdl": 0.0, "otda": 0.0, "sm": 0.0, "n": 0}

    def style(self, n):
        """Per-fighter rates, pulled toward typical UFC values when there are few fights to go on."""
        t = self.totals(n)
        m = t["min"] + 15.0
        return {
            "slpm": (t["ssl"] + 15 * 4.0) / m,
            "sapm": (t["ossl"] + 15 * 4.0) / m,
            "acc": (t["ssl"] + 40 * 0.45) / (t["ssa"] + 40),
            "def": 1 - (t["ossl"] + 40 * 0.45) / (t["ossa"] + 40),
            "td15": (t["tdl"] + 1.3) * 15 / m,
            "tddef": 1 - (t["otdl"] + 3 * 0.35) / (t["otda"] + 3),
            "sub15": (t["sm"] + 0.5) * 15 / m,
            "kd15": (t["kd"] + 0.25) * 15 / m,
            "kdt15": (t["okd"] + 0.25) * 15 / m,
            "n": t["n"],
        }

    def side(self, n, prof, d, wc):
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
        ko_losses = sum(1 for x in h if not x[1] and x[2] == "KO")
        active = sum(1 for x in h if (dd - date.fromisoformat(x[0])).days <= 730)
        return {"layoff": layoff, "ko": ko, "changed": changed, "exp": len(h), "streak": streak, "age": age,
                "ko_losses": ko_losses, "active": active, **self.style(n)}

    def feats(self, d, a, b, pa, pb, wc, ctx=None):
        """Factor values for a fight (a's side minus b's). ctx: {rounds, venue place, apex, class heights}."""
        ctx = ctx or {}
        A, B = self.side(a, pa, d, wc), self.side(b, pb, d, wc)
        def gap(x, y, scale):
            return (x - y) / scale if x is not None and y is not None else 0.0
        sd_a, sd_b = A["slpm"] - A["sapm"], B["slpm"] - B["sapm"]
        grapple = A["td15"] * (1 - B["tddef"]) - B["td15"] * (1 - A["tddef"])
        ev = ctx.get("place") or {}
        home_a = 1.0 if ev.get("cc") and ctx.get("cc_a") == ev.get("cc") else 0.0
        home_b = 1.0 if ev.get("cc") and ctx.get("cc_b") == ev.get("cc") else 0.0
        trav = gap(time_zones(ctx.get("tz_a"), ev.get("tz"), d, ctx.get("lon_a"), ev.get("lon")),
                   time_zones(ctx.get("tz_b"), ev.get("tz"), d, ctx.get("lon_b"), ev.get("lon")), 6) \
            if ev and (ctx.get("tz_a") or ctx.get("lon_a") is not None) and (ctx.get("tz_b") or ctx.get("lon_b") is not None) else 0.0
        alt = min(3.0, (ev.get("elev") or 0) / 1000)
        five = 1.0 if ctx.get("rounds") == 5 else 0.0
        st_a, st_b = (pa.get("stance") or "").lower(), (pb.get("stance") or "").lower()
        south = (1.0 if st_a == "southpaw" and st_b == "orthodox" else 0.0) - (1.0 if st_b == "southpaw" and st_a == "orthodox" else 0.0)
        med = (ctx.get("class_height") or {}).get(wc)
        size = gap((pa.get("height") or med) - med, (pb.get("height") or med) - med, 3) if med else 0.0
        age_gap = gap(A["age"], B["age"], 5)
        return [
            age_gap, gap(pa.get("reach"), pb.get("reach"), 3), gap(pa.get("height"), pb.get("height"), 3),
            A["layoff"] / 365 - 1, B["layoff"] / 365 - 1, A["ko"] - B["ko"], abs(A["changed"]) - abs(B["changed"]),
            math.log1p(A["exp"]) - math.log1p(B["exp"]), (A["streak"] - B["streak"]) / 3,
            # style
            (sd_a - sd_b) / 2, (A["acc"] - B["acc"]) / 0.08, (A["def"] - B["def"]) / 0.08, grapple / 0.5,
            (A["sub15"] - B["sub15"]) / 0.5, (A["kd15"] - B["kd15"]) / 0.3,
            # damage
            (math.log1p(A["ko_losses"]) - math.log1p(B["ko_losses"])), (A["kdt15"] - B["kdt15"]) / 0.3,
            # context
            home_a - home_b, -trav, alt * (home_a - home_b), five * age_gap, five * (sd_a - sd_b) / 2,
            (1.0 if ctx.get("apex") else 0.0) * (sd_a - sd_b) / 2, (A["active"] - B["active"]) / 2,
            # physical
            south, size,
        ]

    def add(self, d, a, b, a_won, method, wc, info=None, sa=None, sb=None):
        self.f.setdefault(a, []).append((d, a_won, method, wc))
        self.f.setdefault(b, []).append((d, not a_won, method, wc))
        mins = (info or {}).get("minutes")
        if not mins or sa is None or sb is None:
            return
        for n, me, op in ((a, sa, sb), (b, sb, sa)):
            t = self.s.setdefault(n, self.totals(n))
            t["min"] += mins
            t["n"] += 1
            for k, v in (("ssl", me.get("ssl")), ("ssa", me.get("ssa")), ("ossl", op.get("ssl")), ("ossa", op.get("ssa")),
                         ("kd", me.get("kd")), ("okd", op.get("kd")), ("tdl", me.get("tdl")), ("tda", me.get("tda")),
                         ("otdl", op.get("tdl")), ("otda", op.get("tda")), ("sm", me.get("sm"))):
                t[k] += v or 0.0


def class_heights(profiles):
    """Typical (median) height in each weight class, from the fighters' listed class."""
    by = {}
    for p in profiles.values():
        if p.get("class") in CLASSES and p.get("height"):
            by.setdefault(p["class"], []).append(p["height"])
    return {c: sorted(v)[len(v) // 2] for c, v in by.items() if len(v) >= 5}


def context(info, pa, pb, heights):
    v = info.get("venue") or {}
    pa_home, pb_home = place(COUNTRY_ALIASES.get(pa.get("country"), pa.get("country"))) if pa.get("country") else None, \
        place(COUNTRY_ALIASES.get(pb.get("country"), pb.get("country"))) if pb.get("country") else None
    return {"rounds": info.get("rounds"), "place": venue_place(v), "apex": "apex" in (v.get("name") or "").lower(),
            "cc_a": (pa_home or {}).get("cc"), "cc_b": (pb_home or {}).get("cc"),
            "tz_a": (pa_home or {}).get("tz"), "tz_b": (pb_home or {}).get("tz"),
            "lon_a": (pa_home or {}).get("lon"), "lon_b": (pb_home or {}).get("lon"), "class_height": heights}


def build(games, preds, profiles, stats):
    X, car = [], Career()
    heights = class_heights(profiles)
    for i, g in enumerate(games):
        d, a, b = g[0], g[1], g[2]
        info = sports.UFC_INFO.get((d, a, b), {})
        ids = info.get("ids") or [None, None]
        pa, pb = profiles.get(ids[0], {}), profiles.get(ids[1], {})
        X.append([matchup.logit(preds[i])] + car.feats(d, a, b, pa, pb, info.get("class", ""), context(info, pa, pb, heights)))
        cid = str(info.get("cid") or "")
        sa, sb = (stats.get(ids[0]) or {}).get(cid), (stats.get(ids[1]) or {}).get(cid)
        car.add(d, a, b, g[3] > g[4], info.get("method", ""), info.get("class", ""), info, sa, sb)
    return X, car


# ------------------------------------------------------------ news and the fight card
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


_DEAD = set()   # news sources that refused us this run: skip them instead of waiting on retries


def _rss(url, name, opponent, source, cutoff):
    if source in _DEAD:
        return []
    raw = sports.get(url, raw=True, tries=1)
    if not raw:
        _DEAD.add(source) if source != "news" else None
        return []
    text = raw.decode("utf-8", "replace")
    out = []
    last = name.split()[-1].lower()
    for item in re.findall(r"<(?:item|entry)>(.*?)</(?:item|entry)>", text, re.S)[:25]:
        title = html.unescape(re.sub(r"<.*?>", "", (re.search(r"<title[^>]*>(.*?)</title>", item, re.S) or [None, ""])[1])).strip()
        link = (re.search(r"<link>(.*?)</link>", item, re.S) or re.search(r'<link href="([^"]+)"', item) or [None, ""])[1].strip()
        when = (re.search(r"<pubDate>(.*?)</pubDate>", item, re.S) or re.search(r"<updated>(.*?)</updated>", item, re.S) or [None, ""])[1].strip()
        dt = None
        for fmt, n in (("%a, %d %b %Y %H:%M:%S", 25), ("%Y-%m-%dT%H:%M:%S", 19)):
            try:
                dt = datetime.strptime(when[:n], fmt)
                break
            except ValueError:
                pass
        if dt is None or dt < cutoff or last not in title.lower():
            continue
        kinds = [k for k, pat in FLAGS.items() if about(title, name, pat, opponent)]
        if kinds:
            out.append({"title": title[:200], "link": link, "date": dt.date().isoformat(), "kinds": kinds, "source": source})
    return out


def news_flags(name, opponent=""):
    """Headlines from the last 3 weeks about injuries, illness, pull-outs, weight, visas, suspensions or camp changes.
    Google News covers MMA sites and the fighters' own posts once reported; Reddit's r/MMA is checked too."""
    cutoff = datetime.utcnow() - timedelta(days=21)
    words = '(injury OR injured OR "pulled out" OR withdraws OR "missed weight" OR "weight cut" OR hospital OR replacement ' \
            'OR illness OR sick OR visa OR suspended OR USADA OR "parts ways")'
    out = _rss("https://news.google.com/rss/search?q=" + urllib.parse.quote(f'"{name}" {words}') + "&hl=en-US&gl=US&ceid=US:en",
               name, opponent, "news", cutoff)
    out += _rss("https://www.reddit.com/r/MMA/search.rss?restrict_sr=1&sort=new&t=month&q=" + urllib.parse.quote(f'"{name}"'),
                name, opponent, "r/MMA", cutoff)
    seen, uniq = set(), []
    for x in sorted(out, key=lambda x: x["date"], reverse=True):
        k = x["title"].lower()[:60]
        if k not in seen:
            seen.add(k)
            uniq.append(x)
    return uniq[:6]


def card_changes(bouts, today):
    """Tracks the fight card night to night. Returns {fighter: {"booked_days": n or None, "was": old opponent or None}}.
    A bout first seen within 3 weeks of fight night was booked on short notice; a fighter whose opponent changed
    since an earlier night has a new opponent. Nothing is flagged on the very first night (no history yet)."""
    path = os.path.join(sports.CACHE, "ufc3/card.json.gz")
    st = {"since": today.isoformat(), "bouts": {}, "opp": {}}
    if os.path.exists(path):
        try:
            with gzip.open(path, "rt") as f:
                st = json.load(f)
        except Exception:
            pass
    first_night = st["since"] == today.isoformat()
    out = {}
    for cid, d, a, b in bouts:
        seen = st["bouts"].setdefault(cid, today.isoformat())
        days = (date.fromisoformat(d) - date.fromisoformat(seen)).days
        short = days if (not first_night and seen > st["since"] and days <= 21) else None
        for me, op in ((a, b), (b, a)):
            was = st["opp"].get(me)
            out[me] = {"booked_days": short, "was": was if (was and was != op and not first_night) else None}
            st["opp"][me] = op
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt") as f:
        json.dump(st, f)
    return out


# ------------------------------------------------------------ run
def reasons(w, cols, row, a, b):
    """The biggest factors behind a fight's chance, in plain words, with whom they favor."""
    out = []
    for k, c in enumerate(cols):
        if c == 0:
            continue
        v = float(w[k + 1]) * row[c]
        if abs(v) >= 0.04:
            out.append((abs(v), (a if v > 0 else b), FACTORS[c][2], round(v, 3)))
    out.sort(reverse=True)
    return [{"favors": f, "why": why, "size": s} for _, f, why, s in out[:4]]


def run(games, home_edge):
    L = sports.learn_league.last
    ids = {i for info in sports.UFC_INFO.values() for i in info.get("ids") or [] if i}
    profiles = {i: profile(i) for i in ids}
    stats = {i: fight_stats(i) for i in ids}
    seasons = L["seasons"]
    y = [1.0 if g[3] > g[4] else 0.0 for g in games]
    scored = {v: [0.5] * len(games) for v in VARIANTS}
    by_year = {}   # (variant, year) -> list of indices scored
    test_idx = []
    years = sorted(set(x for x in seasons if x >= 1))
    # score every variant on every year from the 3rd on (each fit only on earlier years); keep the last 3 as tests
    eval_years = [s for s in years if s >= min(L["tests"]) - 2 and s >= 2]
    for s in eval_years:
        preds = L["runs"][L["chosen"].get(s, L["best"])][0]
        X, _ = build(games, preds, profiles, stats)
        tr = [i for i, x in enumerate(seasons) if 1 <= x < s]
        te = [i for i, x in enumerate(seasons) if x == s]
        if s in L["tests"]:
            test_idx += te
        for v, cols in VARIANTS.items():
            w = matchup.fit([X[i] for i in tr], [y[i] for i in tr], cols)
            for i in te:
                scored[v][i] = matchup.predict(w, X[i], cols)
            by_year[(v, s)] = te

    def ll(v, idx):
        return sum(-math.log(max(1e-9, scored[v][i] if y[i] else 1 - scored[v][i])) for i in idx) / max(1, len(idx))

    # honest version: for each test year use the variant that did best on all the years before it
    def earlier(v, s):
        return [i for yr in eval_years if yr < s for i in by_year.get((v, yr), [])]
    picked = {}
    scored["picked"] = [0.5] * len(games)
    for s in L["tests"]:
        cand = [v for v in VARIANTS if earlier(v, s)]
        best = min(cand, key=lambda v: ll(v, earlier(v, s))) if cand else "elo"
        picked[s] = best
        for i in by_year.get((best, s), []):
            scored["picked"][i] = scored[best][i]
    res = {v: sports.score_games(games, scored[v], test_idx, False) for v in list(VARIANTS) + ["picked"]}
    use_factors = res["picked"]["logloss"] < res["elo"]["logloss"] - 0.0005
    # today's choice: the variant that did best on every year scored so far (the same rule, one year later)
    final = min(VARIANTS, key=lambda v: ll(v, earlier(v, 10 ** 6))) if use_factors else "elo"
    rec = {"tested": res, "picked_each_year": {str(k): v for k, v in picked.items()}, "used": final,
           "groups": {g: [NAMES[i] for i in c] for g, c in GROUPS.items()}}
    preds, _, ratings = L["runs"][L["best"]]
    X, car = build(games, preds, profiles, stats)
    tr = [i for i, x in enumerate(seasons) if x >= 1]
    model = None
    if final != "elo":
        cols = VARIANTS[final]
        w = matchup.fit([X[i] for i in tr], [y[i] for i in tr], cols)
        model = (w, cols)
        rec["weights"] = {NAMES[c]: round(float(w[k + 1]), 3) for k, c in enumerate(cols)}
    print(f"  UFC fight factors (log loss on unseen years): " +
          ", ".join(f"{v} {r['logloss']:.4f}" for v, r in res.items()) + f" -> using {final}")

    # upcoming fights in the next 5 weeks
    today = date.today()
    heights = class_heights(profiles)
    j = sports.get(f"{sports.ESPN}/mma/ufc/scoreboard?dates={today:%Y%m%d}-{today + timedelta(days=35):%Y%m%d}&limit=1000") or {}
    fights = []
    for ev in j.get("events", []):
        for c in ev.get("competitions") or []:
            comp = c.get("competitors") or []
            if len(comp) != 2 or ((c.get("status") or {}).get("type") or {}).get("completed"):
                continue
            a, b = ((x.get("athlete") or {}).get("displayName") for x in comp)
            if a and b:
                fights.append((ev, c, comp, a, b))
    changes = card_changes([(str(c.get("id")), (c.get("date") or ev.get("date") or "")[:10], a, b) for ev, c, comp, a, b in fights], today)
    up = []
    for ev, c, comp, a, b in fights:
        fa, fb = comp[0].get("id"), comp[1].get("id")
        pa, pb = profile(fa) if fa else {}, profile(fb) if fb else {}
        for fid in (fa, fb):
            if fid:
                stats[fid] = fight_stats(fid, keep_days=6)
        ra, rb = ratings.get(a, 1500), ratings.get(b, 1500)
        e = 1 / (1 + 10 ** (-(ra - rb) / 400))
        d = (c.get("date") or ev.get("date") or "")[:10]
        wc = (c.get("type") or {}).get("abbreviation") or ""
        v = c.get("venue") or {}
        addr = v.get("address") or {}
        info = {"venue": {"name": v.get("fullName") or "", "city": addr.get("city") or "", "state": addr.get("state") or "",
                          "country": addr.get("country") or ""},
                "rounds": ((c.get("format") or {}).get("regulation") or {}).get("periods") or 3}
        ctx = context(info, pa, pb, heights)
        f = car.feats(d, a, b, pa, pb, wc, ctx)
        where = ", ".join(x for x in (addr.get("city"), addr.get("state") or addr.get("country")) if x)
        row = {"date": d, "home": a, "away": b, "venue": (v.get("fullName") or "") + (f" ({where})" if where else ""), "indoor": True,
               "elo": round(e, 4), "class": wc, "rounds": info["rounds"], "fighters": {}}
        if model:
            row["full"] = round(matchup.predict(model[0], [matchup.logit(e)] + f, model[1]), 4)
            row["reasons"] = reasons(model[0], model[1], [matchup.logit(e)] + f, a, b)
        ev_place = ctx.get("place") or {}
        for n, prof, side in ((a, pa, "a"), (b, pb, "b")):
            h = car.f.get(n, [])
            last = h[-1] if h else None
            sty = car.style(n)
            S = car.side(n, prof, d, wc)
            home = bool(ev_place.get("cc") and ctx.get("cc_" + side) == ev_place.get("cc"))
            tz = time_zones(ctx.get("tz_" + side), ev_place.get("tz"), d, ctx.get("lon_" + side), ev_place.get("lon")) \
                if ev_place and (ctx.get("tz_" + side) or ctx.get("lon_" + side) is not None) else None
            ch = changes.get(n) or {}
            row["fighters"][n] = {
                "age": prof.get("age"), "reach": prof.get("reach"), "height": prof.get("height"), "stance": prof.get("stance"),
                "country": prof.get("country") or "", "gym": prof.get("gym") or "",
                "ufc_fights": len(h), "record": f"{sum(1 for x in h if x[1])}-{sum(1 for x in h if not x[1])}",
                "layoff_days": (date.fromisoformat(d) - date.fromisoformat(last[0])).days if last else None,
                "last": (("Won" if last[1] else "Lost") + (f" by {last[2]}" if last[2] else "") + f" ({last[0]})") if last else "",
                "class_change": (CLASSES.index(wc) - CLASSES.index(last[3])) if last and last[3] in CLASSES and wc in CLASSES else 0,
                "stats_fights": sty["n"],
                "slpm": round(sty["slpm"], 2), "sapm": round(sty["sapm"], 2), "acc": round(sty["acc"], 3), "def": round(sty["def"], 3),
                "td15": round(sty["td15"], 2), "tddef": round(sty["tddef"], 3), "sub15": round(sty["sub15"], 2),
                "kd_taken15": round(sty["kdt15"], 2), "ko_losses": S["ko_losses"], "active_2y": S["active"],
                "home": home, "time_zones": round(tz, 1) if tz is not None else None,
                "booked_days": ch.get("booked_days"), "new_opponent_was": ch.get("was"),
                "news": news_flags(n, b if n == a else a),
            }
        up.append(row)
    return rec, up

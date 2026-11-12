"""
Letter grades (A+ to F) for every U.S. stock, three ways, like the Briefs Score but with the method shown:

  Growth  - sales and profit rising fast (and speeding up), margins improving, price outperforming.
  Income  - a dividend that is paid reliably, covered by cash, growing, and a fair yield.
  Wealth  - wealth preservation: steady, profitable, low-debt businesses whose stock holds up when the market drops
            (the "drawdown shield": how much of the S&P 500's down days it took, its worst drop, how bumpy it is).

Plus "Next big thing": smaller fast growers whose growth is speeding up and whose stock is already outperforming.

Each measure becomes a percentile against every other graded stock; the weighted average becomes the grade.
The same rules are also run as of one year ago, and the next year's returns by grade are reported, so you can see
whether the grades actually worked (only stocks still trading today are included, which flatters the results a bit).

Company financials come from SEC XBRL "frames" (one file per number per year for every company).
"""
import gzip
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from datetime import date, timedelta

import numpy as np

import history
import model

SEC_UA = os.environ.get("SEC_USER_AGENT") or "StockTracker-Scanner admin@example.com"
M = 1_000_000.0
REVENUE = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"]
DIVIDENDS = ["PaymentsOfDividendsCommonStock", "PaymentsOfDividends", "PaymentsOfOrdinaryDividends",
             "DividendsCommonStockCash", "DividendsCommonStock", "Dividends"]
DIV_PER_SHARE = ["CommonStockDividendsPerShareDeclared", "CommonStockDividendsPerShareCashPaid"]
UNITS = {"EarningsPerShareDiluted": "USD-per-shares", "WeightedAverageNumberOfDilutedSharesOutstanding": "shares",
         "CommonStockDividendsPerShareDeclared": "USD-per-shares", "CommonStockDividendsPerShareCashPaid": "USD-per-shares"}
LETTERS = [(97, "A+"), (90, "A"), (84, "A-"), (77, "B+"), (69, "B"), (61, "B-"), (53, "C+"), (45, "C"), (37, "C-"),
           (29, "D+"), (21, "D"), (13, "D-"), (0, "F")]
_last = [0.0]


# ------------------------------------------------------------ SEC data
def _get(url, retries=4):
    for attempt in range(retries):
        wait = 0.15 - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": SEC_UA, "Accept-Encoding": "gzip"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    data = gzip.decompress(data)
                return data.decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code in (403, 429, 503):
                time.sleep(10 * (attempt + 1))
                continue
            raise
        except Exception:
            time.sleep(3)
    raise RuntimeError(f"SEC kept refusing {url}")


class Sec:
    """SEC frames and the company list, cached on disk (a week for frames, a year for industries)."""

    def __init__(self, cache_dir):
        self.dir = os.path.join(cache_dir, "sec-frames")
        os.makedirs(self.dir, exist_ok=True)
        self.mem = {}

    def frame(self, tag, period):
        key = f"{tag}/{period}"
        if key in self.mem:
            return self.mem[key]
        path = os.path.join(self.dir, f"{tag}_{period}.json")
        m = re.match(r"CY(\d{4})", period)
        keep = 60 if m and int(m.group(1)) <= date.today().year - 3 else 6
        if os.path.exists(path) and time.time() - os.path.getmtime(path) < keep * 86400:
            vals = json.load(open(path))
        else:
            unit = UNITS.get(tag, "USD")
            body = _get(f"https://data.sec.gov/api/xbrl/frames/us-gaap/{tag}/{unit}/{period}.json")
            vals = {}
            if body:
                for d in json.loads(body).get("data", []):
                    v = d.get("val")
                    if isinstance(v, (int, float)) and math.isfinite(v):
                        vals[str(d["cik"])] = v
            json.dump(vals, open(path, "w"))
        m = {int(k): float(v) for k, v in vals.items()}
        self.mem[key] = m
        return m

    def annual(self, tag, year):
        return self.frame(tag, f"CY{year}")

    def companies(self):
        """[(cik, ticker, name)] for NYSE and Nasdaq stocks, one ticker per company. Tickers use '.' (BRK.B)."""
        path = os.path.join(self.dir, "companies.json")
        if os.path.exists(path) and time.time() - os.path.getmtime(path) < 6 * 86400:
            body = open(path).read()
        else:
            body = _get("https://www.sec.gov/files/company_tickers_exchange.json")
            open(path, "w").write(body)
        j = json.loads(body)
        idx = {f: i for i, f in enumerate(j["fields"])}
        seen, out = set(), []
        for row in j["data"]:
            ex = str(row[idx["exchange"]] or "").upper()
            t = str(row[idx["ticker"]] or "").upper()
            cik = int(row[idx["cik"]])
            if ex not in ("NYSE", "NASDAQ") or not re.fullmatch(r"[A-Z]{1,5}(-[A-Z])?", t) or cik in seen:
                continue
            seen.add(cik)
            out.append((cik, t.replace("-", "."), _title(str(row[idx["name"]]))))
        return out

    def industries(self, ciks):
        """{cik: (sic code, description)} for the given companies (cached for a year; new ones fetched)."""
        path = os.path.join(self.dir, "industries.json")
        known = json.load(open(path)) if os.path.exists(path) else {}
        added = 0
        for cik in ciks:
            if str(cik) in known:
                continue
            try:
                body = _get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
                j = json.loads(body) if body else {}
                known[str(cik)] = [int(j.get("sic") or 0), _title(j.get("sicDescription") or "")]
                added += 1
            except Exception:
                pass
            if added and added % 200 == 0:
                json.dump(known, open(path, "w"))
        json.dump(known, open(path, "w"))
        return {int(k): tuple(v) for k, v in known.items()}


def _title(s):
    return s if any(ch.islower() for ch in s) else " ".join(w.capitalize() for w in s.lower().split())


# ------------------------------------------------------------ measures
def _cagr(a, b, years):
    if a is None or b is None or a <= 0 or b <= 0:
        return None
    return (b / a) ** (1 / years) - 1


def _pct(v):
    return f"{v * 100:+.0f}%"


def fundamentals(sec, companies, fy, balance_periods):
    """Per-company measures from SEC filings for fiscal year `fy`. {cik: dict}."""
    def series(tags, cik, years):
        best = {}
        for tag in tags:
            m = {}
            for yr in years:
                v = sec.annual(tag, yr).get(cik)
                if v is not None:
                    m[yr] = v
            if len(m) > len(best):
                best = m
        return best

    def one(tag, yr, cik):
        return sec.annual(tag, yr).get(cik)

    assets, equity = {}, {}
    for p in balance_periods:   # newer last, so it wins
        assets.update(sec.frame("Assets", p))
        equity.update(sec.frame("StockholdersEquity", p))

    y = fy
    out = {}
    for cik, ticker, name in companies:
        rev = series(REVENUE, cik, range(y - 5, y + 1))
        ni = series(["NetIncomeLoss"], cik, range(y - 5, y + 1))
        if y not in ni:
            continue
        n0, r0 = ni[y], rev.get(y)
        if r0 is not None and abs(n0) > 1.5 * abs(r0):
            continue   # a profit far bigger than sales is a mislabeled number
        f = {"rev": rev, "ni": ni, "eps": one("EarningsPerShareDiluted", y, cik),
             "eps_prev": one("EarningsPerShareDiluted", y - 1, cik),
             "shares": one("WeightedAverageNumberOfDilutedSharesOutstanding", y, cik),
             "shares_prev": one("WeightedAverageNumberOfDilutedSharesOutstanding", y - 1, cik)}
        rp = rev.get(y - 1)
        if r0 is not None and rp and rp > 0:
            f["rev_g1"] = r0 / rp - 1
            rpp = rev.get(y - 2)
            if rpp and rpp > 0:
                f["rev_g_prev"] = rp / rpp - 1
        f["rev_c3"] = _cagr(rev.get(y - 3), r0, 3)
        f["rev_c5"] = _cagr(rev.get(y - 5), r0, 5)
        eps, epsp = f["eps"], f["eps_prev"]
        if eps is not None and epsp is not None:
            f["eps_g"] = -1.0 if eps <= 0 else 0.5 if epsp <= 0 else max(-1, min(3, eps / epsp - 1))
        op, op_old, rev_old = one("OperatingIncomeLoss", y, cik), one("OperatingIncomeLoss", y - 2, cik), rev.get(y - 2)
        if None not in (op, op_old, r0, rev_old) and r0 > 0 and rev_old > 0:
            f["margin_trend"] = op / r0 - op_old / rev_old
        gp = one("GrossProfit", y, cik)
        if gp is None:
            cost = one("CostOfRevenue", y, cik) or one("CostOfGoodsAndServicesSold", y, cik)
            if r0 is not None and cost is not None:
                gp = r0 - cost
        gp_prev = one("GrossProfit", y - 1, cik)
        if gp is not None and r0 and r0 > 0:
            f["gross_margin"] = gp / r0
            if gp_prev is not None and rp and rp > 0:
                f["gross_trend"] = gp / r0 - gp_prev / rp
        fcf_sum = ni_sum = 0.0
        yrs = 0
        for yr in range(y - 2, y + 1):
            ocf = one("NetCashProvidedByUsedInOperatingActivities", yr, cik)
            if ocf is None or yr not in ni:
                continue
            capex = abs(one("PaymentsToAcquirePropertyPlantAndEquipment", yr, cik) or 0.0)
            if yr == y:
                f["fcf"] = ocf - capex
            fcf_sum += ocf - capex
            ni_sum += ni[yr]
            yrs += 1
        if f.get("fcf") is not None and r0 and r0 > 0:
            f["fcf_margin"] = f["fcf"] / r0
        if yrs == 3 and ni_sum > 0:
            f["fcf_conv"] = max(-1, min(1.5, fcf_sum / ni_sum))
        a, e = assets.get(cik), equity.get(cik)
        if a and a > 0:
            f["roa"] = n0 / a
            if gp is not None:
                f["gross_prof"] = gp / a
            if e is not None:
                f["equity_ratio"] = e / a
                f["equity"] = e
                if e / a >= 0.05:
                    f["roe"] = n0 / e
                liab = a - e
                if f.get("fcf") is not None:
                    f["fcf_liab"] = 10.0 if liab <= 0 else min(10.0, f["fcf"] / liab)
        f["profitable_years"] = sum(1 for v in ni.values() if v > 0)
        f["ni_c5"] = _cagr(ni.get(y - 5), n0, 5)
        margins = [ni[yr] / rev[yr] for yr in range(y - 5, y + 1) if yr in ni and rev.get(yr, 0) > 0]
        if len(margins) >= 5:
            f["margin_swing"] = float(np.std(margins))
        steps = ups = 0
        for yr in range(y - 4, y + 1):
            for s in (rev, ni):
                if yr - 1 in s and yr in s:
                    steps += 1
                    ups += s[yr] > s[yr - 1]
        if steps >= 8:
            f["consistency"] = ups / steps
        sp, sn = f["shares_prev"], f["shares"]
        if sp and sn and sp > 0 and -0.3 < sn / sp - 1 < 0.4:
            f["dilution"] = sn / sp - 1
        div = {k: abs(v) for k, v in series(DIVIDENDS, cik, range(y - 9, y + 1)).items()}
        if y not in div:
            # Some companies only report dividends per share: times shares outstanding.
            dps = series(DIV_PER_SHARE, cik, range(y - 9, y + 1))
            sh = f["shares"]
            if sh and dps.get(y):
                div = {k: v * sh for k, v in dps.items() if v > 0}
        f["div"] = div
        d = div.get(y)
        if d and d > 0:
            f["div_years"] = 0 if any(v <= 0 for v in div.values()) else sum(1 for v in div.values() if v > 0)
            f["div_c5"] = _cagr(div.get(y - 5), d, 5)
            streak = 0
            for yr in range(y, y - 9, -1):
                if yr in div and yr - 1 in div and div[yr] >= div[yr - 1] * 0.97:
                    streak += 1
                else:
                    break
            f["div_streak"] = float(streak)
            earlier = [div[yr] for yr in range(y - 4, y) if yr in div]
            if earlier and sum(earlier) > 0:
                f["div_vs_avg"] = d / (sum(earlier) / len(earlier))
            if f.get("fcf") and f["fcf"] > 0:
                f["fcf_payout"] = d / f["fcf"]
            if n0 > 0:
                f["profit_payout"] = d / n0
        out[cik] = f
    return out


def price_measures(c, v, h, l, spy_ret, t):
    """Price-based measures as of day index t (inclusive), from a stock's arrays aligned with spy_ret."""
    if t < 260 or c[t] <= 0:
        return None
    win = slice(t - 251, t + 1)
    r = c[t - 251:t + 1] / c[t - 252:t] - 1
    s = spy_ret[t - 251:t + 1]
    ok = np.isfinite(r) & np.isfinite(s)
    down = ok & (s < 0)
    m = {"price": float(c[t]),
         "dollar_vol": float(np.mean(c[t - 19:t + 1] * v[t - 19:t + 1])),
         "chg12": float(c[t] / c[t - 252] - 1),
         "momentum": float(c[t - 21] / c[t - 252] - 1),
         "rs6": float(c[t] / c[t - 126] - 1) - float(np.prod(1 + s[-126:][np.isfinite(s[-126:])]) - 1),
         "near_high": float(c[t] / np.max(h[win])),
         "vol": float(np.std(r[ok]) * math.sqrt(252)) if ok.sum() > 100 else None}
    if down.sum() > 20 and s[down].sum() < 0:
        m["down_capture"] = float(r[down].sum() / s[down].sum())
    peak = np.maximum.accumulate(c[win])
    m["max_dd"] = float(np.min(c[win] / peak - 1))
    return m


# ------------------------------------------------------------ grading
def composite(rows, metrics):
    """rows: {key: dict}. metrics: [(weight, field, higher_is_better, required)]. Returns {key: 0-100 score}."""
    keys = list(rows)
    tot = np.zeros(len(keys))
    wsum = np.zeros(len(keys))
    ok = np.ones(len(keys), bool)
    for w, field, hib, req in metrics:
        vals = [(i, rows[k].get(field)) for i, k in enumerate(keys)]
        vals = [(i, x) for i, x in vals if isinstance(x, (int, float)) and math.isfinite(x)]
        has = np.zeros(len(keys), bool)
        if vals:
            order = sorted(vals, key=lambda p: p[1])
            n = len(order)
            # ties share a percentile
            j = 0
            while j < n:
                k = j
                while k + 1 < n and order[k + 1][1] == order[j][1]:
                    k += 1
                p = 50.0 if n == 1 else (j + k) / 2 * 100 / (n - 1)
                if not hib:
                    p = 100 - p
                for q in range(j, k + 1):
                    i = order[q][0]
                    tot[i] += p * w
                    wsum[i] += w
                    has[i] = True
                j = k + 1
        if req:
            ok &= has
    total = sum(m[0] for m in metrics)
    return {keys[i]: tot[i] / wsum[i] for i in range(len(keys)) if ok[i] and wsum[i] >= 0.7 * total}


def letters(scores, caps=None):
    """Turns scores into letter grades by rank: the top 3% get A+, and so on. caps: {key: worst-allowed letter}."""
    keys = sorted(scores, key=lambda k: scores[k])
    n = len(keys)
    out = {}
    order = [x[1] for x in LETTERS]
    for i, k in enumerate(keys):
        pct = 100 * i / max(1, n - 1)
        g = next(letter for cut, letter in LETTERS if pct >= cut)
        cap = (caps or {}).get(k)
        if cap and order.index(g) < order.index(cap):
            g = cap
        out[k] = (g, round(pct))
    return out


GROWTH = [(20, "rev_g1", True, True), (15, "rev_c3", True, False), (10, "accel", True, False),
          (12, "eps_g", True, False), (8, "margin_trend", True, False), (15, "momentum", True, False),
          (10, "rs6", True, False), (10, "fcf_margin", True, False)]
INCOME = [(25, "yield", True, True), (20, "div_c5", True, False), (15, "div_streak", True, False),
          (15, "fcf_payout", False, False), (15, "fcf_liab", True, False), (10, "roe", True, False)]
WEALTH = [(10, "roe", True, False), (8, "roa", True, False), (8, "gross_prof", True, False),
          (8, "margin_swing", False, False), (8, "equity_ratio", True, False), (6, "rev_c5", True, False),
          (6, "consistency", True, False), (6, "fcf_conv", True, False), (4, "dilution", False, False),
          # drawdown shield
          (14, "down_capture", False, True), (12, "max_dd", True, False), (10, "vol", False, False)]
RISING = [(20, "rev_g1", True, True), (15, "accel", True, False), (10, "gross_trend", True, False),
          (10, "margin_trend", True, False), (20, "rs6", True, True), (10, "near_high", True, False),
          (5, "fcf_margin", True, False), (10, "insider", True, False)]


def grade_all(companies, fund, prices, insiders=None):
    """companies: [(cik, ticker, name)]; fund: {cik: fundamentals}; prices: {ticker: price measures}.
    Returns {ticker: {...}} plus the rising-star ranking."""
    rows = {}
    for cik, t, name in companies:
        f, p = fund.get(cik), prices.get(t)
        if f is None or p is None or p["price"] < 3 or p["dollar_vol"] < 1 * M:
            continue
        shares = f.get("shares")
        n0, eps = f["ni"][max(f["ni"])], f.get("eps")
        implied = n0 / eps if eps not in (None, 0) and n0 / eps > 0 else None
        if not shares or shares <= 0:
            shares = implied
        elif implied and not 0.5 <= shares / implied <= 2:
            shares = implied
        if not shares or shares <= 0:
            continue
        cap = shares * p["price"]
        if cap < 100 * M or (n0 > 0 and cap < 2 * n0):
            continue
        r = {**{k: v for k, v in f.items() if not isinstance(v, dict)}, **p, "cik": cik, "ticker": t, "name": name,
             "cap": cap, "rev_now": f["rev"].get(max(f["ni"])), "ni_now": n0}
        if r.get("rev_g1") is not None and r.get("rev_g_prev") is not None:
            r["accel"] = r["rev_g1"] - r["rev_g_prev"]
        d = f["div"].get(max(f["ni"]))
        if d and d > 0:
            r["yield"] = d / cap
        if insiders and t in insiders:
            r["insider"] = insiders[t]
        rows[t] = r

    g_rows = {t: r for t, r in rows.items() if r.get("rev_g1") is not None}
    growth = composite(g_rows, GROWTH)

    payers = {t: r for t, r in rows.items() if r.get("yield")}
    income = composite(payers, INCOME)
    icaps = {}
    for t, r in payers.items():
        if r["yield"] < 0.01:
            icaps[t] = "D"    # a token dividend isn't income
            continue
        if r["yield"] < 0.02:
            icaps[t] = "B-"
        if r["yield"] > 0.10 or (r.get("fcf_payout") or 0) > 1.0 or (r.get("div_vs_avg") or 1) < 0.9 \
                or r.get("div_years", 0) < 5 or r["ni_now"] <= 0:
            icaps[t] = "C-"   # risky: very high yield, not covered by cash, recently cut, short record, or a loss
    wealth = composite(rows, WEALTH)
    wcaps = {t: "D+" for t, r in rows.items() if r["ni_now"] <= 0 or r.get("profitable_years", 0) < 3}

    rise_rows = {t: r for t, r in rows.items() if 300 * M <= r["cap"] <= 15_000 * M and (r.get("rev_now") or 0) >= 50 * M
                 and (r.get("rev_g1") or 0) >= 0.20}
    rising = composite(rise_rows, RISING)

    gl, il, wl = letters(growth), letters(income, icaps), letters(wealth, wcaps)
    out = {}
    for t, r in rows.items():
        e = {"name": r["name"], "cap": round(r["cap"] / M)}
        if t in gl:
            e["g"], e["gp"] = gl[t]
        if t in il:
            e["i"], e["ip"] = il[t]
        elif not r.get("yield"):
            e["i"] = "F"
        if t in wl:
            e["w"], e["wp"] = wl[t]
        e["why"] = reasons(r)
        out[t] = e
    stars = sorted(rising, key=lambda t: -rising[t])
    return out, stars, rows


def reasons(r):
    """Plain-English facts behind each grade."""
    g, i, w = [], [], []
    if r.get("rev_g1") is not None:
        g.append(f"Sales {_pct(r['rev_g1'])} in a year")
    if r.get("accel") is not None and abs(r["accel"]) >= 0.05:
        g.append(f"Growth {'speeding up' if r['accel'] > 0 else 'slowing'} ({_pct(r['accel'])} pts)")
    if r.get("rev_c3") is not None:
        g.append(f"{_pct(r['rev_c3'])} a year over 3 years")
    if r.get("rs6") is not None:
        g.append(f"{'Beat' if r['rs6'] >= 0 else 'Trailed'} the S&P 500 by {abs(r['rs6']) * 100:.0f} pts in 6 months")
    if r.get("yield"):
        i.append(f"Yield {r['yield'] * 100:.1f}%")
        if r.get("fcf_payout") is not None:
            i.append(f"Pays out {r['fcf_payout'] * 100:.0f}% of its cash flow")
        if r.get("div_c5") is not None:
            i.append(f"Dividend {_pct(r['div_c5'])} a year over 5 years")
        if r.get("div_years"):
            i.append(f"Paid {r['div_years']} of the last 10 years")
    else:
        i.append("Pays no dividend")
    if r.get("down_capture") is not None:
        dc = r["down_capture"]
        w.append(f"On the market's down days it fell {dc * 100:.0f}% as much as the S&P 500")
    if r.get("max_dd") is not None:
        w.append(f"Worst drop in the last year: {r['max_dd'] * 100:.0f}%")
    if r.get("roe") is not None:
        w.append(f"Earns {r['roe'] * 100:.0f}% on shareholders' money")
    if r.get("profitable_years") is not None:
        w.append(f"Profitable {r['profitable_years']} of the last 6 years")
    m = []
    if r.get("gross_prof") is not None:
        m.append(f"Gross profit is {r['gross_prof'] * 100:.0f}% of its assets each year")
    if r.get("roe") is not None:
        m.append(f"Earns {r['roe'] * 100:.0f}% on shareholders' money")
    if r.get("fcf_conv") is not None:
        m.append(f"Turned {r['fcf_conv'] * 100:.0f}% of 3 years' profit into cash")
    if r.get("equity_ratio") is not None:
        m.append(f"Shareholders own {r['equity_ratio'] * 100:.0f}% of its assets (the rest is debt and other bills)")
    if r.get("consistency") is not None:
        m.append(f"Sales and profit rose in {r['consistency'] * 100:.0f}% of the last 5 years")
    return {"g": g[:4], "i": i[:4], "w": w[:4], "m": m[:5]}


def star_reasons(r):
    out = [f"Sales {_pct(r['rev_g1'])} in a year"]
    if r.get("accel") is not None and r["accel"] > 0:
        out.append(f"growth speeding up by {r['accel'] * 100:.0f} pts")
    if r.get("gross_trend") is not None and r["gross_trend"] > 0.01:
        out.append(f"gross margin up {r['gross_trend'] * 100:.0f} pts")
    out.append(f"{'beat' if r['rs6'] >= 0 else 'trailed'} the S&P 500 by {abs(r['rs6']) * 100:.0f} pts in 6 months")
    if r.get("insider"):
        out.append("insiders buying")
    return "; ".join(out)


# ------------------------------------------------------------ main entry
def model_reasons(L, i, goal):
    """The measures that pushed this stock's learned score up or down the most."""
    c = (L["contrib_g"] if goal == "g" else L["contrib_w"])[i]
    x = L["X"][i]
    names = L["names"]
    order = np.argsort(-c)

    def trait(k):   # describe what the stock is, not which way the weight points
        good, bad = model.FEATURES[names[k]]
        return good if x[k] >= 0 else bad
    up = [trait(k) for k in order[:2] if c[k] > 0.002]
    down = [trait(k) for k in order[::-1][:1] if c[k] < -0.002]
    out = []
    if up:
        out.append("Helped by " + " and ".join(up))
    if down:
        out.append("Held back by " + down[0])
    return out


def apply_model(L, grades, rows, companies, sics_get):
    """Replaces the hand-weighted growth, wealth and income grades with the learned, tested ones."""
    ix = {t: i for i, t in enumerate(L["tickers"])}
    oos = L["oos"]
    cal = {"g": model.calibration(oos, 0), "w": model.calibration(oos, 1), "i": model.calibration(oos, 2)}
    line = {"g": model.smooth_odds(oos, 0), "w": model.smooth_odds(oos, 1), "i": model.smooth_odds(oos, 2)}
    cal["m"], line["m"] = model.calibration(oos, 3), model.smooth_odds(oos, 3)
    def letter(p):
        return next(l for cut, l in LETTERS if p * 100 >= cut)
    for t, e in grades.items():
        i = ix.get(t)
        if i is None or not L["uni"][i]:
            continue
        for goal, key in (("g", "pg"), ("w", "pw"), ("i", "pi"), ("m", "pm")):
            p = float(L[key][i])
            if not np.isfinite(p):
                continue
            g = letter(p)
            cap = None
            r = rows.get(t, {})
            if goal == "i":   # keep the safety rules for risky dividends
                y = r.get("yield") or 0
                if y < 0.01:
                    cap = "D"
                elif r.get("yield", 0) > 0.10 or (r.get("fcf_payout") or 0) > 1.0 or (r.get("div_vs_avg") or 1) < 0.9 \
                        or r.get("div_years", 0) < 5 or r.get("ni_now", 1) <= 0:
                    cap = "C-"
                elif y < 0.02:
                    cap = "B-"
            order = [x[1] for x in LETTERS]
            if cap and order.index(g) < order.index(cap):
                g = cap
            e[goal], e[goal + "p"] = g, round(p * 100)
            fit = line[goal]
            if fit:
                a, b, c, d = fit
                e[goal + "b"] = round(min(0.95, max(0.05, a + b * (p - 0.5))), 3)   # measured chance to beat the S&P 500 (3 months)
                e[goal + "d"] = round(min(0.0, c + d * (p - 0.5)), 4)               # typical worst drop over 3 months
            if goal in ("g", "w"):
                e["why"][goal] = model_reasons(L, i, goal) + e["why"][goal][:3]
    # Next big thing from the learned growth score
    pick = model.star_pick(L["raw"], L["pg"], L["uni"], top=120)
    cands = [L["tickers"][i] for i in np.where(pick)[0]]
    cands.sort(key=lambda t: -L["pg"][ix[t]])
    return cal, cands


def run(cache_dir, days, series, insiders=None, data=None, hist_tickers=None):
    """days: [date]; series: {ticker: (o, h, l, c, v, idx)} from scan.py. Returns the grades file contents."""
    sec = Sec(cache_dir)
    companies = sec.companies()
    n = len(days)
    spy = series["SPY"]
    spy_c = np.full(n, np.nan)
    spy_c[spy[5]] = spy[3]
    spy_ret = np.full(n, np.nan)
    spy_ret[1:] = spy_c[1:] / spy_c[:-1] - 1

    def aligned(t):
        o, h, l, c, v, idx = series[t]
        C, H, L, V = (np.full(n, np.nan) for _ in range(4))
        C[idx], H[idx], L[idx], V[idx] = c, h, l, v
        # carry the last price over missing days so windows line up
        for arr in (C, H, L):
            for i in range(1, n):
                if not np.isfinite(arr[i]):
                    arr[i] = arr[i - 1]
        V[~np.isfinite(V)] = 0
        return C, V, H, L

    def prices_at(t):
        out = {}
        for _, tk, _ in companies:
            if tk not in series or len(series[tk][3]) < 270:
                continue
            C, V, H, L = aligned(tk)
            if not np.isfinite(C[t]) or not np.isfinite(C[t - 252]):
                continue
            pm = price_measures(C, V, H, L, spy_ret, t)
            if pm:
                out[tk] = pm
        return out

    today = days[-1]
    fy = today.year - 1
    if len(sec.annual("NetIncomeLoss", fy)) < 2500:
        fy -= 1
    q = today - timedelta(days=150)

    def quarter(d):
        return f"CY{d.year}Q{(d.month - 1) // 3 + 1}I"
    newer = quarter(date(q.year + (q.month + 2) // 12, (q.month + 2) % 12 + 1, 1))
    balance = [quarter(q), newer]

    print("Grades: reading company financials")
    fund = fundamentals(sec, companies, fy, balance)
    now_prices = prices_at(n - 1)
    grades, stars, rows = grade_all(companies, fund, now_prices, insiders)
    print(f"Grades: {len(grades)} stocks graded (fiscal year {fy})")

    sics = sec.industries([rows[t]["cik"] for t in stars[:120]])
    stars = [t for t in stars if not 6000 <= sics.get(rows[t]["cik"], (0, ""))[0] <= 6799]
    star_list = []
    for t in stars[:30]:
        r = rows[t]
        star_list.append({"t": t, "name": r["name"], "cap": round(r["cap"] / M), "industry": sics.get(r["cik"], (0, ""))[1],
                          "why": star_reasons(r), "g": grades[t].get("g", "")})

    # ---- learned model: fit to history and tested on months it never saw ----
    if data is not None:
        try:
            panel, span = None, days
            if hist_tickers:
                try:
                    want = sorted(set(hist_tickers) | {"SPY"})
                    have = history.backfill(want, cache_dir, budget_s=int(os.environ.get("HISTORY_BUDGET", "2400")))
                    if have >= 0.98 * len(want):   # wait for (nearly) every stock, not just the biggest
                        all_t = sorted({t for d in data for t in d if model.TICKER.match(t)} | set(want))
                        dates, C, V = history.merged_panel(cache_dir, days, data, all_t)
                        panel, span = (dates, all_t, C, V), dates
                        print(f"Grades: training on {len(dates)} trading days ({dates[0]} to {dates[-1]})")
                    else:
                        print(f"Grades: long history still downloading ({have} of {len(want)}); using 2 years for now")
                except Exception as e:
                    print(f"::warning title=Long price history failed, using 2 years::{e}")
            need = {model.fy_for(d) for d in span[model.LOOKBACK:]}
            print(f"Grades: learning from history (fiscal years {sorted(need)})")
            fund_by_fy = {y: (fund if y == fy else fundamentals(sec, companies, y, [f"CY{y}Q4I"])) for y in need}
            L = model.learn(days, data, companies, fund_by_fy, panel=panel)
            cal, cands = apply_model(L, grades, rows, companies, None)
            sics2 = sec.industries([rows[t]["cik"] for t in cands if t in rows])
            picks = [t for t in cands if t in rows and not 6000 <= sics2.get(rows[t]["cik"], (0, ""))[0] <= 6799][:30]
            star_list = [{"t": t, "name": rows[t]["name"], "cap": round(rows[t]["cap"] / M),
                          "industry": sics2.get(rows[t]["cik"], (0, ""))[1], "why": star_reasons(rows[t]),
                          "g": grades[t].get("g", "")} for t in picks]
            n_cases = len(L["oos"])
            base = float(np.mean([r[4] > 0 for r in L["oos"] if np.isfinite(r[4])])) if L["oos"] else None
            record = {"g": cal["g"][0], "w": cal["w"][0], "i": cal["i"][0], "m": cal["m"][0], "stars": model.star_record(L["oos"]),
                      "spy": round(float(np.mean(L["spy_fwds"])), 4) if L["spy_fwds"] else None,
                      "from": L["tests"][0] if L["tests"] else "", "to": L["tests"][-1] if L["tests"] else "",
                      "months": 3, "method": "model", "cases": n_cases, "snapshots": len(L["tests"]),
                      "base": round(base, 3) if base is not None else None,
                      "ic": {k: round(float(np.mean(v)), 3) for k, v in L["ics"].items() if v},
                      "ic_by_model": {m: {k: round(float(np.mean(v)), 3) for k, v in d.items() if v} for m, d in L["ics_by"].items()},
                      "weights": {"g": L["w_g"], "w": L["w_w"]},
                      "note": (f"Tested on {len(L['tests'])} monthly snapshots it never trained on "
                               f"({L['tests'][0]} to {L['tests'][-1]}), {n_cases:,} cases, each held 3 months. "
                               f"An average stock beat the S&P 500 {base:.0%} of the time." if L["tests"] else "")}
            print(f"Grades: learned model tested on {len(L['tests'])} snapshots, {n_cases} cases; "
                  f"rank correlation growth {record['ic'].get('g')}, wealth {record['ic'].get('w')} "
                  f"(straight-line model {record['ic_by_model']['ridge']}, trees {record['ic_by_model']['trees']})")
            for goal in ("g", "w", "i", "m"):
                print(f"  {goal}: " + ", ".join(f"{b}: beat {v['beat']:.0%}, median {v['ret']:+.1%} vs S&P, drop {v['dd']:.0%} (n={v['n']})"
                                               for b, v in record[goal].items()))
            print(f"  next big thing: {record['stars']}")
            return {"date": today.isoformat(), "fy": fy, "graded": len(grades), "grades": grades,
                    "stars": star_list, "record": record}
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"::warning title=Learned grades failed, using the rule-based ones::{e}")

    # ---- track record: the same rules as of a year ago, and what happened next ----
    record = {}
    HOLD = 126   # six months: the price measures need a year of history before the grading day
    t0 = n - 1 - HOLD
    if t0 >= 260:
        print("Grades: checking how last year's grades did")
        # Six months ago the same fiscal year's annual reports were already out (they're filed by about March).
        fund0 = fundamentals(sec, companies, fy, [f"CY{fy}Q4I"])
        p0 = prices_at(t0)
        g0, stars0, rows0 = grade_all(companies, fund0, p0)
        spy_fwd = float(spy_c[n - 1] / spy_c[t0] - 1)
        fwd = {}
        for tk in g0:
            C = aligned(tk)[0]
            if np.isfinite(C[t0]) and C[t0] > 0 and np.isfinite(C[n - 1]):
                fwd[tk] = float(C[n - 1] / C[t0] - 1)
        # wealth preservation is about holding up: also measure the worst drop over the year
        dd = {}
        for tk in g0:
            C = aligned(tk)[0][t0:]
            if np.all(np.isfinite(C)):
                dd[tk] = float(np.min(C / np.maximum.accumulate(C) - 1))
        spy_win = spy_c[t0:]
        spy_dd = float(np.min(spy_win / np.maximum.accumulate(spy_win) - 1))
        for goal in ("g", "i", "w"):
            buckets = {}
            for tk, e in g0.items():
                if goal in e and tk in fwd and (goal != "i" or e.get("ip") is not None):
                    buckets.setdefault(e[goal][0], []).append(tk)
            record[goal] = {b: {"n": len(v), "ret": round(float(np.median([fwd[x] for x in v])), 4),
                                "beat": round(float(np.mean([fwd[x] > spy_fwd for x in v])), 3),
                                "dd": round(float(np.median([dd[x] for x in v if x in dd])), 4) if any(x in dd for x in v) else None}
                            for b, v in sorted(buckets.items())}
        sic0 = sec.industries([rows0[x]["cik"] for x in stars0[:120]])
        stars0 = [x for x in stars0 if not 6000 <= sic0.get(rows0[x]["cik"], (0, ""))[0] <= 6799]
        s0 = [x for x in stars0[:30] if x in fwd]
        record["stars"] = {"n": len(s0), "ret": round(float(np.median([fwd[x] for x in s0])), 4) if s0 else None,
                           "beat": round(float(np.mean([fwd[x] > spy_fwd for x in s0])), 3) if s0 else None,
                           "picks": [[x, round(fwd[x], 3)] for x in s0[:10]]}
        record["spy"] = round(spy_fwd, 4)
        record["spy_dd"] = round(spy_dd, 4)
        record["from"] = days[t0].isoformat()
        record["months"] = 6

    return {"date": today.isoformat(), "fy": fy, "graded": len(grades), "grades": grades,
            "stars": star_list, "record": record}

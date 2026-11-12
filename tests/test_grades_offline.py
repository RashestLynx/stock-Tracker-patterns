"""Grades end to end on a fake market and fake SEC filings (no network)."""
import os, sys, tempfile, gzip, json
import numpy as np
from datetime import date, timedelta
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import grades, scan

rng = np.random.default_rng(1)
N = 120
tick = ["SPY"] + [f"G{chr(65 + i % 26)}{chr(65 + i // 26)}" for i in range(N)]
# Business quality and dividends per company; good growers also get a price drift so grades should line up.
growth = {t: rng.uniform(-0.1, 0.5) for t in tick}
payer = {t: rng.random() < 0.5 for t in tick}
days, data, px = [], [], {t: 50.0 for t in tick}
d = date.today() - timedelta(days=730)
while d <= date.today():
    if d.weekday() < 5:
        mkt = rng.normal(0.0004, 0.01)
        rows = {}
        for t in tick:
            beta = 1.0 if t == "SPY" else (0.5 if payer[t] else 1.4)
            drift = 0 if t == "SPY" else growth[t] / 252
            px[t] *= 1 + beta * mkt + drift + (0 if t == "SPY" else rng.normal(0, 0.012))
            p = px[t]
            rows[t] = (p, p * 1.01, p * 0.99, p, 1e6)
        days.append(d); data.append(rows)
    d += timedelta(days=1)
series = scan.series(days, data)

companies = [(1000 + i, t, f"Company {t}") for i, t in enumerate(tick[1:])]
FY = date.today().year - 1
def fake_frame(self, tag, period):
    out = {}
    for cik, t, _ in companies:
        g = growth[t]
        if period.startswith("CY") and "Q" not in period:
            yr = int(period[2:6]); k = yr - (FY - 9)
            rev = 1e9 * (1 + g) ** k
            ni = rev * (0.12 if g > 0 else -0.02)
            v = {"Revenues": rev, "NetIncomeLoss": ni, "EarningsPerShareDiluted": ni / 1e8,
                 "WeightedAverageNumberOfDilutedSharesOutstanding": 1e8, "OperatingIncomeLoss": ni * 1.3,
                 "GrossProfit": rev * 0.5, "NetCashProvidedByUsedInOperatingActivities": ni * 1.2,
                 "PaymentsToAcquirePropertyPlantAndEquipment": ni * 0.2,
                 "PaymentsOfDividendsCommonStock": (ni * 0.4 if payer[t] and ni > 0 else None)}.get(tag)
        else:
            v = {"Assets": 3e9, "StockholdersEquity": 1.5e9}.get(tag)
        if v is not None:
            out[cik] = float(v)
    if tag == "NetIncomeLoss":   # pad so the 2500-company check passes
        out.update({900000 + i: 1.0 for i in range(2600)})
    return out
grades.Sec.frame = fake_frame
grades.Sec.companies = lambda self: companies
grades.Sec.industries = lambda self, ciks: {c: (7372, "Software") for c in ciks}

g = grades.run(tempfile.mkdtemp(), days, series, insiders={companies[0][1]: 1})
print("graded", g["graded"], "stars", len(g["stars"]))
assert g["graded"] > 100, g["graded"]
gr = g["grades"]
# Fast growers should get better growth grades than shrinking companies.
fast = [gr[t]["gp"] for t in gr if growth[t] > 0.35 and "gp" in gr[t]]
slow = [gr[t]["gp"] for t in gr if growth[t] < 0.0 and "gp" in gr[t]]
print("growth percentile: fast", np.mean(fast), "shrinking", np.mean(slow))
assert np.mean(fast) > np.mean(slow) + 20
# Non-payers get F for income; payers get real grades.
assert all(gr[t]["i"] == "F" for t in gr if not payer[t])
assert any(gr[t]["i"].startswith("A") for t in gr if payer[t])
# Low-beta payers should hold up better: better wealth grades on average.
wp = [gr[t]["wp"] for t in gr if payer[t] and "wp" in gr[t]]; wn = [gr[t]["wp"] for t in gr if not payer[t] and "wp" in gr[t]]
print("wealth percentile: steady payers", np.mean(wp), "others", np.mean(wn))
assert np.mean(wp) > np.mean(wn)
assert all(s["why"] for s in g["stars"])
rec = g["record"]
print("record", json.dumps({k: rec[k] for k in ("g", "spy", "from")})[:400])
assert "g" in rec and rec["g"], rec
print(json.dumps(gr[companies[3][1]], indent=0)[:500])
print("grades ok")

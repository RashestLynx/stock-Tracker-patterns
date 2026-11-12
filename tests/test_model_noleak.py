"""On a market of pure coin flips, the walk-forward test must find no skill (no peeking at the future)."""
import os, sys
import numpy as np
from datetime import date, timedelta
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import model

rng = np.random.default_rng(7)
tick = ["SPY"] + [f"N{chr(65 + i % 26)}{chr(65 + (i // 26) % 26)}{chr(65 + i // 676)}" for i in range(800)]
days, data, px = [], [], {t: 40.0 for t in tick}
d = date(2024, 11, 1)
while len(days) < 500:
    if d.weekday() < 5:
        mkt = rng.normal(0.0003, 0.01)
        row = {}
        for t in tick:
            px[t] *= 1 + mkt + (0 if t == "SPY" else rng.normal(0, 0.02))
            row[t] = (px[t], px[t], px[t], px[t], 1e6)
        days.append(d); data.append(row)
    d += timedelta(days=1)
companies = [(i, t, t) for i, t in enumerate(tick[1:])]
fund = {i: {"rev_g1": rng.normal(0.1, 0.2), "rev_g_prev": rng.normal(0.1, 0.2), "roe": rng.normal(0.1, 0.1),
            "shares": 1e8, "ni": {2024: 1e7}, "rev": {2024: 1e8}, "div": {}, "equity": 5e8} for i, _, _ in companies}
L = model.learn(days, data, companies, {2023: fund, 2024: fund, 2025: fund})
ic_g, ic_w = np.mean(L["ics"]["g"]), np.mean(L["ics"]["w"])
by, _ = model.calibration(L["oos"], 0)
print("snapshots", len(L["tests"]), "IC growth", round(ic_g, 3), "IC wealth", round(ic_w, 3))
print({k: v["beat"] for k, v in by.items()})
assert abs(ic_g) < 0.04, ic_g
assert max(v["beat"] for v in by.values()) - min(v["beat"] for v in by.values()) < 0.12, by
print("no leak ok")

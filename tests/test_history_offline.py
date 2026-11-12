"""Long history merge: Nasdaq years + Massive days, with a split at the seam, and the model on the merged panel."""
import gzip, json, math, os, sys, tempfile
import numpy as np
from datetime import date, timedelta
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import history, model

tmp = tempfile.mkdtemp(); os.makedirs(os.path.join(tmp, "hist"))
rng = np.random.default_rng(3)
cal = []; d = date(2020, 1, 2)
while len(cal) < 900:
    if d.weekday() < 5: cal.append(d)
    d += timedelta(days=1)
old, new = cal[:600], cal[600:]
tick = ["SPY", "AAA", "BBB", "CCC"]
px = {t: 100.0 for t in tick}; full = {t: [] for t in tick}
for _ in cal:
    for t in tick:
        px[t] *= 1 + rng.normal(0.0003, 0.015); full[t].append(px[t])
# Nasdaq files: split-adjusted. AAA did a 2-for-1 split on the 10th Massive day, so Nasdaq's old prices are halved.
for t in tick:
    adj = 0.5 if t == "AAA" else 1.0
    h = {"d": [x.isoformat() for x in cal[:600]], "c": [p * adj for p in full[t][:600]], "v": [1e6] * 600}
    with gzip.open(history.path(tmp, t), "wt") as f: json.dump(h, f)
# Massive days (cached raw): AAA shows the pre-split price for its first 10 days, then halves.
data = []
for k, day in enumerate(new):
    row = {}
    for t in tick:
        p = full[t][600 + k] * (0.5 if (t == "AAA" and k >= 10) else 1.0)
        row[t] = (p, p, p, p, 1e6)
    data.append(row)
dates, C, V = history.merged_panel(tmp, new, data, tick)
assert len(dates) == 900 and dates[0] == cal[0], (len(dates), dates[0])
C2, V2, first = model.clean_panel(C.copy(), V.copy())
a = C2[tick.index("AAA")]
jumps = np.abs(np.log(a[1:] / a[:-1]))
print("largest AAA day move after cleaning:", round(float(jumps.max()), 3))
assert jumps.max() < 0.2, jumps.max()
# ratio vs truth is constant (one consistent adjustment)
ratio = a / np.array(full["AAA"])
assert np.nanstd(ratio) / np.nanmean(ratio) < 1e-6, (ratio.min(), ratio.max())
print("history ok")

"""Offline checks for the covered-call history test."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import covered

# Black-Scholes sanity: the textbook value
p = covered.call_price(np.array([100.0]), np.array([100.0]), 0.2, 1.0, 0.05)[0]
assert abs(p - 10.4506) < 0.01, p

rng = np.random.default_rng(1)
n = 2600
def path(drift, vol, k):
    r = rng.normal(drift / 252, vol / np.sqrt(252), (k, n))
    return 50 * np.exp(np.cumsum(r, axis=1))

flat = path(0.0, 0.25, 200)      # going nowhere: the premium should add return
rocket = path(0.45, 0.25, 200)   # strong grower: capping the upside should cost return
C = np.vstack([flat, rocket])
C[5, :1500] = np.nan             # a stock with only a few years of history
tick = [f"F{i}" for i in range(200)] + [f"R{i}" for i in range(200)]
V = np.full_like(C, 1e8); V[:50] = 10.0                 # first 50 stocks barely trade
out = covered.run(tick, C, V=V)
rows = out["rows"]
fl = np.array([rows[t] for t in tick[:200] if t in rows]); ro = np.array([rows[t] for t in tick[200:]])
j = {c: k for k, c in enumerate(covered.COLS)}
beat_flat = np.mean(fl[:, j["balanced"]] > fl[:, j["hold"]])
beat_rocket = np.mean(ro[:, j["balanced"]] > ro[:, j["hold"]])
print("flat stocks: calls beat holding", round(beat_flat, 2), "| fast growers:", round(beat_rocket, 2))
print("called away (balanced) flat", round(np.median(fl[:, j["called_balanced"]]), 2), "rocket", round(np.median(ro[:, j["called_balanced"]]), 2))
print("summary", out["summary"])
assert beat_flat > beat_rocket + 0.3, (beat_flat, beat_rocket)
assert beat_rocket < 0.3, beat_rocket
assert 0.15 < np.median(fl[:, j["called_balanced"]]) < 0.35          # delta 0.25 -> called about a quarter of the time
assert np.median(fl[:, j["called_safer"]]) < np.median(fl[:, j["called_balanced"]])
assert np.median(fl[:, j["drop_balanced"]]) > np.median(fl[:, j["drop_hold"]])   # smaller worst drop
assert rows["F5"][j["years"]] < 5
assert out["summary"]["stocks"] == len([t for t in rows if not (t.startswith("F") and int(t[1:]) < 50)])   # thin stocks left out
print("covered ok")

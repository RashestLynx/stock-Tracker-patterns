import sys, time, numpy as np
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), ".."))
from patterns import detect, PATTERNS

def make(closes, noise=0.004, seed=1, vol=None):
    rng = np.random.default_rng(seed)
    c = np.array(closes, float)
    o = np.concatenate([[c[0]], c[:-1]]) * (1 + rng.normal(0, noise, len(c)))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, noise, len(c))))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, noise, len(c))))
    v = np.full(len(c), 1e6) if vol is None else np.array(vol, float)
    return o, h, l, c, v

def first_hit(name, arrs):
    d = detect(*arrs)
    idx = np.where(d[name])[0]
    return idx

# Double bottom: fall to 80, rally to 90, fall to 80, rally to 95
seq = list(np.linspace(100, 80, 40)) + list(np.linspace(80, 90, 20)) + list(np.linspace(90, 80.5, 20)) + list(np.linspace(80.5, 95, 25))
print("double_bottom at", first_hit("double_bottom", make(seq)), "(rally passes 90 around day ~98)")
# Head and shoulders top
seq = list(np.linspace(80, 100, 30)) + list(np.linspace(100, 90, 15)) + list(np.linspace(90, 110, 15)) + list(np.linspace(110, 90, 15)) + list(np.linspace(90, 100, 15)) + list(np.linspace(100, 80, 25))
print("head_shoulders at", first_hit("head_shoulders", make(seq)))
# Ascending triangle: flat top at 100, rising lows, then breakout
seq = []
lows = [80, 85, 90, 94]
for lo_ in lows:
    seq += list(np.linspace(lo_, 100, 8)) + list(np.linspace(100, lo_ + 4, 8))
seq += list(np.linspace(seq[-1], 106, 6))
print("asc_triangle at", first_hit("asc_triangle", make([60]*0 + list(np.linspace(70, 80, 20)) + seq)))
# Bull flag
seq = list(np.full(40, 50.0) + np.sin(np.arange(40)) * 0.3) + list(np.linspace(50, 58, 8)) + list(np.linspace(58, 56.5, 8)) + list(np.linspace(56.5, 60, 4))
print("bull_flag at", first_hit("bull_flag", make(seq, noise=0.002)))
# Cup and handle
seq = list(np.full(30, 100.0)) + list(100 - 20 * np.sin(np.linspace(0, np.pi, 60))) + list(np.linspace(100, 94, 6)) + list(np.linspace(94, 99, 6)) + list(np.linspace(99, 104, 4))
print("cup_handle at", first_hit("cup_handle", make(seq, noise=0.002)))
# Golden cross on a decline then rise
seq = list(np.linspace(100, 60, 220)) + list(np.linspace(60, 120, 120))
print("golden_cross at", first_hit("golden_cross", make(seq)))
# Hammer after decline
o,h,l,c,v = make(list(np.linspace(100, 85, 30)) + [84.0]*0)
o[-1], c[-1], h[-1], l[-1] = 85.0, 85.4, 85.5, 81.0
print("hammer last day:", detect(o,h,l,c,v)["hammer"][-1])
# Fuzz + speed
rng = np.random.default_rng(7)
t = time.time(); total = 0
for k in range(100):
    n = int(rng.integers(60, 520))
    c = 50 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    d = detect(*make(c, seed=k, vol=rng.uniform(1e5, 5e6, n) * np.where(rng.random(n) < 0.03, 5, 1)))
    assert set(d) <= set(PATTERNS), set(d) - set(PATTERNS)
    total += sum(int(x.sum()) for x in d.values())
dt = time.time() - t
print(f"fuzz ok: 100 random stocks, {total} signals, {dt/100*1000:.0f} ms per stock -> ~{dt/100*4000/60:.0f} min for 4,000")
missing = set(PATTERNS) - set(d) - {k for k in PATTERNS if PATTERNS[k][1] == 'Smart money'}
print("detectors missing:", missing)

"""
Learned grades: instead of hand-picked weights, the weights are fit to history and the accuracy is measured on months
the model never saw.

How it works:
  1. Snapshots: every 21 trading days, for every stock that traded that day (including ones that later stopped trading,
     so failures count), ~20 measures: price trends, how bumpy and how deep its drops were, and business numbers from
     the SEC filings that were public at that time.
  2. Each measure becomes a rank from -0.5 to +0.5 among that day's stocks (robust to outliers and units).
  3. Ridge regression (a linear model that resists overfitting) learns how much each measure mattered for:
       growth -> return over the next 63 trading days (~3 months), beyond the average stock
       wealth -> how small the worst drop was over the next 63 trading days
  4. Walk-forward test: for each snapshot, the model is refit using only snapshots whose 3-month outcome was already
     known by then, and it grades that snapshot. Those out-of-sample grades are compared with what really happened.
     That gives each letter grade a measured hit rate, which becomes each stock's probability.
  5. The model is then fit on everything and grades today's stocks.

Limits: only as much history as the price data allows (2 years on the free plan, so ~9 test snapshots from one market
period). Prices of stocks that stopped trading are held at their last price (a bankruptcy's true loss can be worse).
"""
import math
import re

import numpy as np

HORIZON = 63          # trading days ahead (~3 months)
STEP = 21             # snapshot every ~month
LOOKBACK = 126        # trading days of history the price measures need
MIN_TRAIN = 4         # snapshots needed before the first out-of-sample test
EMBARGO = STEP        # extra gap between the last training outcome and a test snapshot (purge + embargo)
GBM_REFIT = 6         # refit the boosted-tree model every 6 test snapshots (it's slower than ridge)
TICKER = re.compile(r"^[A-Z]{1,5}(\.[A-Z])?$")
SPLITS = np.array([2, 3, 4, 5, 8, 10, 15, 20, 1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 8, 1 / 10, 1 / 15, 1 / 20, 3 / 2, 2 / 3])

FEATURES = {   # name -> (what a high rank means, what a low rank means), for plain-English reasons
    "mom6": ("a strong 6-month trend", "a weak 6-month trend"), "mom1": ("a rise last month", "a drop last month"),
    "vol": ("calm day-to-day moves", "big day-to-day swings"),
    "down_cap": ("staying steady when the market falls", "swinging hard with the market"),
    "max_dd": ("small recent drops", "a deep recent drop"), "near_high": ("trading near its high", "trading far below its high"),
    "size": ("being a large company", "being a small company"), "liquidity": ("heavy trading", "light trading"),
    "rev_g1": ("fast sales growth", "slow or falling sales"), "accel": ("growth speeding up", "growth slowing down"),
    "eps_g": ("rising earnings", "falling earnings"), "margin_trend": ("widening margins", "shrinking margins"),
    "gross_prof": ("high gross profit vs assets", "low gross profit vs assets"), "roe": ("a high return on equity", "a low return on equity"),
    "fcf_margin": ("strong free cash flow", "weak free cash flow"), "fcf_conv": ("profits backed by cash", "profits not backed by cash"),
    "dilution": ("few new shares issued", "many new shares issued"), "equity_ratio": ("low debt", "high debt"),
    "consistency": ("a steady growth record", "an uneven growth record"), "ey": ("a cheap price vs earnings", "a high price vs earnings"),
    "bm": ("a cheap price vs book value", "a high price vs book value"), "sp": ("a cheap price vs sales", "a high price vs sales"),
    "yield": ("a high dividend yield", "a low or no dividend"),
    "mom12": ("a strong 12-month trend", "a weak 12-month trend"),
    "sue": ("a big jump in quarterly earnings", "a drop in quarterly earnings"),
    "asset_g": ("modest asset growth", "fast asset growth (often a warning sign)"),
    "accruals": ("profits backed by cash flow", "profits running ahead of cash flow"),
}
# measures where lower raw numbers are better are flipped so +rank always means "more of the good thing"
FLIP = {"vol", "down_cap", "dilution", "asset_g", "accruals"}


def build_panel(days, data):
    """Close/volume matrices (stocks x days) for every ticker in the Massive days, split-adjusted, forward-filled."""
    tickers = sorted({t for d in data for t in d if TICKER.match(t)})
    ix = {t: i for i, t in enumerate(tickers)}
    n, N = len(days), len(tickers)
    C = np.full((N, n), np.nan, np.float64)
    V = np.zeros((N, n), np.float32)
    for j, d in enumerate(data):
        for t, row in d.items():
            i = ix.get(t)
            if i is not None and row[3] > 0:
                C[i, j] = row[3]
                V[i, j] = row[4]
    C, V, first = clean_panel(C, V)
    return tickers, C, V, first


def clean_panel(C, V):
    """Undoes jumps that exactly match a stock split (cached days and the Nasdaq/Massive seam aren't adjusted alike),
    then carries the last price over missing days and after a stock stops trading."""
    N, n = C.shape
    for i in range(N):
        row = C[i]
        ok = np.where(np.isfinite(row))[0]
        if len(ok) < 2:
            continue
        r = row[ok[:-1]] / row[ok[1:]]
        cand = np.where(np.abs(np.log(r)) > math.log(1.4))[0]
        for k in cand:
            hit = SPLITS[np.abs(r[k] / SPLITS - 1) < 0.03]
            if len(hit):
                s = hit[0]
                row[:ok[k + 1]] /= s
                V[i, :ok[k + 1]] *= s
    good = np.isfinite(C)
    first = np.where(good.any(axis=1), good.argmax(axis=1), n)
    idx = np.where(good, np.arange(n), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    C = C[np.arange(N)[:, None], idx]
    C[np.arange(n)[None, :] < first[:, None]] = np.nan
    return C, V, first


def ranks(x, mask):
    """Rank from -0.5 to +0.5 among rows in mask; missing -> 0 (the middle)."""
    out = np.zeros(len(x))
    good = mask & np.isfinite(x)
    k = good.sum()
    if k > 1:
        order = x[good].argsort().argsort()
        out[good] = order / (k - 1) - 0.5
    return out


def price_features(C, V, spy, t, first):
    """Price measures for every stock as of day t (arrays)."""
    w0 = t - LOOKBACK
    win = C[:, w0:t + 1]
    R = win[:, 1:] / win[:, :-1] - 1
    s = spy[w0:t + 1]
    sr = s[1:] / s[:-1] - 1
    down = sr < 0
    f = {
        "mom6": C[:, t - 21] / C[:, w0] - 1,
        "mom1": C[:, t] / C[:, t - 21] - 1,
        "mom12": (C[:, t - 21] / C[:, t - 252] - 1) if t >= 252 else np.full(len(C), np.nan),
        "vol": np.nanstd(R, axis=1),
        "down_cap": np.nansum(R[:, down], axis=1) / sr[down].sum() if down.any() else np.full(len(C), np.nan),
        "max_dd": np.nanmin(win / np.fmax.accumulate(win, axis=1) - 1, axis=1),
        "near_high": C[:, t] / np.nanmax(win, axis=1),
        "liquidity": np.log1p(np.nanmean(C[:, t - 19:t + 1] * V[:, t - 19:t + 1], axis=1)),
        "price": C[:, t],
        "dollar_vol": np.nanmean(C[:, t - 19:t + 1] * V[:, t - 19:t + 1], axis=1),
    }
    has = (first <= w0) & np.isfinite(C[:, t])
    return f, has


def fund_features(tickers, cik_of, fund, price):
    """Business measures from one fiscal year's filings, valued at today's price (arrays)."""
    N = len(tickers)
    out = {k: np.full(N, np.nan) for k in ("rev_g1", "accel", "eps_g", "margin_trend", "gross_prof", "roe", "fcf_margin",
                                           "fcf_conv", "dilution", "equity_ratio", "consistency", "ey", "bm", "sp", "yield", "size",
                                           "asset_g", "accruals", "sue")}
    for i, t in enumerate(tickers):
        f = fund.get(cik_of.get(t))
        if not f or not np.isfinite(price[i]):
            continue
        for k in ("rev_g1", "eps_g", "margin_trend", "gross_prof", "roe", "fcf_margin", "fcf_conv", "dilution",
                  "equity_ratio", "consistency", "asset_g", "accruals"):
            v = f.get(k)
            if isinstance(v, (int, float)):
                out[k][i] = v
        if f.get("rev_g1") is not None and f.get("rev_g_prev") is not None:
            out["accel"][i] = f["rev_g1"] - f["rev_g_prev"]
        sh = f.get("shares")
        if sh and sh > 0:
            cap = sh * price[i]
            if cap > 0:
                y = max(f["ni"])
                out["size"][i] = math.log(cap)
                out["ey"][i] = f["ni"][y] / cap
                if f.get("equity") is not None:
                    out["bm"][i] = f["equity"] / cap
                if f["rev"].get(y):
                    out["sp"][i] = f["rev"][y] / cap
                d = f["div"].get(y)
                out["yield"][i] = (d / cap) if d else 0.0
    return out


def quarter_for(d):
    """(year, quarter) of the newest quarter whose 10-Q was public on date d (quarter end + 50 days)."""
    from datetime import timedelta
    q = d - timedelta(days=50)
    return q.year, (q.month - 1) // 3 + 1


def sue_feature(tickers, cik_of, eps_q, d, price):
    """(EPS this quarter - EPS same quarter a year before) / price: the 'earnings surprise' behind post-earnings drift.
    eps_q: {(year, quarter): {cik: eps}}. Uses the newest quarter that was public on date d (quarter end + 50 days),
    falling back one quarter if a company hasn't filed it."""
    out = np.full(len(tickers), np.nan)
    if not eps_q:
        return out
    y, q = quarter_for(d)
    tries = [(y, q), (y - (q == 1), 4 if q == 1 else q - 1)]
    for i, t in enumerate(tickers):
        c = cik_of.get(t)
        if c is None or not np.isfinite(price[i]) or price[i] <= 0:
            continue
        for (yy, qq) in tries:
            now, before = eps_q.get((yy, qq), {}).get(c), eps_q.get((yy - 1, qq), {}).get(c)
            if now is not None and before is not None:
                out[i] = (now - before) / price[i]
                break
    return out


def fy_for(d):
    """Fiscal year whose annual reports were public on date d (10-Ks are filed by about mid-April)."""
    return d.year - 1 if (d.month, d.day) >= (4, 15) else d.year - 2


MOAT = ["gross_prof", "roe", "fcf_conv", "equity_ratio", "consistency"]   # a durable business, in the filings


def moat_score(X, names, uni):
    """Our own moat score: the average rank of high gross profit vs assets, high return on equity, profits backed by
    cash, low debt and a steady growth record. A fixed formula (nothing fit), so testing it can't overfit.
    0-1 among stocks with filings; nan for those without."""
    cols = [names.index(m) for m in MOAT]
    sub = X[:, cols]
    known = (sub != 0).sum(axis=1) >= 3   # a rank of exactly 0 means 'missing'
    out = np.full(len(X), np.nan)
    m = uni & known
    if m.sum() > 1:
        out[m] = ranks(np.where(m, sub.mean(axis=1), np.nan), m)[m] + 0.5
    return out


def gbm_fit(train, target):
    """Gradient-boosted trees on the same ranked measures (catches interactions a straight line can't).
    Returns a predict function, or None if LightGBM isn't installed."""
    try:
        import lightgbm as lgb
    except ImportError:
        return None
    Xs = np.vstack([s["X"][s["uni"]] for s in train])
    ys = np.concatenate([s[target][s["uni"]] for s in train])
    if len(ys) > 400_000:   # keep fits fast: a random sample is plenty for ~23 measures
        pick = np.random.default_rng(0).choice(len(ys), 400_000, replace=False)
        Xs, ys = Xs[pick], ys[pick]
    try:
        return _gbm(lgb, Xs, ys)
    except Exception as e:   # trees are optional: never let them stop the grades
        print(f"  boosted trees skipped: {e}")
        return None


def _gbm(lgb, Xs, ys):
    m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=500,
                          subsample=0.5, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, verbose=-1)
    m.fit(Xs, ys)
    return m.predict


def ridge(X, y, lam):
    k = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(k), X.T @ y)


def learn(days, data, companies, fund_by_fy, spy_ticker="SPY", panel=None, eps_q=None):
    """Returns dict with today's model scores, weights, and out-of-sample results. See module docstring.
    panel: optional (dates, tickers, C, V) with longer history; otherwise built from the Massive days."""
    if panel is not None:
        days, tickers, C, V = panel
        C, V, first = clean_panel(C, V)
    else:
        tickers, C, V, first = build_panel(days, data)
    n = len(days)
    si = tickers.index(spy_ticker)
    spy = C[si]
    cik_of = {t: cik for cik, t, _ in companies}
    names = list(FEATURES)
    snaps = []
    dates = list(range(LOOKBACK, n, STEP))
    if dates[-1] != n - 1:
        dates.append(n - 1)
    for t in dates:
        with np.errstate(all="ignore"):
            pf, has = price_features(C, V, spy, t, first)
        fund = fund_by_fy.get(fy_for(days[t]), {})
        ff = fund_features(tickers, cik_of, fund, pf["price"])
        uni = has & (pf["price"] >= 3) & (pf["dollar_vol"] >= 1e6) & (np.arange(len(tickers)) != si)
        uni &= ~(np.isfinite(ff["size"]) & (ff["size"] < math.log(1e8)))   # skip companies under $100M when size is known
        ff["sue"] = sue_feature(tickers, cik_of, eps_q, days[t], pf["price"])
        allf = {**pf, **ff}
        X = np.column_stack([(-1 if k in FLIP else 1) * ranks(allf[k], uni) for k in names])
        snap = {"t": t, "uni": uni, "X": X, "raw": allf}
        if t + HORIZON <= n - 1:
            fwd = C[:, t + HORIZON] / C[:, t] - 1
            path = C[:, t:t + HORIZON + 1] / C[:, t:t + 1]
            snap["fwd"] = fwd
            snap["spy_fwd"] = spy[t + HORIZON] / spy[t] - 1
            snap["dd"] = np.nanmin(path, axis=1) - 1
            snap["y_g"] = ranks(fwd, uni)
            snap["y_w"] = ranks(snap["dd"], uni)
        snaps.append(snap)

    # Each snapshot's X'X and X'y, so refitting on any set of snapshots is just a sum.
    for s in snaps:
        if "fwd" in s:
            Xu = s["X"][s["uni"]]
            s["xtx"], s["m"] = Xu.T @ Xu, len(Xu)
            s["xty_g"], s["xty_w"] = Xu.T @ s["y_g"][s["uni"]], Xu.T @ s["y_w"][s["uni"]]

    NEW = {"mom12", "sue", "asset_g", "accruals"}
    base_idx = [i for i, n in enumerate(names) if n not in NEW]

    def fit(train, target, cols=None):
        A = sum(s["xtx"] for s in train)
        b = sum(s["xty_" + target[-1]] for s in train)
        m = sum(s["m"] for s in train)
        if cols is not None:   # the same model on a subset of measures (to check what the new ones add)
            A, b = A[np.ix_(cols, cols)], b[cols]
            w = np.zeros(len(names))
            w[cols] = np.linalg.solve(A + 0.05 * m * np.eye(len(cols)), b)
            return w
        return np.linalg.solve(A + 0.05 * m * np.eye(A.shape[0]), b)

    # ---- walk-forward test ----
    oos = []          # one row per stock per test snapshot
    ics = {"g": [], "w": []}
    ics_by = {m: {"g": [], "w": []} for m in ("ridge", "trees", "without_new")}
    gbm = {}
    tests, spy_fwds = [], []

    def ic(a, b):
        return float(np.corrcoef(a.argsort().argsort(), b.argsort().argsort())[0, 1])
    for k, s in enumerate(snaps):
        if "fwd" not in s:
            continue
        # Purge + embargo: only snapshots whose 3-month outcome was over a month before this one.
        train = [p for p in snaps[:k] if "fwd" in p and p["t"] + HORIZON + EMBARGO <= s["t"]]
        if len(train) < MIN_TRAIN:
            continue
        tests.append(days[s["t"]].isoformat())
        spy_fwds.append(float(s["spy_fwd"]))
        u = s["uni"]
        pr = {}
        refit = (len(tests) - 1) % GBM_REFIT == 0
        for goal, target in (("g", "y_g"), ("w", "y_w")):
            w = fit(train, target)
            r_ridge = ranks(s["X"] @ w, u)
            if refit or goal not in gbm:
                gbm[goal] = gbm_fit(train, target)
            both = r_ridge
            if gbm[goal] is not None:
                r_tree = ranks(gbm[goal](s["X"]), u)
                # Use the trees only once they've proven themselves on EARLIER test snapshots (no peeking).
                prior_t, prior_r = ics_by["trees"][goal], ics_by["ridge"][goal]
                if len(prior_t) >= 3 and np.mean(prior_t) > 0.5 * np.mean(prior_r) and np.mean(prior_t) > 0:
                    both = ranks(r_ridge + r_tree, u)   # average of the two models' ranks
                ics_by["trees"][goal].append(ic(r_tree[u], s[target][u]))
            ics_by["ridge"][goal].append(ic(r_ridge[u], s[target][u]))
            ics_by["without_new"][goal].append(ic(ranks(s["X"] @ fit(train, target, base_idx), u)[u], s[target][u]))
            pr[goal] = both + 0.5
            ics[goal].append(ic(pr[goal][u], s[target][u]))
        inc = income_score(s["raw"]["yield"], pr["g"], pr["w"], u)
        star = star_pick(s["raw"], pr["g"], u)
        moat = moat_score(s["X"], names, u)
        for i in np.where(u)[0]:
            y = s["raw"]["yield"][i]
            oos.append((pr["g"][i], pr["w"][i], inc[i], float(star[i]), s["fwd"][i] - s["spy_fwd"], s["dd"][i],
                        (s["fwd"][i] + (y if np.isfinite(y) else 0) * HORIZON / 252) - s["spy_fwd"], moat[i]))

    # ---- final fit on everything known, grade today ----
    known = [s for s in snaps if "fwd" in s]
    today = snaps[-1]
    w_g, w_w = fit(known, "y_g"), fit(known, "y_w")
    u = today["uni"]
    pg, pw = ranks(today["X"] @ w_g, u), ranks(today["X"] @ w_w, u)
    for goal in ("g", "w"):
        t_ic, r_ic = ics_by["trees"][goal], ics_by["ridge"][goal]
        if not t_ic or not (np.mean(t_ic) > 0.5 * np.mean(r_ic) and np.mean(t_ic) > 0):
            continue   # trees didn't earn a place in testing
        f = gbm_fit(known, "y_" + goal)
        if f is not None:
            r = ranks(f(today["X"]), u)
            if goal == "g":
                pg = ranks(pg + r, u)
            else:
                pw = ranks(pw + r, u)
    contrib_g = today["X"] * w_g
    pi = income_score(today["raw"]["yield"], pg + 0.5, pw + 0.5, u)
    pm = moat_score(today["X"], names, u)
    return {"tickers": tickers, "uni": u, "pg": pg + 0.5, "pw": pw + 0.5, "pi": pi, "pm": pm, "X": today["X"], "raw": today["raw"],
            "spy_fwds": spy_fwds, "ics_by": ics_by, "ics": ics, "C": C, "days": days,
            "w_g": dict(zip(names, w_g.round(4).tolist())), "w_w": dict(zip(names, w_w.round(4).tolist())),
            "contrib_g": contrib_g, "contrib_w": today["X"] * w_w, "names": names, "oos": oos, "tests": tests,
            "snaps": snaps, "C": C}


def income_score(yld, pg, pw, uni):
    """Dividend payers only: half yield (known today), a quarter each predicted return and predicted steadiness.
    0-1 rank among payers; nan for non-payers."""
    out = np.full(len(yld), np.nan)
    payers = uni & np.isfinite(yld) & (yld > 0)
    if payers.sum() > 1:
        score = 0.5 * ranks(yld, payers) + 0.25 * (pg - 0.5) + 0.25 * (pw - 0.5)
        out[payers] = ranks(score, payers)[payers] + 0.5
    return out


def star_pick(raw, pg, uni, top=30):
    """'Next big thing': $300M-$15B companies with sales up 20%+, the model's top return picks."""
    size = raw["size"]
    cand = uni & np.isfinite(size) & (size >= math.log(3e8)) & (size <= math.log(1.5e10)) & (np.nan_to_num(raw["rev_g1"]) >= 0.2)
    pick = np.zeros(len(pg), bool)
    idx = np.where(cand)[0]
    if len(idx):
        pick[idx[np.argsort(-pg[idx])[:top]]] = True
    return pick


LETTER_CUTS = [(0.84, "A"), (0.61, "B"), (0.37, "C"), (0.13, "D"), (0.0, "F")]


def bucket(p):
    return next(l for c, l in LETTER_CUTS if p >= c)


def calibration(rows, col, outcome="ex"):
    """Out-of-sample results by letter, and by tenth (each stock's probability). col: 0 growth, 1 wealth, 2 income."""
    arr = np.array(rows, float) if rows else np.zeros((0, 8))
    col = 7 if col == 3 else col
    p = arr[:, col]
    ex = arr[:, 6] if col == 2 else arr[:, 4]
    dd = arr[:, 5]
    ok = np.isfinite(p) & np.isfinite(ex) & np.isfinite(dd)
    p, ex, dd = p[ok], ex[ok], dd[ok]
    by = {}
    letters = np.array([bucket(x) for x in p]) if len(p) else np.array([])
    for l in "ABCDF":
        m = letters == l
        if m.sum():
            by[l] = {"n": int(m.sum()), "ret": round(float(np.median(ex[m])), 4), "beat": round(float((ex[m] > 0).mean()), 3),
                     "dd": round(float(np.median(dd[m])), 4)}
    tenths = []
    for k in range(10):
        m = (p >= k / 10) & ((p < (k + 1) / 10) if k < 9 else (p <= 1))
        tenths.append({"beat": round(float((ex[m] > 0).mean()), 3) if m.sum() else None,
                       "dd": round(float(np.median(dd[m])), 4) if m.sum() else None, "n": int(m.sum())})
    return by, tenths


def smooth_odds(rows, col):
    """Chance to beat the S&P 500 and typical worst drop as a straight-line function of the score, fit on the
    out-of-sample cases, so a better score never shows worse odds. Returns (a, b, c, d): beat = a + b*(p-0.5), drop = c + d*(p-0.5)."""
    arr = np.array(rows, float) if rows else np.zeros((0, 8))
    col = 7 if col == 3 else col
    p = arr[:, col] - 0.5
    ex = arr[:, 6] if col == 2 else arr[:, 4]
    dd = arr[:, 5]
    ok = np.isfinite(p) & np.isfinite(ex) & np.isfinite(dd)
    if ok.sum() < 200:
        return None
    p, beat, dd = p[ok], (ex[ok] > 0).astype(float), dd[ok]
    A = np.column_stack([np.ones(len(p)), p])
    a, b = np.linalg.lstsq(A, beat, rcond=None)[0]
    c, d = np.linalg.lstsq(A, dd, rcond=None)[0]
    return float(a), float(b), float(c), float(d)


def star_record(rows):
    arr = np.array(rows, float) if rows else np.zeros((0, 8))
    m = arr[:, 3] > 0
    ex = arr[m, 4]
    ex = ex[np.isfinite(ex)]
    return {"n": int(len(ex)), "ret": round(float(np.median(ex)), 4) if len(ex) else None,
            "beat": round(float((ex > 0).mean()), 3) if len(ex) else None}


def stability(tests, ics):
    """Accuracy year by year on the unseen months: a signal that only worked in one lucky stretch isn't solid."""
    by = {}
    for d, g, w in zip(tests, ics["g"], ics["w"]):
        by.setdefault(d[:4], {"g": [], "w": []})
        by[d[:4]]["g"].append(g)
        by[d[:4]]["w"].append(w)
    years = {y: {"g": round(float(np.mean(v["g"])), 3), "w": round(float(np.mean(v["w"])), 3), "months": len(v["g"])}
             for y, v in sorted(by.items())}
    g = np.array(ics["g"]) if ics["g"] else np.zeros(1)
    w = np.array(ics["w"]) if ics["w"] else np.zeros(1)
    # t-statistics on monthly accuracy (3-month outcomes overlap, so divide the count by 3 to be conservative)
    n_eff = max(1, len(g) / 3)
    return {"years": years,
            "years_positive": {"g": sum(1 for v in years.values() if v["g"] > 0), "w": sum(1 for v in years.values() if v["w"] > 0),
                               "of": len(years)},
            "months_positive": {"g": round(float((g > 0).mean()), 3), "w": round(float((w > 0).mean()), 3)},
            "t": {"g": round(float(g.mean() / (g.std() + 1e-9) * math.sqrt(n_eff)), 2),
                  "w": round(float(w.mean() / (w.std() + 1e-9) * math.sqrt(n_eff)), 2)}}


def live_record(cache_dir, today_iso, tickers, pg, pw, uni, C, days):
    """The real test: today's grades are saved once a month, and graded 3 months later against what actually happened.
    No hindsight is possible here, because the grades were saved before the outcome existed."""
    import json, os
    path = os.path.join(cache_dir, "live_grades.json")
    book = json.load(open(path)) if os.path.exists(path) else {}
    month = today_iso[:7]
    if not any(k.startswith(month) for k in book):
        book[today_iso] = {t: [round(float(pg[i]), 4), round(float(pw[i]), 4)] for i, t in enumerate(tickers) if uni[i]}
    json.dump(book, open(path, "w"))
    ix = {t: i for i, t in enumerate(tickers)}
    dpos = {d.isoformat(): k for k, d in enumerate(days)}
    results = []
    for d0, grades_then in sorted(book.items()):
        k0 = dpos.get(d0)
        if k0 is None or k0 + HORIZON >= len(days):
            continue
        k1 = k0 + HORIZON
        rows = [(v[0], v[1], C[ix[t], k1] / C[ix[t], k0] - 1, np.nanmin(C[ix[t], k0:k1 + 1]) / C[ix[t], k0] - 1)
                for t, v in grades_then.items() if t in ix and np.isfinite(C[ix[t], k0]) and np.isfinite(C[ix[t], k1])]
        if len(rows) < 100:
            continue
        a = np.array(rows)
        def ic(x, y):
            return float(np.corrcoef(x.argsort().argsort(), y.argsort().argsort())[0, 1])
        results.append({"graded": d0, "n": len(rows), "ic_g": round(ic(a[:, 0], a[:, 2]), 3), "ic_w": round(ic(a[:, 1], a[:, 3]), 3)})
    return {"saved": sorted(book), "checked": results}

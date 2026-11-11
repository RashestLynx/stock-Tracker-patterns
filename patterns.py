"""
Pattern detectors for one stock's daily prices.

Every detector returns a boolean array: True on each day the pattern completed. Running them over the whole history
(not just today) is what lets the scanner measure how each pattern has actually performed afterwards.

Arrays: o, h, l, c, v are numpy float arrays, oldest first.
"""
import numpy as np

# id: (name, kind, mood, plain-English description). mood: 1 bullish, -1 bearish, 0 neutral.
PATTERNS = {
    # ---- candlesticks ----
    "hammer": ("Hammer", "Candlestick", 1, "Long lower wick after a decline: sellers pushed it down, buyers pushed it back up."),
    "inv_hammer": ("Inverted hammer", "Candlestick", 1, "Long upper wick after a decline, a possible early sign of buyers stepping in."),
    "shooting_star": ("Shooting star", "Candlestick", -1, "Long upper wick after a rise: buyers pushed it up, sellers pushed it back down."),
    "hanging_man": ("Hanging man", "Candlestick", -1, "Hammer shape after a rise, a warning that selling is starting."),
    "doji": ("Doji", "Candlestick", 0, "Opened and closed at almost the same price: buyers and sellers are evenly matched."),
    "bull_engulf": ("Bullish engulfing", "Candlestick", 1, "An up day whose body swallows the previous down day, after a decline."),
    "bear_engulf": ("Bearish engulfing", "Candlestick", -1, "A down day whose body swallows the previous up day, after a rise."),
    "piercing": ("Piercing line", "Candlestick", 1, "After a down day, opens lower but closes above the middle of that day."),
    "dark_cloud": ("Dark cloud cover", "Candlestick", -1, "After an up day, opens higher but closes below the middle of that day."),
    "morning_star": ("Morning star", "Candlestick", 1, "Three-day bottom: a big down day, a small indecisive day, then a big up day."),
    "evening_star": ("Evening star", "Candlestick", -1, "Three-day top: a big up day, a small indecisive day, then a big down day."),
    "three_soldiers": ("Three white soldiers", "Candlestick", 1, "Three strong up days in a row, each closing near its high."),
    "three_crows": ("Three black crows", "Candlestick", -1, "Three strong down days in a row, each closing near its low."),
    "bull_harami": ("Bullish harami", "Candlestick", 1, "A small up day inside the previous big down day, after a decline."),
    "bear_harami": ("Bearish harami", "Candlestick", -1, "A small down day inside the previous big up day, after a rise."),
    "bull_marubozu": ("Bullish marubozu", "Candlestick", 1, "A big up day with almost no wicks: buyers in control all day."),
    "bear_marubozu": ("Bearish marubozu", "Candlestick", -1, "A big down day with almost no wicks: sellers in control all day."),
    # ---- chart shapes ----
    "double_top": ("Double top", "Chart shape", -1, "Two similar peaks with a dip between, confirmed when price falls below the dip."),
    "double_bottom": ("Double bottom", "Chart shape", 1, "Two similar lows with a rise between, confirmed when price rises above it."),
    "triple_top": ("Triple top", "Chart shape", -1, "Three similar peaks, confirmed when price falls below the dips between them."),
    "triple_bottom": ("Triple bottom", "Chart shape", 1, "Three similar lows, confirmed when price rises above the peaks between them."),
    "head_shoulders": ("Head and shoulders", "Chart shape", -1, "A high peak between two lower ones, confirmed when price breaks the neckline."),
    "inv_head_shoulders": ("Inverse head and shoulders", "Chart shape", 1, "A low between two higher lows, confirmed when price breaks above the neckline."),
    "asc_triangle": ("Ascending triangle breakout", "Chart shape", 1, "Flat ceiling with rising lows, then a close above the ceiling."),
    "desc_triangle": ("Descending triangle breakdown", "Chart shape", -1, "Flat floor with falling highs, then a close below the floor."),
    "sym_triangle_up": ("Triangle breakout (up)", "Chart shape", 1, "Highs falling and lows rising into a point, then a close above the top line."),
    "sym_triangle_down": ("Triangle breakdown (down)", "Chart shape", -1, "Highs falling and lows rising into a point, then a close below the bottom line."),
    "rising_wedge": ("Rising wedge breakdown", "Chart shape", -1, "Rising but narrowing price range, then a close below its bottom line."),
    "falling_wedge": ("Falling wedge breakout", "Chart shape", 1, "Falling but narrowing price range, then a close above its top line."),
    "channel_up_break": ("Rising channel breakout", "Chart shape", 1, "Close above the top of a steady rising channel."),
    "channel_down_break": ("Falling channel breakdown", "Chart shape", -1, "Close below the bottom of a steady falling channel."),
    "bull_flag": ("Bull flag", "Chart shape", 1, "A sharp rise, a short tight pause, then a close above the pause."),
    "bear_flag": ("Bear flag", "Chart shape", -1, "A sharp drop, a short tight pause, then a close below the pause."),
    "cup_handle": ("Cup and handle", "Chart shape", 1, "A rounded dip and recovery, a small pullback, then a close above the old high."),
    "range_breakout": ("Range breakout", "Chart shape", 1, "Close above a tight 30-day trading range."),
    "range_breakdown": ("Range breakdown", "Chart shape", -1, "Close below a tight 30-day trading range."),
    # ---- indicators ----
    "golden_cross": ("Golden cross", "Indicator", 1, "The 50-day average crossed above the 200-day average."),
    "death_cross": ("Death cross", "Indicator", -1, "The 50-day average crossed below the 200-day average."),
    "ma20_50_up": ("20/50-day cross up", "Indicator", 1, "The 20-day average crossed above the 50-day average."),
    "ma20_50_down": ("20/50-day cross down", "Indicator", -1, "The 20-day average crossed below the 50-day average."),
    "macd_up": ("MACD bullish cross", "Indicator", 1, "MACD crossed above its signal line: short-term momentum turning up."),
    "macd_down": ("MACD bearish cross", "Indicator", -1, "MACD crossed below its signal line: short-term momentum turning down."),
    "rsi_oversold": ("RSI oversold", "Indicator", 1, "RSI fell below 30: it has dropped fast and often bounces."),
    "rsi_overbought": ("RSI overbought", "Indicator", -1, "RSI rose above 70: it has risen fast and often pauses."),
    "rsi_bull_div": ("RSI bullish divergence", "Indicator", 1, "Price made a lower low but RSI made a higher low: selling is weakening."),
    "rsi_bear_div": ("RSI bearish divergence", "Indicator", -1, "Price made a higher high but RSI made a lower high: buying is weakening."),
    "stoch_up": ("Stochastic oversold cross", "Indicator", 1, "Stochastic crossed up from below 20."),
    "stoch_down": ("Stochastic overbought cross", "Indicator", -1, "Stochastic crossed down from above 80."),
    "bb_squeeze_up": ("Bollinger squeeze breakout", "Indicator", 1, "After its quietest stretch in 6 months, price closed above the upper band."),
    "bb_squeeze_down": ("Bollinger squeeze breakdown", "Indicator", -1, "After its quietest stretch in 6 months, price closed below the lower band."),
    "adx_up": ("Strong uptrend starting (ADX)", "Indicator", 1, "ADX rose above 25 with buyers in control: a trend is getting strong."),
    "adx_down": ("Strong downtrend starting (ADX)", "Indicator", -1, "ADX rose above 25 with sellers in control."),
    "obv_bull_div": ("Volume bullish divergence", "Indicator", 1, "Price made a lower low but on-balance volume made a higher low: buyers accumulating."),
    "obv_bear_div": ("Volume bearish divergence", "Indicator", -1, "Price made a higher high but on-balance volume made a lower high."),
    "high_52w": ("New 52-week high", "Indicator", 1, "Closed at its highest price in a year."),
    "low_52w": ("New 52-week low", "Indicator", -1, "Closed at its lowest price in a year."),
    "volume_spike": ("Volume spike", "Indicator", 0, "Traded more than 3× its usual volume."),
    "gap_up": ("Gap up", "Indicator", 1, "Opened at least 3% above the previous close."),
    "gap_down": ("Gap down", "Indicator", -1, "Opened at least 3% below the previous close."),
    "gap_and_hold": ("Big gap that held", "Indicator", 1, "Gapped up 5%+ on heavy volume (often earnings news) and stayed above the gap for 5 days."),
    "gap_and_fade": ("Big gap down that held", "Indicator", -1, "Gapped down 5%+ on heavy volume (often earnings news) and stayed below for 5 days."),
    # ---- smart money (filled in by smart_money.py, not by price) ----
    "insider_cluster": ("Insider buying cluster", "Smart money", 1, "Two or more company insiders bought shares on the open market within 30 days."),
    "insider_big_buy": ("Big insider purchase", "Smart money", 1, "A company insider bought $250,000+ of shares on the open market."),
    "insider_ceo_buy": ("CEO or CFO bought", "Smart money", 1, "The CEO or CFO bought shares on the open market."),
    "congress_buy": ("Congress member bought", "Smart money", 1, "A member of Congress disclosed buying this stock."),
}


def _shift(a, n):
    out = np.full_like(a, np.nan)
    if n > 0:
        out[n:] = a[:-n]
    else:
        out[:n] = a[-n:]
    return out


def sma(a, n):
    out = np.full_like(a, np.nan)
    if len(a) >= n:
        cs = np.cumsum(np.insert(a, 0, 0.0))
        out[n - 1:] = (cs[n:] - cs[:-n]) / n
    return out


def ema(a, n):
    out = np.full_like(a, np.nan)
    if len(a) < n:
        return out
    k = 2 / (n + 1)
    out[n - 1] = a[:n].mean()
    for i in range(n, len(a)):
        out[i] = a[i] * k + out[i - 1] * (1 - k)
    return out


def rsi(c, n=14):
    out = np.full_like(c, np.nan)
    if len(c) <= n:
        return out
    d = np.diff(c)
    g = np.where(d > 0, d, 0.0)
    lo = np.where(d < 0, -d, 0.0)
    ag, al = g[:n].mean(), lo[:n].mean()
    out[n] = 100 if al == 0 else 100 - 100 / (1 + ag / al)
    for i in range(n + 1, len(c)):
        ag = (ag * (n - 1) + g[i - 1]) / n
        al = (al * (n - 1) + lo[i - 1]) / n
        out[i] = 100 if al == 0 else 100 - 100 / (1 + ag / al)
    return out


def rolling_max(a, n):
    out = np.full_like(a, np.nan)
    if len(a) >= n:
        w = np.lib.stride_tricks.sliding_window_view(a, n)
        out[n - 1:] = w.max(axis=1)
    return out


def rolling_min(a, n):
    out = np.full_like(a, np.nan)
    if len(a) >= n:
        w = np.lib.stride_tricks.sliding_window_view(a, n)
        out[n - 1:] = w.min(axis=1)
    return out


def cross_up(a, b):
    pa, pb = _shift(a, 1), _shift(b, 1)
    with np.errstate(invalid="ignore"):
        return (a > b) & (pa <= pb)


def first_true(mask):
    """Only the first day of each run of True (so a signal counts once, not every day it stays on)."""
    prev = np.concatenate([[False], mask[:-1]])
    return mask & ~prev


def pivots(h, l, order=5):
    """Swing highs/lows: indexes higher (lower) than the `order` days on each side. Known only `order` days later."""
    n = len(h)
    hi, lo = [], []
    for i in range(order, n - order):
        wh = h[i - order:i + order + 1]
        wl = l[i - order:i + order + 1]
        if h[i] == wh.max() and (wh == h[i]).sum() == 1:
            hi.append(i)
        if l[i] == wl.min() and (wl == l[i]).sum() == 1:
            lo.append(i)
    return np.array(hi, dtype=int), np.array(lo, dtype=int)


def detect(o, h, l, c, v):
    """All price-based patterns. Returns {pattern_id: bool array}."""
    n = len(c)
    out = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        body = np.abs(c - o)
        rng = np.maximum(h - l, 1e-9)
        upper = h - np.maximum(o, c)
        lower = np.minimum(o, c) - l
        up = c > o
        down = c < o
        avg_body = sma(body, 20)
        prior = _shift(c, 10)
        downtrend = c < prior * 0.97
        uptrend = c > prior * 1.03
        po, pc, ph, plo = _shift(o, 1), _shift(c, 1), _shift(h, 1), _shift(l, 1)
        pbody = np.abs(pc - po)
        pup, pdown = pc > po, pc < po
        big = body > 1.2 * avg_body
        small = body < 0.5 * avg_body

        hammer_shape = (lower >= 2 * body) & (upper <= 0.3 * np.maximum(body, 1e-9) + 0.1 * rng) & (body > 0.05 * rng)
        inv_shape = (upper >= 2 * body) & (lower <= 0.3 * np.maximum(body, 1e-9) + 0.1 * rng) & (body > 0.05 * rng)
        out["hammer"] = hammer_shape & _shift(downtrend.astype(float), 1).astype(bool)
        out["hanging_man"] = hammer_shape & _shift(uptrend.astype(float), 1).astype(bool)
        out["inv_hammer"] = inv_shape & _shift(downtrend.astype(float), 1).astype(bool)
        out["shooting_star"] = inv_shape & _shift(uptrend.astype(float), 1).astype(bool)
        out["doji"] = (body <= 0.1 * rng) & (rng > 0.5 * sma(rng, 20))
        out["bull_engulf"] = pdown & up & (o <= pc) & (c >= po) & (body > pbody) & _shift(downtrend.astype(float), 1).astype(bool)
        out["bear_engulf"] = pup & down & (o >= pc) & (c <= po) & (body > pbody) & _shift(uptrend.astype(float), 1).astype(bool)
        mid = (po + pc) / 2
        out["piercing"] = pdown & (pbody > avg_body) & up & (o < plo) & (c > mid) & (c < po)
        out["dark_cloud"] = pup & (pbody > avg_body) & down & (o > ph) & (c < mid) & (c > po)
        o2, c2 = _shift(o, 2), _shift(c, 2)
        b2 = np.abs(c2 - o2)
        avg2 = _shift(avg_body, 2)
        out["morning_star"] = (c2 < o2) & (b2 > avg2) & (pbody < 0.5 * avg2) & up & big & (c > (o2 + c2) / 2)
        out["evening_star"] = (c2 > o2) & (b2 > avg2) & (pbody < 0.5 * avg2) & down & big & (c < (o2 + c2) / 2)
        strong_up = up & big & (upper < 0.25 * body)
        strong_dn = down & big & (lower < 0.25 * body)
        su1, su2 = _shift(strong_up.astype(float), 1).astype(bool), _shift(strong_up.astype(float), 2).astype(bool)
        sd1, sd2 = _shift(strong_dn.astype(float), 1).astype(bool), _shift(strong_dn.astype(float), 2).astype(bool)
        out["three_soldiers"] = first_true(strong_up & su1 & su2 & (c > pc) & (pc > c2))
        out["three_crows"] = first_true(strong_dn & sd1 & sd2 & (c < pc) & (pc < c2))
        out["bull_harami"] = pdown & (pbody > 1.2 * avg_body) & up & small & (o > pc) & (c < po) & _shift(downtrend.astype(float), 1).astype(bool)
        out["bear_harami"] = pup & (pbody > 1.2 * avg_body) & down & small & (o < pc) & (c > po) & _shift(uptrend.astype(float), 1).astype(bool)
        out["bull_marubozu"] = up & (body > 2 * avg_body) & (upper < 0.05 * rng) & (lower < 0.05 * rng)
        out["bear_marubozu"] = down & (body > 2 * avg_body) & (upper < 0.05 * rng) & (lower < 0.05 * rng)

        # ---- indicators ----
        s20, s50, s200 = sma(c, 20), sma(c, 50), sma(c, 200)
        out["golden_cross"] = cross_up(s50, s200)
        out["death_cross"] = cross_up(s200, s50)
        out["ma20_50_up"] = cross_up(s20, s50)
        out["ma20_50_down"] = cross_up(s50, s20)
        macd = ema(c, 12) - ema(c, 26)
        sig = np.full_like(c, np.nan)
        valid = ~np.isnan(macd)
        if valid.sum() >= 9:
            sig[valid] = ema(macd[valid], 9)
        out["macd_up"] = cross_up(macd, sig)
        out["macd_down"] = cross_up(sig, macd)
        r = rsi(c)
        pr = _shift(r, 1)
        out["rsi_oversold"] = (r < 30) & (pr >= 30)
        out["rsi_overbought"] = (r > 70) & (pr <= 70)
        hh, ll = rolling_max(h, 14), rolling_min(l, 14)
        k = 100 * (c - ll) / np.maximum(hh - ll, 1e-9)
        d = sma(np.nan_to_num(k, nan=50.0), 3)
        out["stoch_up"] = cross_up(k, d) & (_shift(k, 1) < 20)
        out["stoch_down"] = cross_up(d, k) & (_shift(k, 1) > 80)
        sd20 = np.full_like(c, np.nan)
        if n >= 20:
            w = np.lib.stride_tricks.sliding_window_view(c, 20)
            sd20[19:] = w.std(axis=1)
        bw = 4 * sd20 / s20
        squeeze = _shift((bw <= rolling_min(bw, 120) * 1.05).astype(float), 1).astype(bool)
        recent_squeeze = rolling_max(squeeze.astype(float), 5) > 0
        out["bb_squeeze_up"] = first_true(recent_squeeze & (c > s20 + 2 * sd20))
        out["bb_squeeze_down"] = first_true(recent_squeeze & (c < s20 - 2 * sd20))
        # ADX (Wilder)
        tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
        upm, dnm = h - ph, plo - l
        pdm = np.where((upm > dnm) & (upm > 0), upm, 0.0)
        ndm = np.where((dnm > upm) & (dnm > 0), dnm, 0.0)
        atr, pdi_s, ndi_s = ema(np.nan_to_num(tr), 27), ema(pdm, 27), ema(ndm, 27)
        pdi, ndi = 100 * pdi_s / atr, 100 * ndi_s / atr
        dx = 100 * np.abs(pdi - ndi) / np.maximum(pdi + ndi, 1e-9)
        adx = ema(np.nan_to_num(dx), 27)
        adx_cross = (adx > 25) & (_shift(adx, 1) <= 25)
        out["adx_up"] = adx_cross & (pdi > ndi)
        out["adx_down"] = adx_cross & (ndi > pdi)
        out["high_52w"] = first_true(c >= rolling_max(_shift(c, 1), 250))
        out["low_52w"] = first_true(c <= rolling_min(_shift(c, 1), 250))
        v50 = _shift(sma(v, 50), 1)
        out["volume_spike"] = first_true(v > 3 * v50)
        out["gap_up"] = o >= pc * 1.03
        out["gap_down"] = o <= pc * 0.97
        big_gap_up = (o >= pc * 1.05) & (v > 2 * v50)
        big_gap_dn = (o <= pc * 0.95) & (v > 2 * v50)
        hold_up = np.zeros(n, bool)
        hold_dn = np.zeros(n, bool)
        for i in np.where(big_gap_up)[0]:
            if i + 5 < n and (l[i + 1:i + 6] > pc[i]).all():
                hold_up[i + 5] = True
        for i in np.where(big_gap_dn)[0]:
            if i + 5 < n and (h[i + 1:i + 6] < pc[i]).all():
                hold_dn[i + 5] = True
        out["gap_and_hold"] = hold_up
        out["gap_and_fade"] = hold_dn
        hi30, lo30 = rolling_max(_shift(h, 1), 30), rolling_min(_shift(l, 1), 30)
        tight = (hi30 / lo30 - 1) < 0.10
        out["range_breakout"] = first_true(tight & (c > hi30))
        out["range_breakdown"] = first_true(tight & (c < lo30))

    for key in list(out):
        out[key] = np.nan_to_num(out[key].astype(float)).astype(bool)

    # ---- divergences and chart shapes (from swing points) ----
    out.update(shapes(o, h, l, c, v, r))
    return out


def _fit(xs, ys):
    """Least-squares line through points: (slope per day, intercept)."""
    if len(xs) < 2:
        return None
    a, b = np.polyfit(xs, ys, 1)
    return a, b


def shapes(o, h, l, c, v, r):
    n = len(c)
    res = {k: np.zeros(n, bool) for k in (
        "double_top", "double_bottom", "triple_top", "triple_bottom", "head_shoulders", "inv_head_shoulders",
        "asc_triangle", "desc_triangle", "sym_triangle_up", "sym_triangle_down", "rising_wedge", "falling_wedge",
        "channel_up_break", "channel_down_break", "bull_flag", "bear_flag", "cup_handle",
        "rsi_bull_div", "rsi_bear_div", "obv_bull_div", "obv_bear_div")}
    if n < 30:
        return res
    order = 5
    ph, pl = pivots(h, l, order)
    obv = np.concatenate([[0.0], np.cumsum(np.sign(np.diff(c)) * v[1:])])

    def known(piv, day):
        """Pivots confirmed by `day` (a pivot needs `order` days after it)."""
        return piv[piv <= day - order]

    fired_at = {}

    def fire(name, day, ref):
        # One signal per formation: skip if this exact formation (ref) already fired.
        key = (name, ref)
        if key in fired_at:
            return
        fired_at[key] = day
        res[name][day] = True

    for day in range(30, n):
        H = known(ph, day)
        L = known(pl, day)
        H = H[H >= day - 150]
        L = L[L >= day - 150]
        p = c[day]
        pp = c[day - 1]

        # Double / triple tops and bottoms: confirmed on the day price crosses the dip / peak between them.
        if len(H) >= 2:
            a, b = H[-2], H[-1]
            if b - a >= 10 and abs(h[a] - h[b]) / max(h[a], h[b]) <= 0.03:
                trough = l[a:b + 1].min()
                if trough <= min(h[a], h[b]) * 0.93 and p < trough <= pp:
                    fire("double_top", day, (a, b))
            if len(H) >= 3:
                a, b, d = H[-3], H[-2], H[-1]
                tops = h[[a, b, d]]
                if d - a >= 20 and (tops.max() - tops.min()) / tops.max() <= 0.03:
                    floor = min(l[a:b + 1].min(), l[b:d + 1].min())
                    if p < floor <= pp:
                        fire("triple_top", day, (a, b, d))
        if len(L) >= 2:
            a, b = L[-2], L[-1]
            if b - a >= 10 and abs(l[a] - l[b]) / max(l[a], l[b]) <= 0.03:
                peak = h[a:b + 1].max()
                if peak >= max(l[a], l[b]) * 1.07 and p > peak >= pp:
                    fire("double_bottom", day, (a, b))
            if len(L) >= 3:
                a, b, d = L[-3], L[-2], L[-1]
                lows = l[[a, b, d]]
                if d - a >= 20 and (lows.max() - lows.min()) / lows.max() <= 0.03:
                    ceil = max(h[a:b + 1].max(), h[b:d + 1].max())
                    if p > ceil >= pp:
                        fire("triple_bottom", day, (a, b, d))

        # Head and shoulders: last three highs, middle one highest, shoulders within 5%; neckline from the lows between.
        if len(H) >= 3:
            s1, hd, s2 = H[-3], H[-2], H[-1]
            if h[hd] > h[s1] * 1.03 and h[hd] > h[s2] * 1.03 and abs(h[s1] - h[s2]) / max(h[s1], h[s2]) <= 0.05:
                n1 = l[s1:hd + 1].argmin() + s1
                n2 = l[hd:s2 + 1].argmin() + hd
                slope = (l[n2] - l[n1]) / max(n2 - n1, 1)
                neck = l[n2] + slope * (day - n2)
                neck_prev = l[n2] + slope * (day - 1 - n2)
                if p < neck and pp >= neck_prev and day - s2 <= 30:
                    fire("head_shoulders", day, (s1, hd, s2))
        if len(L) >= 3:
            s1, hd, s2 = L[-3], L[-2], L[-1]
            if l[hd] < l[s1] * 0.97 and l[hd] < l[s2] * 0.97 and abs(l[s1] - l[s2]) / max(l[s1], l[s2]) <= 0.05:
                n1 = h[s1:hd + 1].argmax() + s1
                n2 = h[hd:s2 + 1].argmax() + hd
                slope = (h[n2] - h[n1]) / max(n2 - n1, 1)
                neck = h[n2] + slope * (day - n2)
                neck_prev = h[n2] + slope * (day - 1 - n2)
                if p > neck and pp <= neck_prev and day - s2 <= 30:
                    fire("inv_head_shoulders", day, (s1, hd, s2))

        # Triangles, wedges and channels: lines through the swing highs and lows of the last ~60 days.
        Hw = H[H >= day - 60]
        Lw = L[L >= day - 60]
        if len(Hw) >= 2 and len(Lw) >= 2 and min(Hw.min(), Lw.min()) <= day - 20:
            top = _fit(Hw, h[Hw])
            bot = _fit(Lw, l[Lw])
            if top and bot:
                ts, tb = top
                bs, bb = bot
                t_now, b_now = ts * day + tb, bs * day + bb
                t_prev, b_prev = ts * (day - 1) + tb, bs * (day - 1) + bb
                start = min(Hw.min(), Lw.min())
                width_start = (ts * start + tb) - (bs * start + bb)
                width_now = t_now - b_now
                if width_now > 0 and width_start > 0:
                    rel = lambda s: s / p * 20  # slope as a share of price over 20 days
                    flat_t, flat_b = abs(rel(ts)) < 0.02, abs(rel(bs)) < 0.02
                    converging = width_now < width_start * 0.75
                    parallel = abs(width_now - width_start) / width_start < 0.25
                    brk_up = p > t_now and pp <= t_prev
                    brk_dn = p < b_now and pp >= b_prev
                    ref = (int(start), int(Hw.max()), int(Lw.max()))
                    if flat_t and rel(bs) > 0.02 and converging and brk_up:
                        fire("asc_triangle", day, ref)
                    if flat_b and rel(ts) < -0.02 and converging and brk_dn:
                        fire("desc_triangle", day, ref)
                    if rel(ts) < -0.02 and rel(bs) > 0.02 and converging:
                        if brk_up:
                            fire("sym_triangle_up", day, ref)
                        if brk_dn:
                            fire("sym_triangle_down", day, ref)
                    if rel(ts) > 0.02 and rel(bs) > 0.02 and converging and bs > ts and brk_dn:
                        fire("rising_wedge", day, ref)
                    if rel(ts) < -0.02 and rel(bs) < -0.02 and converging and ts > bs and brk_up:
                        fire("falling_wedge", day, ref)
                    if parallel and rel(ts) > 0.03 and rel(bs) > 0.03 and brk_up:
                        fire("channel_up_break", day, ref)
                    if parallel and rel(ts) < -0.03 and rel(bs) < -0.03 and brk_dn:
                        fire("channel_down_break", day, ref)

        # Flags: a 10%+ move in 10 days or less, then a 5-15 day tight pause, then a break in the same direction.
        for pause in range(5, 16):
            ps, pe = day - pause, day - 1
            if ps - 10 < 0:
                break
            hi_p, lo_p = h[ps:pe + 1].max(), l[ps:pe + 1].min()
            pole_start = c[ps - 10:ps].min(), c[ps - 10:ps].max()
            gain = c[ps] / pole_start[0] - 1
            drop = c[ps] / pole_start[1] - 1
            tight = (hi_p - lo_p) / c[ps] < 0.6 * max(abs(gain), abs(drop))
            if gain >= 0.10 and tight and p > hi_p and pp <= hi_p:
                fire("bull_flag", day, ("bf", ps))
                break
            if drop <= -0.10 and tight and p < lo_p and pp >= lo_p:
                fire("bear_flag", day, ("brf", ps))
                break

        # Cup and handle: high (rim), 12-35% dip, recovery to within 5% of the rim, small pullback, break above the rim.
        if len(H) >= 2:
            right = H[-1]
            lefts = H[(H < right - 30) & (H >= right - 200)]
            for left in lefts[::-1]:
                rim = max(h[left], h[right])
                if abs(h[left] - h[right]) / rim > 0.05:
                    continue
                bottom = l[left:right + 1].min()
                bi = l[left:right + 1].argmin() + left
                depth = 1 - bottom / rim
                if not (0.12 <= depth <= 0.35) or not (left + (right - left) * 0.25 <= bi <= left + (right - left) * 0.75):
                    continue
                if day - right < 5 or day - right > 25:
                    continue
                handle_low = l[right:day].min()
                if handle_low < rim * 0.88 or handle_low < bottom + (rim - bottom) * 0.5:
                    continue
                if p > rim and pp <= rim:
                    fire("cup_handle", day, (int(left), int(right)))
                break

        # Divergences between price and RSI / on-balance volume on the last two swing lows / highs.
        if len(L) >= 2:
            a, b = L[-2], L[-1]
            if day - b == order and b - a >= 5:
                if l[b] < l[a] and r[b] > r[a] + 2 and r[a] < 40:
                    fire("rsi_bull_div", day, (a, b))
                if l[b] < l[a] and obv[b] > obv[a]:
                    fire("obv_bull_div", day, (a, b))
        if len(H) >= 2:
            a, b = H[-2], H[-1]
            if day - b == order and b - a >= 5:
                if h[b] > h[a] and r[b] < r[a] - 2 and r[a] > 60:
                    fire("rsi_bear_div", day, (a, b))
                if h[b] > h[a] and obv[b] < obv[a]:
                    fire("obv_bear_div", day, (a, b))
    return res

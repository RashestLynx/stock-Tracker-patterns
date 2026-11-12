"""
Smart-money signals:
  - Insider open-market purchases from SEC Form 4 filings (EDGAR daily index; one request per filing, cached by day)
  - Congress members' disclosed purchases (Bargo's free Congress Trades API)
"""
import gzip
import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta

SEC_UA = os.environ.get("SEC_USER_AGENT") or "StockTracker-Scanner admin@example.com"
_last = [0.0]


def _sec(url, retries=3):
    for attempt in range(retries):
        wait = 0.13 - (time.time() - _last[0])  # stay under the SEC's 10 requests a second
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": SEC_UA, "Accept-Encoding": "gzip"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    data = gzip.decompress(data)
                return data.decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code in (403, 429, 503):
                time.sleep(5 * (attempt + 1))
                continue
            raise
        except Exception:
            time.sleep(2)
    return None


def _tag(xml, name):
    m = re.search(rf"<{name}>\s*(?:<value>)?\s*([^<]*?)\s*(?:</value>)?\s*</{name}>", xml, re.S)
    return m.group(1).strip() if m else ""


def parse_form4(text):
    """Open-market purchases (code P) in one Form 4 filing: list of dicts."""
    m = re.search(r"<ownershipDocument>.*?</ownershipDocument>", text, re.S)
    if not m:
        return []
    xml = m.group(0)
    ticker = _tag(xml, "issuerTradingSymbol").upper()
    owner = _tag(xml, "rptOwnerName")
    title = _tag(xml, "officerTitle")
    is_officer = _tag(xml, "isOfficer") in ("1", "true")
    out = []
    for t in re.findall(r"<nonDerivativeTransaction>.*?</nonDerivativeTransaction>", xml, re.S):
        if _tag(t, "transactionCode") != "P":
            continue
        try:
            shares = float(_tag(t, "transactionShares") or 0)
            price = float(_tag(t, "transactionPricePerShare") or 0)
        except ValueError:
            continue
        if shares <= 0 or price <= 0:
            continue
        out.append({"ticker": ticker, "owner": owner, "title": title if is_officer else "",
                    "date": _tag(t, "transactionDate")[:10], "value": shares * price})
    return out


def insider_day(day, cache_dir):
    """All open-market insider purchases filed on `day` (cached)."""
    path = os.path.join(cache_dir, f"form4-{day.isoformat()}.json")
    if os.path.exists(path):
        return json.load(open(path))
    q = (day.month - 1) // 3 + 1
    idx = _sec(f"https://www.sec.gov/Archives/edgar/daily-index/{day.year}/QTR{q}/form.{day.strftime('%Y%m%d')}.idx")
    buys = []
    if idx is not None:
        files = set()
        for line in idx.splitlines():
            if line[:3] in ("4  ", "4/A"):
                m = re.search(r"(edgar/data/\S+\.txt)", line)
                if m:
                    files.add(m.group(1).split("/")[-1] + "|" + m.group(1))
        # Each filing is listed once per company it involves; fetch each accession once.
        seen = set()
        for f in sorted(files):
            acc, rel = f.split("|")
            if acc in seen:
                continue
            seen.add(acc)
            text = _sec("https://www.sec.gov/Archives/" + rel)
            if text:
                buys += parse_form4(text)
    if day < date.today():  # don't cache today's partial index
        json.dump(buys, open(path, "w"))
    return buys


def insider_signals(cache_dir, days=30):
    """{ticker: [(pattern_id, note), ...]} for insider buying in the last `days` calendar days."""
    os.makedirs(cache_dir, exist_ok=True)
    buys = []
    d = date.today()
    for k in range(days + 1):
        day = d - timedelta(days=k)
        if day.weekday() < 5:
            try:
                buys += insider_day(day, cache_dir)
            except Exception as e:
                print(f"insider data for {day} failed: {e}")
    by = {}
    for b in buys:
        if b["ticker"] and re.fullmatch(r"[A-Z]{1,5}([.-][A-Z])?", b["ticker"]):
            by.setdefault(b["ticker"].replace("-", "."), []).append(b)
    out = {}
    for t, lst in by.items():
        sig = []
        owners = {b["owner"] for b in lst}
        total = sum(b["value"] for b in lst)
        if len(owners) >= 2 and total >= 100_000:
            sig.append(("insider_cluster", f"{len(owners)} insiders bought ${total:,.0f} in the last {days} days"))
        big = max(lst, key=lambda b: b["value"])
        if big["value"] >= 250_000:
            sig.append(("insider_big_buy", f"{big['owner'].title()} bought ${big['value']:,.0f} on {big['date']}"))
        ceo = [b for b in lst if re.search(r"\b(ceo|chief executive|cfo|chief financial)\b", b["title"], re.I)]
        if ceo:
            b = max(ceo, key=lambda x: x["value"])
            sig.append(("insider_ceo_buy", f"{b['title'] or 'Officer'} {b['owner'].title()} bought ${b['value']:,.0f} on {b['date']}"))
        if sig:
            out[t] = sig
    return out


def congress_signals(days=45):
    """{ticker: [("congress_buy", note)]} from members' disclosed purchases in the last `days` days."""
    key = os.environ.get("BARGO_KEY", "")
    url = f"https://www.bargo.ai/free-apis/congress/v1/trades?limit={1000 if key else 100}"
    req = urllib.request.Request(url, headers={"User-Agent": "StockTracker-Scanner/1.0", "Accept": "application/json",
                                               **({"X-Api-Key": key} if key else {})})
    try:
        trades = json.loads(urllib.request.urlopen(req, timeout=30).read()).get("trades", [])
    except Exception as e:
        print(f"Congress data failed: {e}")
        return {}
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    out = {}
    for t in trades:
        ticker = (t.get("ticker") or "").upper()
        kind = (t.get("type") or "").lower()
        when = (t.get("disclosure_date") or t.get("transaction_date") or "")[:10]
        if not ticker or ticker in ("--", "N/A") or "purchase" not in kind or when < cutoff:
            continue
        note = f"{t.get('member', 'A member')} disclosed a purchase ({t.get('amount_range') or 'amount not given'}) on {when}"
        out.setdefault(ticker, [])
        if len(out[ticker]) < 3:
            out[ticker].append(("congress_buy", note))
    return out

# Runs scan.py end to end against a fake Massive (synthetic market) and fake SEC/Bargo, offline.
import gzip, json, os, sys, numpy as np
from datetime import date, timedelta
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
TMP = tempfile.mkdtemp()
os.environ["MASSIVE_KEY"] = "test"; os.environ["CACHE_DIR"] = TMP + "/cache"; os.environ["OUT_DIR"] = TMP + "/out"
import scan, smart_money
rng = np.random.default_rng(0)
tick = ["SPY"] + [f"T{chr(65+i%26)}{chr(65+i//26)}" for i in range(300)] + ["BAD-W", "low"]
paths = {t: 20 + rng.random() * 200 for t in tick}
def fake_day(d):
    if d.weekday() >= 5: return None
    rows = {}
    for t in tick:
        p = paths[t]; p *= 1 + rng.normal(0.0003, 0.02); paths[t] = p
        o = p * (1 + rng.normal(0, 0.005)); rows[t] = (o, max(o, p) * 1.01, min(o, p) * 0.99, p, float(rng.uniform(2e5, 3e6)))
    return rows
cache = {}
def massive_day(d):
    if d not in cache: cache[d] = fake_day(d)
    return cache[d]
scan.massive_day = massive_day
scan.grades.run = lambda *a, **k: {"graded": 0, "stars": [], "record": {}, "grades": {}}  # tested in test_grades_offline.py
form4 = """<XML><ownershipDocument><issuer><issuerTradingSymbol>TBA</issuerTradingSymbol></issuer>
<reportingOwner><reportingOwnerId><rptOwnerName>DOE JANE</rptOwnerName></reportingOwnerId><reportingOwnerRelationship><isOfficer>1</isOfficer><officerTitle>Chief Executive Officer</officerTitle></reportingOwnerRelationship></reportingOwner>
<nonDerivativeTable><nonDerivativeTransaction><transactionDate><value>2026-11-05</value></transactionDate><transactionCoding><transactionCode>P</transactionCode></transactionCoding>
<transactionAmounts><transactionShares><value>10000</value></transactionShares><transactionPricePerShare><value>30.5</value></transactionPricePerShare></transactionAmounts></nonDerivativeTransaction></nonDerivativeTable></ownershipDocument></XML>"""
print(smart_money.parse_form4(form4))
smart_money.insider_signals = lambda cache_dir, days=30: {"TBA": [("insider_big_buy", "Jane Doe bought $305,000"), ("insider_ceo_buy", "CEO Jane Doe bought $305,000")]}
smart_money.congress_signals = lambda days=45: {"TCA": [("congress_buy", "A member disclosed a purchase")]}
scan.main()
r = json.load(gzip.open(TMP + "/out/patterns.json.gz", "rt"))
print(r["date"], r["stocks_scanned"], len(r["stocks"]))
print(json.dumps(r["stocks"].get("TBA"), indent=0)[:600])
print({k: v for k, v in list(r["patterns"].items())[:3]})
assert "BAD-W" not in r["stocks"] and "low" not in r["stocks"]
s = r["stocks"]["TBA"]; assert s["lo20"] <= s["hi20"] and s["lo52"] <= s["lo20"] and s["hi52"] >= s["hi20"], s
print("levels ok")

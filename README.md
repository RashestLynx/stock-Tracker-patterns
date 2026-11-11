# Stock Tracker patterns

A nightly scan of every liquid U.S. stock (price at least $3, at least $2M traded a day) for about 70 patterns:

- **Candlesticks**: hammer, engulfing, morning/evening star, doji, three soldiers/crows, harami, and more.
- **Chart shapes**: double/triple tops and bottoms, head and shoulders, triangles, wedges, flags, cup and handle, channel and range breakouts.
- **Indicators**: golden/death cross, MACD, RSI levels and divergences, stochastic, Bollinger squeeze, ADX, OBV divergence, 52-week highs/lows, volume spikes, gaps.
- **Smart money**: insider purchases from SEC Form 4 filings (clusters, big buys, CEO/CFO buys) and Congress members' disclosed purchases.

For each price pattern, the scan also keeps a **track record**: every time it showed up across all stocks over the last two years, how the stock did over the next 20 trading days compared with the S&P 500 (SPY). Past results don't guarantee future ones.

Results are published nightly as `patterns.json.gz` on the [latest release](../../releases/tag/latest); the Stock Tracker app downloads them from there.

## Setup

- Secret `MASSIVE_KEY` (required): a Massive (formerly Polygon) API key.
- Secret `BARGO_KEY` (optional): more Congress trades.

The first run downloads two years of prices (about 100 minutes on the free plan); after that each night adds one day.

## Files

- `patterns.py`: the pattern detectors.
- `scan.py`: downloads prices, runs the detectors, computes track records, writes the results.
- `smart_money.py`: insider and Congress purchases.
- `tests/`: detector tests on made-up price shapes, and a full offline run on a fake market.

import re
import time
import requests
import pandas as pd

DERIBIT = "https://www.deribit.com/api/v2"
INS_RE = re.compile(r"^(BTC|ETH)-(\d{2}[A-Z]{3}\d{2})-(\d+)-(C|P)$")

def deribit_get(path: str, params: dict | None = None) -> dict:
    # Deribit can be picky about booleans -> force lowercase strings
    if params:
        params = {k: ("true" if v is True else "false" if v is False else v) for k, v in params.items()}
        # Avoid sending expired=false (can cause 400); only send if true
        if params.get("expired") == "false":
            params.pop("expired", None)

    r = requests.get(f"{DERIBIT}/{path.lstrip('/')}", params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()
    if payload.get("error"):
        raise RuntimeError(payload["error"])
    return payload["result"]

def parse_option_name(name: str):
    # Example: BTC-29MAR24-65000-C
    m = INS_RE.match(name)
    if not m:
        return None
    underlying, exp_code, strike, cp = m.groups()
    return {
        "underlying": underlying,
        "expiration_code": exp_code,
        "strike": float(strike),
        "type": "call" if cp == "C" else "put",
    }

def get_btc_options_data(window: float = 0.10, expired: bool = False, sleep_s: float = 0.02):
    """
    Equivalent to the yfinance example but for BTC options on Deribit.

    Returns:
      options_data: DataFrame with calls+puts + expiration + type + OI/volume/IV/greeks
      recent_price: spot reference (median underlying_price across tickers)
    """
    params = {"currency": "BTC", "kind": "option"}
    if expired:
        params["expired"] = True  # only when you explicitly want expired instruments

    instruments = deribit_get("public/get_instruments", params=params)
    if not instruments:
        print("No options data available for BTC on Deribit.")
        return None, None

    # Filter to parsable option names and collect tickers
    rows = []
    underlying_prices = []

    for ins in instruments:
        name = ins.get("instrument_name")
        meta = parse_option_name(name) if name else None
        if meta is None:
            continue

        t = deribit_get("public/ticker", {"instrument_name": name})
        stats = t.get("stats") or {}
        greeks = t.get("greeks") or {}

        up = t.get("underlying_price")
        if isinstance(up, (int, float)):
            underlying_prices.append(float(up))

        rows.append({
            "contractSymbol": name,  # similar to yfinance naming
            "lastTradeDate": None,   # Deribit ticker doesn't give last trade time consistently
            "strike": meta["strike"],
            "lastPrice": t.get("last_price"),
            "bid": t.get("best_bid_price"),
            "ask": t.get("best_ask_price"),
            "change": None,
            "percentChange": None,
            "volume": stats.get("volume"),          # 24h volume
            "openInterest": t.get("open_interest"),
            "impliedVolatility": t.get("mark_iv"),  # already in %
            "inTheMoney": None,                     # can compute if you want
            "expiration": pd.to_datetime(ins.get("expiration_timestamp"), unit="ms", utc=True, errors="coerce"),
            "type": meta["type"],

            # extra fields you likely want on crypto
            "mark_price": t.get("mark_price"),
            "underlying_price": up,
            "delta": greeks.get("delta"),
            "gamma": greeks.get("gamma"),
            "theta": greeks.get("theta"),
            "vega": greeks.get("vega"),
        })

        time.sleep(sleep_s)

    if not rows:
        print("No valid options rows built.")
        return None, None

    options_data = pd.DataFrame(rows)

    # Spot reference (similar to recent_price)
    if underlying_prices:
        recent_price = float(pd.Series(underlying_prices).median())
    else:
        # fallback to Deribit index price
        recent_price = float(deribit_get("public/get_index_price", {"index_name": "btc_usd"}))

    # Strike filter around spot (same idea as yfinance example)
    if window and window > 0:
        options_data = options_data[
            options_data["strike"].between(recent_price * (1 - window), recent_price * (1 + window))
        ].copy()

    # Match the yfinance-style transformation: implied_volatility column in %
    # (Deribit mark_iv is already a % value)
    options_data["implied_volatility"] = pd.to_numeric(options_data["impliedVolatility"], errors="coerce")

    return options_data, recent_price


# --- example usage ---
options_data, recent_price = get_btc_options_data(window=0.10, expired=False)
print("Spot (recent_price):", recent_price)
print(options_data.head(10))

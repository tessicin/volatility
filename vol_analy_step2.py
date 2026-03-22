#!/usr/bin/env python3
"""
BTC options IV analysis (Deribit) — fetch chain + volatility smile plots + interpretation.

Run:
  pip install requests pandas plotly
  python vol_analy.py --currency BTC --window 0.10 --max-exp 12 --sleep 0.02

Notes:
- Deribit mark_iv / bid_iv / ask_iv are ALREADY in percent (e.g. 64.06). Do NOT *100.
- "volume" from ticker.stats is typically 24h.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from typing import Any, Dict, Optional, List

import requests
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


DERIBIT_BASE = "https://www.deribit.com/api/v2"
INS_RE = re.compile(r"^(BTC|ETH)-(\d{2}[A-Z]{3}\d{2})-(\d+)-(C|P)$")


def deribit_get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    # Normalize booleans to Deribit-friendly "true"/"false"
    if params:
        params = {k: ("true" if v is True else "false" if v is False else v) for k, v in params.items()}
        # Avoid sending expired=false (Deribit can reject); only send if true
        if params.get("expired") == "false":
            params.pop("expired", None)

    url = f"{DERIBIT_BASE}/{path.lstrip('/')}"
    r = requests.get(url, params=params, timeout=30)
    if not r.ok:
        try:
            body = r.json()
        except Exception:
            body = r.text
        raise requests.HTTPError(f"HTTP {r.status_code} for {r.url}\nResponse: {body}", response=r)

    payload = r.json()
    if payload.get("error"):
        raise RuntimeError(payload["error"])
    return payload["result"]


def parse_option_name(name: str) -> Optional[Dict[str, Any]]:
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


def get_crypto_options_data(
    currency: str = "BTC",
    window: float = 0.10,
    expired: bool = False,
    sleep_s: float = 0.02,
    max_instruments: int = 0,
) -> tuple[pd.DataFrame, float]:
    """
    Equivalent of the yfinance get_options_data(), but for Deribit BTC/ETH options.

    Returns:
      options_data: DataFrame with calls+puts, columns compatible with your plotting functions
      recent_price: spot reference (median underlying_price across tickers)
    """
    params: Dict[str, Any] = {"currency": currency, "kind": "option"}
    if expired:
        params["expired"] = True  # only if you explicitly want expired

    instruments = deribit_get("public/get_instruments", params=params)
    if not instruments:
        raise RuntimeError(f"No options data for {currency}.")

    rows: List[Dict[str, Any]] = []
    underlying_prices: List[float] = []

    # Collect tickers
    for ins in instruments:
        name = ins.get("instrument_name")
        if not name:
            continue

        meta = parse_option_name(name)
        if meta is None:
            continue

        if max_instruments and max_instruments > 0 and len(rows) >= max_instruments:
            break

        t = deribit_get("public/ticker", {"instrument_name": name})
        stats = t.get("stats") or {}
        greeks = t.get("greeks") or {}

        up = t.get("underlying_price")
        if isinstance(up, (int, float)):
            underlying_prices.append(float(up))

        rows.append({
            # yfinance-like
            "contractSymbol": name,
            "lastTradeDate": None,  # not provided consistently via this endpoint
            "strike": meta["strike"],
            "lastPrice": t.get("last_price"),
            "bid": t.get("best_bid_price"),
            "ask": t.get("best_ask_price"),
            "change": None,
            "percentChange": None,
            "volume": stats.get("volume"),          # typically 24h
            "openInterest": t.get("open_interest"),
            "impliedVolatility": t.get("mark_iv"),  # already in %
            "inTheMoney": None,
            "expiration": pd.to_datetime(ins.get("expiration_timestamp"), unit="ms", utc=True, errors="coerce"),
            "type": meta["type"],

            # extras (useful later)
            "mark_price": t.get("mark_price"),
            "underlying_price": up,
            "delta": greeks.get("delta"),
            "gamma": greeks.get("gamma"),
            "theta": greeks.get("theta"),
            "vega": greeks.get("vega"),
        })

        time.sleep(sleep_s)

    if not rows:
        raise RuntimeError("No valid option rows built (parsing/filtering removed everything).")

    options_data = pd.DataFrame(rows)

    # Spot reference
    if underlying_prices:
        recent_price = float(pd.Series(underlying_prices).median())
    else:
        recent_price = float(deribit_get("public/get_index_price", {"index_name": f"{currency.lower()}_usd"}))

    # Strike filter around spot
    if window and window > 0:
        options_data = options_data[
            options_data["strike"].between(recent_price * (1 - window), recent_price * (1 + window))
        ].copy()

    # Create the same derived column used in your smile code
    options_data["implied_volatility"] = pd.to_numeric(options_data["impliedVolatility"], errors="coerce")

    # Clean numeric cols
    numeric_cols = [
        "strike", "lastPrice", "bid", "ask", "volume", "openInterest",
        "impliedVolatility", "implied_volatility", "delta", "gamma", "theta", "vega",
        "mark_price", "underlying_price"
    ]
    for c in numeric_cols:
        if c in options_data.columns:
            options_data[c] = pd.to_numeric(options_data[c], errors="coerce")

    return options_data, recent_price


# ------------------- VOL SMILE PLOT + INTERPRETATION -------------------

def plot_volatility_smile(options_data: pd.DataFrame, recent_price: float, ticker: str, max_exp: int = 12):
    if options_data is None or options_data.empty:
        print("No options_data to plot.")
        return None, None, None, None, None

    calls_data = options_data[options_data["type"] == "call"].copy()
    puts_data = options_data[options_data["type"] == "put"].copy()

    # Keep only first N expirations to avoid super crowded plots
    expirations = sorted(options_data["expiration"].dropna().unique())
    expirations = expirations[:max_exp]

    color_map_2d = px.colors.qualitative.Prism

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Calls", "Puts"], shared_yaxes=True)

    for i, exp in enumerate(expirations):
        color = color_map_2d[i % len(color_map_2d)]
        exp_calls = calls_data[calls_data["expiration"] == exp]
        exp_puts = puts_data[puts_data["expiration"] == exp]

        fig.add_trace(
            go.Scatter(
                x=exp_calls["strike"], y=exp_calls["implied_volatility"],
                mode="markers", marker=dict(color=color),
                name=str(pd.to_datetime(exp).date())
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=exp_puts["strike"], y=exp_puts["implied_volatility"],
                mode="markers", marker=dict(color=color),
                name=str(pd.to_datetime(exp).date()),
                showlegend=False
            ),
            row=1, col=2
        )

    # Average IV by strike (across expirations) — dashed black line
    avg_iv_by_strike_calls = calls_data.groupby("strike")["implied_volatility"].mean()
    avg_iv_by_strike_puts = puts_data.groupby("strike")["implied_volatility"].mean()

    fig.add_trace(
        go.Scatter(
            x=avg_iv_by_strike_calls.index, y=avg_iv_by_strike_calls.values,
            mode="lines", line=dict(color="black", dash="dash"),
            name="Avg. IV by Strike (Calls)"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=avg_iv_by_strike_puts.index, y=avg_iv_by_strike_puts.values,
            mode="lines", line=dict(color="black", dash="dash"),
            name="Avg. IV by Strike (Puts)",
            showlegend=False
        ),
        row=1, col=2
    )

    overall_avg_iv_calls = calls_data["implied_volatility"].mean()
    overall_avg_iv_puts = puts_data["implied_volatility"].mean()

    fig.add_hline(
        y=overall_avg_iv_calls,
        line=dict(color="gray", dash="dash"),
        annotation_text=f"Overall Avg. IV (Calls): {overall_avg_iv_calls:.2f}%",
        row=1, col=1
    )
    fig.add_hline(
        y=overall_avg_iv_puts,
        line=dict(color="gray", dash="dash"),
        annotation_text=f"Overall Avg. IV (Puts): {overall_avg_iv_puts:.2f}%",
        row=1, col=2
    )

    fig.update_layout(
        title=f"{ticker} Volatility Smile - Spot: {recent_price:.2f}",
        showlegend=True,
        legend_title_text="Expiration Date"
    )
    fig.update_xaxes(title_text="Strike Price")
    fig.update_yaxes(title_text="Implied Volatility (%)")

    fig.show()

    return overall_avg_iv_calls, overall_avg_iv_puts, avg_iv_by_strike_calls, avg_iv_by_strike_puts, expirations


def interpret_volatility_smile(
    ticker: str,
    overall_avg_iv_calls: float,
    overall_avg_iv_puts: float,
    avg_iv_by_strike_calls: pd.Series,
    avg_iv_by_strike_puts: pd.Series
):
    interpretation = f"**Interpretation of {ticker} Volatility Smile:**\n"
    interpretation += f"- Avg IV calls: {overall_avg_iv_calls:.2f}%\n"
    interpretation += f"- Avg IV puts:  {overall_avg_iv_puts:.2f}%\n"

    # Variance threshold here is arbitrary; keep it small because IV is in % scale
    call_var = float(avg_iv_by_strike_calls.var()) if avg_iv_by_strike_calls is not None else 0.0
    put_var = float(avg_iv_by_strike_puts.var()) if avg_iv_by_strike_puts is not None else 0.0

    if call_var > 1.0:
        interpretation += "- Calls show a noticeable IV variation across strikes (smile/skew).\n"
    else:
        interpretation += "- Calls show relatively stable IV across strikes.\n"

    if put_var > 1.0:
        interpretation += "- Puts show a noticeable IV variation across strikes (smile/skew).\n"
    else:
        interpretation += "- Puts show relatively stable IV across strikes.\n"

    market_sentiment = "bullish" if overall_avg_iv_calls > overall_avg_iv_puts else "bearish"
    interpretation += f"- Simple sentiment (by avg IV): {market_sentiment}\n"

    print(interpretation)


# ------------------- MAIN -------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--currency", default="BTC", choices=["BTC", "ETH"])
    ap.add_argument("--window", type=float, default=0.10, help="Strike window around spot (0.10 = ±10%).")
    ap.add_argument("--expired", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.02, help="Sleep between ticker calls.")
    ap.add_argument("--max-instruments", type=int, default=0, help="Limit instruments for speed (0=all).")
    ap.add_argument("--max-exp", type=int, default=12, help="Max expirations to plot (avoid overcrowding).")
    ap.add_argument("--out", default="deribit_options.csv", help="CSV output path.")
    args = ap.parse_args()

    ticker = args.currency  # for plot title consistency

    try:
        options_data, recent_price = get_crypto_options_data(
            currency=args.currency,
            window=args.window,
            expired=args.expired,
            sleep_s=args.sleep,
            max_instruments=args.max_instruments,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    # Save snapshot
    options_data.to_csv(args.out, index=False)
    print(f"Saved CSV: {args.out}")
    print(f"Spot reference: {recent_price:.2f}")
    print(options_data.head(10).to_string(index=False))

    # Plot + interpret
    res = plot_volatility_smile(options_data, recent_price, ticker, max_exp=args.max_exp)
    if res[0] is not None:
        overall_avg_iv_calls, overall_avg_iv_puts, avg_iv_by_strike_calls, avg_iv_by_strike_puts, _ = res
        interpret_volatility_smile(
            ticker,
            overall_avg_iv_calls,
            overall_avg_iv_puts,
            avg_iv_by_strike_calls,
            avg_iv_by_strike_puts,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
import re
import time
import requests
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

DERIBIT = "https://www.deribit.com/api/v2"
INS_RE = re.compile(r"^(BTC|ETH)-(\d{2}[A-Z]{3}\d{2})-(\d+)-(C|P)$")

ticker = "BTC"  # keep same name as original

def deribit_get(path: str, params: dict | None = None) -> dict:
    # Deribit wants booleans as lowercase strings; also avoid expired=false (can 400)
    if params:
        params = {k: ("true" if v is True else "false" if v is False else v) for k, v in params.items()}
        if params.get("expired") == "false":
            params.pop("expired", None)

    r = requests.get(f"{DERIBIT}/{path.lstrip('/')}", params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()
    if payload.get("error"):
        raise RuntimeError(payload["error"])
    return payload["result"]

def parse_option_name(name: str):
    # Example: BTC-13MAR26-61000-C
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

def get_options_data(ticker: str, window: float = 0.10, expired: bool = False, sleep_s: float = 0.02):
    """
    Original yfinance-style function name, but implemented for Deribit BTC/ETH options.

    Returns:
      options_data: DataFrame containing both calls and puts with:
        strike, bid, ask, volume, openInterest, impliedVolatility, expiration, type, implied_volatility
      recent_price: spot reference
    """
    if ticker not in ("BTC", "ETH"):
        raise ValueError("ticker must be 'BTC' or 'ETH' for Deribit in this script.")

    params = {"currency": ticker, "kind": "option"}
    if expired:
        params["expired"] = True  # ONLY if you want expired

    instruments = deribit_get("public/get_instruments", params=params)
    if not instruments:
        print(f"No options data available for {ticker}.")
        return None, None

    rows = []
    underlying_prices = []

    for ins in instruments:
        name = ins.get("instrument_name")
        meta = parse_option_name(name) if name else None
        if meta is None:
            continue

        t = deribit_get("public/ticker", {"instrument_name": name})
        stats = t.get("stats") or {}

        up = t.get("underlying_price")
        if isinstance(up, (int, float)):
            underlying_prices.append(float(up))

        rows.append({
            "contractSymbol": name,
            "lastTradeDate": None,
            "strike": meta["strike"],
            "lastPrice": t.get("last_price"),
            "bid": t.get("best_bid_price"),
            "ask": t.get("best_ask_price"),
            "change": None,
            "percentChange": None,
            "volume": stats.get("volume"),
            "openInterest": t.get("open_interest"),
            "impliedVolatility": t.get("mark_iv"),  # already %
            "inTheMoney": None,
            "expiration": pd.to_datetime(ins.get("expiration_timestamp"), unit="ms", utc=True, errors="coerce"),
            "type": meta["type"],
        })

        time.sleep(sleep_s)

    if not rows:
        print(f"No valid options data available for {ticker}.")
        return None, None

    options_data = pd.DataFrame(rows)

    # recent_price = spot reference
    if underlying_prices:
        recent_price = float(pd.Series(underlying_prices).median())
    else:
        recent_price = float(deribit_get("public/get_index_price", {"index_name": f"{ticker.lower()}_usd"}))

    # Filter strikes around spot (same logic as original yfinance example)
    options_data = options_data[
        options_data["strike"].between(recent_price * (1 - window), recent_price * (1 + window))
    ].copy()

    # In original code they did *100; here Deribit is already in %
    options_data["implied_volatility"] = pd.to_numeric(options_data["impliedVolatility"], errors="coerce")

    return options_data, recent_price


# Function to plot volatility smile for call and put options (same as original)
def plot_volatility_smile(options_data, recent_price, ticker, max_expirations=10):
    if options_data is not None and not options_data.empty:
        calls_data = options_data[options_data["type"] == "call"]
        puts_data = options_data[options_data["type"] == "put"]

        expirations = sorted(options_data["expiration"].dropna().unique())
        expirations = expirations[:max_expirations]  # avoid too many legend entries

        color_map_2d = px.colors.qualitative.Prism

        fig = make_subplots(rows=1, cols=2, subplot_titles=["Calls", "Puts"], shared_yaxes=True)

        for i, exp in enumerate(expirations):
            color = color_map_2d[i % len(color_map_2d)]
            exp_calls = calls_data[calls_data["expiration"] == exp]
            exp_puts = puts_data[puts_data["expiration"] == exp]

            fig.add_trace(
                go.Scatter(
                    x=exp_calls["strike"],
                    y=exp_calls["implied_volatility"],
                    mode="markers",
                    marker=dict(color=color),
                    name=str(pd.to_datetime(exp).date()),
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=exp_puts["strike"],
                    y=exp_puts["implied_volatility"],
                    mode="markers",
                    marker=dict(color=color),
                    name=str(pd.to_datetime(exp).date()),
                    showlegend=False
                ),
                row=1, col=2
            )

        avg_iv_by_strike_calls = calls_data.groupby("strike")["implied_volatility"].mean()
        avg_iv_by_strike_puts = puts_data.groupby("strike")["implied_volatility"].mean()

        fig.add_trace(
            go.Scatter(
                x=avg_iv_by_strike_calls.index,
                y=avg_iv_by_strike_calls.values,
                mode="lines",
                line=dict(color="black", dash="dash"),
                name="Avg. IV by Strike (Calls)"
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=avg_iv_by_strike_puts.index,
                y=avg_iv_by_strike_puts.values,
                mode="lines",
                line=dict(color="black", dash="dash"),
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

    return None, None, None, None, None


def interpret_volatility_smile(ticker, overall_avg_iv_calls, overall_avg_iv_puts, avg_iv_by_strike_calls, avg_iv_by_strike_puts):
    interpretation = f"**Interpretation of {ticker} Volatility Smile:**\n"
    interpretation += f"- The average implied volatility for call options is {overall_avg_iv_calls:.2f}%.\n"
    interpretation += f"- The average implied volatility for put options is {overall_avg_iv_puts:.2f}%.\n"

    # With IV in % units, variance threshold should be larger than 0.1 (original was for smaller scales)
    if avg_iv_by_strike_calls.var() > 1.0:
        interpretation += "- Calls show noticeable IV variation across strikes (smile/skew).\n"
    else:
        interpretation += "- Calls do not show strong IV variation across strikes.\n"

    if avg_iv_by_strike_puts.var() > 1.0:
        interpretation += "- Puts show noticeable IV variation across strikes (smile/skew).\n"
    else:
        interpretation += "- Puts do not show strong IV variation across strikes.\n"

    market_sentiment = "bullish" if overall_avg_iv_calls > overall_avg_iv_puts else "bearish"
    interpretation += f"- Simple sentiment (avg IV calls vs puts): {market_sentiment}.\n"

    print(interpretation)


if __name__ == "__main__":
    options_data, recent_price = get_options_data(ticker, window=0.10, expired=False)
    print(options_data.head(10))

    overall_avg_iv_calls, overall_avg_iv_puts, avg_iv_by_strike_calls, avg_iv_by_strike_puts, expirations = \
        plot_volatility_smile(options_data, recent_price, ticker, max_expirations=10)

    if overall_avg_iv_calls is not None:
        interpret_volatility_smile(
            ticker,
            overall_avg_iv_calls,
            overall_avg_iv_puts,
            avg_iv_by_strike_calls,
            avg_iv_by_strike_puts
        )

#!/usr/bin/env python3
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# IMPORTANT: no "www"
DERIBIT = "https://deribit.com/api/v2"


def _get(path: str, params: dict):
    r = requests.get(f"{DERIBIT}/{path.lstrip('/')}", params=params, timeout=30)
    if not r.ok:
        # print body for debugging
        try:
            body = r.json()
        except Exception:
            body = r.text
        raise RuntimeError(f"HTTP {r.status_code} for {r.url}\nResponse: {body}")
    j = r.json()
    if j.get("error"):
        raise RuntimeError(f"API error: {j['error']}")
    return j["result"]


def _fetch_tv_chunk(instrument_name: str, start_ts: int, end_ts: int, resolution: str):
    params = {
        "instrument_name": instrument_name,
        "start_timestamp": start_ts,
        "end_timestamp": end_ts,
        "resolution": resolution,
    }
    return _get("public/get_tradingview_chart_data", params)


def fetch_tradingview_history(
    instrument_name: str,
    start_date: str,
    end_date: str,
    resolution_candidates=("1D", "D", "1440", "1"),
    chunk_days: int = 180,
):
    """
    Robust downloader for Deribit TradingView chart data.
    - Tries multiple resolution formats.
    - Downloads in chunks to avoid 400 errors on big ranges.
    """
    start = pd.Timestamp(start_date).tz_localize("UTC")
    end = pd.Timestamp(end_date).tz_localize("UTC")

    # convert to ms timestamps
    def to_ms(ts: pd.Timestamp) -> int:
        return int(ts.timestamp() * 1000)

    last_err = None

    for res in resolution_candidates:
        try:
            ticks_all, o_all, h_all, l_all, c_all = [], [], [], [], []
            cur = start

            while cur < end:
                nxt = min(cur + pd.Timedelta(days=chunk_days), end)
                data = _fetch_tv_chunk(instrument_name, to_ms(cur), to_ms(nxt), res)

                # validate payload
                if not data or "ticks" not in data or len(data["ticks"]) == 0:
                    # empty chunk: just move on
                    cur = nxt
                    continue

                ticks_all.extend(data.get("ticks", []))
                o_all.extend(data.get("open", []))
                h_all.extend(data.get("high", []))
                l_all.extend(data.get("low", []))
                c_all.extend(data.get("close", []))

                cur = nxt

            if not ticks_all:
                raise RuntimeError(f"No data returned for {instrument_name} with resolution={res}")

            df = pd.DataFrame({
                "date": pd.to_datetime(ticks_all, unit="ms", utc=True),
                "open": o_all,
                "high": h_all,
                "low": l_all,
                "close": c_all,
            }).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

            return df, res

        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"Failed all resolutions for {instrument_name}. Last error:\n{last_err}")


def plot_iv_vs_price(iv_df: pd.DataFrame, price_df: pd.DataFrame, title: str):
    df = pd.merge(iv_df, price_df, on="date", how="inner")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["close_x"], name="IV (DVOL close)", yaxis="y1"))
    fig.add_trace(go.Scatter(x=df["date"], y=df["close_y"], name="BTC close", yaxis="y2"))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis=dict(title="Implied Volatility (%)"),
        yaxis2=dict(title="BTC Price (USD)", overlaying="y", side="right"),
        legend=dict(orientation="h"),
    )
    fig.show()


if __name__ == "__main__":
    start_date = "2023-01-01"
    end_date = datetime.utcnow().strftime("%Y-%m-%d")

    # 1) IV index history (DVOL)
    iv_df, iv_res = fetch_tradingview_history("BTC-DVOL", start_date, end_date)
    print(f"Fetched BTC-DVOL with resolution={iv_res}, rows={len(iv_df)}")

    # 2) BTC price history (use perpetual as proxy)
    px_df, px_res = fetch_tradingview_history("BTC-PERPETUAL", start_date, end_date)
    print(f"Fetched BTC-PERPETUAL with resolution={px_res}, rows={len(px_df)}")

    plot_iv_vs_price(iv_df, px_df, title="BTC DVOL (IV) vs BTC Price (Deribit)")

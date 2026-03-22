#!/usr/bin/env python3
"""
Deribit options IV workflow (BTC/ETH) — CLI + CSV export (no plots).

What it does:
- Downloads all option instruments for BTC or ETH from Deribit (public API).
- Fetches per-instrument ticker (mark_iv, bid/ask, OI, volume, greeks, underlying price).
- Builds a clean pandas DataFrame.
- Optional strike filter around spot (e.g. ±10%).
- Prints summaries (IV, OI, volume, put/call ratio, top OI & volume strikes).
- Saves CSV.

Usage examples:
  python vol_analy.py --currency BTC --window 0.10 --out deribit_btc_options.csv
  python vol_analy.py --currency ETH --window 0.15 --max-instruments 800 --out eth.csv
  python vol_analy.py --currency BTC --expired --out btc_expired.csv
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

import requests
import pandas as pd


DERIBIT_BASE = "https://www.deribit.com/api/v2"
INS_RE = re.compile(r"^(BTC|ETH)-(\d{2}[A-Z]{3}\d{2})-(\d+)-(C|P)$")


@dataclass
class DeribitClient:
    base_url: str = DERIBIT_BASE
    timeout_s: int = 30
    sleep_s: float = 0.02  # polite pacing for public API

    def _normalize_params(self, params: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not params:
            return params
        out: Dict[str, Any] = {}
        for k, v in params.items():
            if isinstance(v, bool):
                out[k] = "true" if v else "false"  # Deribit is picky: lowercase
            else:
                out[k] = v
        return out

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        params = self._normalize_params(params)
        url = f"{self.base_url}/{path.lstrip('/')}"
        r = requests.get(url, params=params, timeout=self.timeout_s)
        # helpful error body on HTTP errors
        if not r.ok:
            try:
                body = r.json()
            except Exception:
                body = r.text
            raise requests.HTTPError(f"HTTP {r.status_code} for {r.url}\nResponse: {body}", response=r)
        payload = r.json()
        if "error" in payload and payload["error"]:
            raise RuntimeError(f"Deribit API error: {payload['error']}")
        return payload.get("result")

    def get_instruments(self, currency: str, kind: str = "option", expired: bool = False) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"currency": currency, "kind": kind}
        # IMPORTANT: don't send expired=false (can cause 400 depending on parsing); omit it
        if expired:
            params["expired"] = True
        return self.get("public/get_instruments", params=params)

    def get_ticker(self, instrument_name: str) -> Dict[str, Any]:
        res = self.get("public/ticker", params={"instrument_name": instrument_name})
        time.sleep(self.sleep_s)
        return res


def parse_option_name(name: str) -> Optional[Dict[str, Any]]:
    """
    Parse Deribit option instrument name like: BTC-29MAR24-65000-C
    """
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


def fetch_deribit_options_chain(
    currency: str = "BTC",
    expired: bool = False,
    window: float = 0.10,
    max_instruments: int = 0,
    sleep_s: float = 0.02,
) -> pd.DataFrame:
    """
    Fetch full option chain snapshot and return a DataFrame.
    window: strike filter around spot (e.g. 0.10 = ±10%). If <=0 => no filter.
    max_instruments: 0 means all; otherwise limit to first N instruments (useful for quick tests).
    """
    client = DeribitClient(sleep_s=sleep_s)

    instruments = client.get_instruments(currency=currency, kind="option", expired=expired)
    if not instruments:
        raise RuntimeError(f"No instruments returned for {currency} (expired={expired}).")

    # Keep only option-like names we can parse (defensive)
    names = []
    meta_by_name = {}
    for ins in instruments:
        name = ins.get("instrument_name")
        if not name:
            continue
        meta = parse_option_name(name)
        if meta is None:
            continue
        names.append(name)
        meta_by_name[name] = (ins, meta)

    if max_instruments and max_instruments > 0:
        names = names[:max_instruments]

    if not names:
        raise RuntimeError("No parsable option instrument names found.")

    rows: List[Dict[str, Any]] = []

    # First pass: fetch tickers; also collect underlying_price to estimate spot
    underlying_prices = []
    tickers_cache: Dict[str, Dict[str, Any]] = {}

    for name in names:
        t = client.get_ticker(name)
        tickers_cache[name] = t
        up = t.get("underlying_price")
        if isinstance(up, (int, float)) and pd.notna(up):
            underlying_prices.append(float(up))

    if not underlying_prices:
        # Fallback: Deribit index price
        idx = client.get("public/get_index_price", params={"index_name": f"{currency.lower()}_usd"})
        spot = float(idx)
    else:
        spot = float(pd.Series(underlying_prices).median())

    # Second pass: build rows, apply strike window filter later
    for name in names:
        ins, meta = meta_by_name[name]
        t = tickers_cache.get(name, {})

        stats = t.get("stats") or {}
        greeks = t.get("greeks") or {}

        row = {
            "instrument_name": name,
            "currency": currency,
            "type": meta["type"],
            "strike": meta["strike"],
            "expiration_timestamp": ins.get("expiration_timestamp"),
            "expiration": pd.to_datetime(ins.get("expiration_timestamp"), unit="ms", utc=True, errors="coerce"),
            "creation_timestamp": ins.get("creation_timestamp"),
            "contract_size": ins.get("contract_size"),
            "tick_size": ins.get("tick_size"),
            "min_trade_amount": ins.get("min_trade_amount"),
            "spot_ref": spot,

            # prices
            "mark_price": t.get("mark_price"),
            "best_bid_price": t.get("best_bid_price"),
            "best_ask_price": t.get("best_ask_price"),
            "underlying_price": t.get("underlying_price"),

            # IV (% on Deribit)
            "mark_iv": t.get("mark_iv"),
            "bid_iv": t.get("bid_iv"),
            "ask_iv": t.get("ask_iv"),

            # activity / positioning
            "open_interest": t.get("open_interest"),
            "volume": stats.get("volume"),                 # 24h volume (contracts)
            "volume_usd": stats.get("volume_usd"),         # 24h volume USD

            # greeks
            "delta": greeks.get("delta"),
            "gamma": greeks.get("gamma"),
            "theta": greeks.get("theta"),
            "vega": greeks.get("vega"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Basic cleanup
    for col in ["mark_iv", "bid_iv", "ask_iv", "mark_price", "best_bid_price", "best_ask_price",
                "open_interest", "volume", "volume_usd", "delta", "gamma", "theta", "vega", "underlying_price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Strike window filter around spot
    if window and window > 0:
        lo = (1.0 - window) * spot
        hi = (1.0 + window) * spot
        df = df[df["strike"].between(lo, hi)].copy()

    # Sort
    df = df.sort_values(["expiration", "type", "strike"], ascending=[True, True, True]).reset_index(drop=True)
    return df


def print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("Empty dataframe (after filters).")
        return

    spot = float(df["spot_ref"].iloc[0])
    currency = df["currency"].iloc[0]

    print(f"\n=== Deribit {currency} options snapshot ===")
    print(f"Spot reference (median underlying_price): {spot:,.2f}")
    print(f"Rows: {len(df):,} | Expirations: {df['expiration'].nunique():,}\n")

    # Summary by type
    summary = (
        df.groupby("type", dropna=True)
          .agg(
              avg_iv=("mark_iv", "mean"),
              median_iv=("mark_iv", "median"),
              avg_oi=("open_interest", "mean"),
              total_oi=("open_interest", "sum"),
              total_vol=("volume", "sum"),
              total_vol_usd=("volume_usd", "sum"),
          )
          .reset_index()
    )
    print("=== Summary by type ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    # Put/Call volume ratio
    put_vol = df.loc[df["type"] == "put", "volume"].sum(skipna=True)
    call_vol = df.loc[df["type"] == "call", "volume"].sum(skipna=True)
    if call_vol and call_vol > 0:
        print(f"\nPut/Call (24h volume) ratio: {put_vol / call_vol:.3f}")
    else:
        print("\nPut/Call (24h volume) ratio: N/A (call volume=0 or missing)")

    # Top OI strikes (aggregated across expiries)
    oi_by_strike = (
        df.groupby(["type", "strike"], dropna=True)["open_interest"]
          .sum(min_count=1)
          .reset_index()
          .sort_values(["type", "open_interest"], ascending=[True, False])
    )

    print("\n=== Top 5 OI strikes (aggregated across expiries) ===")
    for opt_type in ["call", "put"]:
        sub = oi_by_strike[oi_by_strike["type"] == opt_type].head(5)
        if sub.empty:
            continue
        print(f"\n{opt_type.upper()}:")
        for _, r in sub.iterrows():
            print(f"  strike={r['strike']:,.0f}  OI={int(r['open_interest']) if pd.notna(r['open_interest']) else 'NA'}")

    # Smile dispersion metric (std of IV across strikes, per type, per expiry)
    smile_std = (
        df.groupby(["type", "expiration"], dropna=True)["mark_iv"]
          .std()
          .reset_index()
          .rename(columns={"mark_iv": "iv_std_across_strikes"})
          .sort_values(["type", "expiration"])
    )
    # Print a compact tail/head
    print("\n=== Smile dispersion (IV std across strikes) — first 6 rows ===")
    print(smile_std.head(6).to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    # Expiry-level IV term structure (ATM-ish: closest strike to spot per expiry & type)
    df2 = df.copy()
    df2["moneyness_abs"] = (df2["strike"] - spot).abs()
    atm = (
        df2.sort_values(["type", "expiration", "moneyness_abs"])
           .groupby(["type", "expiration"], as_index=False)
           .first()[["type", "expiration", "strike", "mark_iv", "open_interest", "volume"]]
           .rename(columns={"strike": "atm_strike", "mark_iv": "atm_mark_iv"})
    )
    print("\n=== ATM-ish term structure (closest strike to spot) — first 10 rows ===")
    print(atm.head(10).to_string(index=False, float_format=lambda x: f"{x:,.4f}"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--currency", default="BTC", choices=["BTC", "ETH"], help="Underlying currency")
    ap.add_argument("--expired", action="store_true", help="Include expired instruments (can be large)")
    ap.add_argument("--window", type=float, default=0.10, help="Strike filter around spot (e.g. 0.10 = ±10%%). <=0 disables.")
    ap.add_argument("--max-instruments", type=int, default=0, help="Limit number of instruments (0 = all)")
    ap.add_argument("--sleep", type=float, default=0.02, help="Sleep between ticker calls (seconds)")
    ap.add_argument("--out", default="deribit_options.csv", help="Output CSV path")
    args = ap.parse_args()

    try:
        df = fetch_deribit_options_chain(
            currency=args.currency,
            expired=args.expired,
            window=args.window,
            max_instruments=args.max_instruments,
            sleep_s=args.sleep,
        )
    except Exception as e:
        print(f"\nERROR: {e}\n", file=sys.stderr)
        return 1

    print_summary(df)

    df.to_csv(args.out, index=False)
    print(f"\nSaved: {args.out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

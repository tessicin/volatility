#!/usr/bin/env python3
"""
Volatility Dashboard — Deribit Options Analysis
Run with:  streamlit run vol_dashboard.py
"""

import re
import io
import json
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timezone

# ──────────────────────────────────────────────
#  PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Vol Dashboard · Deribit",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
#  CUSTOM CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

  /* Dark theme overrides */
  .stApp { background-color: #0d0f14; color: #e8eaf0; }

  h1, h2, h3 { font-family: 'Space Mono', monospace; color: #f0f2ff; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #12151d;
    border-right: 1px solid #1e2330;
  }

  /* Metric cards */
  .metric-card {
    background: linear-gradient(135deg, #161a26 0%, #1a1f30 100%);
    border: 1px solid #252b3d;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px;
  }
  .metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: #6b7394;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 6px;
  }
  .metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 28px;
    font-weight: 700;
    color: #f0f2ff;
  }
  .metric-sub {
    font-size: 13px;
    color: #8890b0;
    margin-top: 4px;
  }

  /* Signal badges */
  .badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.05em;
  }
  .badge-bullish { background: #0d2a1a; color: #2ecc71; border: 1px solid #2ecc71; }
  .badge-bearish { background: #2a0d0d; color: #e74c3c; border: 1px solid #e74c3c; }
  .badge-neutral { background: #1a1a0d; color: #f39c12; border: 1px solid #f39c12; }

  /* Insight box */
  .insight-box {
    background: #13162000;
    border-left: 3px solid #5468ff;
    padding: 14px 18px;
    border-radius: 0 8px 8px 0;
    margin: 10px 0;
    font-size: 14px;
    line-height: 1.7;
    color: #c8ccde;
  }

  /* Decision panel */
  .decision-panel {
    background: linear-gradient(135deg, #13162080 0%, #1a1a2e80 100%);
    border: 1px solid #2a2f4a;
    border-radius: 14px;
    padding: 24px;
    margin-top: 16px;
  }

  /* Separator */
  hr { border-color: #1e2330; }

  /* Streamlit overrides */
  .stButton > button {
    background: linear-gradient(135deg, #5468ff, #7b6bfa);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    padding: 10px 28px;
    font-weight: 700;
    letter-spacing: 0.05em;
    transition: opacity 0.2s;
    width: 100%;
  }
  .stButton > button:hover { opacity: 0.85; }

  .stSelectbox label, .stSlider label, .stRadio label {
    color: #8890b0 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 12px !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }

  div[data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace;
    font-size: 24px;
  }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
#  DERIBIT API HELPERS
# ──────────────────────────────────────────────
DERIBIT = "https://deribit.com/api/v2"
INS_RE = re.compile(r"^(BTC|ETH)-(\d{2}[A-Z]{3}\d{2})-(\d+)-(C|P)$")

# ── Asset registry ─────────────────────────────────────────────────────────
# Deribit has options only for BTC and ETH.
# Other assets get price/OHLCV from Binance but NO options chain.
ASSET_REGISTRY = {
    # label           : (deribit_currency, binance_symbol, has_options, category)
    "BTC — Bitcoin"   : ("BTC", "BTCUSDT",  True,  "Major Crypto"),
    "ETH — Ethereum"  : ("ETH", "ETHUSDT",  True,  "Major Crypto"),
    "SOL — Solana"    : ("SOL", "SOLUSDT",  False, "Major Crypto"),
    "BNB — BNB Chain" : ("BNB", "BNBUSDT",  False, "Major Crypto"),
    "XRP — Ripple"    : ("XRP", "XRPUSDT",  False, "Major Crypto"),
    "ADA — Cardano"   : ("ADA", "ADAUSDT",  False, "Major Crypto"),
    "DOGE — Dogecoin" : ("DOG", "DOGEUSDT", False, "Meme"),
    "AVAX — Avalanche": ("AVX", "AVAXUSDT", False, "Major Crypto"),
    "LINK — Chainlink": ("LNK", "LINKUSDT", False, "Major Crypto"),
    "MATIC — Polygon" : ("MTK", "MATICUSDT",False, "Major Crypto"),
    "ARB — Arbitrum"  : ("ARB", "ARBUSDT",  False, "L2"),
    "OP — Optimism"   : ("OPT", "OPUSDT",   False, "L2"),
    "SUI — Sui"       : ("SUI", "SUIUSDT",  False, "Emerging"),
    "TIA — Celestia"  : ("TIA", "TIAUSDT",  False, "Emerging"),
    "INJ — Injective" : ("INJ", "INJUSDT",  False, "Emerging"),
}
BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
BINANCE_PRICE  = "https://api.binance.com/api/v3/ticker/price"

def asset_info(label: str) -> dict:
    rec = ASSET_REGISTRY.get(label, ("BTC", "BTCUSDT", True, "Major Crypto"))
    return {"deribit": rec[0], "binance": rec[1], "has_options": rec[2], "category": rec[3]}


@st.cache_data(ttl=60, show_spinner=False)
def binance_spot(symbol: str) -> float | None:
    try:
        r = requests.get(BINANCE_PRICE, params={"symbol": symbol}, timeout=10)
        return float(r.json()["price"])
    except Exception:
        return None


@st.cache_data(ttl=900, show_spinner=False)
def binance_ohlcv(symbol: str, interval: str, limit: int = 1000) -> pd.DataFrame:
    """Fetch OHLCV from Binance for non-Deribit assets."""
    interval_map = {"1D": "1d", "4H": "4h", "1H": "1h", "15m": "15m"}
    try:
        r = requests.get(BINANCE_KLINES, params={
            "symbol": symbol, "interval": interval_map.get(interval, "1d"), "limit": limit
        }, timeout=20)
        raw = r.json()
        df = pd.DataFrame(raw, columns=[
            "ts","open","high","low","close","volume","close_time",
            "qav","n_trades","taker_base","taker_quote","ignore"])
        df["ts"]    = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["high"]  = pd.to_numeric(df["high"],  errors="coerce")
        df["low"]   = pd.to_numeric(df["low"],   errors="coerce")
        df["volume"]= pd.to_numeric(df["volume"],errors="coerce")
        df = df.set_index("ts").sort_index()[["close","high","low","volume"]]
        df = df[~df.index.duplicated(keep="last")]
        # Compute all indicators
        c = df["close"]
        df["ma20"]     = c.rolling(20).mean()
        df["ma50"]     = c.rolling(50).mean()
        df["ma200"]    = c.rolling(200, min_periods=50).mean()
        delta = c.diff(); gain = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        df["rsi"]      = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
        ema12 = c.ewm(span=12).mean(); ema26 = c.ewm(span=26).mean()
        df["macd"]     = ema12 - ema26
        df["macd_sig"] = df["macd"].ewm(span=9).mean()
        df["macd_hist"]= df["macd"] - df["macd_sig"]
        bb_mid = c.rolling(20).mean(); bb_std = c.rolling(20).std()
        df["bb_upper"] = bb_mid + 2*bb_std; df["bb_lower"] = bb_mid - 2*bb_std
        df["bb_pct"]   = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
        df["ath"]      = c.expanding().max()
        df["drawdown"] = (c - df["ath"]) / df["ath"] * 100
        log_r = np.log(c / c.shift(1))
        bars_year = {"1D":365,"4H":365*6,"1H":365*24,"15m":365*96}.get(interval,365)
        df["rv30"]     = log_r.rolling(30).std() * np.sqrt(bars_year) * 100
        df["vol_z"]    = (df["volume"] - df["volume"].rolling(20).mean()) / (df["volume"].rolling(20).std() + 1e-9)
        df["roc5"]     = c.pct_change(5)  * 100
        df["roc20"]    = c.pct_change(20) * 100
        return df
    except Exception:
        return pd.DataFrame()


def deribit_get(path: str, params: dict | None = None) -> dict:
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
    m = INS_RE.match(name)
    if not m:
        return None
    underlying, exp_code, strike, cp = m.groups()
    return {"underlying": underlying, "expiration_code": exp_code,
            "strike": float(strike), "type": "call" if cp == "C" else "put"}


# ──────────────────────────────────────────────
#  DATA FETCHING (cached)
# ──────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def get_options_data(ticker: str, window: float = 0.12, sleep_s: float = 0.02):
    instruments = deribit_get("public/get_instruments", {"currency": ticker, "kind": "option"})
    if not instruments:
        return None, None

    rows, underlying_prices = [], []

    progress = st.progress(0, text="Fetching options chain…")
    total = len(instruments)

    for idx, ins in enumerate(instruments):
        name = ins.get("instrument_name")
        meta = parse_option_name(name) if name else None
        if meta is None:
            continue
        try:
            t = deribit_get("public/ticker", {"instrument_name": name})
        except Exception:
            continue
        stats = t.get("stats") or {}
        up = t.get("underlying_price")
        if isinstance(up, (int, float)):
            underlying_prices.append(float(up))

        rows.append({
            "contractSymbol": name,
            "strike": meta["strike"],
            "bid": t.get("best_bid_price"),
            "ask": t.get("best_ask_price"),
            "volume": stats.get("volume"),
            "openInterest": t.get("open_interest"),
            "impliedVolatility": t.get("mark_iv"),
            "delta": (t.get("greeks") or {}).get("delta"),
            "gamma": (t.get("greeks") or {}).get("gamma"),
            "vega": (t.get("greeks") or {}).get("vega"),
            "theta": (t.get("greeks") or {}).get("theta"),
            "expiration": pd.to_datetime(ins.get("expiration_timestamp"), unit="ms", utc=True, errors="coerce"),
            "type": meta["type"],
        })
        progress.progress(min((idx + 1) / total, 1.0), text=f"Loading… {idx+1}/{total}")
        time.sleep(sleep_s)

    progress.empty()

    if not rows:
        return None, None

    options_data = pd.DataFrame(rows)
    recent_price = float(pd.Series(underlying_prices).median()) if underlying_prices else float(
        deribit_get("public/get_index_price", {"index_name": f"{ticker.lower()}_usd"})["index_price"])

    options_data = options_data[
        options_data["strike"].between(recent_price * (1 - window), recent_price * (1 + window))
    ].copy()

    for col in ["openInterest", "volume", "strike", "impliedVolatility", "delta", "gamma", "vega", "theta"]:
        options_data[col] = pd.to_numeric(options_data[col], errors="coerce")

    return options_data, recent_price


# ──────────────────────────────────────────────
#  ANALYSIS HELPERS
# ──────────────────────────────────────────────
def compute_skew(calls, puts):
    """25-delta skew proxy: OTM put IV − OTM call IV (positive = fear/bearish)"""
    try:
        otm_puts = puts[puts["delta"].notna() & (puts["delta"].abs() < 0.4) & (puts["delta"].abs() > 0.1)]
        otm_calls = calls[calls["delta"].notna() & (calls["delta"] > 0.1) & (calls["delta"] < 0.4)]
        if otm_puts.empty or otm_calls.empty:
            return None
        return otm_puts["impliedVolatility"].mean() - otm_calls["impliedVolatility"].mean()
    except Exception:
        return None


def term_structure(data: pd.DataFrame) -> pd.DataFrame:
    """ATM IV per expiration (proxy ATM = median strike bucket)"""
    rows = []
    for exp, grp in data.groupby("expiration"):
        atm_iv = grp["impliedVolatility"].median()
        oi_total = grp["openInterest"].sum()
        rows.append({"expiration": exp, "atm_iv": atm_iv, "oi": oi_total})
    df = pd.DataFrame(rows).sort_values("expiration")
    df["days"] = (pd.to_datetime(df["expiration"], utc=True) - pd.Timestamp.now(tz="UTC")).dt.days
    return df


def put_call_ratio(calls, puts):
    oi_c = calls["openInterest"].sum()
    oi_p = puts["openInterest"].sum()
    if oi_c == 0:
        return None
    return oi_p / oi_c


def sentiment_signal(skew, pcr, iv_calls, iv_puts):
    """Returns (signal_text, badge_class, explanation)"""
    signals = []

    if skew is not None:
        if skew > 5:
            signals.append(("bearish", f"Skew +{skew:.1f}% → il mercato paga di più le put OTM (protezione dal ribasso)"))
        elif skew < -5:
            signals.append(("bullish", f"Skew {skew:.1f}% → il mercato paga di più le call OTM (domanda rialzista)"))
        else:
            signals.append(("neutral", f"Skew {skew:.1f}% → asimmetria contenuta"))

    if pcr is not None:
        if pcr > 1.2:
            signals.append(("bearish", f"Put/Call OI ratio {pcr:.2f} → più copertura/scommesse ribassiste"))
        elif pcr < 0.8:
            signals.append(("bullish", f"Put/Call OI ratio {pcr:.2f} → posizionamento netto rialzista"))
        else:
            signals.append(("neutral", f"Put/Call OI ratio {pcr:.2f} → bilanciato"))

    if iv_calls > iv_puts + 3:
        signals.append(("bullish", f"IV calls ({iv_calls:.1f}%) > IV puts ({iv_puts:.1f}%) → premi call elevati"))
    elif iv_puts > iv_calls + 3:
        signals.append(("bearish", f"IV puts ({iv_puts:.1f}%) > IV calls ({iv_calls:.1f}%) → premi put elevati"))

    bullish = sum(1 for s, _ in signals if s == "bullish")
    bearish = sum(1 for s, _ in signals if s == "bearish")

    if bullish > bearish:
        overall = "bullish"
    elif bearish > bullish:
        overall = "bearish"
    else:
        overall = "neutral"

    return overall, signals




@st.cache_data(ttl=900, show_spinner=False)
def get_perpetual_history(ticker: str, resolution: str = "60", lookback_days: int = 240):
    end_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
    start_ms = int((pd.Timestamp.utcnow() - pd.Timedelta(days=lookback_days)).timestamp() * 1000)
    payload = deribit_get(
        "public/get_tradingview_chart_data",
        {
            "instrument_name": f"{ticker}-PERPETUAL",
            "start_timestamp": start_ms,
            "end_timestamp": end_ms,
            "resolution": resolution,
        },
    )
    closes = payload.get("close", [])
    ticks = payload.get("ticks", [])
    if not closes or not ticks:
        return pd.DataFrame()
    hist = pd.DataFrame({
        "timestamp": pd.to_datetime(ticks, unit="ms", utc=True),
        "close": pd.to_numeric(closes, errors="coerce"),
    }).dropna()
    hist = hist[hist["close"] > 0].copy()
    hist["log_ret"] = np.log(hist["close"]).diff()
    hist["date"] = hist["timestamp"].dt.floor("D")
    rv = hist.groupby("date", as_index=False)["log_ret"].apply(lambda s: np.square(s.dropna()).sum())
    rv = rv.rename(columns={"log_ret": "rv_var"})
    rv["rv_var"] = rv["rv_var"].clip(lower=1e-12)
    rv["rv_ann"] = np.sqrt(rv["rv_var"] * 365.0)
    rv["rv_ann_pct"] = rv["rv_ann"] * 100.0
    rv["log_rv"] = np.log(rv["rv_ann"].clip(lower=1e-12))
    return rv


def estimate_hurst_variogram(series: pd.Series, max_lag: int = 20):
    x = pd.Series(series).dropna().astype(float).values
    if len(x) < max_lag + 10:
        return np.nan
    lags = np.arange(2, min(max_lag, len(x) // 3) + 1)
    tau = []
    valid_lags = []
    for lag in lags:
        diffs = x[lag:] - x[:-lag]
        if len(diffs) > 5 and np.std(diffs) > 0:
            tau.append(np.std(diffs))
            valid_lags.append(lag)
    if len(valid_lags) < 4:
        return np.nan
    slope = np.polyfit(np.log(valid_lags), np.log(tau), 1)[0]
    return float(np.clip(slope, 0.01, 0.99))


def _fit_linear(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    beta = np.linalg.pinv(X.T @ X + 1e-8 * np.eye(X.shape[1])) @ X.T @ y
    return beta


def _predict_linear(X, beta):
    return np.asarray(X, dtype=float) @ np.asarray(beta, dtype=float)


def frac_weights(d: float, size: int):
    w = [1.0]
    for k in range(1, size):
        w.append(w[-1] * ((k - 1 - d) / k))
    return np.array(w, dtype=float)


def frac_feature(values: np.ndarray, t: int, d: float, depth: int = 30):
    k = min(depth, t + 1)
    w = frac_weights(d, k)
    window = values[t - k + 1:t + 1][::-1]
    return float(np.dot(w, window))


def rough_feature(values: np.ndarray, t: int, H: float, depth: int = 30):
    if t < 2:
        return 0.0
    k = min(depth, t)
    diffs = np.diff(values[t - k:t + 1])
    idx = np.arange(1, len(diffs) + 1)
    kernel = idx ** (H - 0.5)
    return float(np.dot(kernel[::-1], diffs))


def build_direct_dataset(y: pd.Series, horizon: int, model: str, d: float | None = None, H: float | None = None, depth: int = 30):
    values = pd.Series(y).dropna().astype(float).values
    max_back = max(30, depth)
    rows, target, idx = [], [], []
    for t in range(max_back, len(values) - horizon):
        yt = values[t]
        feats = [1.0]
        if model == "ar1":
            feats += [yt]
        elif model == "har":
            feats += [yt, values[t-6:t+1].mean(), values[t-29:t+1].mean()]
        elif model == "arfima":
            feats += [yt, frac_feature(values, t, d=d if d is not None else 0.2, depth=depth)]
        elif model == "rough":
            feats += [yt, rough_feature(values, t, H=H if H is not None else 0.2, depth=depth)]
        else:
            raise ValueError(f"Unknown model: {model}")
        rows.append(feats)
        target.append(values[t + horizon])
        idx.append(t)
    return np.asarray(rows, dtype=float), np.asarray(target, dtype=float), np.asarray(idx, dtype=int)


def tune_fractional_model(y: pd.Series, horizon: int, model: str, base_h: float | None = None):
    if model == "arfima":
        grid = np.arange(0.05, 0.46, 0.05)
        param_name = "d"
    else:
        h0 = 0.18 if base_h is None or np.isnan(base_h) else float(np.clip(base_h, 0.05, 0.45))
        low = max(0.05, h0 - 0.12)
        high = min(0.49, h0 + 0.12)
        grid = np.round(np.linspace(low, high, 6), 3)
        param_name = "H"
    best = None
    for param in grid:
        kwargs = {param_name: float(param)}
        X, yt, _ = build_direct_dataset(y, horizon, model=model, **kwargs)
        if len(yt) < 80:
            continue
        split = max(40, int(len(yt) * 0.8))
        beta = _fit_linear(X[:split], yt[:split])
        pred = _predict_linear(X[split:], beta)
        rmse = float(np.sqrt(np.mean((yt[split:] - pred) ** 2)))
        if best is None or rmse < best["rmse"]:
            best = {"param": float(param), "rmse": rmse}
    return best


def evaluate_forecast_models(log_rv: pd.Series):
    log_rv = pd.Series(log_rv).dropna().astype(float).reset_index(drop=True)
    base_h = estimate_hurst_variogram(log_rv)
    horizons = [1, 3, 7]
    rows = []
    pred_rows = []
    models = [("ar1", "AR(1)"), ("har", "HAR-RV"), ("arfima", "ARFIMA-like"), ("rough", "Rough Vol")]
    for horizon in horizons:
        tuned = {
            "arfima": tune_fractional_model(log_rv, horizon, "arfima"),
            "rough": tune_fractional_model(log_rv, horizon, "rough", base_h=base_h),
        }
        for model_key, model_name in models:
            kwargs = {}
            if model_key in tuned and tuned[model_key] is not None:
                if model_key == "arfima":
                    kwargs["d"] = tuned[model_key]["param"]
                else:
                    kwargs["H"] = tuned[model_key]["param"]
            X, y_target, _ = build_direct_dataset(log_rv, horizon, model_key, **kwargs)
            if len(y_target) < 60:
                continue
            split = max(40, int(len(y_target) * 0.8))
            beta = _fit_linear(X[:split], y_target[:split])
            pred_test = _predict_linear(X[split:], beta)
            rmse = float(np.sqrt(np.mean((y_target[split:] - pred_test) ** 2)))
            mae = float(np.mean(np.abs(y_target[split:] - pred_test)))

            beta_full = _fit_linear(X, y_target)
            next_pred_log = float(_predict_linear(X[-1:], beta_full)[0])
            next_pred_ann = float(np.exp(next_pred_log) * 100.0)
            actual_test_ann = np.exp(y_target[split:]) * 100.0
            pred_test_ann = np.exp(pred_test) * 100.0
            rmse_ann = float(np.sqrt(np.mean((actual_test_ann - pred_test_ann) ** 2)))
            pred_rows.append({
                "horizon": horizon,
                "model": model_name,
                "forecast_log_rv": next_pred_log,
                "forecast_rv_pct": next_pred_ann,
                "rmse_log": rmse,
                "rmse_rv_pct": rmse_ann,
                "param": None if model_key not in tuned or tuned[model_key] is None else tuned[model_key]["param"],
            })
            rows.append({
                "horizon": horizon,
                "model": model_name,
                "rmse_log": rmse,
                "mae_log": mae,
                "rmse_rv_pct": rmse_ann,
                "param": None if model_key not in tuned or tuned[model_key] is None else tuned[model_key]["param"],
            })
    score_df = pd.DataFrame(rows)
    forecast_df = pd.DataFrame(pred_rows)
    if score_df.empty:
        return {"hurst": base_h, "scores": score_df, "forecasts": forecast_df, "best_overall": None}
    overall = score_df.groupby("model", as_index=False)["rmse_log"].mean().sort_values("rmse_log")
    best_overall = overall.iloc[0]["model"] if not overall.empty else None
    return {
        "hurst": base_h,
        "scores": score_df,
        "forecasts": forecast_df,
        "best_overall": best_overall,
        "overall": overall,
    }


def chart_rv_history(rv_df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rv_df["date"], y=rv_df["rv_ann_pct"],
        mode="lines", line=dict(color="#5468ff", width=2),
        name="Realized Vol",
        hovertemplate="%{x|%Y-%m-%d}<br>RV: %{y:.2f}%<extra></extra>"
    ))
    fig.update_layout(**DARK_LAYOUT, title="Realized Volatility (annualized)", height=320, xaxis_title="Date", yaxis_title="RV (%)")
    return fig


def chart_log_rv(rv_df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rv_df["date"], y=rv_df["log_rv"],
        mode="lines", line=dict(color="#2ecc71", width=2),
        name="log(RV)",
        hovertemplate="%{x|%Y-%m-%d}<br>log(RV): %{y:.3f}<extra></extra>"
    ))
    fig.update_layout(**DARK_LAYOUT, title="Target series: log(RV)", height=320, xaxis_title="Date", yaxis_title="log(RV)")
    return fig


def chart_model_rmse(score_df: pd.DataFrame):
    if score_df.empty:
        return go.Figure()
    fig = px.bar(score_df, x="model", y="rmse_rv_pct", color="horizon", barmode="group",
                 title="Out-of-sample RMSE on annualized RV forecast")
    fig.update_layout(**DARK_LAYOUT, height=360, xaxis_title="Model", yaxis_title="RMSE (vol points)")
    return fig


def build_vol_signal(atm_iv_pct: float, forecast_rv_pct: float, model_rmse_pct: float, horizon: int):
    edge = float(atm_iv_pct - forecast_rv_pct)
    threshold = max(3.0, 0.75 * float(model_rmse_pct))
    if edge > threshold:
        signal = "SELL VOL"
        cls = "badge-bearish"
        why = (
            f"L'IV ATM di mercato è {atm_iv_pct:.2f}%, sopra la RV prevista a {horizon}g ({forecast_rv_pct:.2f}%) "
            f"di {edge:.2f} punti, oltre la soglia operativa di {threshold:.2f}. "
            f"Il mercato sta pagando volatilità più cara di quanto il modello si aspetti."
        )
    elif edge < -threshold:
        signal = "BUY VOL"
        cls = "badge-bullish"
        why = (
            f"L'IV ATM di mercato è {atm_iv_pct:.2f}%, sotto la RV prevista a {horizon}g ({forecast_rv_pct:.2f}%) "
            f"di {abs(edge):.2f} punti, oltre la soglia operativa di {threshold:.2f}. "
            f"Il mercato sta prezzando meno volatilità di quella attesa dal modello."
        )
    else:
        signal = "HOLD / NO EDGE"
        cls = "badge-neutral"
        why = (
            f"La differenza tra IV ATM ({atm_iv_pct:.2f}%) e RV prevista ({forecast_rv_pct:.2f}%) è {edge:.2f} punti, "
            f"dentro la fascia di rumore del modello ({threshold:.2f}). Non emerge un edge pulito."
        )
    return {"signal": signal, "badge_class": cls, "edge": edge, "threshold": threshold, "why": why}


# ──────────────────────────────────────────────
#  CHART FACTORIES
# ──────────────────────────────────────────────
DARK_LAYOUT = dict(
    paper_bgcolor="#0d0f14",
    plot_bgcolor="#0d0f14",
    font=dict(family="DM Sans", color="#c8ccde"),
    title_font=dict(family="Space Mono", color="#f0f2ff", size=14),
    legend=dict(bgcolor="#12151d", bordercolor="#1e2330", borderwidth=1),
    xaxis=dict(gridcolor="#1e2330", zerolinecolor="#252b3d"),
    yaxis=dict(gridcolor="#1e2330", zerolinecolor="#252b3d"),
)


def chart_vol_smile(calls, puts, recent_price, max_exp=8):
    expirations = sorted(calls["expiration"].dropna().unique())[:max_exp]
    palette = px.colors.qualitative.Prism

    fig = make_subplots(rows=1, cols=2, subplot_titles=["CALLS", "PUTS"], shared_yaxes=True,
                        horizontal_spacing=0.06)

    for i, exp in enumerate(expirations):
        color = palette[i % len(palette)]
        label = str(pd.to_datetime(exp).date())
        ec = calls[calls["expiration"] == exp].sort_values("strike")
        ep = puts[puts["expiration"] == exp].sort_values("strike")

        for df, col, show in [(ec, 1, True), (ep, 2, False)]:
            fig.add_trace(go.Scatter(
                x=df["strike"], y=df["impliedVolatility"],
                mode="markers+lines", marker=dict(color=color, size=6),
                line=dict(color=color, width=1.2, dash="dot"),
                name=label, showlegend=show,
            ), row=1, col=col)

    # Avg IV line
    for df, col in [(calls, 1), (puts, 2)]:
        avg = df.groupby("strike")["impliedVolatility"].mean()
        fig.add_trace(go.Scatter(
            x=avg.index, y=avg.values, mode="lines",
            line=dict(color="#f0f2ff", dash="dash", width=2),
            name="Avg IV", showlegend=(col == 1),
        ), row=1, col=col)

    # Spot line
    for col in [1, 2]:
        fig.add_vline(x=recent_price, line=dict(color="#5468ff", dash="dash", width=1.5),
                      annotation_text=f"SPOT {recent_price:,.0f}", row=1, col=col)

    fig.update_layout(**DARK_LAYOUT,
                      title="Volatility Smile — IV per strike",
                      height=440, showlegend=True,
                      legend_title_text="Scadenza")
    fig.update_xaxes(title_text="Strike Price")
    fig.update_yaxes(title_text="IV (%)")
    return fig


def chart_term_structure(ts_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts_df["days"], y=ts_df["atm_iv"],
        mode="markers+lines",
        marker=dict(color="#5468ff", size=9),
        line=dict(color="#5468ff", width=2),
        text=[str(pd.to_datetime(e).date()) for e in ts_df["expiration"]],
        hovertemplate="<b>%{text}</b><br>Days: %{x}<br>ATM IV: %{y:.1f}%<extra></extra>",
    ))
    # Backwardation / contango shading
    if len(ts_df) > 1:
        slope = np.polyfit(ts_df["days"], ts_df["atm_iv"], 1)[0]
        color = "#e74c3c" if slope < 0 else "#2ecc71"
        fig.add_trace(go.Scatter(
            x=ts_df["days"], y=ts_df["atm_iv"],
            fill="tozeroy", fillcolor=f"rgba({'231,76,60' if slope < 0 else '46,204,113'},0.08)",
            mode="none", showlegend=False,
        ))

    fig.update_layout(**DARK_LAYOUT,
                      title="Term Structure — ATM IV vs giorni alla scadenza",
                      xaxis_title="Giorni alla scadenza",
                      yaxis_title="ATM IV (%)",
                      height=360)
    return fig


def chart_open_interest(calls, puts, recent_price, max_exp=8):
    expirations = sorted(calls["expiration"].dropna().unique())[:max_exp]
    palette = px.colors.qualitative.Prism

    fig = make_subplots(rows=1, cols=2, subplot_titles=["CALLS OI", "PUTS OI"], shared_yaxes=True,
                        horizontal_spacing=0.06)

    for i, exp in enumerate(expirations):
        color = palette[i % len(palette)]
        label = str(pd.to_datetime(exp).date())
        ec = calls[calls["expiration"] == exp].sort_values("strike")
        ep = puts[puts["expiration"] == exp].sort_values("strike")

        for df, col, show in [(ec, 1, True), (ep, 2, False)]:
            fig.add_trace(go.Bar(
                x=df["strike"], y=df["openInterest"],
                marker_color=color, name=label, showlegend=show,
                opacity=0.75,
            ), row=1, col=col)

    for col in [1, 2]:
        fig.add_vline(x=recent_price, line=dict(color="#5468ff", dash="dash", width=1.5),
                      annotation_text=f"SPOT", row=1, col=col)

    fig.update_layout(**DARK_LAYOUT,
                      title="Open Interest per strike",
                      barmode="overlay", height=420,
                      showlegend=True, legend_title_text="Scadenza")
    fig.update_xaxes(title_text="Strike Price")
    fig.update_yaxes(title_text="Open Interest")
    return fig


def chart_pcr_by_expiry(calls, puts):
    exps = sorted(calls["expiration"].dropna().unique())
    rows = []
    for exp in exps:
        oi_c = calls[calls["expiration"] == exp]["openInterest"].sum()
        oi_p = puts[puts["expiration"] == exp]["openInterest"].sum()
        if oi_c > 0:
            rows.append({"exp": str(pd.to_datetime(exp).date()), "pcr": oi_p / oi_c})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    colors = ["#e74c3c" if v > 1.0 else "#2ecc71" for v in df["pcr"]]
    fig = go.Figure(go.Bar(x=df["exp"], y=df["pcr"], marker_color=colors, text=df["pcr"].round(2),
                           textposition="outside"))
    fig.add_hline(y=1.0, line=dict(color="#f39c12", dash="dash"), annotation_text="PCR = 1")
    fig.update_layout(**DARK_LAYOUT,
                      title="Put/Call Ratio OI per scadenza",
                      xaxis_title="Scadenza",
                      yaxis_title="PCR",
                      height=320)
    return fig


def chart_skew_by_expiry(calls, puts):
    """Bar chart of put-call IV skew per expiry"""
    exps = sorted(calls["expiration"].dropna().unique())
    rows = []
    for exp in exps:
        ec = calls[calls["expiration"] == exp]
        ep = puts[puts["expiration"] == exp]
        iv_c = ec["impliedVolatility"].mean()
        iv_p = ep["impliedVolatility"].mean()
        if pd.notna(iv_c) and pd.notna(iv_p):
            rows.append({"exp": str(pd.to_datetime(exp).date()), "skew": iv_p - iv_c})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in df["skew"]]
    fig = go.Figure(go.Bar(x=df["exp"], y=df["skew"], marker_color=colors,
                           text=df["skew"].round(1), textposition="outside"))
    fig.add_hline(y=0, line=dict(color="#f0f2ff", dash="dash"))
    fig.update_layout(**DARK_LAYOUT,
                      title="IV Skew per scadenza (put IV − call IV)",
                      xaxis_title="Scadenza", yaxis_title="Skew (%)",
                      height=320)
    return fig


# ──────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ VOL DASHBOARD")
    st.markdown("---")

    # ── Asset selector ────────────────────────────────────────────
    asset_labels   = list(ASSET_REGISTRY.keys())
    asset_label    = st.selectbox("Asset", asset_labels, index=0, key="sidebar_asset")
    ainfo          = asset_info(asset_label)
    ticker         = ainfo["deribit"] if ainfo["has_options"] else asset_label.split(" — ")[0]
    has_options    = ainfo["has_options"]
    binance_sym    = ainfo["binance"]

    if not has_options:
        st.markdown(f"""<div style='background:#1a1a0a;border:1px solid #f39c1240;border-radius:8px;
                    padding:8px 12px;font-size:11px;color:#f39c12;font-family:Space Mono,monospace;'>
                    ⚠️ {asset_label.split(' — ')[0]}: nessuna catena opzioni su Deribit.<br>
                    Dati OHLCV da Binance. Tab opzioni disabilitate.</div>""", unsafe_allow_html=True)

    window = st.slider("Strike window (%)", min_value=5, max_value=30, value=12, step=1) / 100
    max_exp = st.slider("Max scadenze", 3, 15, 8)
    st.markdown("---")
    run = st.button("🔄 Carica dati live")
    st.markdown("---")
    st.markdown("""
    <div style='font-family:Space Mono,monospace;font-size:10px;color:#3a4060;line-height:1.8;'>
    Dati: Deribit API v2<br>
    Aggiornamento: ogni 5 min<br>
    Strike window: ± % dal spot<br><br>
    ⚠️ Non è consulenza finanziaria
    </div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  HEADER
# ──────────────────────────────────────────────
st.markdown(f"""
<h1 style='font-family:Space Mono,monospace;font-size:28px;margin-bottom:0;'>
  {ticker} OPTIONS VOLATILITY DASHBOARD
</h1>
<p style='color:#6b7394;font-family:Space Mono,monospace;font-size:12px;margin-top:4px;'>
  LIVE · DERIBIT · {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ──────────────────────────────────────────────
#  DATA LOAD
# ──────────────────────────────────────────────
if "data" not in st.session_state:
    st.session_state.data = None
    st.session_state.recent_price = None
    st.session_state.ticker_loaded = None

if run or (st.session_state.data is None) or (st.session_state.ticker_loaded != asset_label):
    if has_options:
        with st.spinner(f"Connessione a Deribit per {ticker}…"):
            try:
                data, recent_price = get_options_data.clear() or get_options_data(ticker, window)
                st.session_state.data = data
                st.session_state.recent_price = recent_price
                st.session_state.ticker_loaded = asset_label
            except Exception as e:
                st.error(f"Errore API Deribit: {e}")
                st.stop()
    else:
        # Non-options asset: get spot from Binance, set data to empty DF
        with st.spinner(f"Caricamento spot {asset_label} da Binance…"):
            spot = binance_spot(binance_sym)
            if spot is None:
                st.error(f"Impossibile caricare il prezzo di {asset_label} da Binance.")
                st.stop()
            st.session_state.data = pd.DataFrame()   # no options chain
            st.session_state.recent_price = spot
            st.session_state.ticker_loaded = asset_label

data = st.session_state.data
recent_price = st.session_state.recent_price

# For non-options assets data may be empty — guard downstream calls
_has_data = data is not None and not data.empty

if has_options and not _has_data:
    st.warning("Nessun dato opzioni disponibile. Premi 'Carica dati live'.")
    st.stop()

if _has_data:
    calls = data[data["type"] == "call"].copy()
    puts  = data[data["type"] == "put"].copy()
else:
    calls = pd.DataFrame()
    puts  = pd.DataFrame()

# ──────────────────────────────────────────────
#  KEY METRICS ROW
# ──────────────────────────────────────────────
avg_iv_c = calls["impliedVolatility"].mean() if _has_data else float("nan")
avg_iv_p = puts["impliedVolatility"].mean()  if _has_data else float("nan")
pcr  = put_call_ratio(calls, puts)  if _has_data else None
skew = compute_skew(calls, puts)    if _has_data else None
overall, signals = sentiment_signal(skew, pcr, avg_iv_c if pd.notna(avg_iv_c) else 0,
                                    avg_iv_p if pd.notna(avg_iv_p) else 0)
ts_df = term_structure(data) if _has_data else pd.DataFrame()

badge_map = {"bullish": "badge-bullish", "bearish": "badge-bearish", "neutral": "badge-neutral"}
emoji_map = {"bullish": "🟢 RIALZISTA", "bearish": "🔴 RIBASSISTA", "neutral": "🟡 NEUTRALE"}

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-label'>Spot Price</div>
      <div class='metric-value'>${recent_price:,.0f}</div>
      <div class='metric-sub'>{ticker}/USD</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-label'>Avg IV — Calls</div>
      <div class='metric-value'>{avg_iv_c:.1f}%</div>
      <div class='metric-sub'>Volatilità implicita media</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-label'>Avg IV — Puts</div>
      <div class='metric-value'>{avg_iv_p:.1f}%</div>
      <div class='metric-sub'>Volatilità implicita media</div>
    </div>""", unsafe_allow_html=True)

with col4:
    pcr_str = f"{pcr:.2f}" if pcr else "N/A"
    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-label'>Put/Call OI Ratio</div>
      <div class='metric-value'>{pcr_str}</div>
      <div class='metric-sub'>&gt;1 bearish · &lt;1 bullish</div>
    </div>""", unsafe_allow_html=True)

with col5:
    skew_str = f"{skew:+.1f}%" if skew else "N/A"
    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-label'>IV Skew (OTM)</div>
      <div class='metric-value'>{skew_str}</div>
      <div class='metric-sub'>put IV − call IV</div>
    </div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  SENTIMENT SIGNAL BOX
# ──────────────────────────────────────────────
st.markdown(f"""
<div class='decision-panel'>
  <div style='display:flex;align-items:center;gap:16px;margin-bottom:14px;'>
    <span style='font-family:Space Mono,monospace;font-size:13px;color:#6b7394;'>SEGNALE DI MERCATO</span>
    <span class='badge {badge_map[overall]}'>{emoji_map[overall]}</span>
  </div>
  <div>
""", unsafe_allow_html=True)

for s, explanation in signals:
    icon = "▲" if s == "bullish" else ("▼" if s == "bearish" else "◆")
    color = "#2ecc71" if s == "bullish" else ("#e74c3c" if s == "bearish" else "#f39c12")
    st.markdown(f"""
    <div class='insight-box' style='border-left-color:{color};margin:6px 0;'>
      <span style='color:{color};font-weight:700;'>{icon}</span> {explanation}
    </div>""", unsafe_allow_html=True)

# Term structure interpretation
if len(ts_df) > 1:
    slope = np.polyfit(ts_df["days"], ts_df["atm_iv"], 1)[0]
    if slope < -0.05:
        ts_msg = "📉 <b>Backwardation:</b> la IV a breve termine è più alta → stress/incertezza imminente. Il mercato teme movimenti bruschi nel breve."
        ts_color = "#e74c3c"
    elif slope > 0.05:
        ts_msg = "📈 <b>Contango:</b> la IV cresce con la scadenza → situazione normale, il mercato è relativamente calmo nel breve."
        ts_color = "#2ecc71"
    else:
        ts_msg = "➡️ <b>Term structure piatta:</b> incertezza distribuita su tutte le scadenze."
        ts_color = "#f39c12"

    st.markdown(f"""
    <div class='insight-box' style='border-left-color:{ts_color};margin:6px 0;'>
      {ts_msg}
    </div>""", unsafe_allow_html=True)

st.markdown("</div></div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  TABS: GRAFICI
# ──────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 Volatility Smile", "📈 Term Structure", "📦 Open Interest", "⚖️ Put/Call & Skew",
    "🎯 Decision Tool", "🧠 RV Forecast Lab",
    "🧭 Buy Signal Board", "📉 Backtest Engine"
])

with tab1:
    st.plotly_chart(chart_vol_smile(calls, puts, recent_price, max_exp), use_container_width=True)
    st.markdown("""
    <div class='insight-box'>
    <b>Come leggere il Volatility Smile:</b><br>
    • <b>Smile simmetrico</b> → il mercato prezza ugualmente i movimenti al rialzo e al ribasso.<br>
    • <b>Skew negativo (smirk)</b> → IV più alta sulle put OTM → paura del ribasso dominante (tipico per crypto/equity).<br>
    • <b>Skew positivo</b> → IV più alta sulle call OTM → domanda di upside, momentum rialzista.<br>
    • <b>Curva ripida</b> → alta incertezza; <b>piatta</b> → mercato tranquillo.
    </div>""", unsafe_allow_html=True)

with tab2:
    st.plotly_chart(chart_term_structure(ts_df), use_container_width=True)
    st.markdown("""
    <div class='insight-box'>
    <b>Come leggere la Term Structure:</b><br>
    • <b>Contango (crescente)</b> → normale; il mercato si aspetta più volatilità nel tempo ma è calmo ora.<br>
    • <b>Backwardation (decrescente)</b> → attenzione! Il mercato teme un evento imminente. Costoso comprare protezione breve.<br>
    • <b>Hump (gobba)</b> → incertezza concentrata attorno a una specifica scadenza (es. evento macro).
    </div>""", unsafe_allow_html=True)

with tab3:
    st.plotly_chart(chart_open_interest(calls, puts, recent_price, max_exp), use_container_width=True)

    # Max pain
    all_strikes = sorted(data["strike"].dropna().unique())
    if all_strikes:
        pain_values = []
        for s in all_strikes:
            c_pain = calls[calls["strike"] < s]["openInterest"].fillna(0).sum() * 0  # simplified
            p_val = puts[puts["strike"] > s].apply(lambda r: (r["strike"] - s) * r["openInterest"], axis=1).fillna(0).sum()
            c_val = calls[calls["strike"] < s].apply(lambda r: (s - r["strike"]) * r["openInterest"], axis=1).fillna(0).sum()
            pain_values.append(p_val + c_val)
        max_pain_strike = all_strikes[np.argmin(pain_values)]
        st.markdown(f"""
        <div class='insight-box' style='border-left-color:#f39c12;'>
        🎯 <b>Max Pain stimato: ${max_pain_strike:,.0f}</b> — il prezzo a cui scadono con perdita massima per i compratori di opzioni (e minima per i writer). 
        Il mercato tende a gravitare verso questo livello vicino alla scadenza.
        </div>""", unsafe_allow_html=True)

with tab4:
    c1, c2 = st.columns(2)
    pcr_fig = chart_pcr_by_expiry(calls, puts)
    skew_fig = chart_skew_by_expiry(calls, puts)
    if pcr_fig:
        c1.plotly_chart(pcr_fig, use_container_width=True)
    if skew_fig:
        c2.plotly_chart(skew_fig, use_container_width=True)

    st.markdown("""
    <div class='insight-box'>
    <b>PCR > 1.2</b> → pressione ribassista o hedging massiccio. <b>PCR < 0.8</b> → posizionamento rialzista o call overwriting.<br>
    <b>Skew positivo</b> (puts più care) → fear premium → mercato in difesa. <b>Skew negativo</b> → greed premium → momentum rialzista.
    </div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  TAB 5: DECISION TOOL
# ──────────────────────────────────────────────
with tab5:
    st.markdown("### 🎯 Decision Support Tool")
    st.markdown("""
    <div class='insight-box'>
    Usa questo pannello per simulare e valutare strategie in base al contesto di volatilità attuale.
    </div>""", unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("#### Parametri della tua view")
        user_view = st.radio("La tua view di mercato", ["Rialzista", "Ribassista", "Neutrale / Range"])
        horizon = st.selectbox("Orizzonte temporale", ["< 1 settimana", "1–4 settimane", "1–3 mesi", "> 3 mesi"])
        risk_appetite = st.radio("Profilo di rischio", ["Conservativo", "Moderato", "Aggressivo"])
        target_strike = st.number_input(
            f"Strike di interesse (spot = {recent_price:,.0f})",
            min_value=float(data["strike"].min()),
            max_value=float(data["strike"].max()),
            value=float(recent_price),
            step=500.0,
        )

    with col_b:
        st.markdown("#### Analisi opzione selezionata")

        nearby = data[abs(data["strike"] - target_strike) <= 1500].copy()
        if not nearby.empty:
            st.dataframe(
                nearby[["contractSymbol", "type", "strike", "bid", "ask", "impliedVolatility",
                         "openInterest", "volume", "delta", "gamma", "theta"]].round(4),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("Nessuna opzione trovata vicino allo strike selezionato.")

    # Strategy recommendation
    st.markdown("---")
    st.markdown("#### 💡 Strategia suggerita")

    strategies = {
        ("Rialzista", "< 1 settimana", "Aggressivo"):
            ("Long Call OTM", "Alta leva, costo contenuto. Adatta se attendi un movimento rapido."),
        ("Rialzista", "< 1 settimana", "Moderato"):
            ("Bull Call Spread", "Riduci il costo del premio limitando il profilo di guadagno. Rischio definito."),
        ("Rialzista", "< 1 settimana", "Conservativo"):
            ("Sell Put OTM (Cash Secured)", "Incassa il premio, sei disposto a comprare spot se il prezzo scende."),
        ("Rialzista", "1–4 settimane", "Aggressivo"):
            ("Long Call ATM", "Delta elevato, cattura il movimento direttamente."),
        ("Rialzista", "1–4 settimane", "Moderato"):
            ("Bull Call Spread", "Costo netto limitato, esposizione direzionale."),
        ("Rialzista", "1–4 settimane", "Conservativo"):
            ("Covered Call + Long Spot", "Genera reddito da premi call se sei già long."),
        ("Rialzista", "1–3 mesi", "Aggressivo"):
            ("Long Call ITM / ATM", "Esposizione lunga al rialzo con delta > 0.5."),
        ("Rialzista", "1–3 mesi", "Moderato"):
            ("Bull Call Spread o LEAPS", "Basso theta decay rispetto alla scadenza lunga."),
        ("Rialzista", "1–3 mesi", "Conservativo"):
            ("Sell Put spread OTM", "Incassi il premio, rischio contenuto tra i due strike."),
        ("Ribassista", "< 1 settimana", "Aggressivo"):
            ("Long Put OTM", "Alta leva sul ribasso. Utile se attendi sell-off rapido."),
        ("Ribassista", "< 1 settimana", "Moderato"):
            ("Bear Put Spread", "Riduce il costo della protezione limitando il massimo guadagno."),
        ("Ribassista", "< 1 settimana", "Conservativo"):
            ("Sell Call OTM (Covered)", "Incassa premio, protegge parzialmente dal ribasso."),
        ("Ribassista", "1–4 settimane", "Aggressivo"):
            ("Long Put ATM", "Esposizione diretta al ribasso con delta negativo elevato."),
        ("Ribassista", "1–4 settimane", "Moderato"):
            ("Bear Put Spread", "Rischio/rendimento definito, costo netto ridotto."),
        ("Ribassista", "1–4 settimane", "Conservativo"):
            ("Protective Put su posizione long", "Copri il rischio di ribasso mantenendo l'upside."),
        ("Neutrale / Range", "< 1 settimana", "Aggressivo"):
            ("Short Straddle / Strangle", "Incassi il premio da entrambi i lati. Rischio illimitato se il mercato si muove."),
        ("Neutrale / Range", "< 1 settimana", "Moderato"):
            ("Iron Condor", "Vendi spread call e put OTM. Profitto se il prezzo resta in range."),
        ("Neutrale / Range", "< 1 settimana", "Conservativo"):
            ("Iron Butterfly", "Massimo profitto ATM, rischio definito sui due lati."),
        ("Neutrale / Range", "1–4 settimane", "Aggressivo"):
            ("Short Strangle OTM", "Incassi alto premio; attento agli spike di volatilità."),
        ("Neutrale / Range", "1–4 settimane", "Moderato"):
            ("Iron Condor largo", "Ampia finestra di profitto, premio più basso ma più sicuro."),
        ("Neutrale / Range", "1–4 settimane", "Conservativo"):
            ("Calendar Spread", "Vendi scadenza breve, compra lunga. Guadagni da theta decay differenziale."),
    }

    key = (user_view, horizon, risk_appetite)
    default_strategy = {
        "Rialzista": ("Long Call ATM", "Esposizione diretta al rialzo con rischio limitato al premio pagato."),
        "Ribassista": ("Long Put ATM", "Esposizione diretta al ribasso con rischio limitato al premio pagato."),
        "Neutrale / Range": ("Iron Condor", "Strategia range-bound che incassa premio se il prezzo rimane tra i due strike."),
    }
    strat_name, strat_desc = strategies.get(key, default_strategy.get(user_view, ("N/A", "")))

    iv_context = "alta" if avg_iv_c > 80 else ("media" if avg_iv_c > 50 else "bassa")
    iv_advice = {
        "alta": "⚠️ IV alta → i premi sono costosi. Preferisci <b>vendere volatilità</b> (spread, condor) invece di comprare opzioni nude.",
        "media": "✅ IV nella norma → buon equilibrio tra costo del premio e protezione. Strategie direzionali moderate.",
        "bassa": "💡 IV bassa → i premi sono economici. Momento favorevole per <b>comprare opzioni</b> o long straddle.",
    }

    st.markdown(f"""
    <div class='decision-panel'>
      <div style='font-family:Space Mono,monospace;font-size:18px;color:#f0f2ff;margin-bottom:8px;'>
        {strat_name}
      </div>
      <div class='insight-box' style='border-left-color:#5468ff;'>
        {strat_desc}
      </div>
      <div class='insight-box' style='border-left-color:#f39c12;margin-top:10px;'>
        {iv_advice[iv_context]}
      </div>
      <div style='font-size:12px;color:#3a4060;margin-top:16px;font-family:Space Mono,monospace;'>
        ⚠️ Questo tool è solo a scopo educativo. Non costituisce consulenza finanziaria.
      </div>
    </div>""", unsafe_allow_html=True)


with tab6:
    st.markdown("### 🧠 Realized Volatility Forecast Lab")
    st.markdown("""
    <div class='insight-box'>
    In questa tab il target non è il prezzo di BTC ma <b>log(RV)</b>, dove RV è la realized volatility annualizzata ricostruita dai log returns del perpetual.
    L'obiettivo è confrontare modelli semplici e modelli con memoria lunga, poi usare il modello con il miglior errore out-of-sample per confrontare <b>IV ATM attuale</b> e <b>RV prevista</b>.
    </div>""", unsafe_allow_html=True)

    lookback_days = st.slider("Storico per RV model (giorni)", 120, 720, 240, step=30, key="rv_lookback_days")
    rv_hist = get_perpetual_history(ticker, resolution="60", lookback_days=lookback_days)

    if rv_hist.empty or len(rv_hist) < 80:
        st.warning("Storico insufficiente per stimare la RV e confrontare i modelli.")
    else:
        rv_results = evaluate_forecast_models(rv_hist["log_rv"])
        hurst_h = rv_results.get("hurst")
        score_df = rv_results.get("scores", pd.DataFrame())
        forecast_df = rv_results.get("forecasts", pd.DataFrame())
        overall_df = rv_results.get("overall", pd.DataFrame())
        best_model = rv_results.get("best_overall")

        nearest_exp_days = int(max(1, ts_df["days"].dropna().clip(lower=1).min())) if not ts_df.empty else 7
        horizon_map = 1 if nearest_exp_days <= 2 else (3 if nearest_exp_days <= 5 else 7)
        atm_iv_pct = float(ts_df.sort_values("days").iloc[0]["atm_iv"]) if not ts_df.empty else float(data["impliedVolatility"].median())

        best_h_rows = score_df[score_df["horizon"] == horizon_map].sort_values("rmse_log") if not score_df.empty else pd.DataFrame()
        chosen_model = best_h_rows.iloc[0]["model"] if not best_h_rows.empty else best_model
        chosen_forecast = forecast_df[(forecast_df["horizon"] == horizon_map) & (forecast_df["model"] == chosen_model)] if not forecast_df.empty else pd.DataFrame()
        if chosen_forecast.empty and not forecast_df.empty:
            chosen_forecast = forecast_df.sort_values(["horizon", "rmse_log"]).head(1)
        forecast_rv_pct = float(chosen_forecast.iloc[0]["forecast_rv_pct"]) if not chosen_forecast.empty else np.nan
        model_rmse_pct = float(chosen_forecast.iloc[0]["rmse_rv_pct"]) if not chosen_forecast.empty else np.nan
        vol_signal = build_vol_signal(atm_iv_pct, forecast_rv_pct, model_rmse_pct if pd.notna(model_rmse_pct) else 5.0, horizon_map) if pd.notna(forecast_rv_pct) else None

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class='metric-card'>
              <div class='metric-label'>Hurst H</div>
              <div class='metric-value'>{hurst_h:.3f}</div>
              <div class='metric-sub'>{'rough / anti-persistent' if pd.notna(hurst_h) and hurst_h < 0.5 else 'persistent / smooth'}</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class='metric-card'>
              <div class='metric-label'>Best overall model</div>
              <div class='metric-value' style='font-size:20px;'>{best_model or 'N/A'}</div>
              <div class='metric-sub'>min mean OOS RMSE su 1d / 3d / 7d</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            latest_rv = float(rv_hist.iloc[-1]["rv_ann_pct"])
            st.markdown(f"""
            <div class='metric-card'>
              <div class='metric-label'>Last realized vol</div>
              <div class='metric-value'>{latest_rv:.2f}%</div>
              <div class='metric-sub'>annualized from hourly log returns</div>
            </div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class='metric-card'>
              <div class='metric-label'>Current ATM IV</div>
              <div class='metric-value'>{atm_iv_pct:.2f}%</div>
              <div class='metric-sub'>proxy dalla term structure live</div>
            </div>""", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(chart_rv_history(rv_hist), use_container_width=True)
        with c2:
            st.plotly_chart(chart_log_rv(rv_hist), use_container_width=True)

        c3, c4 = st.columns([1.15, 0.85])
        with c3:
            st.plotly_chart(chart_model_rmse(score_df), use_container_width=True)
        with c4:
            if not overall_df.empty:
                table_df = overall_df.copy()
                table_df["rmse_log"] = table_df["rmse_log"].round(4)
                st.dataframe(table_df.rename(columns={"model": "Model", "rmse_log": "Mean OOS RMSE log(RV)"}), hide_index=True, use_container_width=True)

        st.markdown("#### Forecast table")
        if not forecast_df.empty:
            disp = forecast_df.copy()
            disp["forecast_rv_pct"] = disp["forecast_rv_pct"].round(2)
            disp["rmse_rv_pct"] = disp["rmse_rv_pct"].round(2)
            disp["forecast_log_rv"] = disp["forecast_log_rv"].round(4)
            disp["param"] = disp["param"].round(3)
            st.dataframe(disp.rename(columns={
                "horizon": "Horizon (days)",
                "model": "Model",
                "forecast_log_rv": "Forecast log(RV)",
                "forecast_rv_pct": "Forecast RV %",
                "rmse_rv_pct": "OOS RMSE vol pts",
                "param": "Tuned param"
            }), hide_index=True, use_container_width=True)

        st.markdown("#### Segnale IV vs forecast RV")
        horizon_choice = st.selectbox("Orizzonte operativo del segnale", [1, 3, 7], index=[1,3,7].index(horizon_map), key="rv_horizon_choice")
        model_choice_df = score_df[score_df["horizon"] == horizon_choice].sort_values("rmse_log") if not score_df.empty else pd.DataFrame()
        model_choice = model_choice_df.iloc[0]["model"] if not model_choice_df.empty else best_model
        forecast_choice = forecast_df[(forecast_df["horizon"] == horizon_choice) & (forecast_df["model"] == model_choice)] if not forecast_df.empty else pd.DataFrame()

        if not forecast_choice.empty:
            f_rv = float(forecast_choice.iloc[0]["forecast_rv_pct"])
            f_rmse = float(forecast_choice.iloc[0]["rmse_rv_pct"])
            sig = build_vol_signal(atm_iv_pct, f_rv, f_rmse, horizon_choice)
            st.markdown(f"""
            <div class='decision-panel'>
              <div style='display:flex;align-items:center;gap:16px;margin-bottom:12px;'>
                <span style='font-family:Space Mono,monospace;font-size:13px;color:#6b7394;'>MODELLO SCELTO</span>
                <span class='badge badge-neutral'>{model_choice}</span>
                <span class='badge {sig['badge_class']}'>{sig['signal']}</span>
              </div>
              <div class='insight-box'>
                <b>Perché questo è il segnale:</b> {sig['why']}
              </div>
              <div class='insight-box' style='border-left-color:#5468ff;'>
                <b>Lettura operativa:</b> {'Preferisci strutture short-vol con rischio definito, perché IV > RV attesa.' if sig['signal']=='SELL VOL' else ('Preferisci strutture long-vol, perché la volatilità implicita è sotto la volatilità attesa.' if sig['signal']=='BUY VOL' else 'Nessun edge netto: evitare di forzare trade di volatilità pura.') }
              </div>
              <div class='insight-box' style='border-left-color:#f39c12;'>
                <b>Dati usati:</b> RV costruita da log returns orari del perpetual, target = log(RV), forecast diretto a {horizon_choice} giorni, confronto con IV ATM live della scadenza più vicina.
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class='insight-box'>
        <b>Note metodologiche.</b><br>
        • <b>RV</b>: somma dei quadrati dei log returns orari aggregata per giorno, poi annualizzata.<br>
        • <b>H</b>: stimato su log(RV) con metodo vario-gramma; H &lt; 0.5 suggerisce struttura rough.<br>
        • <b>AR(1)</b> e <b>HAR-RV</b>: benchmark semplici e robusti.<br>
        • <b>ARFIMA-like</b>: regressione con memoria frazionaria, usata qui come approssimazione pratica senza librerie specialistiche.<br>
        • <b>Rough Vol</b>: regressione con kernel power-law guidato da H.
        • La scelta finale è fatta sul <b>minimo errore out-of-sample</b>, non sull'eleganza del modello.
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  TAB 7 — BUY SIGNAL BOARD  (breve-medio termine)
# ══════════════════════════════════════════════
with tab7:
    st.markdown("### 🧭 Buy Signal Board — Breve/Medio Termine")
    st.markdown("""
    <div class='insight-box'>
    Questo pannello aggrega <b>tutti i segnali delle altre tab</b> in un unico cruscotto operativo.
    Ogni dimensione di analisi contribuisce a uno <b>score composito 0–100</b>. La raccomandazione
    finale integra anche l'orizzonte temporale scelto e il regime di volatilità corrente.
    </div>""", unsafe_allow_html=True)

    # ── Orizzonte utente ──────────────────────────────────────────
    sb_col_left, sb_col_right = st.columns([1, 2])
    with sb_col_left:
        sb_horizon = st.radio(
            "Orizzonte decisionale",
            ["Breve (1–4 sett.)", "Medio (1–3 mesi)", "Lungo (3–6 mesi)"],
            index=1, key="sb_horizon"
        )
        sb_capital = st.slider("Capitale da deployare (%)", 5, 100, 30, 5, key="sb_capital")
        sb_dca = st.checkbox("Piano DCA (ingresso graduale)?", value=True, key="sb_dca")

    # ── Raccolta dati per i segnali ───────────────────────────────
    # 1. Realized vol (perpetual history, cached già)
    @st.cache_data(ttl=900, show_spinner=False)
    def _rv_quick(ticker):
        try:
            return get_perpetual_history(ticker, resolution="60", lookback_days=120)
        except Exception:
            return pd.DataFrame()

    rv_df = _rv_quick(ticker)
    latest_rv  = float(rv_df.iloc[-1]["rv_ann_pct"]) if not rv_df.empty else None
    rv_7d_mean = float(rv_df.tail(7)["rv_ann_pct"].mean()) if not rv_df.empty else None

    # 2. Indicatori tecnici (MA, RSI) dal perpetual
    def _compute_ta(rv_df_raw):
        """Compute daily close + basic TA from hourly perpetual history."""
        if rv_df_raw.empty:
            return pd.DataFrame()
        # get perpetual hourly closes aggregated daily
        # rv_df already has rv_ann_pct per day; we need price
        return pd.DataFrame()  # placeholder — ta uses prices below

    @st.cache_data(ttl=900, show_spinner=False)
    def _price_history(ticker, lookback=180):
        try:
            end_ms   = int(pd.Timestamp.utcnow().timestamp() * 1000)
            start_ms = int((pd.Timestamp.utcnow() - pd.Timedelta(days=lookback)).timestamp() * 1000)
            payload  = deribit_get("public/get_tradingview_chart_data", {
                "instrument_name": f"{ticker}-PERPETUAL",
                "start_timestamp": start_ms,
                "end_timestamp":   end_ms,
                "resolution":      "1D",
            })
            closes = payload.get("close", [])
            ticks  = payload.get("ticks",  [])
            if not closes or not ticks:
                return pd.DataFrame()
            df = pd.DataFrame({"date": pd.to_datetime(ticks, unit="ms", utc=True),
                               "close": pd.to_numeric(closes, errors="coerce")}).dropna()
            df = df.set_index("date").sort_index()
            c = df["close"]
            df["ma20"]  = c.rolling(20).mean()
            df["ma50"]  = c.rolling(50).mean()
            df["ma200"] = c.rolling(200, min_periods=50).mean()
            delta = c.diff()
            gain  = delta.clip(lower=0).rolling(14).mean()
            loss  = (-delta.clip(upper=0)).rolling(14).mean()
            rs    = gain / loss.replace(0, np.nan)
            df["rsi"] = 100 - 100 / (1 + rs)
            df["ema12"] = c.ewm(span=12).mean()
            df["ema26"] = c.ewm(span=26).mean()
            df["macd"]  = df["ema12"] - df["ema26"]
            df["macd_sig"] = df["macd"].ewm(span=9).mean()
            df["bb_mid"]   = c.rolling(20).mean()
            bb_std         = c.rolling(20).std()
            df["bb_upper"] = df["bb_mid"] + 2 * bb_std
            df["bb_lower"] = df["bb_mid"] - 2 * bb_std
            df["bb_pct"]   = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
            df["ath"]      = c.expanding().max()
            df["drawdown"] = (c - df["ath"]) / df["ath"] * 100
            return df
        except Exception:
            return pd.DataFrame()

    if has_options:
        price_df = _price_history(ticker)
    else:
        price_df = binance_ohlcv(binance_sym, "1D", 500)
    last_p   = price_df.iloc[-1] if not price_df.empty else pd.Series(dtype=float)

    # ── Score dimensions ─────────────────────────────────────────
    dims = {}  # name -> (score, max, detail, color)

    # A. Trend (MA) — 20 pts
    ta_score = 0
    ta_detail = []
    if not last_p.empty:
        if last_p.get("close", 0) > (last_p.get("ma50") or 0):
            ta_score += 7;  ta_detail.append("✅ Prezzo > MA50")
        else:
            ta_detail.append("❌ Prezzo < MA50")
        if last_p.get("close", 0) > (last_p.get("ma200") or 0):
            ta_score += 8;  ta_detail.append("✅ Prezzo > MA200")
        else:
            ta_detail.append("❌ Prezzo < MA200")
        if (last_p.get("ma50") or 0) > (last_p.get("ma200") or 0):
            ta_score += 5;  ta_detail.append("✅ Golden Cross MA50 > MA200")
        else:
            ta_detail.append("⚠️ Death Cross MA50 < MA200")
    else:
        ta_detail.append("Dati TA non disponibili")
    ta_color = "#2ecc71" if ta_score >= 15 else "#f39c12" if ta_score >= 8 else "#e74c3c"
    dims["📈 Trend (MA)"] = (ta_score, 20, " · ".join(ta_detail), ta_color)

    # B. Momentum (RSI + MACD) — 20 pts
    mo_score = 0
    mo_detail = []
    rsi_val = last_p.get("rsi") if not last_p.empty else None
    if rsi_val is not None and pd.notna(rsi_val):
        if rsi_val < 30:
            mo_score += 14; mo_detail.append(f"✅ RSI {rsi_val:.0f} — ipervenduto (buy zone)")
        elif rsi_val < 45:
            mo_score += 10; mo_detail.append(f"✅ RSI {rsi_val:.0f} — zona di accumulazione")
        elif rsi_val < 60:
            mo_score += 6;  mo_detail.append(f"⚠️ RSI {rsi_val:.0f} — neutro")
        else:
            mo_score += 0;  mo_detail.append(f"❌ RSI {rsi_val:.0f} — ipercomprato")
    if not last_p.empty:
        macd_h = last_p.get("macd", 0) - (last_p.get("macd_sig") or 0)
        if pd.notna(macd_h) and macd_h > 0:
            mo_score += 6;  mo_detail.append("✅ MACD hist positivo")
        elif pd.notna(macd_h):
            mo_detail.append("❌ MACD hist negativo")
    mo_score = min(mo_score, 20)
    mo_color = "#2ecc71" if mo_score >= 14 else "#f39c12" if mo_score >= 8 else "#e74c3c"
    dims["⚡ Momentum (RSI/MACD)"] = (mo_score, 20, " · ".join(mo_detail) or "N/A", mo_color)

    # C. Volatility regime — 15 pts
    vr_score = 0
    vr_detail = []
    avg_iv_all = data["impliedVolatility"].mean()
    if latest_rv is not None:
        vr_premium = avg_iv_all - latest_rv
        if avg_iv_all < 60:
            vr_score += 8; vr_detail.append(f"✅ IV bassa ({avg_iv_all:.0f}%) → premi economici")
        elif avg_iv_all < 90:
            vr_score += 5; vr_detail.append(f"⚠️ IV media ({avg_iv_all:.0f}%)")
        else:
            vr_score += 1; vr_detail.append(f"❌ IV alta ({avg_iv_all:.0f}%) → protezione cara")
        if vr_premium < 10:
            vr_score += 4; vr_detail.append(f"✅ IV/RV premium contenuto ({vr_premium:+.0f}%)")
        elif vr_premium < 25:
            vr_score += 2; vr_detail.append(f"⚠️ IV/RV premium {vr_premium:+.0f}%")
        else:
            vr_detail.append(f"❌ IV/RV premium elevato ({vr_premium:+.0f}%) → mercato teme volatilità")
        # Term structure slope
        if len(ts_df) > 1:
            ts_slope = np.polyfit(ts_df["days"], ts_df["atm_iv"], 1)[0]
            if ts_slope > 0:
                vr_score += 3; vr_detail.append("✅ Term structure in contango")
            else:
                vr_detail.append("⚠️ Term structure in backwardation (stress breve)")
    vr_score = min(vr_score, 15)
    vr_color = "#2ecc71" if vr_score >= 10 else "#f39c12" if vr_score >= 5 else "#e74c3c"
    dims["🌡️ Regime Volatilità"] = (vr_score, 15, " · ".join(vr_detail) or "N/A", vr_color)

    # D. Options sentiment (PCR + Skew) — 20 pts
    os_score = 0
    os_detail = []
    if pcr is not None:
        if pcr > 1.3:
            os_score += 9; os_detail.append(f"✅ PCR {pcr:.2f} → hedging massiccio (contrarian bullish)")
        elif pcr > 1.0:
            os_score += 5; os_detail.append(f"⚠️ PCR {pcr:.2f} → lieve pressione put")
        elif pcr < 0.7:
            os_score += 1; os_detail.append(f"❌ PCR {pcr:.2f} → euforia (rischio top)")
        else:
            os_score += 3; os_detail.append(f"⚠️ PCR {pcr:.2f} → neutro")
    if skew is not None:
        if skew > 8:
            os_score += 8; os_detail.append(f"✅ Skew {skew:+.1f}% → fear premium (opportunità contrarian)")
        elif skew > 3:
            os_score += 4; os_detail.append(f"⚠️ Skew {skew:+.1f}% → lieve paura")
        elif skew < -5:
            os_score += 0; os_detail.append(f"❌ Skew {skew:+.1f}% → greed, call molto care")
        else:
            os_score += 2; os_detail.append(f"Skew {skew:+.1f}% → neutro")
    # Max pain proximity
    all_strikes_sb = sorted(data["strike"].dropna().unique())
    if all_strikes_sb:
        pain_vals = []
        for s in all_strikes_sb:
            pv = puts[puts["strike"] > s].apply(lambda r: (r["strike"] - s) * r["openInterest"], axis=1).fillna(0).sum()
            cv = calls[calls["strike"] < s].apply(lambda r: (s - r["strike"]) * r["openInterest"], axis=1).fillna(0).sum()
            pain_vals.append(pv + cv)
        mp_strike = all_strikes_sb[int(np.argmin(pain_vals))]
        mp_dist_pct = (recent_price - mp_strike) / recent_price * 100
        if abs(mp_dist_pct) < 3:
            os_score += 3; os_detail.append(f"✅ Spot vicino al Max Pain (${mp_strike:,.0f}, {mp_dist_pct:+.1f}%)")
        elif mp_dist_pct > 0:
            os_detail.append(f"⚠️ Spot sopra Max Pain ({mp_dist_pct:+.1f}%) — pressione ribassista a scadenza")
        else:
            os_detail.append(f"⚠️ Spot sotto Max Pain ({mp_dist_pct:+.1f}%) — pressione rialzista a scadenza")
    os_score = min(os_score, 20)
    os_color = "#2ecc71" if os_score >= 14 else "#f39c12" if os_score >= 8 else "#e74c3c"
    dims["🎭 Options Sentiment"] = (os_score, 20, " · ".join(os_detail) or "N/A", os_color)

    # E. Posizionamento / BB / Drawdown — 15 pts
    pos_score = 0
    pos_detail = []
    if not last_p.empty:
        bb_pct = last_p.get("bb_pct")
        if pd.notna(bb_pct):
            if bb_pct < 0.15:
                pos_score += 8; pos_detail.append(f"✅ BB% {bb_pct:.2f} — zona di ipervenduto estremo")
            elif bb_pct < 0.30:
                pos_score += 5; pos_detail.append(f"✅ BB% {bb_pct:.2f} — basso")
            elif bb_pct > 0.85:
                pos_score += 0; pos_detail.append(f"❌ BB% {bb_pct:.2f} — ipercomprato")
            else:
                pos_score += 3; pos_detail.append(f"⚠️ BB% {bb_pct:.2f} — neutro")
        dd = last_p.get("drawdown")
        if pd.notna(dd):
            if dd < -60:
                pos_score += 7; pos_detail.append(f"✅ Drawdown {dd:.0f}% — area capitolazione")
            elif dd < -40:
                pos_score += 5; pos_detail.append(f"✅ Drawdown {dd:.0f}% — sconto significativo")
            elif dd < -20:
                pos_score += 2; pos_detail.append(f"⚠️ Drawdown {dd:.0f}%")
            else:
                pos_detail.append(f"⚠️ Drawdown {dd:.0f}% — vicino ai massimi")
    pos_score = min(pos_score, 15)
    pos_color = "#2ecc71" if pos_score >= 10 else "#f39c12" if pos_score >= 5 else "#e74c3c"
    dims["📐 Posizionamento (BB/DD)"] = (pos_score, 15, " · ".join(pos_detail) or "N/A", pos_color)

    # F. RV Forecast Lab link-in — 10 pts (bonus)
    rvf_score = 0
    rvf_detail = []
    if not rv_df.empty and latest_rv is not None:
        rv_ratio = avg_iv_all / max(latest_rv, 1)
        if rv_ratio > 1.4:
            rvf_score += 5; rvf_detail.append(f"✅ IV/RV ratio {rv_ratio:.1f} — IV sovrastimata (sell vol edge)")
        elif rv_ratio < 0.85:
            rvf_score += 3; rvf_detail.append(f"⚠️ IV/RV {rv_ratio:.1f} — IV bassa vs RV realizzata")
        else:
            rvf_score += 2; rvf_detail.append(f"IV/RV ratio neutro ({rv_ratio:.1f})")
        rv7_trend = rv_df.tail(7)["rv_ann_pct"]
        if len(rv7_trend) >= 4:
            rv_slope_7 = np.polyfit(range(len(rv7_trend)), rv7_trend.values, 1)[0]
            if rv_slope_7 < -2:
                rvf_score += 5; rvf_detail.append("✅ RV in calo ultimi 7g — volatilità si sta comprimendo")
            elif rv_slope_7 > 2:
                rvf_score += 1; rvf_detail.append("⚠️ RV in aumento — mercato si sta muovendo")
            else:
                rvf_score += 3; rvf_detail.append("RV stabile")
    rvf_score = min(rvf_score, 10)
    rvf_color = "#2ecc71" if rvf_score >= 7 else "#f39c12" if rvf_score >= 4 else "#e74c3c"
    dims["🔬 RV Forecast"] = (rvf_score, 10, " · ".join(rvf_detail) or "N/A", rvf_color)

    # ── Composite score ───────────────────────────────────────────
    total_score  = sum(v[0] for v in dims.values())
    max_score    = sum(v[1] for v in dims.values())
    score_pct    = int(total_score / max_score * 100)

    # Horizon weight adjustment
    horizon_adj = {"Breve (1–4 sett.)": 0, "Medio (1–3 mesi)": 5, "Lungo (3–6 mesi)": 8}
    adj = horizon_adj.get(sb_horizon, 0)
    score_final = min(100, score_pct + adj)

    if score_final >= 70:
        verdict = "🟢 FORTE SEGNALE DI ACQUISTO"
        v_color = "#2ecc71"
        v_action = f"Deploya {sb_capital}% del capitale. " + ("DCA su 3–4 tranche nelle prossime settimane." if sb_dca else "Ingresso unico.")
        v_card   = "card-green"
    elif score_final >= 52:
        verdict = "🟡 ACCUMULO GRADUALE"
        v_color = "#f39c12"
        v_action = f"Inizia con {sb_capital // 2}% ora. " + ("Rinfuerza a ogni dip significativo." if sb_dca else f"Considera di riservarti {sb_capital // 2}% per ingressi migliori.")
        v_card   = "card-yellow"
    elif score_final >= 38:
        verdict = "🟠 ATTENDERE / OSSERVARE"
        v_color = "#e67e22"
        v_action = f"Non è il momento ideale. Limita l'esposizione al {min(10, sb_capital // 3)}%. Monitora il trend."
        v_card   = ""
    else:
        verdict = "🔴 EVITARE / HEDGE"
        v_color = "#e74c3c"
        v_action = "Segnali tecnici e di sentiment negativi. Considera protezione con put o resta flat."
        v_card   = "card-red"

    # ── Score gauge chart ─────────────────────────────────────────
    def _gauge(score):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            number={"font": {"family": "Space Mono", "color": "#f0f2ff", "size": 44}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#4a5278", "tickfont": {"color": "#4a5278"}},
                "bar":  {"color": ("#2ecc71" if score >= 70 else "#f39c12" if score >= 52 else "#e74c3c"), "thickness": 0.28},
                "bgcolor":   "#0d0f14",
                "bordercolor": "#1e2330",
                "steps": [
                    {"range": [0,   38], "color": "rgba(231,76,60,0.12)"},
                    {"range": [38,  52], "color": "rgba(230,126,34,0.12)"},
                    {"range": [52,  70], "color": "rgba(243,156,18,0.12)"},
                    {"range": [70, 100], "color": "rgba(46,204,113,0.12)"},
                ],
                "threshold": {"line": {"color": "#5468ff", "width": 2}, "thickness": 0.75, "value": score},
            },
            title={"text": "COMPOSITE BUY SCORE", "font": {"family": "Space Mono", "color": "#6b7394", "size": 12}},
        ))
        fig.update_layout(paper_bgcolor="#0d0f14", height=260, margin=dict(t=30, b=10, l=20, r=20))
        return fig

    # ── Radar / spider chart ──────────────────────────────────────
    def _radar(dims):
        cats   = list(dims.keys())
        scores = [v[0] / v[1] * 100 for v in dims.values()]
        cats_c = cats + [cats[0]]
        vals_c = scores + [scores[0]]
        fig = go.Figure(go.Scatterpolar(
            r=vals_c, theta=cats_c,
            fill="toself", fillcolor="rgba(84,104,255,0.15)",
            line=dict(color="#5468ff", width=2),
            marker=dict(color="#5468ff", size=7),
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(color="#4a5278", size=9),
                                gridcolor="#1e2330", linecolor="#1e2330"),
                angularaxis=dict(tickfont=dict(color="#8890b0", size=11), gridcolor="#1e2330"),
                bgcolor="#0d0f14",
            ),
            paper_bgcolor="#0d0f14", showlegend=False, height=320,
            title=dict(text="Score per dimensione (%)", font=dict(family="Space Mono", color="#6b7394", size=11)),
            margin=dict(t=40, b=20, l=40, r=40),
        )
        return fig

    # ── Price + MA + BB chart (rescalable Y axis) ─────────────────
    def _price_ta_chart(df, ticker, spot):
        if df.empty or len(df) < 30:
            return None
        # Controls for this chart
        ch_c1, ch_c2, ch_c3, ch_c4 = st.columns(4)
        with ch_c1:
            n_bars = st.slider("Barre da visualizzare", 20, min(len(df), 500),
                               min(120, len(df)), 10, key="pta_bars")
        with ch_c2:
            y_scale = st.radio("Scala Y", ["Lineare", "Logaritmica"], horizontal=True, key="pta_yscale")
        with ch_c3:
            y_padding = st.slider("Padding Y (%)", 0, 30, 5, 1, key="pta_ypad",
                                  help="Restringe la finestra Y attorno ai dati visibili")
        with ch_c4:
            show_bb = st.checkbox("Bande di Bollinger", True, key="pta_bb")

        d = df.tail(n_bars).copy()
        use_log = (y_scale == "Logaritmica")

        # Compute tight Y range
        price_min = d["close"].min()
        price_max = d["close"].max()
        ma_vals   = pd.concat([d.get("ma20", pd.Series(dtype=float)),
                                d.get("ma50", pd.Series(dtype=float)),
                                d.get("ma200", pd.Series(dtype=float))]).dropna()
        if not ma_vals.empty:
            price_min = min(price_min, ma_vals.min())
            price_max = max(price_max, ma_vals.max())
        if show_bb and "bb_lower" in d.columns and "bb_upper" in d.columns:
            price_min = min(price_min, d["bb_lower"].min())
            price_max = max(price_max, d["bb_upper"].max())
        pad = (price_max - price_min) * y_padding / 100
        y_lo = max(price_min - pad, 1e-3)
        y_hi = price_max + pad

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            row_heights=[0.60, 0.22, 0.18], vertical_spacing=0.03,
                            subplot_titles=["Prezzo + MA + BB", "RSI (14)", "MACD"])

        if show_bb and "bb_upper" in d.columns:
            fig.add_trace(go.Scatter(x=d.index, y=d["bb_upper"], mode="lines",
                line=dict(color="#5468ff", width=0.7, dash="dot"), name="BB Upper", showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=d.index, y=d["bb_lower"], mode="lines",
                line=dict(color="#5468ff", width=0.7, dash="dot"), fill="tonexty",
                fillcolor="rgba(84,104,255,0.06)", name="BB Band", showlegend=False), row=1, col=1)

        fig.add_trace(go.Scatter(x=d.index, y=d["close"], mode="lines",
            line=dict(color="#f0f2ff", width=2), name="Prezzo"), row=1, col=1)
        if "ma20" in d.columns:
            fig.add_trace(go.Scatter(x=d.index, y=d["ma20"], mode="lines",
                line=dict(color="#3498db", width=1.2), name="MA20"), row=1, col=1)
        if "ma50" in d.columns:
            fig.add_trace(go.Scatter(x=d.index, y=d["ma50"], mode="lines",
                line=dict(color="#f39c12", width=1.4), name="MA50"), row=1, col=1)
        if "ma200" in d.columns and d["ma200"].notna().any():
            fig.add_trace(go.Scatter(x=d.index, y=d["ma200"], mode="lines",
                line=dict(color="#e74c3c", width=1.4), name="MA200"), row=1, col=1)
        fig.add_hline(y=spot, line=dict(color="#5468ff", dash="dash", width=1),
                      annotation_text=f"SPOT ${spot:,.2f}", row=1, col=1)

        # RSI
        if "rsi" in d.columns:
            fig.add_trace(go.Scatter(x=d.index, y=d["rsi"], mode="lines",
                line=dict(color="#7b6bfa", width=1.8), name="RSI"), row=2, col=1)
            fig.add_hline(y=70, line=dict(color="#e74c3c", dash="dot", width=1), row=2, col=1)
            fig.add_hline(y=30, line=dict(color="#2ecc71", dash="dot", width=1), row=2, col=1)
            fig.add_hrect(y0=30, y1=70, fillcolor="rgba(84,104,255,0.04)", line_width=0, row=2, col=1)

        # MACD
        if "macd_hist" in d.columns:
            hist_colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in d["macd_hist"].fillna(0)]
            fig.add_trace(go.Bar(x=d.index, y=d["macd_hist"], marker_color=hist_colors,
                name="MACD Hist", showlegend=False), row=3, col=1)
            fig.add_trace(go.Scatter(x=d.index, y=d["macd"], mode="lines",
                line=dict(color="#5468ff", width=1.3), name="MACD"), row=3, col=1)
            fig.add_trace(go.Scatter(x=d.index, y=d["macd_sig"], mode="lines",
                line=dict(color="#f39c12", width=1.2), name="Signal"), row=3, col=1)

        # Apply tight Y range to price panel
        y_axis_type = "log" if use_log else "linear"
        fig.update_yaxes(range=[np.log10(y_lo) if use_log else y_lo,
                                 np.log10(y_hi) if use_log else y_hi],
                         type=y_axis_type, title_text="Prezzo ($)", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
        fig.update_yaxes(title_text="MACD", row=3, col=1)

        fig.update_layout(**DARK_LAYOUT, height=560,
                          title=f"{ticker} — Analisi Tecnica ({n_bars} barre · scala {y_scale})")
        return fig

    # ── LAYOUT ───────────────────────────────────────────────────
    with sb_col_right:
        st.markdown(f"""
        <div class='decision-panel' style='text-align:center;'>
          <div style='font-family:Space Mono,monospace;font-size:22px;font-weight:700;
                      color:{v_color};margin-bottom:6px;'>{verdict}</div>
          <div style='font-family:Space Mono,monospace;font-size:48px;font-weight:700;
                      color:{v_color};'>{score_final}</div>
          <div style='color:#6b7394;font-family:Space Mono,monospace;font-size:11px;'>/ 100</div>
          <div class='insight-box' style='border-left-color:{v_color};text-align:left;margin-top:14px;'>
            <b>Azione suggerita:</b> {v_action}
          </div>
          <div style='font-size:11px;color:#3a4060;margin-top:8px;'>
            Orizzonte: {sb_horizon} · Adj +{adj}pts per orizzonte
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    g_col, r_col = st.columns([1, 1])
    with g_col:
        st.plotly_chart(_gauge(score_final), use_container_width=True)
    with r_col:
        st.plotly_chart(_radar(dims), use_container_width=True)

    # ── Dimension breakdown ───────────────────────────────────────
    st.markdown("#### 🔍 Dettaglio Dimensioni")
    for dim_name, (d_score, d_max, d_detail, d_color) in dims.items():
        pct = int(d_score / d_max * 100)
        st.markdown(f"""
        <div style='margin:8px 0;'>
          <div style='display:flex;justify-content:space-between;margin-bottom:3px;'>
            <span style='font-family:Space Mono,monospace;font-size:12px;color:#c8ccde;'>{dim_name}</span>
            <span style='font-family:Space Mono,monospace;font-size:12px;color:{d_color};font-weight:700;'>{d_score}/{d_max} ({pct}%)</span>
          </div>
          <div style='background:#1a1f2e;border-radius:6px;height:7px;overflow:hidden;'>
            <div style='width:{pct}%;height:100%;background:{d_color};border-radius:6px;'></div>
          </div>
          <div style='font-size:11px;color:#6b7394;margin-top:3px;'>{d_detail}</div>
        </div>""", unsafe_allow_html=True)

    # ── Price TA chart ────────────────────────────────────────────
    st.markdown("---")
    if not price_df.empty:
        ta_fig = _price_ta_chart(price_df, ticker, recent_price)
        if ta_fig:
            st.plotly_chart(ta_fig, use_container_width=True)

    # ── DCA planner ───────────────────────────────────────────────
    if sb_dca:
        st.markdown("---")
        st.markdown("#### 📅 Piano DCA Intelligente")
        dca_cap    = st.number_input("Capitale totale (€/$)", min_value=100, value=10000, step=500, key="dca_cap")
        dca_weeks  = st.slider("Distribuzione su N settimane", 2, 24, 8, key="dca_weeks")
        dca_method = st.radio("Metodo", ["Uniforme", "Ponderato (più su dip)", "Front-loaded (50% subito)"], key="dca_method", horizontal=True)

        if dca_method == "Uniforme":
            tranche_sizes = [dca_cap / dca_weeks] * dca_weeks
        elif dca_method == "Front-loaded (50% subito)":
            first = dca_cap * 0.5
            rest  = (dca_cap * 0.5) / max(dca_weeks - 1, 1)
            tranche_sizes = [first] + [rest] * max(dca_weeks - 1, 1)
        else:  # Ponderato
            weights = np.linspace(0.5, 1.5, dca_weeks)
            weights = weights / weights.sum()
            tranche_sizes = (weights * dca_cap).tolist()

        dca_rows = []
        for i, t in enumerate(tranche_sizes):
            entry_date = pd.Timestamp.now(tz="UTC") + pd.Timedelta(weeks=i)
            dca_rows.append({
                "Settimana": i + 1,
                "Data stimata": entry_date.strftime("%Y-%m-%d"),
                "Importo ($)": round(t, 2),
                "% del totale": f"{t/dca_cap*100:.1f}%",
                "Note": ("🟢 Ingresso principale" if i == 0 else
                         "🔵 Rinforzo su dip" if dca_method == "Ponderato (più su dip)" and i >= dca_weeks // 2 else
                         "⚪ Tranche regolare"),
            })
        st.dataframe(pd.DataFrame(dca_rows), hide_index=True, use_container_width=True)

    # ── Checklist operativa ───────────────────────────────────────
    st.markdown("---")
    st.markdown("#### ✅ Checklist Operativa Prima di Entrare")
    checklist_items = [
        ("Controlla il trend di BTC dominance", "Se BTC.D sale → prefer BTC vs altcoin. Se scende → altseason possibile."),
        ("Verifica il funding rate del perpetual", "Funding positivo alto → mercato leva lungo → rischio squeeze. Vai su 'Dati Mercato'."),
        ("Controlla i livelli di Max Pain", "Verifica se spot è sopra/sotto Max Pain → bias di prezzo a scadenza."),
        ("Leggi il Fear & Greed Index", "Comprare vicino a 'Extreme Fear' (< 20) ha storicamente un edge positivo."),
        ("Imposta stop-loss mentale o reale", "Definisci il tuo livello di uscita prima di entrare: -15%, -25%, -40%?"),
        ("Verifica la correlazione con gli indici equity", "BTC correlato con SPX → macro USA conta. Controlla i futures S&P."),
        ("Non over-investire in un solo asset", "Crypto ≤ 5–20% del portafoglio totale per profili non speculativi."),
    ]
    for item, tip in checklist_items:
        st.markdown(f"""
        <div style='display:flex;gap:12px;padding:8px 0;border-bottom:1px solid #1a1f2e;'>
          <span style='color:#5468ff;font-size:16px;'>□</span>
          <div>
            <div style='color:#c8ccde;font-size:13px;font-weight:600;'>{item}</div>
            <div style='color:#6b7394;font-size:11px;margin-top:2px;'>{tip}</div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""<div class='insight-box' style='margin-top:16px;border-left-color:#2a3060;font-size:11px;'>
    ⚠️ Il Buy Score è un aggregato quantitativo di segnali tecnici, di volatilità e di sentiment.
    Non incorpora dati fondamentali on-chain (NVT, MVRV, SOPR), notizie macro, né dati proprietari.
    Usa sempre il tuo giudizio critico e consulta più fonti prima di prendere decisioni di investimento.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  TAB 8 — BACKTEST ENGINE  (v2 — weighted signals + optimizer + multi-TF)
# ══════════════════════════════════════════════

# ── Shared backtest helpers (defined once, outside with-block) ──────────────

@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_bt_ohlcv(ticker: str, resolution: str, lookback: int) -> pd.DataFrame:
    """Fetch OHLCV from Deribit perpetual at any resolution and compute all raw signal features."""
    res_map = {"1D": 1440, "4H": 240, "1H": 60, "15m": 15}
    bars_needed = lookback * (1440 // res_map.get(resolution, 1440)) + 300
    end_ms   = int(pd.Timestamp.utcnow().timestamp() * 1000)
    start_ms = int((pd.Timestamp.utcnow() - pd.Timedelta(days=lookback + 220)).timestamp() * 1000)
    try:
        payload = deribit_get("public/get_tradingview_chart_data", {
            "instrument_name": f"{ticker}-PERPETUAL",
            "start_timestamp": start_ms,
            "end_timestamp":   end_ms,
            "resolution":      res_map.get(resolution, 1440),
        })
        closes = payload.get("close", [])
        highs  = payload.get("high",  closes)
        lows   = payload.get("low",   closes)
        ticks  = payload.get("ticks", [])
        if not closes or not ticks:
            return pd.DataFrame()
        df = pd.DataFrame({
            "ts":    pd.to_datetime(ticks, unit="ms", utc=True),
            "close": pd.to_numeric(closes, errors="coerce"),
            "high":  pd.to_numeric(highs,  errors="coerce"),
            "low":   pd.to_numeric(lows,   errors="coerce"),
        }).dropna(subset=["close"]).set_index("ts").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        c = df["close"]
        # Moving averages
        df["ma20"]     = c.rolling(20).mean()
        df["ma50"]     = c.rolling(50).mean()
        df["ma200"]    = c.rolling(200, min_periods=50).mean()
        # RSI
        delta = c.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - 100 / (1 + rs)
        # MACD
        ema12 = c.ewm(span=12).mean(); ema26 = c.ewm(span=26).mean()
        df["macd"]     = ema12 - ema26
        df["macd_sig"] = df["macd"].ewm(span=9).mean()
        df["macd_hist"]= df["macd"] - df["macd_sig"]
        # Bollinger
        bb_mid = c.rolling(20).mean(); bb_std = c.rolling(20).std()
        df["bb_upper"] = bb_mid + 2 * bb_std
        df["bb_lower"] = bb_mid - 2 * bb_std
        df["bb_pct"]   = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
        # Drawdown
        df["ath"]      = c.expanding().max()
        df["drawdown"] = (c - df["ath"]) / df["ath"] * 100
        # Realized vol (30 bars rolling)
        log_r = np.log(c / c.shift(1))
        bars_per_year = 1440 / res_map.get(resolution, 1440) * 365
        df["rv30"]     = log_r.rolling(30).std() * np.sqrt(bars_per_year) * 100
        # Volume z-score (if volume available — deribit returns volume)
        vols = payload.get("volume", [])
        if vols and len(vols) == len(ticks):
            df["volume"] = pd.to_numeric(vols, errors="coerce")
            df["vol_z"]  = (df["volume"] - df["volume"].rolling(20).mean()) / (df["volume"].rolling(20).std() + 1e-9)
        else:
            df["vol_z"] = 0.0
        # Rate-of-change
        df["roc5"]  = c.pct_change(5)  * 100
        df["roc20"] = c.pct_change(20) * 100
        return df
    except Exception:
        return pd.DataFrame()


def _compute_raw_signals(df: pd.DataFrame, iv_proxy: float = 70.0) -> pd.DataFrame:
    """
    Returns a DataFrame with one column per signal, each normalised to [0, 1].
    1 = fully bullish, 0 = fully bearish.
    """
    out = pd.DataFrame(index=df.index)
    c = df["close"]

    # S1 — Trend MA50
    out["s_ma50"]   = np.where(pd.notna(df["ma50"])  & (c > df["ma50"]),  1.0, 0.0)
    # S2 — Trend MA200
    out["s_ma200"]  = np.where(pd.notna(df["ma200"]) & (c > df["ma200"]), 1.0, 0.0)
    # S3 — Golden cross
    out["s_gc"]     = np.where(pd.notna(df["ma50"]) & pd.notna(df["ma200"]) & (df["ma50"] > df["ma200"]), 1.0, 0.0)
    # S4 — RSI (inverted: low RSI → high bullish signal)
    rsi = df["rsi"].clip(1, 99)
    out["s_rsi"]    = ((70 - rsi) / 70).clip(0, 1)
    # S5 — MACD histogram cross
    out["s_macd"]   = np.where(df["macd_hist"] > 0, 1.0, 0.0)
    # S6 — Bollinger %B low
    bb = df["bb_pct"].clip(0, 1)
    out["s_bb"]     = (1 - bb).clip(0, 1)
    # S7 — Drawdown depth (deeper = more bullish contrarian)
    dd = df["drawdown"].clip(-100, 0)
    out["s_dd"]     = (-dd / 100).clip(0, 1)
    # S8 — RV vs IV proxy (low IV relative to RV = cheap vol / bullish)
    rv = df["rv30"].clip(1, 500)
    iv_ratio = (rv / iv_proxy).clip(0.2, 5)
    out["s_rv_iv"]  = (iv_ratio - 0.2) / (5 - 0.2)           # high RV vs IV → bullish
    # S9 — Rate of change 5-bar
    roc5 = df["roc5"].clip(-30, 30)
    out["s_roc5"]   = ((roc5 + 30) / 60).clip(0, 1)
    # S10 — Rate of change 20-bar
    roc20 = df["roc20"].clip(-60, 60)
    out["s_roc20"]  = ((roc20 + 60) / 120).clip(0, 1)

    return out.fillna(0.5)   # neutral default for NaN


SIGNAL_LABELS = {
    "s_ma50":  "Prezzo > MA50",
    "s_ma200": "Prezzo > MA200",
    "s_gc":    "Golden Cross (MA50>MA200)",
    "s_rsi":   "RSI basso (ipervenduto)",
    "s_macd":  "MACD hist positivo",
    "s_bb":    "BB%B basso (lower band)",
    "s_dd":    "Drawdown da ATH profondo",
    "s_rv_iv": "RV alta vs IV (vol cheap)",
    "s_roc5":  "ROC 5 barre",
    "s_roc20": "ROC 20 barre",
}
SIGNAL_KEYS = list(SIGNAL_LABELS.keys())


def _weighted_score(signals_df: pd.DataFrame, weights: dict) -> pd.Series:
    """Compute weighted composite score [0,100] per bar."""
    w = np.array([weights.get(k, 1.0) for k in SIGNAL_KEYS], dtype=float)
    w = w / w.sum()
    mat = signals_df[SIGNAL_KEYS].values
    return pd.Series((mat @ w) * 100, index=signals_df.index)


def _backtest_run(df: pd.DataFrame, signals_df: pd.DataFrame, weights: dict,
                  threshold: float, fwd_days: int, costs_bps: float = 10) -> pd.DataFrame:
    """Run vectorised backtest; returns enriched DataFrame with score + forward returns."""
    bt = df.copy()
    bt["score"] = _weighted_score(signals_df, weights)
    # Forward returns (log, converted to pct)
    for fwd in [3, 7, 14, 30, 60, 90]:
        fp = bt["close"].shift(-fwd)
        bt[f"fwd_{fwd}"] = (fp / bt["close"] - 1) * 100 - (costs_bps / 100)
    bt["signal"] = (bt["score"] >= threshold).astype(int)
    return bt


def _equity_sim(bt: pd.DataFrame, fwd_days: int, threshold: float) -> tuple:
    """
    Simple equity simulation: enter on signal day, hold fwd_days bars, then cash.
    Returns (strategy equity array, bh equity array).
    """
    ret = bt["close"].pct_change().fillna(0).values
    scores = bt["score"].values
    n = len(ret)
    strat = np.ones(n)
    in_trade = 0
    for i in range(n):
        if scores[i] >= threshold and in_trade == 0:
            in_trade = fwd_days
        if in_trade > 0:
            strat[i] = 1 + ret[i]
            in_trade -= 1
        # else cash → multiply by 1 (already set)
    return np.cumprod(strat), np.cumprod(1 + ret)


def _performance_metrics(returns_pct: pd.Series, label: str, fwd_days: int) -> dict:
    r = returns_pct.dropna() / 100
    if len(r) < 3:
        return {}
    mu   = r.mean()
    sig  = r.std()
    ann  = 252 / max(fwd_days, 1)
    sharpe = mu / sig * np.sqrt(ann) if sig > 0 else np.nan
    cum  = (1 + r).cumprod()
    mdd  = ((cum / cum.cummax()) - 1).min() * 100
    wr   = (r > 0).mean() * 100
    pf   = r[r > 0].sum() / (-r[r < 0].sum()) if r[r < 0].sum() != 0 else np.nan
    avg_w = r[r > 0].mean() * 100 if (r > 0).any() else np.nan
    avg_l = r[r < 0].mean() * 100 if (r < 0).any() else np.nan
    return {
        "Gruppo": label, "n": len(r),
        "Avg Return": f"{mu*100:.2f}%",
        "Std Dev": f"{sig*100:.2f}%",
        "Win Rate": f"{wr:.1f}%",
        "Avg Win": f"{avg_w:.2f}%" if pd.notna(avg_w) else "N/A",
        "Avg Loss": f"{avg_l:.2f}%" if pd.notna(avg_l) else "N/A",
        "Profit Factor": f"{pf:.2f}" if pd.notna(pf) else "N/A",
        "Sharpe": f"{sharpe:.2f}" if pd.notna(sharpe) else "N/A",
        "Max DD": f"{mdd:.1f}%",
    }


def _optimize_weights(df: pd.DataFrame, signals_df: pd.DataFrame,
                      fwd_days: int,
                      objective: str = "sharpe",
                      n_iter: int = 1200,
                      random_seed: int = 42,
                      threshold_range: tuple = (45, 70),
                      min_trades: int = 10,
                      excluded_signals: list | None = None,
                      concentration_penalty: float = 0.0) -> tuple:
    """
    Three-phase optimisation over weights AND threshold jointly.
      Phase 1 — Dirichlet random search with temperature annealing acceptance
      Phase 2 — Hill climbing with adaptive step
      Phase 3 — Fine-tune threshold around best found
    Returns (best_weights dict, best_threshold, best_score float, history list).
    """
    rng = np.random.default_rng(random_seed)
    fwd_col = f"fwd_{fwd_days}"

    # ── Build active signal set ────────────────────────────────────
    active_keys = [k for k in SIGNAL_KEYS if k not in (excluded_signals or [])]
    if not active_keys:
        active_keys = SIGNAL_KEYS[:]

    # ── Align indexes ──────────────────────────────────────────────
    common_idx = df.index.intersection(signals_df.index)
    df_aln  = df.loc[common_idx]
    sig_aln = signals_df.loc[common_idx]

    if fwd_col not in df_aln.columns:
        return ({k: 1.0/len(SIGNAL_KEYS) for k in SIGNAL_KEYS}, 55.0, -999.0, [])

    valid_mask = df_aln[fwd_col].notna().values
    if sum(valid_mask) > fwd_days:
        valid_mask[-fwd_days:] = False

    mat_full  = sig_aln[SIGNAL_KEYS].values.astype(float)[valid_mask]
    # Active columns only
    act_idx   = [SIGNAL_KEYS.index(k) for k in active_keys]
    mat       = mat_full[:, act_idx]
    fwd_vals  = df_aln[fwd_col].values[valid_mask]
    fwd_all   = fwd_vals[~np.isnan(fwd_vals)]

    if len(fwd_all) < max(min_trades, 8):
        return ({k: 1.0/len(SIGNAL_KEYS) for k in SIGNAL_KEYS}, 55.0, -999.0, [])

    def _eval(w_raw, thr):
        w = np.abs(w_raw); s = w.sum()
        if s < 1e-9: return -999.0
        w /= s
        score_arr   = (mat @ w) * 100
        signal_mask = score_arr >= thr
        ret_s       = fwd_vals[signal_mask]
        ret_s       = ret_s[~np.isnan(ret_s)]
        if len(ret_s) < min_trades:
            return -999.0
        r = ret_s / 100.0
        base = fwd_all[~np.isnan(fwd_all)] / 100.0

        if objective == "sharpe":
            sig_ = r.std()
            val = float(r.mean() / sig_) if sig_ > 1e-9 else -999.0
        elif objective == "avg_return":
            val = float(r.mean())
        elif objective == "win_rate":
            val = float((r > 0).mean())
        elif objective == "edge":
            val = float(r.mean() - base.mean()) if len(base) > 0 else -999.0
        elif objective == "calmar":
            cum  = (1 + r).cumprod()
            mdd  = abs(((cum / cum.cummax()) - 1).min())
            val  = float(r.mean() / mdd) if mdd > 1e-6 else -999.0
        elif objective == "profit_factor":
            wins  = r[r > 0].sum(); loss = -r[r < 0].sum()
            val   = float(wins / loss) if loss > 1e-9 else -999.0
        else:
            val = -999.0

        # Concentration penalty: discourages single-signal dominance
        if concentration_penalty > 0 and val > -900:
            hhi = float(np.sum(w ** 2))   # Herfindahl index: 1/n to 1
            val -= concentration_penalty * hhi

        return val

    # Phase 1: random search with simulated annealing acceptance
    thr_lo, thr_hi = threshold_range
    history  = []
    best_w   = np.ones(len(active_keys)) / len(active_keys)
    best_thr = (thr_lo + thr_hi) / 2
    best_val = -999.0
    current_w   = best_w.copy()
    current_thr = best_thr
    current_val = -999.0
    T = 0.15   # initial temperature

    for i in range(n_iter):
        w   = rng.dirichlet(np.ones(len(active_keys)) * 0.8)
        thr = float(rng.uniform(thr_lo, thr_hi))
        val = _eval(w, thr)
        history.append(float(val))
        # Accept if better, or with probability exp(Δ/T) if worse
        if val > best_val:
            best_val = val; best_w = w.copy(); best_thr = thr
        delta = val - current_val
        if delta > 0 or (T > 0 and np.random.random() < np.exp(delta / T)):
            current_w = w.copy(); current_thr = thr; current_val = val
        T *= (1 - 2 / n_iter)   # cool down

    # Phase 2: hill climbing on weights + threshold jointly
    step_w = 0.06; step_t = 2.0
    for outer in range(500):
        improved = False
        for i in range(len(best_w)):
            for delta in [step_w, -step_w]:
                cand = best_w.copy()
                cand[i] = max(1e-4, cand[i] + delta); cand /= cand.sum()
                val = _eval(cand, best_thr)
                if val > best_val:
                    best_val = val; best_w = cand.copy(); improved = True
        for delta_t in [step_t, -step_t]:
            cand_t = float(np.clip(best_thr + delta_t, thr_lo, thr_hi))
            val = _eval(best_w, cand_t)
            if val > best_val:
                best_val = val; best_thr = cand_t; improved = True
        step_w *= 0.94; step_t *= 0.92
        if step_w < 5e-6: break

    # Phase 3: fine-tune threshold ±3 around best
    for thr_c in np.linspace(max(thr_lo, best_thr - 5), min(thr_hi, best_thr + 5), 21):
        val = _eval(best_w, float(thr_c))
        if val > best_val:
            best_val = val; best_thr = float(thr_c)

    # Rebuild full weight dict (excluded signals get 0)
    w_norm = best_w / best_w.sum()
    full_w = {}
    ai = 0
    for k in SIGNAL_KEYS:
        if k in active_keys:
            full_w[k] = float(w_norm[ai]); ai += 1
        else:
            full_w[k] = 0.0

    return full_w, float(best_thr), best_val, history


with tab8:
    st.markdown("### 📉 Backtest Engine v3 — Pesi, Ottimizzatore Avanzato, Multi-Timeframe, Save/Load")
    st.markdown("""
    <div class='insight-box'>
    Testa il <b>Buy Score con pesi personalizzati</b> su qualsiasi asset e timeframe.
    Ottimizzazione a 3 fasi (annealing + hill climbing + fine-tune soglia) con 6 obiettivi disponibili.
    Salva e ricarica le strategie come file <b>JSON</b>.
    </div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════════
    #  SECTION A — CONFIGURATION
    # ════════════════════════════════════════════
    st.markdown("#### ⚙️ Configurazione")
    cfg_c1, cfg_c2, cfg_c3, cfg_c4 = st.columns(4)
    with cfg_c1:
        bt_tf       = st.selectbox("Timeframe", ["1D", "4H", "1H", "15m"], index=0, key="bt_tf")
    with cfg_c2:
        bt_lookback = st.slider("Storico (giorni)", 60, 720, 365, 30, key="bt_lookback")
    with cfg_c3:
        bt_fwd_days = st.selectbox("Orizzonte forward (barre)", [3, 7, 14, 30, 60, 90], index=2, key="bt_fwd")
    with cfg_c4:
        bt_costs    = st.slider("Costi (bps per trade)", 0, 50, 10, 5, key="bt_costs")

    # ════════════════════════════════════════════
    #  SECTION B — SAVE / LOAD STRATEGY
    # ════════════════════════════════════════════
    st.markdown("---")
    st.markdown("#### 💾 Salva / Carica Strategia")
    sl_c1, sl_c2, sl_c3 = st.columns([1.2, 1.2, 1.6])

    with sl_c1:
        strategy_name = st.text_input("Nome strategia", value="my_strategy", key="bt_strat_name")

    with sl_c2:
        # ── Download current strategy ──────────────────────────────
        def _build_strategy_dict():
            return {
                "name":        st.session_state.get("bt_strat_name", "my_strategy"),
                "asset":       asset_label,
                "timeframe":   st.session_state.get("bt_tf", "1D"),
                "lookback":    st.session_state.get("bt_lookback", 365),
                "fwd_days":    st.session_state.get("bt_fwd", 14),
                "costs_bps":   st.session_state.get("bt_costs", 10),
                "weight_mode": st.session_state.get("bt_wmode", "Manuale"),
                "objective":   st.session_state.get("bt_obj", "sharpe"),
                "threshold":   st.session_state.get("bt_thresh", 60),
                "weights":     {k: float(st.session_state.get(f"w_{k}", 1.0)) for k in SIGNAL_KEYS},
                "opt_thr_lo":  st.session_state.get("bt_opt_thr_lo", 45),
                "opt_thr_hi":  st.session_state.get("bt_opt_thr_hi", 70),
                "min_trades":  st.session_state.get("bt_min_trades", 10),
                "n_iter":      st.session_state.get("bt_n_iter", 1200),
                "conc_pen":    st.session_state.get("bt_conc_pen", 0.0),
                "excluded":    st.session_state.get("bt_excluded", []),
                "saved_at":    datetime.now(timezone.utc).isoformat(),
            }

        strat_json = json.dumps(_build_strategy_dict(), indent=2)
        st.download_button(
            "⬇ Scarica strategia (.json)",
            data=strat_json,
            file_name=f"{strategy_name}.json",
            mime="application/json",
            key="bt_download_strat",
        )

    with sl_c3:
        uploaded_strat = st.file_uploader("📂 Carica strategia (.json)", type=["json"],
                                           key="bt_upload_strat")
        if uploaded_strat is not None:
            try:
                loaded = json.load(uploaded_strat)
                # Restore all values into session_state with proper keys
                _tf_map = {"1D": 0, "4H": 1, "1H": 2, "15m": 3}
                st.session_state["bt_strat_name"] = loaded.get("name", "loaded")
                # We can't set widget values directly in Streamlit for selectbox/slider
                # so we store them and show them as read-only info, then auto-populate sliders
                st.session_state["_loaded_strat"] = loaded
                st.success(f"✅ Strategia '{loaded.get('name')}' caricata — premi ▶ Esegui per applicarla")
            except Exception as ex:
                st.error(f"Errore nel parsing JSON: {ex}")

    # If a strategy was loaded, show a summary and offer to apply
    _loaded = st.session_state.get("_loaded_strat")
    if _loaded:
        with st.expander(f"📋 Strategia caricata: {_loaded.get('name', '?')} — dettagli", expanded=False):
            lc1, lc2, lc3 = st.columns(3)
            lc1.markdown(f"**Asset:** {_loaded.get('asset', '?')}")
            lc1.markdown(f"**TF:** {_loaded.get('timeframe', '?')} · **Lookback:** {_loaded.get('lookback', '?')}g")
            lc2.markdown(f"**Fwd:** {_loaded.get('fwd_days', '?')} barre · **Soglia:** {_loaded.get('threshold', '?')}")
            lc2.markdown(f"**Obiettivo:** {_loaded.get('objective', '?')} · **Min trades:** {_loaded.get('min_trades', '?')}")
            if "weights" in _loaded:
                w_sorted = sorted(_loaded["weights"].items(), key=lambda x: -x[1])
                w_str = " · ".join([f"{SIGNAL_LABELS.get(k, k)}: **{v*100:.1f}%**" for k, v in w_sorted[:5]])
                lc3.markdown(f"**Top pesi:** {w_str}")
            lc3.markdown(f"*Salvata il:* {_loaded.get('saved_at', '?')[:16]}")
        if st.button("↩ Applica strategia caricata ai parametri correnti", key="bt_apply_strat"):
            # Push loaded values to session state — widgets will pick them up on next rerun
            for k, v in _loaded.get("weights", {}).items():
                st.session_state[f"w_{k}"] = v * 10  # slider scale is 0-10
            st.session_state["bt_thresh"]      = int(_loaded.get("threshold", 60))
            st.session_state["bt_opt_thr_lo"]  = int(_loaded.get("opt_thr_lo", 45))
            st.session_state["bt_opt_thr_hi"]  = int(_loaded.get("opt_thr_hi", 70))
            st.session_state["bt_min_trades"]  = int(_loaded.get("min_trades", 10))
            st.session_state["bt_n_iter"]      = int(_loaded.get("n_iter", 1200))
            st.session_state["bt_conc_pen"]    = float(_loaded.get("conc_pen", 0.0))
            st.session_state["bt_excluded"]    = _loaded.get("excluded", [])
            st.rerun()

    # ════════════════════════════════════════════
    #  SECTION C — SIGNAL WEIGHTS
    # ════════════════════════════════════════════
    st.markdown("---")
    st.markdown("#### 🎛️ Pesi dei Segnali")
    st.markdown("""
    <div class='insight-box' style='font-size:12px;'>
    Regola i pesi (0 = escludi, 10 = peso massimo). Normalizzati automaticamente.
    In modalità <b>Ottimizzata</b> i pesi vengono calcolati sul dataset selezionato.
    </div>""", unsafe_allow_html=True)

    weight_mode = st.radio(
        "Modalità pesi",
        ["Manuale", "Ottimizzati (annealing + hill climbing)", "Equal Weight"],
        horizontal=True, key="bt_wmode"
    )
    is_opt_mode = (weight_mode == "Ottimizzati (annealing + hill climbing)")

    # ── Optimizer advanced parameters (shown only in opt mode) ────
    if is_opt_mode:
        st.markdown("##### 🔧 Parametri Ottimizzatore")
        op1, op2, op3, op4, op5 = st.columns(5)
        with op1:
            opt_objective = st.selectbox(
                "Obiettivo",
                ["sharpe", "avg_return", "win_rate", "edge", "calmar", "profit_factor"],
                index=0, key="bt_obj",
                help="sharpe=rendimento/rischio · calmar=rendimento/maxDD · profit_factor=guadagni/perdite"
            )
        with op2:
            opt_thr_lo = st.slider("Soglia min ottimizzatore", 30, 60, 45, 5, key="bt_opt_thr_lo")
            opt_thr_hi = st.slider("Soglia max ottimizzatore", 50, 85, 70, 5, key="bt_opt_thr_hi")
        with op3:
            opt_min_trades = st.slider("Min trades richiesti", 5, 50, 10, 5, key="bt_min_trades",
                                       help="Rifiuta combinazioni che generano meno di N segnali")
            opt_n_iter = st.slider("Iterazioni random search", 400, 3000, 1200, 200, key="bt_n_iter")
        with op4:
            opt_conc_pen = st.slider("Penalità concentrazione", 0.0, 2.0, 0.0, 0.1, key="bt_conc_pen",
                                     help="Penalizza soluzioni con un singolo segnale dominante (diversifica i pesi)")
        with op5:
            excluded_opts = st.multiselect(
                "Escludi segnali",
                options=list(SIGNAL_LABELS.keys()),
                format_func=lambda k: SIGNAL_LABELS[k],
                default=st.session_state.get("bt_excluded", []),
                key="bt_excluded",
                help="I segnali esclusi ricevono peso 0 e non vengono ottimizzati"
            )
    else:
        opt_objective  = "sharpe"
        opt_thr_lo     = st.session_state.get("bt_opt_thr_lo", 45)
        opt_thr_hi     = st.session_state.get("bt_opt_thr_hi", 70)
        opt_min_trades = st.session_state.get("bt_min_trades", 10)
        opt_n_iter     = st.session_state.get("bt_n_iter", 1200)
        opt_conc_pen   = st.session_state.get("bt_conc_pen", 0.0)
        excluded_opts  = st.session_state.get("bt_excluded", [])

    # ── Manual weight sliders ──────────────────────────────────────
    w_cols = st.columns(5)
    manual_weights  = {}
    slider_disabled = is_opt_mode
    for idx, (key, label) in enumerate(SIGNAL_LABELS.items()):
        col = w_cols[idx % 5]
        default_val = float(st.session_state.get(f"w_{key}", 1.0))
        manual_weights[key] = col.slider(
            label, 0.0, 10.0, default_val, 0.5,
            key=f"w_{key}",
            disabled=slider_disabled,
            help=f"Segnale interno: {key}"
        )

    if weight_mode == "Equal Weight":
        manual_weights = {k: 1.0 for k in SIGNAL_KEYS}

    # ── Manual threshold (shown only when not in optimizer mode) ──
    if not is_opt_mode:
        bt_threshold = st.slider("Soglia BUY (score ≥)", 30, 80,
                                  int(st.session_state.get("bt_thresh", 60)), 5, key="bt_thresh")
    else:
        bt_threshold = int(st.session_state.get("bt_thresh", 60))
        st.markdown(f"""<div class='insight-box' style='font-size:12px;border-left-color:#f39c12;'>
        ⚙️ Soglia ottimizzata automaticamente nell'intervallo [{opt_thr_lo}, {opt_thr_hi}].
        </div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════════
    #  SECTION D — DATA + COMPUTE
    # ════════════════════════════════════════════
    st.markdown("---")
    run_bt_cols = st.columns([1, 3])
    run_bt = run_bt_cols[0].button("▶ Esegui Backtest", key="bt_run")

    # Use _res_ prefix for stored results to avoid colliding with widget keys
    if run_bt or st.session_state.get("_res_bt_results") is None:
        # Fetch OHLCV — Deribit for BTC/ETH, Binance for others
        with st.spinner(f"Caricamento dati {bt_tf} · {bt_lookback} giorni…"):
            if has_options:
                bt_df_raw = _fetch_bt_ohlcv(ticker, bt_tf, bt_lookback)
            else:
                bt_df_raw = binance_ohlcv(binance_sym, bt_tf, min(bt_lookback * 7, 1000))

        if bt_df_raw.empty or len(bt_df_raw) < 80:
            st.warning("Dati storici insufficienti. Riduci il lookback o cambia timeframe.")
            st.session_state["_res_bt_results"] = None
        else:
            iv_proxy_bt = float(data["impliedVolatility"].median()) if _has_data else 70.0
            sigs_df = _compute_raw_signals(bt_df_raw, iv_proxy=iv_proxy_bt)

            for fwd in [3, 7, 14, 30, 60, 90]:
                fp = bt_df_raw["close"].shift(-fwd)
                bt_df_raw[f"fwd_{fwd}"] = (fp / bt_df_raw["close"] - 1) * 100 - (bt_costs / 100)

            if is_opt_mode:
                with st.spinner(f"Ottimizzazione in corso ({opt_objective}) · {opt_n_iter} iter…"):
                    opt_weights, opt_best_thr, opt_val, opt_history = _optimize_weights(
                        bt_df_raw, sigs_df,
                        fwd_days=bt_fwd_days,
                        objective=opt_objective,
                        n_iter=opt_n_iter,
                        threshold_range=(opt_thr_lo, opt_thr_hi),
                        min_trades=opt_min_trades,
                        excluded_signals=excluded_opts,
                        concentration_penalty=opt_conc_pen,
                    )
                active_weights = opt_weights
                bt_threshold   = int(round(opt_best_thr))
                st.session_state["bt_thresh"]          = bt_threshold
                st.session_state["_res_opt_weights"]   = opt_weights
                st.session_state["_res_opt_val"]       = opt_val
                st.session_state["_res_opt_history"]   = opt_history
                st.session_state["_res_opt_thr"]       = opt_best_thr
            else:
                active_weights = manual_weights
                st.session_state["_res_opt_weights"]  = None
                st.session_state["_res_opt_val"]      = None
                st.session_state["_res_opt_history"]  = None
                st.session_state["_res_opt_thr"]      = None

            bt_result = _backtest_run(bt_df_raw, sigs_df, active_weights, bt_threshold, bt_fwd_days, bt_costs)
            cutoff_ts = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=bt_lookback)
            bt_result = bt_result[bt_result.index >= cutoff_ts]
            st.session_state["_res_bt_results"]        = bt_result
            st.session_state["_res_bt_active_weights"] = active_weights
            st.session_state["_res_bt_sigs_df"]        = sigs_df
            st.session_state["_res_bt_tf"]             = bt_tf
            st.session_state["_res_bt_fwd_days"]       = bt_fwd_days

    bt_result       = st.session_state.get("_res_bt_results")
    active_weights  = st.session_state.get("_res_bt_active_weights", manual_weights)
    opt_weights_res = st.session_state.get("_res_opt_weights")
    opt_val         = st.session_state.get("_res_opt_val")
    opt_history     = st.session_state.get("_res_opt_history")
    opt_best_thr    = st.session_state.get("_res_opt_thr")

    if bt_result is None or bt_result.empty:
        st.info("Premi ▶ Esegui Backtest per avviare l'analisi.")
    else:
        fwd_col  = f"fwd_{bt_fwd_days}"
        bt_valid = bt_result.dropna(subset=[fwd_col, "score"]).copy()

        # ── If optimizer ran, show results ────────────────────────
        if opt_weights_res is not None:
            st.markdown("---")
            st.markdown("#### 🤖 Risultati Ottimizzatore")
            oc1, oc2 = st.columns([1, 2])
            with oc1:
                st.markdown(f"""<div class='metric-card'>
                  <div class='metric-label'>Score ottimizzato ({opt_objective})</div>
                  <div class='metric-value' style='font-size:20px;'>{opt_val:.4f}</div>
                  <div class='metric-sub'>Soglia ottimale: {opt_best_thr:.1f}</div>
                </div>""", unsafe_allow_html=True)
                w_table = pd.DataFrame([
                    {"Segnale": SIGNAL_LABELS.get(k, k), "Peso": f"{v*100:.1f}%"}
                    for k, v in sorted(opt_weights_res.items(), key=lambda x: -x[1])
                    if v > 0.001
                ])
                st.dataframe(w_table, hide_index=True, use_container_width=True)
            with oc2:
                if opt_history:
                    h_clean = [v for v in opt_history if v > -900]
                    if h_clean:
                        fig_opt = go.Figure(go.Scatter(
                            y=h_clean, mode="lines",
                            line=dict(color="#5468ff", width=1.2), name="Score per tentativo"))
                        fig_opt.add_hline(y=opt_val, line=dict(color="#2ecc71", dash="dash"),
                                          annotation_text=f"Best: {opt_val:.4f}")
                        fig_opt.update_layout(**DARK_LAYOUT, height=280,
                            title=f"Convergenza ottimizzatore — {opt_objective}",
                            xaxis_title="Tentativo", yaxis_title=opt_objective)
                        st.plotly_chart(fig_opt, use_container_width=True)

        # ── Weight bar chart ──────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 📊 Distribuzione Pesi Attivi")
        w_norm = {k: v for k, v in active_weights.items()}
        total_w = sum(w_norm.values()) or 1
        w_pct = {k: v / total_w * 100 for k, v in w_norm.items()}
        fig_w = go.Figure(go.Bar(
            x=[SIGNAL_LABELS[k] for k in SIGNAL_KEYS],
            y=[w_pct[k] for k in SIGNAL_KEYS],
            marker_color=["#5468ff" if w_pct[k] >= 12 else "#7b6bfa" if w_pct[k] >= 7 else "#3a4060"
                          for k in SIGNAL_KEYS],
            text=[f"{w_pct[k]:.1f}%" for k in SIGNAL_KEYS],
            textposition="outside",
        ))
        fig_w.update_layout(**DARK_LAYOUT, height=280,
            title="Peso % di ogni segnale (normalizzato)",
            xaxis_title="Segnale", yaxis_title="Peso (%)",
            xaxis_tickangle=-35)
        st.plotly_chart(fig_w, use_container_width=True)

        # ── Key metrics row ───────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 📈 Metriche Chiave")
        buy_signals  = bt_valid[bt_valid["score"] >= bt_threshold]
        no_signals   = bt_valid[bt_valid["score"] <  bt_threshold]
        avg_ret_buy  = buy_signals[fwd_col].mean()  if not buy_signals.empty else np.nan
        avg_ret_all  = bt_valid[fwd_col].mean()
        win_rate_buy = (buy_signals[fwd_col] > 0).mean() * 100 if not buy_signals.empty else np.nan
        win_rate_all = (bt_valid[fwd_col] > 0).mean() * 100
        n_signals    = len(buy_signals)
        edge         = avg_ret_buy - avg_ret_all if pd.notna(avg_ret_buy) else np.nan

        m1, m2, m3, m4, m5 = st.columns(5)
        def _mc(col, label, val, sub, fmt=".1f"):
            ok = pd.notna(val)
            disp = f"{val:{fmt}}%" if ok else "N/A"
            col.markdown(f"""<div class='metric-card'>
              <div class='metric-label'>{label}</div>
              <div class='metric-value' style='font-size:20px;color:{"#2ecc71" if ok and val>0 else "#e74c3c" if ok else "#6b7394"};'>{disp}</div>
              <div class='metric-sub'>{sub}</div>
            </div>""", unsafe_allow_html=True)
        _mc(m1, f"Avg Ret BUY≥{bt_threshold}", avg_ret_buy,  f"{bt_fwd_days} barre fwd · n={n_signals}")
        _mc(m2, "Avg Ret Tutti",                avg_ret_all,  f"n={len(bt_valid)}")
        _mc(m3, f"Win Rate BUY",                win_rate_buy, "% barre con ret>0")
        _mc(m4, "Win Rate BASE",                win_rate_all, "% barre con ret>0")
        _mc(m5, "Edge vs Base",                 edge,         "extra return BUY vs media")

        # ── Score time series + price (rescalable Y) ─────────────
        st.markdown("---")
        st.markdown("#### 📉 Score nel Tempo & Punti di Ingresso")
        ts_yscale = st.radio("Scala Y prezzo (backtest)", ["Lineare", "Logaritmica"],
                             horizontal=True, key="bt_ts_yscale")
        ts_ypad   = st.slider("Padding Y% (backtest)", 0, 20, 3, 1, key="bt_ts_ypad")

        fig_ts = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               row_heights=[0.42, 0.58], vertical_spacing=0.04,
                               subplot_titles=["Buy Score", f"Prezzo {asset_label}"])
        buy_mask = bt_valid["score"] >= bt_threshold
        fig_ts.add_trace(go.Scatter(x=bt_valid.index, y=bt_valid["score"], mode="lines",
            line=dict(color="#5468ff", width=1.6), name="Score"), row=1, col=1)
        fig_ts.add_trace(go.Scatter(x=bt_valid.index[buy_mask], y=bt_valid["score"][buy_mask],
            mode="markers", marker=dict(color="#2ecc71", size=7, symbol="triangle-up"),
            name=f"BUY ≥ {bt_threshold}"), row=1, col=1)
        fig_ts.add_hline(y=bt_threshold, line=dict(color="#2ecc71", dash="dash", width=1.2),
                         annotation_text=f"Soglia {bt_threshold}", row=1, col=1)
        fig_ts.add_trace(go.Scatter(x=bt_valid.index, y=bt_valid["close"], mode="lines",
            line=dict(color="#f0f2ff", width=1.4), name="Prezzo"), row=2, col=1)
        fig_ts.add_trace(go.Scatter(x=bt_valid.index[buy_mask], y=bt_valid["close"][buy_mask],
            mode="markers", marker=dict(color="#2ecc71", size=6, symbol="triangle-up"),
            name="Entry", showlegend=False), row=2, col=1)
        # Tight Y range on price panel
        p_min = bt_valid["close"].min(); p_max = bt_valid["close"].max()
        pad_ts = (p_max - p_min) * ts_ypad / 100
        y_lo_ts = max(p_min - pad_ts, 1e-3); y_hi_ts = p_max + pad_ts
        use_log_ts = (ts_yscale == "Logaritmica")
        fig_ts.update_layout(**DARK_LAYOUT, height=520, showlegend=True,
            title=f"{asset_label} — Score & ingresso su {bt_tf}")
        fig_ts.update_yaxes(title_text="Score [0–100]", row=1, col=1, range=[-5, 108])
        fig_ts.update_yaxes(
            title_text="Prezzo ($)", row=2, col=1,
            type="log" if use_log_ts else "linear",
            range=[np.log10(y_lo_ts) if use_log_ts else y_lo_ts,
                   np.log10(y_hi_ts) if use_log_ts else y_hi_ts],
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        # ── Distribution + Scatter ────────────────────────────────
        dc1, dc2 = st.columns(2)
        with dc1:
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(x=bt_valid[fwd_col], nbinsx=40,
                name="Tutti", marker_color="rgba(107,115,148,0.5)", histnorm="probability density"))
            if not buy_signals.empty:
                fig_dist.add_trace(go.Histogram(x=buy_signals[fwd_col], nbinsx=30,
                    name=f"BUY ≥ {bt_threshold}", marker_color="rgba(84,104,255,0.75)", histnorm="probability density"))
            fig_dist.add_vline(x=0, line=dict(color="#f0f2ff", dash="dash", width=1.4))
            if not buy_signals.empty and buy_signals[fwd_col].notna().any():
                fig_dist.add_vline(x=buy_signals[fwd_col].mean(),
                    line=dict(color="#2ecc71", dash="dot", width=2),
                    annotation_text=f"Avg BUY {buy_signals[fwd_col].mean():.1f}%",
                    annotation_font_color="#2ecc71")
            fig_dist.update_layout(**DARK_LAYOUT, barmode="overlay", height=320,
                title=f"Distribuzione rendimenti — {bt_fwd_days} barre fwd",
                xaxis_title="Return (%)", yaxis_title="Densità")
            st.plotly_chart(fig_dist, use_container_width=True)
        with dc2:
            scatter_colors = ["#2ecc71" if s >= bt_threshold else "#6b7394" for s in bt_valid["score"]]
            fig_sc = go.Figure(go.Scatter(
                x=bt_valid["score"], y=bt_valid[fwd_col], mode="markers",
                marker=dict(color=scatter_colors, size=4, opacity=0.6),
                text=bt_valid.index.strftime("%Y-%m-%d %H:%M"),
                hovertemplate="<b>%{text}</b><br>Score: %{x:.1f}<br>Return: %{y:.1f}%<extra></extra>",
            ))
            v_ = bt_valid.dropna(subset=[fwd_col, "score"])
            if len(v_) > 10:
                z_ = np.polyfit(v_["score"], v_[fwd_col], 1)
                xl = np.linspace(v_["score"].min(), v_["score"].max(), 80)
                fig_sc.add_trace(go.Scatter(x=xl, y=np.polyval(z_, xl),
                    mode="lines", line=dict(color="#f39c12", width=2, dash="dash"),
                    name=f"Trend (slope {z_[0]:.2f})"))
            fig_sc.add_hline(y=0, line=dict(color="#f0f2ff", dash="dot", width=1))
            fig_sc.add_vline(x=bt_threshold, line=dict(color="#5468ff", dash="dash", width=1.5),
                             annotation_text=f"Soglia {bt_threshold}")
            fig_sc.update_layout(**DARK_LAYOUT, height=320,
                title=f"Score vs Return {bt_fwd_days} barre (scatter)",
                xaxis_title="Buy Score", yaxis_title="Return (%)")
            st.plotly_chart(fig_sc, use_container_width=True)

        # ── Equity curve ──────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 💹 Equity Curve — Strategia vs Buy & Hold")
        strat_eq, bh_eq = _equity_sim(bt_valid, bt_fwd_days, bt_threshold)
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=bt_valid.index, y=bh_eq,
            mode="lines", line=dict(color="#6b7394", width=1.8), name="Buy & Hold"))
        fig_eq.add_trace(go.Scatter(x=bt_valid.index, y=strat_eq,
            mode="lines", line=dict(color="#5468ff", width=2.5), name=f"Strategia Score≥{bt_threshold}"))
        fig_eq.add_hline(y=1, line=dict(color="#f0f2ff", dash="dot", width=1))
        fig_eq.update_layout(**DARK_LAYOUT, height=340,
            title=f"Equity multiplo (base 1.0) — {bt_tf} · costi {bt_costs}bps",
            xaxis_title="Data", yaxis_title="Equity")
        st.plotly_chart(fig_eq, use_container_width=True)

        # ── Signal contribution heatmap ───────────────────────────
        st.markdown("---")
        st.markdown("#### 🔬 Contributo dei Segnali al Return (correlazione)")
        bt_sigs = st.session_state.get("_res_bt_sigs_df")
        if bt_sigs is not None:
            sigs_aligned = bt_sigs.reindex(bt_valid.index).fillna(0.5)
            corr_rows = []
            for key, label in SIGNAL_LABELS.items():
                r_corr = bt_valid[fwd_col].corr(sigs_aligned[key])
                corr_rows.append({"Segnale": label, "Correlazione con return fwd": r_corr,
                                  "Peso attivo": w_pct.get(key, 0)})
            corr_df = pd.DataFrame(corr_rows).sort_values("Correlazione con return fwd", ascending=False)
            bar_colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in corr_df["Correlazione con return fwd"]]
            fig_corr = go.Figure(go.Bar(
                x=corr_df["Segnale"], y=corr_df["Correlazione con return fwd"],
                marker_color=bar_colors,
                text=corr_df["Correlazione con return fwd"].round(3),
                textposition="outside"))
            fig_corr.add_hline(y=0, line=dict(color="#f0f2ff", dash="dot", width=1))
            fig_corr.update_layout(**DARK_LAYOUT, height=300,
                title=f"Correlazione segnale → return {bt_fwd_days} barre",
                xaxis_tickangle=-35, yaxis_title="Pearson r")
            st.plotly_chart(fig_corr, use_container_width=True)

        # ── Return by score bucket ─────────────────────────────────
        st.markdown("---")
        st.markdown("#### 🪣 Return Medio per Bucket di Score")
        bt_valid["bucket"] = pd.cut(bt_valid["score"],
            bins=[0, 25, 40, 55, 65, 75, 100],
            labels=["0–25", "26–40", "41–55", "56–65", "66–75", "76–100"])
        grp_ = bt_valid.groupby("bucket", observed=True)[fwd_col].agg(["mean", "count", "std"]).reset_index()
        grp_["se"] = grp_["std"] / grp_["count"].pow(0.5)
        b_colors = ["#e74c3c", "#e67e22", "#f39c12", "#3498db", "#2ecc71", "#27ae60"]
        fig_bkt = go.Figure(go.Bar(
            x=grp_["bucket"].astype(str), y=grp_["mean"],
            error_y=dict(type="data", array=grp_["se"].tolist(), visible=True,
                         color="#6b7394", thickness=1.5),
            marker_color=b_colors[:len(grp_)],
            text=grp_.apply(lambda r: f"{r['mean']:.1f}%<br>n={int(r['count'])}", axis=1),
            textposition="outside"))
        fig_bkt.add_hline(y=0, line=dict(color="#f0f2ff", dash="dot", width=1))
        fig_bkt.update_layout(**DARK_LAYOUT, height=320,
            title=f"Return medio ({bt_fwd_days} barre) per bucket di score",
            xaxis_title="Score bucket", yaxis_title="Avg Return (%)")
        st.plotly_chart(fig_bkt, use_container_width=True)

        # ── Risk metrics table ─────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 📊 Indici di Rischio/Rendimento")
        risk_rows = []
        for grp_lbl, grp_df in [("Tutti", bt_valid), (f"BUY ≥ {bt_threshold}", buy_signals)]:
            if grp_df.empty: continue
            m = _performance_metrics(grp_df[fwd_col], grp_lbl, bt_fwd_days)
            if m: risk_rows.append(m)
        if risk_rows:
            st.dataframe(pd.DataFrame(risk_rows), hide_index=True, use_container_width=True)

        # ── Multi-horizon summary ──────────────────────────────────
        st.markdown("---")
        st.markdown("#### 📆 Riepilogo Multi-Orizzonte")
        hz_rows = []
        for fwd in [3, 7, 14, 30, 60, 90]:
            fc = f"fwd_{fwd}"
            if fc not in bt_valid.columns: continue
            buy_s = bt_valid[bt_valid["score"] >= bt_threshold][fc].dropna()
            all_s = bt_valid[fc].dropna()
            if len(all_s) == 0: continue
            hz_rows.append({
                "Orizzonte": f"{fwd} barre ({bt_tf})",
                "n segnali": len(buy_s),
                "Avg BUY": f"{buy_s.mean():.1f}%" if len(buy_s) else "N/A",
                "Avg ALL": f"{all_s.mean():.1f}%",
                "Win Rate BUY": f"{(buy_s>0).mean()*100:.0f}%" if len(buy_s) else "N/A",
                "Win Rate ALL": f"{(all_s>0).mean()*100:.0f}%",
                "Edge": f"{buy_s.mean()-all_s.mean():.1f}%" if len(buy_s) else "N/A",
                "Sharpe BUY": f"{buy_s.mean()/buy_s.std():.2f}" if len(buy_s) > 2 and buy_s.std() > 0 else "N/A",
            })
        if hz_rows:
            st.dataframe(pd.DataFrame(hz_rows), hide_index=True, use_container_width=True)

        # ── Multi-timeframe comparison ─────────────────────────────
        st.markdown("---")
        st.markdown("#### ⏱️ Confronto Multi-Timeframe")
        st.markdown("""
        <div class='insight-box' style='font-size:12px;'>
        Confronta la stessa strategia (pesi attivi, stessa soglia) su tutti i timeframe disponibili.
        Mostra edge e win rate per ogni TF + orizzonte forward equivalente.
        </div>""", unsafe_allow_html=True)
        run_mtf = st.button("🔄 Calcola confronto multi-TF", key="bt_mtf")
        if run_mtf:
            mtf_rows = []
            for tf in ["1D", "4H", "1H"]:
                with st.spinner(f"Caricamento {tf}…"):
                    if has_options:
                        tf_df = _fetch_bt_ohlcv(ticker, tf, min(bt_lookback, 365))
                    else:
                        tf_df = binance_ohlcv(binance_sym, tf, min(bt_lookback * 7, 1000))
                if tf_df.empty or len(tf_df) < 60:
                    continue
                tf_sigs = _compute_raw_signals(tf_df, iv_proxy=float(data["impliedVolatility"].median()) if _has_data else 70.0)
                # Map fwd_days to bars: 1D=1, 4H=6, 1H=24 per day
                bars_map = {"1D": bt_fwd_days, "4H": bt_fwd_days * 6, "1H": bt_fwd_days * 24}
                fwd_b = min(bars_map.get(tf, bt_fwd_days), len(tf_df) // 3)
                fp = tf_df["close"].shift(-fwd_b)
                tf_df[f"fwd_tf"] = (fp / tf_df["close"] - 1) * 100 - (bt_costs / 100)
                tf_score = _weighted_score(tf_sigs, active_weights)
                tf_df["score"] = tf_score
                tf_v = tf_df.dropna(subset=["fwd_tf", "score"])
                buy_tf = tf_v[tf_v["score"] >= bt_threshold]
                all_tf = tf_v
                if len(all_tf) == 0: continue
                mtf_rows.append({
                    "Timeframe": tf,
                    "n totale": len(all_tf),
                    f"n segnali BUY≥{bt_threshold}": len(buy_tf),
                    "Avg ALL": f"{all_tf['fwd_tf'].mean():.1f}%",
                    "Avg BUY": f"{buy_tf['fwd_tf'].mean():.1f}%" if len(buy_tf) else "N/A",
                    "Win Rate ALL": f"{(all_tf['fwd_tf']>0).mean()*100:.0f}%",
                    "Win Rate BUY": f"{(buy_tf['fwd_tf']>0).mean()*100:.0f}%" if len(buy_tf) else "N/A",
                    "Edge": f"{buy_tf['fwd_tf'].mean()-all_tf['fwd_tf'].mean():.1f}%" if len(buy_tf) else "N/A",
                    "Fwd barre equiv.": fwd_b,
                })
            if mtf_rows:
                st.dataframe(pd.DataFrame(mtf_rows), hide_index=True, use_container_width=True)
            else:
                st.warning("Impossibile calcolare confronto multi-TF (dati insufficienti).")

        # ── Interpretation ─────────────────────────────────────────
        st.markdown("---")
        if pd.notna(edge) and n_signals >= 5:
            if edge > 5:
                bt_v = f"✅ <b>Edge positivo +{edge:.1f}%</b> vs base su {bt_fwd_days} barre · {n_signals} segnali generati."
                bt_c = "#2ecc71"
            elif edge > 0:
                bt_v = f"⚠️ Edge contenuto (<b>+{edge:.1f}%</b>). Il segnale funziona ma il margine è modesto su {bt_fwd_days} barre."
                bt_c = "#f39c12"
            else:
                bt_v = f"❌ <b>Edge negativo ({edge:.1f}%)</b>. Prova a cambiare soglia, pesi o timeframe."
                bt_c = "#e74c3c"
            st.markdown(f"""<div class='insight-box' style='border-left-color:{bt_c};'>
            {bt_v}
            </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class='insight-box' style='border-left-color:#2a3060;font-size:11px;'>
        <b>Limiti metodologici:</b><br>
        • Dati: solo OHLCV del perpetual Deribit — nessun dato opzioni storico, nessun on-chain, nessuna macro.<br>
        • Look-ahead bias ridotto: ogni score usa solo dati disponibili al momento T; i forward return sono fuori-campione per costruzione.<br>
        • Costi inclusi ma costanti: non catturano slippage variabile, funding rate, tasse.<br>
        • Ottimizzatore: i pesi ottimali su dati passati possono soffrire di overfitting — usali come guida, non come oracolo.<br>
        • Performance passata non garantisce risultati futuri.
        </div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
#  FOOTER
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;font-family:Space Mono,monospace;font-size:10px;color:#2a3050;padding:16px 0;'>
  Vol Dashboard · Deribit API v2 · Solo a scopo educativo · Non è consulenza finanziaria
</div>""", unsafe_allow_html=True)

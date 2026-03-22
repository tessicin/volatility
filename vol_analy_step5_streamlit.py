#!/usr/bin/env python3
"""
Volatility Dashboard — Deribit Options Analysis
Run with:  streamlit run vol_dashboard.py
"""

import re
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
    ticker = st.selectbox("Asset", ["BTC", "ETH"], index=0)
    window = st.slider("Strike window (%)", min_value=5, max_value=30, value=12, step=1) / 100
    max_exp = st.slider("Max scadenze da mostrare", 3, 15, 8)
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

if run or (st.session_state.data is None):
    with st.spinner(f"Connessione a Deribit per {ticker}…"):
        try:
            data, recent_price = get_options_data.clear() or get_options_data(ticker, window)
            st.session_state.data = data
            st.session_state.recent_price = recent_price
            st.session_state.ticker_loaded = ticker
        except Exception as e:
            st.error(f"Errore API Deribit: {e}")
            st.stop()

data = st.session_state.data
recent_price = st.session_state.recent_price

if data is None or data.empty:
    st.warning("Nessun dato disponibile. Premi 'Carica dati live'.")
    st.stop()

calls = data[data["type"] == "call"].copy()
puts = data[data["type"] == "put"].copy()

# ──────────────────────────────────────────────
#  KEY METRICS ROW
# ──────────────────────────────────────────────
avg_iv_c = calls["impliedVolatility"].mean()
avg_iv_p = puts["impliedVolatility"].mean()
pcr = put_call_ratio(calls, puts)
skew = compute_skew(calls, puts)
overall, signals = sentiment_signal(skew, pcr, avg_iv_c, avg_iv_p)
ts_df = term_structure(data)

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

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Volatility Smile", "📈 Term Structure", "📦 Open Interest", "⚖️ Put/Call & Skew", "🎯 Decision Tool", "🧠 RV Forecast Lab"
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

# ──────────────────────────────────────────────
#  FOOTER
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;font-family:Space Mono,monospace;font-size:10px;color:#2a3050;padding:16px 0;'>
  Vol Dashboard · Deribit API v2 · Solo a scopo educativo · Non è consulenza finanziaria
</div>""", unsafe_allow_html=True)

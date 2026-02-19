import streamlit as st
import pandas as pd
import json
from pathlib import Path
import joblib

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# =========================
# CONFIG (LIGHT THEME)
# =========================
st.set_page_config(page_title="Doji-X Platform", layout="wide", page_icon="📊")

# --- CSS FIX FOR TOP VISIBILITY ---
st.markdown(
    """
    <style>
      /* Increase top padding so badges aren't hidden behind the header */
      .block-container { 
          padding-top: 3.5rem; 
          padding-bottom: 2rem; 
      }
      
      div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #eee;
        padding: 14px 14px;
        border-radius: 14px;
      }
      section[data-testid="stSidebar"] { background: #fafafa; }
      
      /* Style for the static top bar badges */
      .top-badge {
        background-color: #f0f2f6;
        padding: 8px 12px;
        border-radius: 8px;
        font-weight: 600;
        color: #31333F;
        display: inline-block;
        border: 1px solid #e0e0e0;
        margin-bottom: 10px; /* Add space below badge */
      }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# PATHS
# =========================
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data/processed/GC_F_5m_60d_with_indicators.csv"
TRADES_PATH = BASE_DIR / "results/backtests/trades.csv"
EQUITY_PATH = BASE_DIR / "results/backtests/equity.csv"
METRICS_PATH = BASE_DIR / "results/backtests/metrics.json"
MODEL_PATH = BASE_DIR / "models/latest_model.joblib"


# =========================
# DATA LOADING 
# =========================
@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    # --- Load Price Data ---
    df = pd.read_csv(DATA_PATH)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        # FIX: Force Timezone Naive
        if df["datetime"].dt.tz is not None:
            df["datetime"] = df["datetime"].dt.tz_localize(None)
        df = df.set_index("datetime")

    # --- Load Trades ---
    if TRADES_PATH.exists():
        trades = pd.read_csv(TRADES_PATH)
        for col in ["entry_dt", "exit_dt"]:
            if col in trades.columns:
                trades[col] = pd.to_datetime(trades[col])
                # FIX: Force Timezone Naive
                if trades[col].dt.tz is not None:
                    trades[col] = trades[col].dt.tz_localize(None)
        
        if "entry_dt" in trades.columns:
            trades["date"] = trades["entry_dt"].dt.date
            trades["month_code"] = trades["entry_dt"].dt.strftime("%Y-%m")
            trades["year"] = trades["entry_dt"].dt.year
    else:
        trades = pd.DataFrame()

    # --- Load Equity ---
    if EQUITY_PATH.exists():
        equity = pd.read_csv(EQUITY_PATH)
        if "datetime" in equity.columns:
            equity["datetime"] = pd.to_datetime(equity["datetime"])
            # FIX: Force Timezone Naive
            if equity["datetime"].dt.tz is not None:
                equity["datetime"] = equity["datetime"].dt.tz_localize(None)
            equity = equity.set_index("datetime")
    else:
        equity = pd.DataFrame()

    # --- Load Metrics ---
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)
    else:
        metrics = {}

    return df, trades, equity, metrics


def stop_if_no_data(df: pd.DataFrame):
    if df.empty:
        st.error("Data not found. Please run the data loader and backtest scripts first.")
        st.stop()


df, trades, equity, metrics = load_data()
stop_if_no_data(df)


# =========================
# HELPERS
# =========================
def kpi_strip(items, cols=5):
    """items = list of tuples: (label, value, delta(optional or None))"""
    col_list = st.columns(cols)
    for i, (label, value, delta) in enumerate(items):
        with col_list[i % cols]:
            if delta is None:
                st.metric(label, value)
            else:
                st.metric(label, value, delta=delta)


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def compute_win_rate(trades_df: pd.DataFrame) -> float:
    if trades_df is None or trades_df.empty or "pnl" not in trades_df.columns:
        return 0.0
    return (trades_df["pnl"] > 0).mean() * 100.0


def compute_max_drawdown(equity_df: pd.DataFrame) -> float:
    if equity_df is None or equity_df.empty or "equity" not in equity_df.columns:
        return 0.0
    s = equity_df["equity"].astype(float)
    peak = s.cummax()
    dd = (s - peak) / peak
    return float(dd.min()) * 100.0  # negative %


def topbar_controls(df: pd.DataFrame):
    """
    Top bar with:
    Static Info | View Mode | Dynamic selector | (No Thesis Toggle)
    Returns: symbol, tf, start_date, end_date, view_mode
    """
    # Hardcoded values for your thesis dataset
    symbol = "GC=F"
    tf = "5m"

    with st.container():
        # Adjusted columns: Info Badge | View Mode | Date Selector | (Empty space for balance)
        c1, c2, c3, c4 = st.columns([1.5, 1.0, 2.0, 0.8])

        with c1:
            # Static Badge Display (No Dropdown needed)
            st.markdown(f'<div class="top-badge">🪙 {symbol} &nbsp; | &nbsp; ⏱ {tf}</div>', unsafe_allow_html=True)

        with c2:
            view_mode = st.selectbox("View Mode", ["Daily", "Weekly", "Monthly", "Yearly"], index=0, label_visibility="collapsed")

        idx = pd.to_datetime(df.index)
        min_date = idx.min().date()
        max_date = idx.max().date()

        # ---- Weekly options ----
        iso = idx.isocalendar()
        df_weeks = pd.DataFrame({"date": idx.date, "year": iso["year"], "week": iso["week"]}).drop_duplicates()
        week_pairs = df_weeks[["year", "week"]].drop_duplicates().sort_values(["year", "week"]).values.tolist()

        week_labels = []
        week_map = {} 
        for y, w in week_pairs:
            start = pd.Timestamp.fromisocalendar(int(y), int(w), 1).date()
            end = pd.Timestamp.fromisocalendar(int(y), int(w), 7).date()
            start_clamped = max(start, min_date)
            end_clamped = min(end, max_date)
            label = f"Week {int(w)} ({int(y)})" 
            week_labels.append(label)
            week_map[label] = (start_clamped, end_clamped)

        # ---- Monthly options ----
        df_months = pd.DataFrame({"dt": idx})
        df_months["year"] = df_months["dt"].dt.year
        df_months["month"] = df_months["dt"].dt.month
        month_pairs = df_months[["year", "month"]].drop_duplicates().sort_values(["year", "month"]).values.tolist()

        month_labels = []
        month_map = {} 
        for y, m in month_pairs:
            p = pd.Period(f"{int(y)}-{int(m):02d}", freq="M")
            start = p.start_time.date()
            end = p.end_time.date()
            start_clamped = max(start, min_date)
            end_clamped = min(end, max_date)
            
            # Clean Label: "November 2025"
            nice = pd.Timestamp(int(y), int(m), 1).strftime("%B %Y")
            label = nice 
            month_labels.append(label)
            month_map[label] = (start_clamped, end_clamped)

        years = sorted(idx.year.unique().tolist())

        with c3:
            if view_mode == "Daily":
                selected_day = st.date_input(
                    "Select Date",
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date,
                    label_visibility="collapsed"
                )
                start_date = selected_day
                end_date = selected_day

            elif view_mode == "Weekly":
                if not week_labels:
                    st.warning("No weeks available.")
                    start_date, end_date = min_date, max_date
                else:
                    selected_week = st.selectbox("Select Week", options=week_labels, index=len(week_labels) - 1, label_visibility="collapsed")
                    start_date, end_date = week_map[selected_week]

            elif view_mode == "Monthly":
                if not month_labels:
                    st.warning("No months available.")
                    start_date, end_date = min_date, max_date
                else:
                    selected_month = st.selectbox("Select Month", options=month_labels, index=len(month_labels) - 1, label_visibility="collapsed")
                    start_date, end_date = month_map[selected_month]

            else:  # Yearly
                if not years:
                    st.warning("No years available.")
                    start_date, end_date = min_date, max_date
                else:
                    selected_year = st.selectbox("Select Year", options=years, index=len(years) - 1, label_visibility="collapsed")
                    start_date = pd.Timestamp(selected_year, 1, 1).date()
                    end_date = pd.Timestamp(selected_year, 12, 31).date()
                    start_date = max(start_date, min_date)
                    end_date = min(end_date, max_date)
        

    return symbol, tf, start_date, end_date, view_mode


def filter_df_by_dates(df: pd.DataFrame, start_date, end_date):
    mask = (df.index.date >= start_date) & (df.index.date <= end_date)
    return df.loc[mask].copy()


def filter_trades_by_dates(trades_df: pd.DataFrame, start_date, end_date):
    if trades_df is None or trades_df.empty or "entry_dt" not in trades_df.columns:
        return trades_df
    mask = (trades_df["entry_dt"].dt.date >= start_date) & (trades_df["entry_dt"].dt.date <= end_date)
    return trades_df.loc[mask].copy()


def make_equity_chart(equity_df: pd.DataFrame, price_df: pd.DataFrame):
    fig = go.Figure()

    # Strategy equity
    if (equity_df is not None) and (not equity_df.empty) and ("equity" in equity_df.columns):
        fig.add_trace(
            go.Scatter(
                x=equity_df.index,
                y=equity_df["equity"].astype(float),
                mode="lines",
                name="Strategy",
            )
        )
        initial_equity = float(equity_df["equity"].iloc[0])
    else:
        initial_equity = 10000.0

    # Buy & Hold normalized
    if (price_df is not None) and (not price_df.empty) and ("close" in price_df.columns):
        if (equity_df is not None) and (not equity_df.empty):
            # The timezone fix in load_data ensures indices are compatible here
            aligned = price_df.reindex(equity_df.index, method="nearest").copy()
        else:
            aligned = price_df.copy()

        aligned = aligned.dropna(subset=["close"])
        if not aligned.empty:
            bh = (aligned["close"].astype(float) / float(aligned["close"].iloc[0])) * initial_equity
            fig.add_trace(
                go.Scatter(
                    x=aligned.index,
                    y=bh,
                    mode="lines",
                    name="Buy & Hold",
                    line=dict(dash="dash"),
                )
            )

    fig.update_layout(
        title="Portfolio Value Over Time (Strategy vs Buy & Hold)",
        template="plotly_white",
        height=360,
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis_title="Datetime",
        yaxis_title="Equity ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def make_heatmap(trades_df: pd.DataFrame):
    daily = trades_df.groupby("date")["pnl"].sum().reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["Week"] = daily["date"].dt.isocalendar().week.astype(int)
    daily["DayOfWeek"] = daily["date"].dt.dayofweek
    daily["Text"] = daily.apply(lambda x: f"{x['date'].date()}: ${x['pnl']:.2f}", axis=1)

    fig = go.Figure(
        data=go.Heatmap(
            z=daily["pnl"],
            x=daily["Week"],
            y=daily["DayOfWeek"],
            text=daily["Text"],
            hoverinfo="text",
            colorscale="RdYlGn",
            xgap=3,
            ygap=3,
        )
    )
    fig.update_layout(
        height=220,
        margin=dict(t=10, b=10, l=10, r=10),
        yaxis=dict(
            tickmode="array",
            tickvals=[0, 1, 2, 3, 4, 5, 6],
            ticktext=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        ),
        xaxis_title="Week Number",
    )
    return fig


def make_price_chart(chart_df: pd.DataFrame, period_trades: pd.DataFrame, title: str):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.75, 0.25])

    fig.add_trace(
        go.Candlestick(
            x=chart_df.index,
            open=chart_df["open"],
            high=chart_df["high"],
            low=chart_df["low"],
            close=chart_df["close"],
            name="Price",
        ),
        row=1,
        col=1,
    )

    # Optional EMA
    if "ema_50" in chart_df.columns:
        fig.add_trace(
            go.Scatter(x=chart_df.index, y=chart_df["ema_50"], name="EMA 50", line=dict(width=1)),
            row=1,
            col=1,
        )

    # Trade markers
    if period_trades is not None and not period_trades.empty:
        if {"entry_dt", "entry_price"}.issubset(period_trades.columns):
            fig.add_trace(
                go.Scatter(
                    x=period_trades["entry_dt"],
                    y=period_trades["entry_price"],
                    mode="markers",
                    marker=dict(symbol="triangle-up", size=12),
                    name="Buy",
                ),
                row=1,
                col=1,
            )
        if {"exit_dt", "exit_price"}.issubset(period_trades.columns):
            fig.add_trace(
                go.Scatter(
                    x=period_trades["exit_dt"],
                    y=period_trades["exit_price"],
                    mode="markers",
                    marker=dict(symbol="x", size=9),
                    name="Sell",
                ),
                row=1,
                col=1,
            )

    # Volume
    if "volume" in chart_df.columns:
        fig.add_trace(go.Bar(x=chart_df.index, y=chart_df["volume"], name="Volume"), row=2, col=1)

    fig.update_layout(title=title, height=520, xaxis_rangeslider_visible=False, template="plotly_white")
    return fig


# =========================
# SIDEBAR NAV
# =========================
st.sidebar.title("Doji-X Platform")
page = st.sidebar.radio(
    "Navigate",
    [
        "📊 Overview",
        "⚡ Live Strategy",
        "🧠 AI Engine",
        "📈 Performance",
        "🛡 Risk & Drawdown",
        "📜 Trade Journal",
        # "⚙ Settings",
    ],
)

# =========================
# GLOBAL TOP BAR
# =========================
# Updated: No thesis_mode variable anymore
symbol, tf, start_date, end_date, view_mode = topbar_controls(df)

# Default: Thesis mode is OFF (Standard display)
thesis_mode = False

df_f = filter_df_by_dates(df, start_date, end_date)
trades_f = filter_trades_by_dates(trades, start_date, end_date)
equity_f = (
    equity.loc[(equity.index.date >= start_date) & (equity.index.date <= end_date)].copy()
    if not equity.empty else equity
)

# =========================
# PAGE: OVERVIEW
# =========================
if page == "📊 Overview":
    st.title("📊 Overview")

    total_return = safe_float(metrics.get("return_pct", 0.0), 0.0)
    win_rate = compute_win_rate(trades_f)
    max_dd = compute_max_drawdown(equity_f)
    total_trades = int(len(trades_f)) if trades_f is not None else 0
    end_val = safe_float(metrics.get("end_value", 10000), 10000)

    # KPI strip
    kpi_strip(
        [
            ("Total Return", f"{total_return:.2f}%", None),
            ("Max Drawdown", f"{max_dd:.2f}%", None),
            ("Win Rate", f"{win_rate:.1f}%", None),
            ("Total Trades", f"{total_trades}", None),
            ("Final Equity", f"${end_val:,.2f}", None),
        ],
        cols=5,
    )

    st.markdown("")

    c_left, c_right = st.columns([2.2, 1.2])

    with c_left:
        st.subheader("Account Growth")
        if not equity_f.empty:
            st.plotly_chart(make_equity_chart(equity_f, df_f), use_container_width=True)
        else:
            st.info("No equity curve found.")

    with c_right:
        st.subheader("Today / Selection Summary")
        
        # --- LOGIC ADDED: Calculate Live Confidence for Overview ---
        live_confidence = "—"
        bias_text = "Neutral"
        
        if not df_f.empty and MODEL_PATH.exists():
            try:
                # Load model to get real-time confidence
                model = joblib.load(MODEL_PATH)
                if hasattr(model, "feature_names_in_"):
                    feats = list(model.feature_names_in_)
                    # Get last candle features
                    last_features = df_f.iloc[[-1]].reindex(columns=feats, fill_value=0)
                    
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(last_features)[0]
                        up_prob = probs[1]
                        live_confidence = f"{up_prob:.1%}"
                        
                        # Simple Bias Logic
                        if up_prob > 0.60: bias_text = "Bullish"
                        elif up_prob < 0.40: bias_text = "Bearish"
            except Exception:
                pass
        # -----------------------------------------------------------

        st.metric("Current Bias", bias_text)
        st.metric("Model Confidence", live_confidence)
        st.metric("PnL (selected)", f"${trades_f['pnl'].sum():.2f}" if (trades_f is not None and not trades_f.empty) else "$0.00")

        if trades_f is not None and not trades_f.empty:
            last = trades_f.sort_values("entry_dt").iloc[-1]
            st.caption("Last Trade")
            st.write(f"- Entry: {last.get('entry_dt', '')}")
            st.write(f"- Exit: {last.get('exit_dt', '')}")
            st.write(f"- PnL: ${float(last.get('pnl', 0.0)):.2f}")

    st.markdown("---")

    c1, c2 = st.columns([1.4, 1.0])
    with c1:
        st.subheader("Weekly Heatmap (PnL)")
        if trades_f is not None and not trades_f.empty:
            st.plotly_chart(make_heatmap(trades_f), use_container_width=True)
        else:
            st.info("No trades found in this range.")

    with c2:
        st.subheader("Top Insights")
        st.write("• Best / worst blocks (add expectancy later)")
        st.write("• Risk notes (drawdown, streaks)")
        st.write("• Thesis note: ML used as confirmation filter")

# =========================
# PAGE: LIVE STRATEGY
# =========================
elif page == "⚡ Live Strategy":
    st.title("⚡ Live Strategy")

    # --- 1. Load Model & Prepare Live Data ---
    latest_signal = "NEUTRAL"
    confidence = 0.0
    ml_agreement = "WAIT"
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    
    # Get the very last candle from your data
    if not df_f.empty:
        last_candle = df_f.iloc[-1]
        current_close = last_candle["close"]
        
        # Try to load model for prediction
        if MODEL_PATH.exists():
            try:
                model = joblib.load(MODEL_PATH)
                if hasattr(model, "feature_names_in_"):
                    feats = list(model.feature_names_in_)
                    # Extract features for the last candle
                    # Note: Ensure your CSV has these columns. If not, fill with 0.
                    live_features = df_f.iloc[[-1]].reindex(columns=feats, fill_value=0)
                    
                    # Get Probability
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(live_features)[0]
                        confidence = probs[1] # Probability of Class 1 (Up)
                    else:
                        confidence = 0.5 # Fallback
            except:
                pass

        # --- 2. Simple Strategy Logic (Heikin Ashi + ML) ---
        # Logic: If Green Candle + Conf > 60% -> BUY
        #        If Red Candle + Conf < 40% -> SELL
        
        # Check Heikin Ashi color (approximation if columns exist, else use Close > Open)
        is_green = last_candle["close"] > last_candle["open"]
        
        if is_green and confidence > 0.60:
            latest_signal = "LONG (BUY)"
            ml_agreement = "BULLISH ✅"
            entry_price = current_close
            # Dynamic Risk limits (using ATR if available, else % based)
            atr = last_candle.get("atr", current_close * 0.005) 
            stop_loss = current_close - (2.0 * atr)
            take_profit = current_close + (3.0 * atr)
            
        elif not is_green and confidence < 0.40:
            latest_signal = "SHORT (SELL)"
            ml_agreement = "BEARISH ✅"
            entry_price = current_close
            atr = last_candle.get("atr", current_close * 0.005)
            stop_loss = current_close + (2.0 * atr)
            take_profit = current_close - (3.0 * atr)
        else:
            latest_signal = "NO TRADE"
            ml_agreement = "NEUTRAL ⚪"

    # --- 3. Render the Page ---
    s1, s2, s3 = st.columns([1.2, 1.2, 1.0])
    with s1:
        session = st.selectbox("Session", ["All", "London", "New York", "Asia"], index=0)
    with s2:
        show_markers = st.toggle("Show trade markers", value=True)
    with s3:
        auto_refresh = st.toggle("Auto refresh", value=False)

    left, right = st.columns([2.2, 1.2])

    with left:
        st.subheader("Chart")
        chart_df = df_f.tail(300).copy() # Show last 300 bars
        period_trades = trades_f.copy() if (show_markers and trades_f is not None) else pd.DataFrame()

        if not chart_df.empty:
            fig = make_price_chart(chart_df, period_trades, title=f"{symbol} • {tf} • Live View")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No chart data available for selected range.")

    with right:
        st.subheader("Strategy State")
        
        # Dynamic Color for Signal
        sig_color = "normal"
        if "LONG" in latest_signal: sig_color = "off" # green-ish in streamlit metric usually
        if "SHORT" in latest_signal: sig_color = "inverse"

        st.metric("Current Signal", latest_signal)
        st.metric("Model Confidence", f"{confidence:.1%}")
        st.metric("ML Agreement", ml_agreement)
        
        # Risk Reward Calculation
        if entry_price > 0 and (entry_price - stop_loss) != 0:
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            rr_ratio = reward / risk
            st.metric("Risk/Reward", f"1 : {rr_ratio:.2f}")
        else:
            st.metric("Risk/Reward", "—")

        # st.markdown("---")
        # st.caption("Projected Levels (Live)")
        # if latest_signal != "NO TRADE":
        #     st.write(f"**Entry:** ${entry_price:,.2f}")
        #     st.write(f"**Stop:** ${stop_loss:,.2f}")
        #     st.write(f"**Target:** ${take_profit:,.2f}")
        # else:
        #     st.write("Waiting for setup...")

    with st.expander("📌 Last 10 trades", expanded=False):
        if trades_f is not None and not trades_f.empty:
            cols = [c for c in ["entry_dt", "entry_price", "exit_dt", "exit_price", "pnl"] if c in trades_f.columns]
            st.dataframe(trades_f.sort_values("entry_dt").tail(10)[cols], use_container_width=True)
        else:
            st.info("No trades available.")

# =========================
# PAGE: AI ENGINE
# =========================
elif page == "🧠 AI Engine":
    st.title("🧠 AI Engine")
    tabs = st.tabs(["Trust the Model", "Explain Decisions"])

    with tabs[0]:
        st.subheader("Model Performance")

        if not MODEL_PATH.exists():
            st.warning("⚠️ Model file not found. Please run training first.")
        else:
            try:
                model = joblib.load(MODEL_PATH)
                from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, accuracy_score, f1_score

                # Detect features
                if hasattr(model, "feature_names_in_"):
                    feature_cols = list(model.feature_names_in_)
                else:
                    feature_cols = [
                        c for c in df.columns
                        if c not in ["datetime", "y", "future_ret", "target"]
                        and pd.api.types.is_numeric_dtype(df[c])
                    ]

                eval_df = df.dropna().copy()

                # Simple target: next 3 bars up/down
                eval_df["future_close"] = eval_df["close"].shift(-3)
                eval_df["target"] = (eval_df["future_close"] > eval_df["close"]).astype(int)
                eval_df = eval_df.dropna()

                X_eval = eval_df[feature_cols].fillna(0)
                y_eval = eval_df["target"]

                y_pred = model.predict(X_eval)
                y_proba = model.predict_proba(X_eval)[:, 1] if hasattr(model, "predict_proba") else None

                acc = accuracy_score(y_eval, y_pred)
                prec = precision_score(y_eval, y_pred, zero_division=0)
                f1 = f1_score(y_eval, y_pred, zero_division=0)

                roc_auc = 0.0
                fpr = tpr = None
                if y_proba is not None:
                    try:
                        fpr, tpr, _ = roc_curve(y_eval, y_proba)
                        roc_auc = auc(fpr, tpr)
                    except Exception:
                        pass

                kpi_strip(
                    [
                        ("Accuracy", f"{acc:.1%}", None),
                        ("Precision", f"{prec:.1%}", None),
                        ("F1", f"{f1:.2f}", None),
                        ("ROC-AUC", f"{roc_auc:.2f}", None),
                        ("Samples", f"{len(y_eval):,}", None),
                    ],
                    cols=5,
                )

                st.info("Thesis note: Model is used as a **confirmation filter**, not a standalone predictor.")

                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_eval, y_pred)
                    fig_cm = px.imshow(
                        cm,
                        text_auto=True,
                        color_continuous_scale="Blues",
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=["DOWN", "UP"],
                        y=["DOWN", "UP"],
                    )
                    fig_cm.update_layout(template="plotly_white", height=340, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig_cm, use_container_width=True)

                with c2:
                    st.subheader("ROC Curve")
                    if fpr is not None and tpr is not None:
                        roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr})
                        fig_roc = px.area(roc_df, x="FPR", y="TPR", title=f"AUC = {roc_auc:.2f}", template="plotly_white")
                        fig_roc.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)
                        fig_roc.update_layout(height=340, margin=dict(l=10, r=10, t=60, b=10))
                        st.plotly_chart(fig_roc, use_container_width=True)
                    else:
                        st.info("ROC not available (need predict_proba + class diversity).")

            except Exception as e:
                st.error(f"Error calculating AI insights: {e}")

    with tabs[1]:
        st.subheader("Feature Importance")
        if not MODEL_PATH.exists():
            st.warning("⚠️ Model file not found.")
        else:
            try:
                model = joblib.load(MODEL_PATH)

                if hasattr(model, "feature_names_in_"):
                    feature_cols = list(model.feature_names_in_)
                else:
                    feature_cols = [
                        c for c in df.columns
                        if c not in ["datetime", "y", "future_ret", "target"]
                        and pd.api.types.is_numeric_dtype(df[c])
                    ]

                # --- 1. Feature Importance Plot ---
                if hasattr(model, "feature_importances_"):
                    imp_df = pd.DataFrame(
                        {"Feature": feature_cols, "Importance": model.feature_importances_}
                    ).sort_values("Importance", ascending=True)

                    fig_imp = px.bar(
                        imp_df,
                        x="Importance",
                        y="Feature",
                        orientation="h",
                        height=520,
                        template="plotly_white",
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)
                else:
                    st.info("This model does not expose feature_importances_ (not a tree-based model).")

                st.markdown("---")

                # --- 2. Live Prediction (Replaces Placeholder) ---
                st.subheader("Last Prediction")
                
                # Get the very last row of data (most recent candle)
                if not df.empty:
                    last_row = df.iloc[[-1]][feature_cols].fillna(0)
                    last_time = df.index[-1]
                    
                    # Predict proba
                    if hasattr(model, "predict_proba"):
                        last_prob = model.predict_proba(last_row)[0][1] # Prob of class 1 (UP)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Prediction Time", str(last_time))
                            st.metric("Bullish Probability", f"{last_prob:.2%}")
                        
                        with col2:
                            decision = "LONG" if last_prob > 0.5 else "SHORT/NEUTRAL"
                            st.metric("Model Bias", decision)
                            st.progress(last_prob)
                    else:
                         st.write("Model does not support probability output.")
                else:
                    st.write("No data available for prediction.")

            except Exception as e:
                st.error(f"Error loading feature importance: {e}")


# =========================
# PAGE: PERFORMANCE
# =========================
elif page == "📈 Performance":
    st.title("📈 Performance")

    total_return = safe_float(metrics.get("return_pct", 0.0), 0.0)
    win_rate = compute_win_rate(trades_f)
    max_dd = compute_max_drawdown(equity_f)
    total_trades = int(len(trades_f)) if trades_f is not None else 0
    end_val = safe_float(metrics.get("end_value", 10000), 10000)

    kpi_strip(
        [
            ("Total Return", f"{total_return:.2f}%", None),
            ("Win Rate", f"{win_rate:.1f}%", None),
            ("Max Drawdown", f"{max_dd:.2f}%", None),
            ("Total Trades", f"{total_trades}", None),
            ("Final Equity", f"${end_val:,.2f}", None),
        ],
        cols=5,
    )

    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Monthly Returns")
        if trades_f is not None and not trades_f.empty and "entry_dt" in trades_f.columns:
            tmp = trades_f.copy()
            tmp["month_code"] = tmp["entry_dt"].dt.strftime("%Y-%m")
            monthly = tmp.groupby("month_code")["pnl"].sum().sort_index()
            fig_m = px.bar(
                x=monthly.index,
                y=monthly.values,
                labels={"x": "Month", "y": "PnL"},
                template="plotly_white",
            )
            fig_m.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_m, use_container_width=True)
        else:
            st.info("No trades in selected range.")

    with c2:
        st.subheader("Win Rate by Month")
        if trades_f is not None and not trades_f.empty and "entry_dt" in trades_f.columns:
            tmp = trades_f.copy()
            tmp["month_code"] = tmp["entry_dt"].dt.strftime("%Y-%m")
            wr = tmp.groupby("month_code")["pnl"].apply(lambda x: (x > 0).mean() * 100).sort_index()
            fig_wr = px.line(
                x=wr.index,
                y=wr.values,
                markers=True,
                labels={"x": "Month", "y": "Win Rate %"},
                template="plotly_white",
            )
            fig_wr.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_wr, use_container_width=True)
        else:
            st.info("No trades in selected range.")

    st.markdown("---")

    c3, c4 = st.columns(2)

    with c3:
        st.subheader("Equity Curve")
        if not equity_f.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=equity_f.index, y=equity_f["equity"], mode="lines", name="Strategy"))
            fig.update_layout(height=360, template="plotly_white", margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No equity found.")

    with c4:
        st.subheader("PnL Distribution")
        if trades_f is not None and not trades_f.empty:
            fig_hist = px.histogram(trades_f, x="pnl", nbins=40, template="plotly_white")
            fig_hist.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("No trades found.")


# =========================
# PAGE: RISK & DRAWDOWN
# =========================
elif page == "🛡 Risk & Drawdown":
    st.title("🛡 Risk & Drawdown")

    max_dd = compute_max_drawdown(equity_f)

    losing_streak = 0
    if trades_f is not None and not trades_f.empty and "pnl" in trades_f.columns:
        streak = 0
        best = 0
        for pnl in trades_f.sort_values("entry_dt")["pnl"].tolist():
            if pnl <= 0:
                streak += 1
                best = max(best, streak)
            else:
                streak = 0
        losing_streak = best

    kpi_strip(
        [
            ("Max Drawdown", f"{max_dd:.2f}%", None),
            ("Losing Streak (max)", f"{losing_streak}", None),
            ("Trades (range)", f"{len(trades_f) if trades_f is not None else 0}", None),
            ("PnL (range)", f"${trades_f['pnl'].sum():.2f}" if (trades_f is not None and not trades_f.empty) else "$0.00", None),
            ("Recovery Time", "—", None),
        ],
        cols=5,
    )

    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Drawdown Curve")
        if not equity_f.empty and "equity" in equity_f.columns:
            s = equity_f["equity"].astype(float)
            peak = s.cummax()
            dd = (s - peak) / peak * 100.0
            fig_dd = px.area(x=dd.index, y=dd.values, labels={"x": "Time", "y": "Drawdown %"}, template="plotly_white")
            fig_dd.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_dd, use_container_width=True)
        else:
            st.info("No equity curve available.")

    with c2:
        st.subheader("Worst / Best Trades")
        if trades_f is not None and not trades_f.empty and "pnl" in trades_f.columns:
            cols = [c for c in ["entry_dt", "exit_dt", "pnl"] if c in trades_f.columns]
            worst = trades_f.nsmallest(10, "pnl")[cols]
            best = trades_f.nlargest(10, "pnl")[cols]
            t1, t2 = st.tabs(["Worst 10", "Best 10"])
            with t1:
                st.dataframe(worst, use_container_width=True)
            with t2:
                st.dataframe(best, use_container_width=True)
        else:
            st.info("No trades found.")


# =========================
# PAGE: TRADE JOURNAL
# =========================
elif page == "📜 Trade Journal":
    st.title("📜 Trade Journal")

    if trades_f is None or trades_f.empty:
        st.info("No trades found in selected range.")
        st.stop()

    f1, f2, f3, f4, f5 = st.columns([1.2, 1.2, 1.2, 1.2, 1.6])

    with f1:
        result = st.selectbox("Result", ["All", "Wins", "Losses"], index=0)
    with f2:
        min_pnl = st.number_input("Min PnL", value=float(trades_f["pnl"].min()))
    with f3:
        max_pnl = st.number_input("Max PnL", value=float(trades_f["pnl"].max()))
    with f4:
        sort_by = st.selectbox("Sort", ["Newest", "Oldest", "PnL High", "PnL Low"], index=0)
    with f5:
        search = st.text_input("Search (date/time)", value="")

    view = trades_f.copy()

    if result == "Wins":
        view = view[view["pnl"] > 0]
    elif result == "Losses":
        view = view[view["pnl"] <= 0]

    view = view[(view["pnl"] >= min_pnl) & (view["pnl"] <= max_pnl)]

    if search.strip():
        s = search.strip().lower()
        view = view[
            view["entry_dt"].astype(str).str.lower().str.contains(s)
            | view["exit_dt"].astype(str).str.lower().str.contains(s)
        ]

    if sort_by == "Newest":
        view = view.sort_values("entry_dt", ascending=False)
    elif sort_by == "Oldest":
        view = view.sort_values("entry_dt", ascending=True)
    elif sort_by == "PnL High":
        view = view.sort_values("pnl", ascending=False)
    else:
        view = view.sort_values("pnl", ascending=True)

    st.markdown("---")

    cols = [c for c in ["entry_dt", "entry_price", "exit_dt", "exit_price", "pnl"] if c in view.columns]
    st.dataframe(view[cols], use_container_width=True)

    csv = view[cols].to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered trades (CSV)", data=csv, file_name="doji_x_trades_filtered.csv", mime="text/csv")


# =========================
# PAGE: SETTINGS
# =========================
# elif page == "⚙ Settings":
#     st.title("⚙ Settings")

#     st.subheader("Paths")
#     st.code(
#         "\n".join(
#             [
#                 f"DATA_PATH   = {DATA_PATH}",
#                 f"TRADES_PATH = {TRADES_PATH}",
#                 f"EQUITY_PATH = {EQUITY_PATH}",
#                 f"METRICS_PATH= {METRICS_PATH}",
#                 f"MODEL_PATH  = {MODEL_PATH}",
#             ]
#         )
#     )

#     st.markdown("---")
#     st.subheader("Notes")
#     #st.write("- If the Thesis checkbox is ON, you can hide live-trading language later.")
#     st.write("- You can add exports (PDF/CSV) here.")
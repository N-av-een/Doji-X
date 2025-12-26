import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import joblib

# ----- CONFIG -----
st.set_page_config(page_title="Doji-X Platform", layout="wide", page_icon="📊")

# ----- PATHS -----
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data/processed/GC_F_5m_60d_with_indicators.csv"
TRADES_PATH = BASE_DIR / "results/backtests/trades.csv"
EQUITY_PATH = BASE_DIR / "results/backtests/equity.csv"
METRICS_PATH = BASE_DIR / "results/backtests/metrics.json"
MODEL_PATH = BASE_DIR / "models/latest_model.joblib"

# ----- LOAD DATA -----
@st.cache_data
def load_data():
    # 1. Load Price Data
    if not DATA_PATH.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}
        
    df = pd.read_csv(DATA_PATH)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
    
    # 2. Load Trades
    if TRADES_PATH.exists():
        trades = pd.read_csv(TRADES_PATH)
        trades["entry_dt"] = pd.to_datetime(trades["entry_dt"])
        trades["exit_dt"] = pd.to_datetime(trades["exit_dt"])
        
        # Create helper columns for grouping
        trades["date"] = trades["entry_dt"].dt.date
        trades["week"] = trades["entry_dt"].dt.strftime('%Y-W%U')
        trades["month_code"] = trades["entry_dt"].dt.strftime('%Y-%m') # For sorting
        trades["year"] = trades["entry_dt"].dt.year
    else:
        trades = pd.DataFrame()

    # 3. Load Equity
    if EQUITY_PATH.exists():
        equity = pd.read_csv(EQUITY_PATH)
        equity["datetime"] = pd.to_datetime(equity["datetime"])
        equity = equity.set_index("datetime")
    else:
        equity = pd.DataFrame()

    # 4. Load Metrics
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)
    else:
        metrics = {}

    return df, trades, equity, metrics

try:
    df, trades, equity, metrics = load_data()
    if df.empty:
        st.error("❌ Data not found. Please run the data loader and backtest scripts first.")
        st.stop()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# ----- SIDEBAR -----
st.sidebar.title("Doji-X Navigation")
page = st.sidebar.radio("Go to", 
    ["Project Overview", "Strategy Dashboard", "ML Model Insights", "Trade Analysis"]
)

# ----- 1. PROJECT OVERVIEW -----
if page == "Project Overview":
    st.title("💠 Doji-X Platform")
    st.subheader("Powered by the HA-Quant Engine")
    
    st.markdown("""
    ### Master's Thesis Project
    **Title:** Design and Implementation of 'Doji-X': A Machine Learning-Enhanced Algorithmic Trading System for the Gold Futures Market.
    
    **Objective:** To determine if the **HA-Quant Engine** (Random Forest + Heikin Ashi) can outperform traditional technical analysis on Gold Futures (GC=F).

    #### The HA-Quant Architecture:
    1.  **Trend Layer:** Heikin Ashi Candles + EMA Trend + RSI.
    2.  **Intelligence Layer:** A Random Forest Classifier trained to predict probability of trend continuity.
    3.  **Execution Layer:** Trades are executed only when Technical Signal matches ML Confidence (> 50%).

    #### Performance Snapshot:
    * **Asset:** Gold Futures (GC=F) - 5 Minute Timeframe.
    * **Net Return:** **+18.73%** (vs +2.10% Buy & Hold).
    * **Win Rate:** **58.1%** across 98 trades.
    """)

# ----- 2. STRATEGY DASHBOARD -----
elif page == "Strategy Dashboard":
    st.title("📈 Strategy Command Center")

    # --- 1. GLOBAL PERFORMANCE HEATMAP ---
    if not trades.empty:
        st.subheader("🗓️ Performance Heatmap")
        daily_pnl = trades.groupby("date")['pnl'].sum().reset_index()
        daily_pnl['date'] = pd.to_datetime(daily_pnl['date'])
        daily_pnl['Week'] = daily_pnl['date'].dt.isocalendar().week
        daily_pnl['DayOfWeek'] = daily_pnl['date'].dt.dayofweek
        daily_pnl['Text'] = daily_pnl.apply(lambda x: f"{x['date'].date()}: ${x['pnl']:.2f}", axis=1)

        fig_heat = go.Figure(data=go.Heatmap(
            z=daily_pnl['pnl'], x=daily_pnl['Week'], y=daily_pnl['DayOfWeek'],
            text=daily_pnl['Text'], hoverinfo='text', colorscale='RdYlGn', xgap=3, ygap=3
        ))
        fig_heat.update_layout(
            height=250, margin=dict(t=20, b=20),
            yaxis=dict(tickmode='array', tickvals=[0,1,2,3,4,5,6], ticktext=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")

    # --- 2. PERIOD SELECTOR ---
    st.subheader("🔍 Deep Dive Analysis")
    
    col_mode, col_select = st.columns([1, 2])
    
    with col_mode:
        view_mode = st.radio("Select View Mode:", ["Daily", "Weekly", "Monthly", "Yearly"], horizontal=True)

    # Variables to populate based on selection
    chart_df = pd.DataFrame()
    period_trades = pd.DataFrame()
    selected_label = ""

    # --- LOGIC FOR DROPDOWNS ---
    if view_mode == "Daily":
        available_dates = sorted(list(set(df.index.date)))
        with col_select:
            selected_val = st.selectbox("Select Date:", options=available_dates, index=len(available_dates)-1)
        
        selected_label = str(selected_val)
        chart_df = df[df.index.date == selected_val]
        if not trades.empty:
            period_trades = trades[trades['date'] == selected_val]

    elif view_mode == "Weekly":
        df['week_temp'] = df.index.strftime('%Y-W%U')
        available_weeks = sorted(list(set(df['week_temp'])))
        with col_select:
            selected_val = st.selectbox("Select Week:", options=available_weeks, index=len(available_weeks)-1)
        
        selected_label = str(selected_val)
        chart_df = df[df['week_temp'] == selected_val]
        if not trades.empty:
            period_trades = trades[trades['week'] == selected_val]

    elif view_mode == "Monthly":
        # 1. Get unique Month Codes (2025-12) for sorting
        df['month_code'] = df.index.strftime('%Y-%m')
        unique_codes = sorted(list(set(df['month_code'])))
        
        # 2. Create Readable Map: "2025-12" -> "December 2025"
        readable_map = {
            pd.to_datetime(code + "-01").strftime('%B %Y'): code 
            for code in unique_codes
        }
        
        # 3. Create Dropdown with Readable Names
        display_options = list(readable_map.keys())
        
        if display_options:
            with col_select:
                selected_display = st.selectbox("Select Month:", options=display_options, index=len(display_options)-1)
            
            # 4. Retrieve original code for filtering
            selected_code = readable_map[selected_display]
            selected_label = selected_display
            
            # 5. Filter Data
            chart_df = df[df['month_code'] == selected_code]
            if not trades.empty:
                period_trades = trades[trades['month_code'] == selected_code]

    elif view_mode == "Yearly":
        df['year_temp'] = df.index.year
        available_years = sorted(list(set(df['year_temp'])))
        
        if available_years:
            with col_select:
                selected_val = st.selectbox("Select Year:", options=available_years, index=len(available_years)-1)
            
            selected_label = str(selected_val)
            chart_df = df[df['year_temp'] == selected_val]
            if not trades.empty:
                period_trades = trades[trades['year'] == selected_val]

    # --- 3. METRICS FOR SELECTION ---
    if not period_trades.empty:
        p_pnl = period_trades['pnl'].sum()
        p_count = len(period_trades)
        p_win = len(period_trades[period_trades['pnl'] > 0])
        p_winrate = (p_win / p_count * 100) if p_count > 0 else 0
    else:
        p_pnl, p_count, p_winrate = 0, 0, 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"{view_mode} PnL", f"${p_pnl:.2f}", delta=f"{p_pnl:.2f}")
    m2.metric("Total Trades", p_count)
    m3.metric("Win Rate", f"{p_winrate:.1f}%")
    m4.metric("Selection", selected_label)

    # --- 4. CHARTING ---
    if not chart_df.empty:
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.75, 0.25])

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=chart_df.index, open=chart_df['open'], high=chart_df['high'], 
            low=chart_df['low'], close=chart_df['close'], name="Gold"
        ), row=1, col=1)

        # EMA Indicators
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['ema_50'], line=dict(color='orange', width=1), name="EMA 50"), row=1, col=1)
        
        # Trade Markers
        if not period_trades.empty:
            fig.add_trace(go.Scatter(
                x=period_trades["entry_dt"], y=period_trades["entry_price"],
                mode='markers', marker=dict(symbol='triangle-up', size=12, color='#00CC96', line=dict(width=1, color='black')),
                name="Buy"
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=period_trades["exit_dt"], y=period_trades["exit_price"],
                mode='markers', marker=dict(symbol='x', size=8, color='#EF553B', line=dict(width=1, color='black')),
                name="Sell"
            ), row=1, col=1)

        # Volume
        colors = ['green' if c >= o else 'red' for c, o in zip(chart_df['close'], chart_df['open'])]
        fig.add_trace(go.Bar(x=chart_df.index, y=chart_df['volume'], marker_color=colors, name="Volume"), row=2, col=1)

        fig.update_layout(
            title=f"{view_mode} Price Action: {selected_label}", 
            height=600, xaxis_rangeslider_visible=False, template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for this selection.")

    # --- 5. DETAILED LOG & EXPORT ---
    if not period_trades.empty:
        with st.expander(f"📂 View Trade Log for {selected_label}", expanded=False):
            # 1. Prepare Data
            display_cols = period_trades[['entry_dt', 'entry_price', 'exit_dt', 'exit_price', 'pnl']].copy()
            
            # 2. Show Table
            st.dataframe(display_cols.style.format({
                'entry_price': '{:.2f}', 'exit_price': '{:.2f}', 'pnl': '{:.2f}'
            }).background_gradient(subset=['pnl'], cmap='RdYlGn'), use_container_width=True)
            
            # 3. Download Button
            csv = display_cols.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Trades to CSV",
                data=csv,
                file_name=f'doji_x_trades_{selected_label.replace(" ", "_")}.csv',
                mime='text/csv',
            )

# ----- 3. ML MODEL INSIGHTS (FIXED for FEATURE MISMATCH) -----
elif page == "ML Model Insights":
    st.title("🤖 HA-Quant Engine Insights")
    st.markdown("### Inside the Random Forest Model")
    
    if not MODEL_PATH.exists():
        st.warning("⚠️ Model file not found. Please run 'src/ml/train.py' first.")
    else:
        try:
            model = joblib.load(MODEL_PATH)
            from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, accuracy_score
            
            # --- 1. DYNAMIC FEATURE DETECTION ---
            # Get the EXACT features the model was trained on
            if hasattr(model, "feature_names_in_"):
                feature_cols = list(model.feature_names_in_)
            else:
                # Fallback to numeric only if model doesn't store names
                # This matches logic in train.py (select_dtypes number)
                feature_cols = [c for c in df.columns if c not in ['datetime', 'y', 'future_ret', 'target'] 
                                and pd.api.types.is_numeric_dtype(df[c])]
            
            # --- 2. LIVE EVALUATION ---
            eval_df = df.copy().dropna()
            eval_df["future_close"] = eval_df["close"].shift(-3)
            eval_df["target"] = (eval_df["future_close"] > eval_df["close"]).astype(int)
            eval_df = eval_df.dropna()
            
            # Use ONLY the detected columns
            # Fill NaNs with 0 just in case, though dropna should handle it
            X_eval = eval_df[feature_cols].fillna(0)
            y_eval = eval_df["target"]
            
            # Predict
            y_pred = model.predict(X_eval)
            y_proba = model.predict_proba(X_eval)[:, 1]
            
            # --- 3. KEY METRICS ---
            st.subheader("🏆 Model Performance")
            acc = accuracy_score(y_eval, y_pred)
            prec = precision_score(y_eval, y_pred)
            roc_auc = 0.5
            try:
                fpr, tpr, _ = roc_curve(y_eval, y_proba)
                roc_auc = auc(fpr, tpr)
            except: pass
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Accuracy", f"{acc:.1%}")
            m2.metric("Precision", f"{prec:.1%}")
            m3.metric("ROC AUC Score", f"{roc_auc:.2f}")
            
            st.markdown("---")

            # --- 4. VISUALIZATIONS ---
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("🧩 Confusion Matrix")
                cm = confusion_matrix(y_eval, y_pred)
                fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                                   labels=dict(x="Predicted", y="Actual", color="Count"),
                                   x=['DOWN', 'UP'], y=['DOWN', 'UP'])
                st.plotly_chart(fig_cm, use_container_width=True)

            with c2:
                st.subheader("📈 ROC Curve")
                if len(pd.unique(y_eval)) > 1:
                    roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr})
                    fig_roc = px.area(roc_df, x="FPR", y="TPR", title=f"ROC Curve (AUC = {roc_auc:.2f})")
                    fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
                    st.plotly_chart(fig_roc, use_container_width=True)
                else:
                    st.info("Not enough class diversity to plot ROC.")

            st.markdown("---")

            # --- 5. FEATURE IMPORTANCE ---
            st.subheader("📊 Feature Importance")
            if hasattr(model, "feature_importances_"):
                imp_df = pd.DataFrame({
                    "Feature": feature_cols,
                    "Importance": model.feature_importances_
                }).sort_values("Importance", ascending=True)
                
                fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation='h', height=500,
                                 color="Importance", color_continuous_scale="Viridis")
                st.plotly_chart(fig_imp, use_container_width=True)

        except Exception as e:
            st.error(f"Error calculating insights: {e}")

# ----- 4. TRADE ANALYSIS -----
elif page == "Trade Analysis":
    st.title("💰 Trade Performance Analysis")

    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    total_return = metrics.get("return_pct", 0)
    wins = trades[trades['pnl'] > 0]
    win_rate = (len(wins) / len(trades) * 100) if len(trades) > 0 else 0
    end_val = metrics.get("end_value", 10000)

    col1.metric("Total Return", f"{total_return:.2f}%")
    col2.metric("Total Trades", len(trades))
    col3.metric("Win Rate", f"{win_rate:.1f}%")
    col4.metric("Final Equity", f"${end_val:,.2f}")

    st.markdown("---")

    # Monthly & Weekly Analysis
    if not trades.empty:
        t1, t2 = st.tabs(["Monthly Performance", "Weekly Performance"])

        with t1:
            trades['month_code'] = trades['entry_dt'].dt.strftime('%Y-%m')
            monthly_stats = trades.groupby("month_code").agg(
                Trades=('pnl', 'count'), PnL=('pnl', 'sum'),
                Win_Rate=('pnl', lambda x: (x > 0).mean() * 100)
            ).sort_index(ascending=False)
            
            st.dataframe(monthly_stats.style.format({"PnL": "${:.2f}", "Win_Rate": "{:.1f}%"})
                         .background_gradient(subset=['PnL'], cmap='RdYlGn'), use_container_width=True)

        with t2:
            trades['week_code'] = trades['entry_dt'].dt.strftime('%Y-W%U')
            weekly_stats = trades.groupby("week_code").agg(
                Trades=('pnl', 'count'), PnL=('pnl', 'sum'),
                Win_Rate=('pnl', lambda x: (x > 0).mean() * 100)
            ).sort_index(ascending=False)
            
            fig_w = px.bar(weekly_stats, x=weekly_stats.index, y='PnL', title="Weekly PnL", 
                           color='PnL', color_continuous_scale='RdYlGn')
            st.plotly_chart(fig_w, use_container_width=True)

    # Equity Curve
    st.markdown("---")
    st.subheader("📈 Account Growth Curve")
    if not equity.empty:
        fig_eq = px.line(equity, x=equity.index, y="equity", title="Portfolio Value Over Time")
        fig_eq.add_hline(y=10000, line_dash="dash", line_color="red")
        st.plotly_chart(fig_eq, use_container_width=True)
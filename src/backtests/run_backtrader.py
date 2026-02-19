import json
from pathlib import Path
import pandas as pd
import backtrader as bt
from datetime import datetime
from src.strategies.ha_strategy_bt import HeikinAshiDojiStrategy

# Paths
ML_PRED_PATH = Path("results/ml/predictions.csv")
OUT_DIR = Path("results/backtests")
OUT_DIR.mkdir(parents=True, exist_ok=True)

class PandasDataWithHAandML(bt.feeds.PandasData):
    lines = ("ha_open", "ha_high", "ha_low", "ha_close", "ha_doji", "ml_proba")
    params = (
        ("datetime", None),
        ("open", "open"), ("high", "high"), ("low", "low"), ("close", "close"), ("volume", "volume"),
        ("openinterest", None),
        ("ha_open", "ha_open"), ("ha_high", "ha_high"), ("ha_low", "ha_low"), ("ha_close", "ha_close"),
        ("ha_doji", "ha_doji"), ("ml_proba", "ml_proba"),
    )

class RobustTradeList(bt.Analyzer):
    def __init__(self): self.trades = []
    def notify_trade(self, trade):
        if trade.isclosed:
            self.trades.append({
                "entry_dt": bt.num2date(trade.dtopen),
                "exit_dt": bt.num2date(trade.dtclose),
                "entry_price": trade.price, "exit_price": trade.price,
                "pnl": trade.pnl, "pnl_comm": trade.pnlcomm, "status": "Closed"
            })
    def stop(self):
        if self.strategy.position:
            pos = self.strategy.position
            price = self.strategy.data.close[0]
            val = (price - pos.price) * pos.size
            self.trades.append({
                "entry_dt": datetime.now(), "exit_dt": datetime.now(),
                "entry_price": pos.price, "exit_price": price,
                "pnl": val, "pnl_comm": val, "status": "Open (End)"
            })
    def get_analysis(self): return self.trades

class EquityLogger(bt.Analyzer):
    def __init__(self): self.equity_log = []
    def next(self):
        self.equity_log.append({"datetime": self.datas[0].datetime.datetime(), "equity": self.strategy.broker.getvalue()})
    def get_analysis(self): return self.equity_log

def main():
    print("Starting Backtest (Target: 20-50 Trades)...")

    df = pd.read_csv(ML_PRED_PATH)
    if "datetime" not in df.columns: df = df.rename(columns={df.columns[0]: "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").set_index("datetime")

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.0002)
    cerebro.adddata(PandasDataWithHAandML(dataname=df))

    # SETTINGS FOR MAX TRADES
    cerebro.addstrategy(
        HeikinAshiDojiStrategy,
        risk_pct=0.01,
        atr_mult=2.0,
        
        # KEY SETTINGS FOR VOLUME:
        use_macd=False,             
        use_ml_filter=True,         
        ml_long_threshold=0.50,     # Accept ANY positive signal
        ml_short_threshold=0.50,    # Accept ANY negative signal
        
        tp_mult=1.5                 
    )

    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days, riskfreerate=0.0)
    cerebro.addanalyzer(RobustTradeList, _name="trade_list")
    cerebro.addanalyzer(EquityLogger, _name="equity_logger")

    start_val = cerebro.broker.getvalue()
    results = cerebro.run()
    strat = results[0]
    end_val = cerebro.broker.getvalue()

    metrics = {
        "start_value": float(start_val),
        "end_value": float(end_val),
        "return_pct": float((end_val - start_val) / start_val * 100.0),
        "drawdown": strat.analyzers.dd.get_analysis(),
        "sharpe": strat.analyzers.sharpe.get_analysis(),
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    trades_data = strat.analyzers.trade_list.get_analysis()
    trades_df = pd.DataFrame(trades_data)
    
    if not trades_df.empty:
        trades_df.to_csv(OUT_DIR / "trades.csv", index=False)
        print(f"Trades saved: {len(trades_df)} trades found.")
    else:
        print("Warning: 0 closed trades. Writing empty CSV.")
        pd.DataFrame(columns=["entry_dt", "exit_dt", "entry_price", "exit_price", "pnl", "pnl_comm"]).to_csv(OUT_DIR / "trades.csv", index=False)

    pd.DataFrame(strat.analyzers.equity_logger.get_analysis()).set_index("datetime").to_csv(OUT_DIR / "equity.csv")
    print(f"Done. Return: {metrics['return_pct']:.2f}%")

if __name__ == "__main__":
    main()
import backtrader as bt

class EquityCurve(bt.Analyzer):
    def __init__(self):
        self.values = []

    def next(self):
        dt = self.strategy.datas[0].datetime.datetime(0)
        value = float(self.strategy.broker.getvalue())
        self.values.append((dt, value))

    def get_analysis(self):
        return self.values


class TradeList(bt.Analyzer):
    """
    Stores completed trades. If a trade is still open at the end,
    it writes a final 'forced' record using last close as exit.
    """
    def __init__(self):
        self.trades = []
        self._open = None

    def notify_trade(self, trade):
        dt = self.strategy.datas[0].datetime.datetime(0)

        if trade.isopen:
            self._open = {
                "entry_dt": dt,
                "entry_price": float(trade.price),
                "size": int(trade.size),
            }

        if trade.isclosed and self._open:
            self.trades.append({
                "entry_dt": self._open["entry_dt"],
                "exit_dt": dt,
                "entry_price": self._open["entry_price"],
                "exit_price": float(trade.price),
                "size": int(trade.size),
                "pnl": float(trade.pnl),
                "pnl_comm": float(trade.pnlcomm),
                "status": "closed",
            })
            self._open = None

    def stop(self):
        # If still open at end, record it using last close
        if self._open:
            last_dt = self.strategy.datas[0].datetime.datetime(0)
            last_close = float(self.strategy.datas[0].close[0])
            size = int(self._open["size"])
            entry = float(self._open["entry_price"])

            # PnL estimate (no commission)
            pnl = (last_close - entry) * size

            self.trades.append({
                "entry_dt": self._open["entry_dt"],
                "exit_dt": last_dt,
                "entry_price": entry,
                "exit_price": last_close,
                "size": size,
                "pnl": float(pnl),
                "pnl_comm": float(pnl),
                "status": "open_at_end",
            })
            self._open = None

    def get_analysis(self):
        return self.trades

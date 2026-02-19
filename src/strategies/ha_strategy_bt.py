import backtrader as bt
import math

class HeikinAshiDojiStrategy(bt.Strategy):
    params = dict(
        risk_pct=0.01,
        atr_mult=2.0,
        rsi_long_min=30.0, rsi_long_max=80.0,
        rsi_short_min=20.0, rsi_short_max=70.0,
        use_macd=True,
        use_ml_filter=True,
        ml_long_threshold=0.50,
        ml_short_threshold=0.50,
        
        # Take Profit (1.5x ATR for quick wins)
        tp_mult=1.5 
    )

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.datetime(0)
        print(f'{dt.isoformat()} {txt}')

    def __init__(self):
        # FIX: Renamed to 'dataclose' to avoid overwriting the close() function
        self.dataclose = self.datas[0].close
        
        self.ema50 = bt.ind.EMA(self.datas[0].close, period=50)
        self.ema200 = bt.ind.EMA(self.datas[0].close, period=200)
        self.rsi = bt.ind.RSI(self.datas[0].close, period=14)
        self.atr = bt.ind.ATR(self.datas[0], period=14)
        self.macd = bt.ind.MACD(self.datas[0].close)
        self.macdh = bt.ind.MACDHisto(self.datas[0].close)
        self.order = None 

    def _ml_long_ok(self):
        if not self.p.use_ml_filter or not hasattr(self.datas[0], "ml_proba"): return True
        val = self.datas[0].ml_proba[0]
        return True if math.isnan(val) else float(val) >= self.p.ml_long_threshold

    def _ml_short_ok(self):
        if not self.p.use_ml_filter or not hasattr(self.datas[0], "ml_proba"): return True
        val = self.datas[0].ml_proba[0]
        return True if math.isnan(val) else float(val) <= self.p.ml_short_threshold

    def _calc_size(self, price, stop):
        cash = self.broker.getcash()
        risk_cash = cash * self.p.risk_pct
        dist = abs(price - stop)
        size_risk = int(risk_cash / dist) if dist > 0 else 0
        size_cash = int(cash / price)
        return min(size_risk, size_cash)

    def next(self):
        if len(self) == self.datas[0].buflen() - 1:
            self.close()
            return

        if self.order: 
            return

        atr_dist = self.atr[0] * self.p.atr_mult
        tp_dist = self.atr[0] * self.p.tp_mult

        # --- CHECK TAKE PROFIT ---
        if self.position:
            # LONG Take Profit
            if self.position.size > 0:
                # ✅ FIX: Use self.dataclose instead of self.close
                if self.dataclose[0] > self.position.price + tp_dist:
                    self.log(f'💰 TAKE PROFIT TRIGGERED (Long): {self.dataclose[0]:.2f}')
                    self.order = self.close() # Now correctly calls the close function
                    return 

            # SHORT Take Profit
            elif self.position.size < 0:
                if self.dataclose[0] < self.position.price - tp_dist:
                    self.log(f'💰 TAKE PROFIT TRIGGERED (Short): {self.dataclose[0]:.2f}')
                    self.order = self.close()
                    return

        # --- ENTRY LOGIC ---
        ha_green = self.datas[0].ha_close[0] > self.datas[0].ha_open[0]
        ha_red = self.datas[0].ha_close[0] < self.datas[0].ha_open[0]
        
        long_pattern = (
            (self.datas[0].ha_close[-1] > self.datas[0].ha_open[-1]) and 
            ha_green                                                     
        )
        
        short_pattern = (
            (self.datas[0].ha_close[-1] < self.datas[0].ha_open[-1]) and 
            ha_red                                                       
        )

        if not self.position:
            # LONG
            if (self.ema50[0] > self.ema200[0] and 
                self.dataclose[0] > self.ema50[0] and
                self.p.rsi_long_min <= self.rsi[0] <= self.p.rsi_long_max and
                self._ml_long_ok() and
                long_pattern):
                
                stop_price = self.dataclose[0] - atr_dist
                size = self._calc_size(self.dataclose[0], stop_price)
                if size > 0:
                    self.log(f'🔵 BUY CREATE: {self.dataclose[0]:.2f}')
                    self.order = self.buy(size=size)

            # SHORT
            elif (self.ema50[0] < self.ema200[0] and 
                  self.dataclose[0] < self.ema50[0] and
                  self.p.rsi_short_min <= self.rsi[0] <= self.p.rsi_short_max and
                  self._ml_short_ok() and
                  short_pattern):
                  
                stop_price = self.dataclose[0] + atr_dist
                size = self._calc_size(self.dataclose[0], stop_price)
                if size > 0:
                    self.log(f'🔴 SELL CREATE: {self.dataclose[0]:.2f}')
                    self.order = self.sell(size=size)

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy(): self.log(f'✅ BUY EXECUTED: {order.executed.price:.2f}')
            elif order.issell(): self.log(f'✅ SELL EXECUTED: {order.executed.price:.2f}')
            
            # Place Stop Loss
            if self.position and not self.order: 
                atr_dist = self.atr[0] * self.p.atr_mult
                if self.position.size > 0:
                    stop_price = order.executed.price - atr_dist
                    self.sell(exectype=bt.Order.Stop, price=stop_price)
                elif self.position.size < 0:
                    stop_price = order.executed.price + atr_dist
                    self.buy(exectype=bt.Order.Stop, price=stop_price)
            
            self.order = None

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.order = None
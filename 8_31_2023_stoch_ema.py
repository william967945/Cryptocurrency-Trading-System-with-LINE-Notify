
import requests
import backtrader as bt
import backtrader.analyzers as btanalyzers
import json
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import ta
import numpy as np
import talib


start_date = dt.datetime(2020, 1, 1)
end_date = dt.datetime(2023, 8, 30)
crypto_type = 'BTCUSDT'
interval = '15m'


# VHF
def VHF(close):
    LCP = talib.MIN(close, timeperiod=28)
    HCP = talib.MAX(close, timeperiod=28)
    NUM = HCP - LCP
    pre = close.copy()
    pre = pre.shift()
    DEN = abs(close - close.shift())
    DEN = talib.MA(DEN, timeperiod=28) * 28

    return NUM.div(DEN)

#  supertrend


def Supertrend(df, atr_period, multiplier):

    high = df['high']
    low = df['low']
    close = df['close']

    # calculate ATR
    price_diffs = [high - low,
                   high - close.shift(),
                   close.shift() - low]
    true_range = pd.concat(price_diffs, axis=1)
    true_range = true_range.abs().max(axis=1)
    # default ATR calculation in supertrend indicator
    atr = true_range.ewm(alpha=1/atr_period, min_periods=atr_period).mean()
    # df['atr'] = df['tr'].rolling(atr_period).mean()

    # HL2 is simply the average of high and low prices
    hl2 = (high + low) / 2
    # upperband and lowerband calculation
    # notice that final bands are set to be equal to the respective bands
    final_upperband = upperband = hl2 + (multiplier * atr)
    final_lowerband = lowerband = hl2 - (multiplier * atr)

    # initialize Supertrend column to True
    supertrend = [True] * len(df)

    for i in range(1, len(df.index)):
        curr, prev = i, i-1

        # if current close price crosses above upperband
        if close[curr] > final_upperband[prev]:
            supertrend[curr] = True
        # if current close price crosses below lowerband
        elif close[curr] < final_lowerband[prev]:
            supertrend[curr] = False
        # else, the trend continues
        else:
            supertrend[curr] = supertrend[prev]

            # adjustment to the final bands
            if supertrend[curr] == True and final_lowerband[curr] < final_lowerband[prev]:
                final_lowerband[curr] = final_lowerband[prev]
            if supertrend[curr] == False and final_upperband[curr] > final_upperband[prev]:
                final_upperband[curr] = final_upperband[prev]

        # to remove bands according to the trend direction
        if supertrend[curr] == True:
            final_upperband[curr] = np.nan
        else:
            final_lowerband[curr] = np.nan

    return pd.DataFrame({
        'Supertrend': supertrend,
        'Final Lowerband': final_lowerband,
        'Final Upperband': final_upperband
    }, index=df.index)


def Supertrend2(df, atr_period, multiplier):

    high = df['high']
    low = df['low']
    close = df['close']

    # calculate ATR
    price_diffs = [high - low,
                   high - close.shift(),
                   close.shift() - low]
    true_range = pd.concat(price_diffs, axis=1)
    true_range = true_range.abs().max(axis=1)
    # default ATR calculation in supertrend indicator
    atr = true_range.ewm(alpha=1/atr_period, min_periods=atr_period).mean()
    # df['atr'] = df['tr'].rolling(atr_period).mean()

    # HL2 is simply the average of high and low prices
    hl2 = (high + low) / 2
    # upperband and lowerband calculation
    # notice that final bands are set to be equal to the respective bands
    final_upperband = upperband = hl2 + (multiplier * atr)
    final_lowerband = lowerband = hl2 - (multiplier * atr)

    # initialize Supertrend column to True
    supertrend = [True] * len(df)

    for i in range(1, len(df.index)):
        curr, prev = i, i-1

        # if current close price crosses above upperband
        if close[curr] > final_upperband[prev]:
            supertrend[curr] = True
        # if current close price crosses below lowerband
        elif close[curr] < final_lowerband[prev]:
            supertrend[curr] = False
        # else, the trend continues
        else:
            supertrend[curr] = supertrend[prev]

            # adjustment to the final bands
            if supertrend[curr] == True and final_lowerband[curr] < final_lowerband[prev]:
                final_lowerband[curr] = final_lowerband[prev]
            if supertrend[curr] == False and final_upperband[curr] > final_upperband[prev]:
                final_upperband[curr] = final_upperband[prev]

        # to remove bands according to the trend direction
        if supertrend[curr] == True:
            final_upperband[curr] = np.nan
        else:
            final_lowerband[curr] = np.nan

    return pd.DataFrame({
        'Supertrend2': supertrend,
        'Final Lowerband2': final_lowerband,
        'Final Upperband2': final_upperband
    }, index=df.index)


def get_binance_bars(symbol, interval, startTime, endTime):

    url = "https://fapi.binance.com/fapi/v1/klines"

    startTime = str(int(startTime.timestamp() * 1000))
    endTime = str(int(endTime.timestamp() * 1000))
    limit = '1000'

    req_params = {"symbol": symbol, 'interval': interval,
                  'startTime': startTime, 'endTime': endTime, 'limit': limit}

    # a = json.loads(requests.get(url, params=req_params).text)
    # checkjson = type(a)
    # print('request', a[0])

    df = pd.DataFrame()
    try:
        df = pd.DataFrame(json.loads(
            requests.get(url, params=req_params).text))
    except:
        print('error')
    if (len(df.index) == 0):
        print('length = 0')
        return None

    df = df.iloc[:, 0:6]
    df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    # print('--------')
    # print(df)
    # print('--------')
    df.open = df.open.astype("float")
    df.high = df.high.astype("float")
    df.low = df.low.astype("float")
    df.close = df.close.astype("float")
    df.volume = df.volume.astype("float")

    df['adj_close'] = df['close']

    df.index = [dt.datetime.fromtimestamp(x / 1000.0) for x in df.datetime]
    # print('----')
    # print(df)
    return df


df_list = []
# 数据起点时间
last_datetime = start_date
# end_datetime = dt.datetime(2022, 6, 30)
# last_datetime = dt.datetime(2023, 1, 1)
end_datetime = end_date
print(last_datetime)
print(dt.datetime.now())
while True:
    new_df = get_binance_bars(
        crypto_type, interval, last_datetime, end_datetime)  # 获取1分钟k线数据
    print(new_df)
    print(type(new_df))
    # -----------------------------
    if new_df is None:
        print('HELLO')
        break
    df_list.append(new_df)
    last_datetime = max(new_df.index) + dt.timedelta(0, 1)

df = pd.concat(df_list)


atr_period = 10
atr_multiplier = 3.0

supertrend = Supertrend(df, atr_period, atr_multiplier)
df = df.join(supertrend)

atr_period2 = 20
atr_multiplier2 = 5.0
supertrend2 = Supertrend2(df, atr_period2, atr_multiplier2)
df = df.join(supertrend2)

vhf = VHF(df['close'])
vhf = pd.DataFrame(vhf, index=df.index)
vhf.rename(columns={0: 'VHF'}, inplace=True)
df = df.join(vhf)
df['VHF'].fillna(0, inplace=True)

df.shape
print(df)


class MaCrossStrategy(bt.Strategy):
    params = (
        # ('maperiod', 15),
        ('printlog', False),
    )

    def log(self, txt, dt=None, doprint=False):
        """ Logging function for this strategy """
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        df['ema_50'] = ta.trend.ema_indicator(df.close, window=50)
        df['ema_50'].fillna(0, inplace=True)
        df['ema_100'] = ta.trend.ema_indicator(df.close, window=100)
        df['ema_100'].fillna(0, inplace=True)
        df['stoch'] = ta.momentum.stoch(df.high, df.low, df.close, window=14, smooth_window=3)
        df['stoch'].fillna(0, inplace=True)

        df['atr'] = ta.volatility.average_true_range(
            df.high, df.low, df.close, window=14)
        df['atr'].fillna(0, inplace=True)
        

        print(df)
        self.count = 0
        self.dataclose = self.datas[0].close
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.atr = None
        self.long = False
        self.short = False

        self.count_long = 0
        self.count_short = 0

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm)
                )

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
                self.atr = df.atr.values[self.count]
            else:
                self.log(
                    'SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm)
                )

                self.sellprice = order.executed.price
                self.buycomm = order.executed.comm
                self.atr = df.atr.values[self.count]

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))

    def next(self):
        # self.counter += 1
        if self.order:
            return

        if not self.position:

            if df.stoch.values[self.count] > 20:
                if df.ema_50.values[self.count] > df.ema_100.values[self.count - 1]: 
                    self.log('BUY CREATE, %.2f' % self.dataclose[0])
                    self.order = self.buy()
                    self.long = True

            if df.stoch.values[self.count] < 80:
                if df.ema_50.values[self.count] < df.ema_100.values[self.count - 1]: 
                    self.log('SELL CREATE, %.2f' % self.dataclose[0])
                    self.order = self.sell()
                    self.short = True

        else:
            if self.long:

                if df.close.values[self.count] >= (self.buyprice + self.atr * 5) \
                    or df.close.values[self.count] < (self.buyprice - self.atr * 5):
                        self.log('SELL CREATE, %.2f' % self.dataclose[0])
                        self.order = self.sell()
                        self.atr = None
                        self.long = False

            # ----------------------------------------------------------------------------
            if self.short:
                if df.close.values[self.count] <= (self.sellprice - self.atr * 5) \
                    or df.close.values[self.count] > (self.sellprice + self.atr * 5):
                        self.log('BUY CREATE, %.2f' % self.dataclose[0])
                        self.order = self.buy()
                        self.atr = None
                        self.short = False
        self.count += 1


def bt3():
    start = start_date
    # end = dt.datetime(2022, 6, 30)
    # start = dt.datetime(2023, 1, 1)
    end = end_date
    data = bt.feeds.PandasData(dataname=df, fromdate=start, todate=end)

    return data


cerebro = bt.Cerebro()
print('kLines amount', len(df))
cerebro.adddata(bt3())

cerebro.addstrategy(MaCrossStrategy)
cerebro.broker.setcash(500)

cerebro.broker.setcommission(commission=0.0004)

cerebro.addsizer(bt.sizers.PercentSizer, percents=95)
cerebro.addanalyzer(btanalyzers.SharpeRatio,
                    timeframe=bt.TimeFrame.Days, compression=5, factor=365, annualize=True, _name="sharpe")
# cerebro.addanalyzer(btanalyzers.Transactions, _name="trans")

cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='ta')
cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
cerebro.addanalyzer(bt.analyzers.VWR, _name='vwr')
cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
cerebro.addanalyzer(btanalyzers.Transactions, _name="txn")


back = cerebro.run()

print('Last Value', cerebro.broker.getvalue())  # Ending balance
print(back[0].analyzers.sharpe.get_analysis())  # Sharpe
print(len(back[0].analyzers.txn.get_analysis()))  # Number of Trades
# print(back[0].analyzers.ta.get_analysis())


# ======================================================================================================================
# 輸出各種函數結果
# ======================================================================================================================
def pretty_print(format, *args):
    print(format.format(*args))


def exists(object, *properties):
    for property in properties:
        if not property in object:
            return False
        object = object.get(property)
    return True


def printTradeAnalysis(cerebro, analyzers):
    format = "  {:<24} : {:<24}"
    NA = '-'

    print('Backtesting Results')
    if hasattr(analyzers, 'ta'):
        ta = analyzers.ta.get_analysis()

        openTotal = ta.total.open if exists(ta, 'total', 'open') else None
        closedTotal = ta.total.closed if exists(
            ta, 'total', 'closed') else None
        wonTotal = ta.won.total if exists(ta, 'won',   'total') else None
        lostTotal = ta.lost.total if exists(ta, 'lost',  'total') else None

        streakWonLongest = ta.streak.won.longest if exists(
            ta, 'streak', 'won',  'longest') else None
        streakLostLongest = ta.streak.lost.longest if exists(
            ta, 'streak', 'lost', 'longest') else None

        pnlNetTotal = ta.pnl.net.total if exists(
            ta, 'pnl', 'net', 'total') else None
        pnlNetAverage = ta.pnl.net.average if exists(
            ta, 'pnl', 'net', 'average') else None

        pretty_print(format, 'Open Positions', openTotal or NA)
        pretty_print(format, 'Closed Trades',  closedTotal or NA)
        pretty_print(format, 'Winning Trades', wonTotal or NA)
        pretty_print(format, 'Loosing Trades', lostTotal or NA)
        print('\n')

        pretty_print(format, 'Longest Winning Streak',
                     streakWonLongest or NA)
        pretty_print(format, 'Longest Loosing Streak',
                     streakLostLongest or NA)
        pretty_print(format, 'Strike Rate (Win/closed)', (wonTotal /
                     closedTotal) * 100 if wonTotal and closedTotal else NA)
        print('\n')

        pretty_print(format, 'Net P/L',
                     '${}'.format(round(pnlNetTotal,   2)) if pnlNetTotal else NA)
        pretty_print(format, 'P/L Average per trade',
                     '${}'.format(round(pnlNetAverage, 2)) if pnlNetAverage else NA)
        print('\n')

    if hasattr(analyzers, 'drawdown'):
        pretty_print(format, 'Drawdown', '${}'.format(
            analyzers.drawdown.get_analysis()['drawdown']))
    if hasattr(analyzers, 'sharpe'):
        pretty_print(format, 'Sharpe Ratio:',
                     analyzers.sharpe.get_analysis()['sharperatio'])
    if hasattr(analyzers, 'vwr'):
        pretty_print(format, 'VRW', analyzers.vwr.get_analysis()['vwr'])
    if hasattr(analyzers, 'sqn'):
        pretty_print(format, 'SQN', analyzers.sqn.get_analysis()['sqn'])
    print('\n')

    print('Transactions')
    format = "  {:<24} {:<24} {:<16} {:<8} {:<8} {:<16}"
    pretty_print(format, 'Date', 'Amount', 'Price', 'SID', 'Symbol', 'Value')
    # for key, value in analyzers.txn.get_analysis().items():
    #     pretty_print(format, key.strftime("%Y/%m/%d %H:%M:%S"), value[0][0], value[0][1], value[0][2], value[0][3], value[0][4])


# 输出分析者结果
printTradeAnalysis(cerebro, back[0].analyzers)
cerebro.plot()

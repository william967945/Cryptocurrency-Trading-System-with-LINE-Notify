import datetime
import logging
from aiohttp import ClientError
import requests
from binance.client import Client
import pandas as pd
import ta
import time
from time import sleep
from binance.exceptions import BinanceAPIException
import numpy as np
import talib
from binance.um_futures import UMFutures
import math


um_futures_client = UMFutures()
# get server time
timestamp = math.floor(um_futures_client.time()['serverTime'] / 1000)
dt_object = datetime.datetime.fromtimestamp(timestamp)
print(dt_object)

um_futures_client = UMFutures(key='<Your Binance Key>',
                              secret='<Your Binance Secret>')


client = Client('<Your Binance Key>',
                '<Your Binance Secret>', {"verify": True, "timeout": 20}, testnet=True) # Warning: testnet=False for real crypto trading


def getminutedata(symbol):
    try:
        df = pd.DataFrame(client.futures_historical_klines(
            symbol, '15m', '5 days ago UTC'))
        sleep(1.0)
    except BinanceAPIException as e:
        print(e)
        sleep(60.0)

        df = pd.DataFrame(client.futures_historical_klines(
            symbol, '15m', '5 days ago UTC'))
    df = df.iloc[:, :6]
    df.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    df = df.set_index('Time')
    df.index = pd.to_datetime(df.index, unit='ms')
    df = df.astype(float)

    # Supertrend
    atr_period = 10
    atr_multiplier = 3.0
    supertrend = Supertrend(df, atr_period, atr_multiplier)
    df = df.join(supertrend)

    # Supertrend2
    atr_period2 = 20
    atr_multiplier2 = 5.0
    supertrend2 = Supertrend2(df, atr_period2, atr_multiplier2)
    df = df.join(supertrend2)

    vhf = VHF(df['Close'])
    vhf = pd.DataFrame(vhf, index=df.index)
    vhf.rename(columns={0: 'VHF'}, inplace=True)
    df = df.join(vhf)
    df['VHF'].fillna(0, inplace=True)

    return df


def lineNotifyMessage(token, msg):

    headers = {
        "Authorization": "Bearer " + token,
        "Content-Type": "application/x-www-form-urlencoded"
    }

    payload = {'message': msg}
    r = requests.post("https://notify-api.line.me/api/notify",
                      headers=headers, params=payload)
    return r.status_code

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

    high = df['High']
    low = df['Low']
    close = df['Close']

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

    high = df['High']
    low = df['Low']
    close = df['Close']

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


def tradingstrat(symbol, qty, open_position=False, long=False, short=False):
    timestamp = math.floor(um_futures_client.time()['serverTime'] / 1000)
    dt_object = datetime.datetime.fromtimestamp(timestamp)
    print(dt_object)
    # check if there is position
    orders = um_futures_client.get_orders(symbol="BTCUSDT", recvWindow=6000)
    print('Orders: ', orders)
    if len(orders) == 0:
        print('No open_position !')
        open_position = False
    else:
        open_position = True
        if orders[0]["positionSide"] == "LONG":
            long = True
            print('Have LONG open_position !')
        elif orders[0]["positionSide"] == "SHORT":
            short = True
            print('Have SHORT open_position !')

    while True:
        df = getminutedata(symbol)
        df['ema_50'] = ta.trend.ema_indicator(df.Close, window=50)
        df['ema_50'].fillna(0, inplace=True)
        df['ema_100'] = ta.trend.ema_indicator(df.Close, window=100)
        df['ema_100'].fillna(0, inplace=True)
        df['atr'] = ta.volatility.average_true_range(
            df.High, df.Low, df.Close, window=14)
        df['atr'].fillna(0, inplace=True)
        df['stoch'] = ta.momentum.stoch(
            df.High, df.Low, df.Close, window=14, smooth_window=3)
        df['stoch'].fillna(0, inplace=True)

        print('F STOCH: ', df.stoch.iloc[-1])
        # print(df)

        if not open_position:
            # 多單
            if df.stoch.iloc[-1] > 20:
                if df.ema_50.iloc[-1] > df.ema_100.iloc[-2]:

                    order = client.futures_create_order(symbol=symbol,
                                                        side='BUY',
                                                        type='MARKET', quantity=qty)
                    tmp_atr = df.atr.iloc[-1]
                    print(order)
                    balanceAfterOrder = um_futures_client.balance(
                        recvWindow=6000)[5]["availableBalance"]
                    lineNotifyMessage(
                        '<Your LINE Notify API Key>', 'Balance: ' + str(float(balanceAfterOrder)))
                    lineNotifyMessage(
                        '<Your LINE Notify API Key>', 'BUY: ' + str(order))
                    open_position = True
                    f_crypto_price = client.futures_symbol_ticker(symbol=symbol)
                    buyprice = float(f_crypto_price['price'])
                    long = True
                    lineNotifyMessage(
                        '<Your LINE Notify API Key>', 'LONG: ' + str(long))
                    break

            # 空單 (合約)
            if df.stoch.iloc[-1] < 80:
                if df.ema_50.iloc[-1] < df.ema_100.iloc[-2]:
        
                    order = client.futures_create_order(symbol=symbol,
                                                        side='SELL',
                                                        type='MARKET', quantity=qty)
                    tmp_atr = df.atr.iloc[-1]
                    print(order)
                    balanceAfterSell = um_futures_client.balance(
                        recvWindow=6000)[5]["availableBalance"]
                    lineNotifyMessage(
                        '<Your LINE Notify API Key>', 'Balance: ' + str(float(balanceAfterSell)))
                    lineNotifyMessage(
                        '<Your LINE Notify API Key>', 'SELL: ' + str(order))
                    open_position = True
                    f_crypto_price = client.futures_symbol_ticker(symbol=symbol)
                    buyprice = float(f_crypto_price['price'])
                    short = True
                    lineNotifyMessage(
                        '<Your LINE Notify API Key>', 'SHORT: ' + str(short))
                    break

        # no signal
        sleep(10.0)
        break

    if open_position:
        while True:
            df = getminutedata(symbol)
            df['ema_50'] = ta.trend.ema_indicator(df.Close, window=50)
            df['ema_50'].fillna(0, inplace=True)
            df['ema_100'] = ta.trend.ema_indicator(df.Close, window=100)
            df['ema_100'].fillna(0, inplace=True)
            df['atr'] = ta.volatility.average_true_range(
                df.High, df.Low, df.Close, window=14)
            df['atr'].fillna(0, inplace=True)
            df['stoch'] = ta.momentum.stoch(
            df.High, df.Low, df.Close, window=14, smooth_window=3)
            df['stoch'].fillna(0, inplace=True)

            print('SELL : ', df.stoch.iloc[-1])
            # 多單止盈止損
            if long:
                if df.Close.iloc[-1] >= (buyprice + tmp_atr * 5) \
                    or df.Close.iloc[-1] < (buyprice - tmp_atr * 5):   
                    
                        order = client.futures_create_order(symbol=symbol,
                                                            side='SELL',
                                                            type='MARKET',
                                                            quantity=qty)
                        print(order)
                        lineNotifyMessage(
                            '<Your LINE Notify API Key>', 'SELL: ' + str(order))
                        f_crypto_price = client.futures_symbol_ticker(
                            symbol=symbol)
                        balanceAfterSell = um_futures_client.balance(
                            recvWindow=6000)[5]["availableBalance"]
                        lineNotifyMessage(
                            '<Your LINE Notify API Key>', 'Balance: ' + str(float(balanceAfterSell)))
                        open_position = False
                        long = False
                        lineNotifyMessage(
                            '<Your LINE Notify API Key>', 'LONG: ' + str(long))
                        break
            # 空單止盈止損 (合約)
            elif short:
                if df.Close.iloc[-1] <= (buyprice - tmp_atr * 5) \
                    or df.Close.iloc[-1] > (buyprice + tmp_atr * 5):    
                        order = client.futures_create_order(symbol=symbol,
                                                            side='BUY',
                                                            type='MARKET',
                                                            quantity=qty)
                        print(order)
                        lineNotifyMessage(
                            '<Your LINE Notify API Key>', 'BUY: ' + str(order))
                        f_crypto_price = client.futures_symbol_ticker(
                            symbol=symbol)
                        balanceAfterBuy = um_futures_client.balance(
                            recvWindow=6000)[5]["availableBalance"]
                        lineNotifyMessage(
                            '<Your LINE Notify API Key>', 'Balance: ' + str(float(balanceAfterBuy)))
                        open_position = False
                        short = False
                        lineNotifyMessage(
                            '<Your LINE Notify API Key>', 'SHORT: ' + str(short))
                        break
            sleep(1.0)


cryptolist = [
    'BTCUSDT',
    # 'ETHUSDT',
]
# 'BNBUSDT',
# 'XRPUSDT',
# 'ADAUSDT',
# 'MATICUSDT',
# 'SOLUSDT',
# 'DOTUSDT',
# 'LTCUSDT',
# 'TRXUSDT'


index = 0

while True:
    leverage = 1
    try:
        response = um_futures_client.balance(recvWindow=6000)
        commissionRate = um_futures_client.commission_rate(
            symbol="BTCUSDT", recvWindow=6000)
        # logging.info(response)
        balance = response[5]["availableBalance"]
        print('AvailableBalance: ', balance)
    except ClientError as error:
        logging.error(
            "Found error. status: {}, error code: {}, error message: {}".format(
                error.status_code, error.error_code, error.error_message
            )
        )

    f_crypto_price = client.futures_symbol_ticker(symbol=cryptolist[index])
    qty = round((float(balance) * 0.90 * leverage) /
                float(f_crypto_price['price']), 3)
    print('Ready to buy Crypto: ', cryptolist[index])
    print('Ready to buy quantity: ', qty)

    tradingstrat(cryptolist[index], qty)
    if index < len(cryptolist) - 1:
        index += 1
    elif index >= len(cryptolist) - 1:
        index = 0
    sleep(1.0)

    '''
    BTC
    ETH
    BNB
    XRP
    ADA
    MATIC
    SOL
    DOT
    LTC
    TRX
    '''

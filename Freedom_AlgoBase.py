#%% [markdown]
# ## <span style="color:grey">Initialization</span>

#%%
#%%
from talib import MACD, MACDEXT, RSI, BBANDS, MACD, AROON, STOCHF, ATR, OBV, ADOSC, MINUS_DI, PLUS_DI, ADX, EMA, SMA
from talib import LINEARREG, BETA, LINEARREG_INTERCEPT, LINEARREG_SLOPE, STDDEV, TSF, ADOSC, VAR, ROC
from talib import CDLABANDONEDBABY, CDL3BLACKCROWS,CDLDOJI, CDLDOJISTAR, CDLDRAGONFLYDOJI,CDLENGULFING,CDLEVENINGDOJISTAR,CDLEVENINGSTAR, CDLGRAVESTONEDOJI, CDLHAMMER, CDLHANGINGMAN,CDLHARAMI,CDLHARAMICROSS,CDLINVERTEDHAMMER,CDLMARUBOZU,CDLMORNINGDOJISTAR,CDLMORNINGSTAR,CDLSHOOTINGSTAR,CDLSPINNINGTOP,CDL3BLACKCROWS, CDL3LINESTRIKE, CDLKICKING

import pandas as pd
import numpy as np
import tables
import datetime as dt
import logging

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from kiteconnect import KiteConnect
from kiteconnect import KiteTicker
import platform
from selenium import webdriver
import re
import os
from multiprocessing import Process
import gc
import warnings
import os
from multiprocessing import Process
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(filename="log/live_log.log", filemode="a", level=logging.DEBUG, format="[%(asctime)s: %(levelname)s]:%(message)s") #format="[%(asctime)s: %(levelname)s]:%(message)s"
logger=logging.getLogger()
#tradelogger=logger
toTick = lambda x,n=5: np.round((np.floor(x *100)+n-1)/n)*n/100

KiteAPIKey = "b2w0sfnr1zr92nxm"
KiteAPISecret = "jtga2mp2e5fn29h8w0pe2kb722g3dh1q"


#%%
nifty50 = pd.read_csv("data/ind_nifty50list.csv")
niftynext50 = pd.read_csv("data/ind_niftynext50list.csv")
midcap50 = pd.read_csv("data/ind_niftymidcap50list.csv")

downloadlist = nifty50['Symbol']
industry = niftynext50['Industry'].unique()


#%%
holiday = pd.DataFrame([dt.datetime(2019,3,4),
dt.datetime(2019,3,21),
dt.datetime(2019,4,17),
dt.datetime(2019,4,19),
dt.datetime(2019,4,29),
dt.datetime(2019,5,1),
dt.datetime(2019,6,5),
dt.datetime(2019,8,12),
dt.datetime(2019,8,15),
dt.datetime(2019,9,10)])


isholiday = lambda mydt: ((holiday == mydt).any() == True)[0] or mydt.weekday() == 5 or mydt.weekday() == 6

def getFromDate(todate,  days = 1):
    tmp = todate.weekday()
    if tmp == 0:
        days = days + 2
    elif tmp >4:
        days = days + tmp - 5
    
    days = days + 1
    
    
    fromdate = todate - dt.timedelta(days=days)
    
    adj = holiday[(holiday > fromdate)&(holiday<todate)].dropna().shape[0]
    fromdate = fromdate - dt.timedelta(days=adj)
    return fromdate

#%% [markdown]
# # Indicators and plots

#%%
get_ipython().run_line_magic('run', '"KiteConnect_Charting.ipynb"')

#%% [markdown]
# # Historical Data Download

#%%
def getInstruments(exchange='NSE'):
    instruments_df = pd.DataFrame(data=kite.instruments(exchange))
    instruments_df = instruments_df.set_index('tradingsymbol')
    return instruments_df

def downloadData(symbol="HDFC", fromDate= dt.datetime.now() - dt.timedelta(days = 1), toDate=dt.datetime.now(), freq="minute"):
    symbolToken = instruments_df.loc[symbol,'instrument_token']
    
    if type(symbolToken).__name__ == 'Series':
        symbolToken = symbolToken[symbol].values[0]
    
    logging.debug(freq)
    raw_data = pd.DataFrame(data=kite.historical_data(symbolToken, fromDate, toDate, freq, continuous=False))
    raw_data = raw_data.set_index('date').tz_localize(None)
    return raw_data

def resample2(data,freq):
    data = data.resample(freq).agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'})
    #data.columns = data.columns.droplevel()
    return data

def getData(symbol, fromDate, toDate, exchange="NSE", freq="minute", force=False, symbolToken=''):
    #symbol = "SBIN"
    key = freq+"/"+exchange+"/"+symbol
    
    try:
        if symbolToken == '':
            symbolToken = instruments_df.loc[symbol,'instrument_token']
    except:
        logger.debug(symbol+":stock not in the list")
        return pd.DataFrame()

    #fromDate = dt.datetime(2019,4,8)
    #toDate = dt.datetime(2019,4,10)
    
    if force:
        temp_data = downloadData(symbol, fromDate, toDate, freq)
        return temp_data
    
    try:
        temp_file = pd.HDFStore("kite_data/kite_cache.h5", mode="r")
        rDate = temp_file.get(key).tail(1).index
        lDate = temp_file.get(key).head(1).index
        
        temp_file.close()
        
        #print(fromDate,toDate, lDate, rDate)
        raw_data = pd.read_hdf("kite_data/kite_cache.h5", key=key, mode="r", format="table")

        if   (fromDate < lDate ) and (toDate <= rDate):
            logging.info("Downloading data from fromDate to lDate")
            temp_data = downloadData(symbol,  fromDate, lDate, freq)
            temp_data = temp_data.append(raw_data.tail(-1))
#            temp_data.to_hdf("kite_data/kite_cache.h5", key=key, mode="a", format="table")
        elif (fromDate >=lDate ) and (toDate <= rDate):
            logging.info("Using cache: Not downloading data")
            temp_data = raw_data
        elif (fromDate >= lDate ) and (toDate > rDate):
            logging.info("Downloading data from rDate to toDate")
            temp_data = downloadData(symbol,  rDate, toDate, freq)
            temp_data = raw_data.append(temp_data.tail(-1))
#            temp_data.to_hdf("kite_data/kite_cache.h5", key=key, mode="a", format="table")
        elif (fromDate < lDate ) and (toDate > rDate):
            logging.info("Downloading data from fromDate to lDate")
            temp_data = downloadData(symbol,  fromDate, lDate, freq)
            temp_data = temp_data.append(raw_data.tail(-1))
            logging.info("Downloading data from rDate to toDate")
            temp_data2 = downloadData(symbol,  rDate, toDate, freq)
            temp_data = temp_data.append(temp_data2.tail(-1))
#            temp_data.to_hdf("kite_data/kite_cache.h5", key=key, mode="a", format="table")

    except Exception as e:
        logging.debug(e)
        temp_data = downloadData(symbol, fromDate, toDate, freq)
    finally:
        temp_data.to_hdf("kite_data/kite_cache.h5", key=key, mode="a", format="table")
        return temp_data[(temp_data.index >= fromDate) & (temp_data.index <= toDate)]
    
def portfolioDownload(stocklist, toDate):
    stocklist_df = pd.DataFrame()
    for index, row in stocklist.iterrows():
        symbol = row[0]
        logging.info("Downloading data for: "+symbol)
        temp_data = getData(symbol,  toDate - dt.timedelta(days = 5), toDate)
        temp_data['symbol'] = symbol
        temp_data.set_index(['symbol',temp_data.index], inplace=True)
        #print(temp_data)
        stocklist_df = stocklist_df.append(temp_data)
    
    #print(stocklist_df)
    return stocklist_df

#%% [markdown]
# # Kite Authentication and wrappers

#%%
kite = KiteConnect(api_key=KiteAPIKey)
reauthentication = False

f = open("kite_data/access_token.txt", mode="r")
access_token = f.readlines()
logger.info(access_token[0])

try:
    kite.set_access_token(access_token[0])
    logger.info("Welcome "+kite.profile()['user_name'])
except:
    logger.critical("Offline Mode: Could not authenticate with the Kite Server")
    offline = True


#%%
try:
    if exchange=="":
        exchange = "NSE"
except:
    logging.debug("Exchange not defined: Using default NSE")
    exchange = "NSE"

try:
    instruments_df = getInstruments(exchange)
    instruments_df.to_hdf('kite_data/kite_cache.h5', key=exchange, mode='a', format="table")
except:
    logger.critical("Error in downloading instrument table from kite")
    
try:
    instruments_df = pd.read_hdf('kite_data/kite_cache.h5', key=exchange, mode='r', format="table")

    EQSYMBOL = lambda x:instruments_df[instruments_df['instrument_token']==x].index[0]
    EQTOKEN = lambda x:instruments_df.loc[x,'instrument_token']
except:
    logger.critical("Error in reading h5 file")

#%% [markdown]
# ## Kite- Order Management

#%%
#logging.critical("BUY"+symbol)
def buy_slm(symbol, price, trigger,quantity=1): 
    logger.info('%12s'%"BUY SLM: "+symbol+", price: "+str('%0.2f'%price)+", stoploss: "+str('%0.2f'%stoploss)+", quantity: "+str(quantity))
    
    if papertrade:
        return
    
    try:
        order_id = kite.place_order(tradingsymbol=symbol,
                                exchange=kite.EXCHANGE_NSE,
                                transaction_type=kite.TRANSACTION_TYPE_BUY,
                                quantity=quantity,
                                order_type=kite.ORDER_TYPE_SLM,
                                product=kite.PRODUCT_MIS,
                                trigger_price=round(trigger,1),
                                #stoploss=round(stoploss,1),
                                #price=price,
                                variety=kite.VARIETY_REGULAR
                                )
        logger.info("Order placed. ID is: {}".format(order_id))
    except Exception as e:
        logger.info("Order placement failed: {}".format(e.message))
        
def sell_slm(symbol, price, trigger, quantity=1):
    
    logger.info('%12s'%"SELL SLM: "+symbol+", price: "+str('%0.2f'%price)+", stoploss: "+str('%0.2f'%stoploss)+", quantity: "+str(quantity))
       
    if papertrade:
         return
    try:
        order_id = kite.place_order(tradingsymbol=symbol,
                            exchange=kite.EXCHANGE_NSE,
                            transaction_type=kite.TRANSACTION_TYPE_SELL,
                            quantity=quantity,
                            order_type=kite.ORDER_TYPE_SLM,
                            product=kite.PRODUCT_MIS,
                            trigger_price=round(trigger,1),
                            #price=price,
                            variety=kite.VARIETY_REGULAR)
        logger.info("Order placed. ID is: {}".format(order_id))
    except Exception as e:
        logger.info("Order placement failed: {}".format(e.message))

def buy_bo(symbol, price, trigger, stoploss, squareoff, quantity=1, tag="bot"): 
    logger.info('%12s'%"BUY BO: "+symbol+", price: "+str('%0.2f'%price)+", squareoff: "+str('%0.2f'%squareoff)+", stoploss: "+str('%0.2f'%stoploss)+", quantity: "+str(quantity))
    if papertrade:
        return
    
    try:
        order_id = kite.place_order(tradingsymbol=symbol, exchange=kite.EXCHANGE_NSE, transaction_type=kite.TRANSACTION_TYPE_BUY,
                        order_type=kite.ORDER_TYPE_LIMIT, product=kite.PRODUCT_MIS, variety=kite.VARIETY_BO, 
                                quantity=quantity, trigger_price=trigger, price=price,
                                squareoff=squareoff,  stoploss=stoploss, tag=tag )
        logger.info("Order placed. ID is: {}".format(order_id))
    except Exception as e:
        logger.info("Order placement failed: {}".format(e.message))



def sell_bo(symbol, price, trigger, stoploss, squareoff, quantity=1, tag="bot"): 
    logger.info('%12s'%"SELL BO: "+symbol+", price: "+str('%0.2f'%price)+", squareoff: "+str('%0.2f'%squareoff)+", stoploss: "+str('%0.2f'%stoploss)+", quantity: "+str(quantity))
    if papertrade:
        return
    
    try:
        order_id = kite.place_order(tradingsymbol=symbol, exchange=kite.EXCHANGE_NSE, transaction_type=kite.TRANSACTION_TYPE_SELL,
                                order_type=kite.ORDER_TYPE_LIMIT, product=kite.PRODUCT_MIS, variety=kite.VARIETY_BO,
                                quantity=quantity, trigger_price=trigger, price=price,
                                stoploss=stoploss, squareoff=squareoff,  tag=tag )
        logger.info("Order placed. ID is: {}".format(order_id))
    except Exception as e:
        logger.info("Order placement failed: {}".format(e.message))
        
def getOrders():    
    # Fetch all orders
    return pd.DataFrame(kite.orders())

def cancelOrder(orderId):
    if papertrade:
        logging.critical("In Paper Trade Mode: Order cancellation not possible")
        return
    
    try:
        kite.cancel_order(variety=kite.VARIETY_REGULAR, order_id=orderId, parent_order_id=None)    
    except Exception as e:
        logger.info("Order Cancellation failed: {}".format(e.message))
        
def squareoff(symbol=None, tag="bot"):
    logger.info('%12s'%"Squareoff: "+symbol)
    if papertrade:
        return
    
    orders_df = pd.DataFrame(kite.orders())
    if symbol != None:
        open_orders = orders_df[(orders_df['tradingsymbol']==symbol) & (orders_df['status'] == 'TRIGGER PENDING')  & (orders_df['tag'] == tag)]
    else:
        open_orders = orders_df[(orders_df['status'] == 'TRIGGER PENDING')  & (orders_df['tag'] == tag)]
        
    for index, row in open_orders.iterrows():
        print(row.order_id, row.parent_order_id)
        #kite.exit_order(variety=kite.VARIETY_AMO, order_id=row.order_id, parent_order_id=row.parent_order_id)
        kite.exit_order(variety=kite.VARIETY_BO, order_id=order_id, parent_order_id=parent_order_id)

#%% [markdown]
# ## Kite - Live Tick Handler

#%%
def resample(ws, freq="1min"):
    #F = open("kite_data/recommendation.csv","a") 
    
    logging.debug(str(ws.prevtimeStamp)+": In resampler function")
    
    if ws.LiveStream.empty:
        logging.debug(str(ws.prevtimeStamp)+": Empty dataframe, Exiting resampler")
        return
      
    LiveStream2 = ws.LiveStream.groupby(['symbol','date']).agg({'price':['first','max','min','last'], 'volume':['last']})
    LiveStream2.columns = LiveStream2.columns.droplevel()
    LiveStream2.columns = ['open', 'high','low','close', 'volume']

    for index, data in LiveStream2.groupby(level=0):
        sampled = data.loc[index].resample(freq).agg({'open':{'open':'first'},'high':{'high':'max'},'low':{'low':'min'},'close':{'close':'last'},'volume':{'volume':'last'}})
        sampled.columns = sampled.columns.droplevel()
        logger.debug(index)
        
        sampled['volume'] = sampled['volume'] - sampled['volume'].shift(1) 
        sampled['symbol'] = index
        sampled.set_index(['symbol',sampled.index], inplace=True)
        #logger.debug(sampled.tail())

        ws.LiveStreamOHLC = ws.LiveStreamOHLC.append(sampled.iloc[-1])
        
    #ws.LiveStreamOHLC.to_csv("kite_data/livestreamohlc.csv", mode='a')

    for symbol in portfolio[0]:
        #symbol = portfolio[0].iloc[-1]
        temp_ohlc_df = ws.LiveStreamOHLC.loc[symbol].tail(120)
        ws.tradebook_df.loc[symbol,'symbol'].trade_manager(symbol, temp_ohlc_df)
    
    
def ticksHandler(ws, ticks):
    #timeStamp = dt.datetime.now().replace(second=0, microsecond=0)
    tick_df = pd.DataFrame(ticks)
    
    try:
        #tick_df.loc[tick_df['timestamp'].isna(), 'timestamp'] = timeStamp
        tick_df = tick_df[['timestamp','instrument_token','last_price','volume']]
        tick_df.instrument_token = tick_df.instrument_token.apply(EQSYMBOL)
        tick_df.columns = ['date','symbol','price','volume']
        tick_df.set_index(['symbol','date'], inplace=True)
        
        timeStamp = tick_df.index[0][-1].to_pydatetime()
        
    except  Exception as e:
        logging.debug("Exception: ticksHandler: "+str(e)+str(tick_df))
        
    if( (timeStamp - ws.prevtimeStamp) >= dt.timedelta(minutes=1)):
        ws.prevtimeStamp = timeStamp
        resample(ws)
    
    ws.LiveStream = ws.LiveStream.append(tick_df)

def orderNotification(ws,data):
    #logger.debug(data)
    order_df = pd.DataFrame.from_dict(data, orient='index')

    symbol = order_df.loc['tradingsymbol'][0]
    
    ws.tradebook_df.loc[symbol,'symbol'].update_order(order_df)
    #logger.debug(order_df)

def initTrade(ws):
    ws.prevtimeStamp = dt.datetime.now() - dt.timedelta(minutes=10)
    toDate = dt.datetime.now()
    
    ws.tradebook_df = pd.DataFrame()
    
    for symbol in portfolio[0]:
        temp_df = pd.DataFrame(data=[algoTrade(symbol)], index=[symbol], columns=['symbol'])
        ws.tradebook_df = ws.tradebook_df.append(temp_df)
        
    #TODO: Convert to multistock handling
    #symbol = portfolio[0].iloc[-1]
    #ws.a = algoTrade(symbol)
    
    ws.LiveStream = pd.DataFrame()
    ws.LiveStreamOHLC = pd.DataFrame()
    ws.LiveStreamOHLC = portfolioDownload(portfolio, toDate)
    

#%% [markdown]
# ## Kite - Streaming Data(Websocket) Handler

#%%
def on_ticks(ws, ticks):
    # Callback to receive ticks.
    #logging.debug("Ticks: {}".format(ticks))
    ticksHandler(ws, ticks)


def on_connect(ws, response):
    initTrade(ws)
    logger.debug(portfolioToken)
    # Callback on successful connect.
    # Subscribe to a list of instrument_tokens (RELIANCE and ACC here).
    #ws.subscribe(portfolioToken)

    ws.subscribe(portfolioToken)
    
    # Set RELIANCE to tick in `full` mode.
    # MODE_LTP, MODE_QUOTE, or MODE_FULL

    ws.set_mode(ws.MODE_FULL, portfolioToken)
    #ws.set_mode(ws.MODE_FULL, [225537]) 
    #ws.set_mode(ws.MODE_LTP, [225537, 3861249]) 
    #ws.set_mode(ws.MODE_MODE_QUOTE, [2714625,779521]) 

def on_close(ws, code, reason):
    # On connection close stop the main loop
    # Reconnection will not happen after executing `ws.stop()`
    ws.stop()

def on_order_update(ws, data):
    #logger.info("New Order Update")
    orderNotification(ws,data)


#%%
# Initialise
kws = KiteTicker(KiteAPIKey, kite.access_token)

# Assign the callbacks.
kws.on_ticks = on_ticks
kws.on_connect = on_connect
kws.on_order_update = on_order_update


open("log/live_log.log", "w")
mode = "Algo"
papertrade = True

logger.setLevel(logging.DEBUG)


#%%
logger.info("Let's do trading")


#%%
pd.set_option('display.max_rows', None)
pd.set_option('precision', 2)
pd.set_option('display.max_columns', None)

#%% [markdown]
# ### Helper functions for CD pattern and backtesting

#%%
from talib import MIN, MAX, CDLCLOSINGMARUBOZU,CDL3WHITESOLDIERS,CDLMORNINGSTAR,CDLMORNINGDOJISTAR,CDL3LINESTRIKE,CDL3OUTSIDE,CDLENGULFING,CDLBELTHOLD,CDLABANDONEDBABY,CDL3INSIDE,CDLPIERCING,CDLBREAKAWAY,CDLXSIDEGAP3METHODS,CDLHAMMER,CDLMARUBOZU,CDL3BLACKCROWS,CDLIDENTICAL3CROWS,CDLEVENINGSTAR,CDLEVENINGDOJISTAR,CDLDARKCLOUDCOVER,CDLSHOOTINGSTAR,CDLHANGINGMAN,CDLHARAMI,CDLHARAMICROSS,CDLINVERTEDHAMMER,CDLSPINNINGTOP,CDLGRAVESTONEDOJI,CDLDOJI,CDLDOJISTAR,CDLDRAGONFLYDOJI
cdlfunc_array = [CDLCLOSINGMARUBOZU,CDL3WHITESOLDIERS,CDLMORNINGSTAR,CDLMORNINGDOJISTAR,CDL3LINESTRIKE,CDL3OUTSIDE,CDLENGULFING,CDLBELTHOLD,CDLABANDONEDBABY,CDL3INSIDE,CDLPIERCING,CDLBREAKAWAY,CDLXSIDEGAP3METHODS,CDLHAMMER,CDLMARUBOZU,CDL3BLACKCROWS,CDLIDENTICAL3CROWS,CDLEVENINGSTAR,CDLEVENINGDOJISTAR,CDLDARKCLOUDCOVER,CDLSHOOTINGSTAR,CDLHANGINGMAN,CDLHARAMI,CDLHARAMICROSS,CDLINVERTEDHAMMER,CDLSPINNINGTOP,CDLGRAVESTONEDOJI,CDLDOJI,CDLDOJISTAR,CDLDRAGONFLYDOJI]

def detectCDPattern(prices, a, noofcandles=15, strPlot="engulfing"):
    #https://github.com/mrjbq7/ta-lib
    
    annotateText = []
    annotateIndex = []
    
    for index, curr in prices.iloc[noofcandles:].iterrows():
        text = ""
        
        index_r = prices.index.get_loc(index)+1
        index_l = max(0, index_r - noofcandles)
        #rice = pd.DataFrame()
        popen = prices.iloc[index_l:index_r]['open']
        phigh = prices.iloc[index_l:index_r]['high']
        pclose = prices.iloc[index_l:index_r]['close']
        plow = prices.iloc[index_l:index_r]['low']
        
        function = CDLDOJI
        
        for cdlfunc in cdlfunc_array:
            ret_val = cdlfunc(popen, phigh, plow, pclose)[-1]
            if (ret_val > 0):
                text += "Bull: "+re.sub('CDL', '', cdlfunc.__name__)+"<br>"
            elif (ret_val < 0):
                text += "Sell: "+re.sub('CDL', '', cdlfunc.__name__)+"<br>"
                
        if text != "":
            annotateText.append(text)
            annotateIndex.append(index)
            
            a.appendHoverText(index,text)
            #logger.debug(index.strftime("%d-%m-%y %H:%M")+": "+text)
days = 1    
def trade_simulator(portfolio, toDate, plot=False, cd=False, mode="backtest", ha=True, prn=True):
    open("log/live_log.log", "w")
    
    fromDate =   getFromDate(toDate, days=days)
    #logger.setLevel(logging.DEBUG)
    papertrade = True
    
    cumsum = 0
    for i,y in portfolio.iterrows():
        logger.info(y[0])
    
        temp_data = getData(y[0], fromDate, toDate, freq="minute",force =False)
        
        #print(temp_data.head())

        a = algoTrade(y[0])
        a.symbol = y[0]
        a.tradeDecision(temp_data)
        
        buysell = pd.DataFrame(pd.concat([a.buy.close, a.sell.close]))
            
        temp_index = toDate.replace(hour=11,minute=31)
        try:
            temp_df = pd.DataFrame(data=[temp_data.loc[temp_index,'close']],index=[temp_index])
            buysell = buysell.append(temp_df)
            temp_index = toDate.replace(hour=15,minute=1)
            temp_df = pd.DataFrame(data=[temp_data.loc[temp_index,'close']],index=[temp_index])
            buysell= buysell.append(temp_df)
        except:
            return 0
        
        
        
        buysell = buysell.sort_index()
        
        if not buysell.empty:
            buysell = buysell[buysell.index>buysell.index[-1].date().strftime("%Y-%m-%d")]
        
        #print(buysell)
        
        payoff = 0
        if portfolio.shape[0] == 1 and cd and mode=="backtest":
            detectCDPattern(temp_data, a=a)
            pass
        
        previndex = buysell.index[0].replace(hour=9, minute=14)
        
        for index in buysell.index:
            
            if index == previndex:
                continue
            
            if mode == "backtest":    
                a.timestamp = index

            r = temp_data.index.get_loc(index)+1
            l = r - 30
            
            l2 = r - int(np.floor((index - previndex).total_seconds()/60))
            
            if l2 != r and True:
                low = min(temp_data.iloc[l2:r]['low'])
                high = max(temp_data.iloc[l2:r]['high'])
                
                if (a.price - a.stoploss) > low and a.STATE == "BUY":
                    a.profit = a.profit + (a.price - a.stoploss)
                    logTrade("SO SELL",a)
                    #logger.info("SO S:"+a.STATE+":"+str('%.2f'%a.profit)+","+str('%.2f'%(a.price - a.stoploss)))
                    #a.STATE = "WAIT"
                    a.update_order(pd.DataFrame.from_dict(orient='index', data={'order_id':173215,'tradingsymbol':a.symbol,'parent_order_id':a.order_id,'order_id':54321, 'status':'COMPLETE', 'filled_quantity':1,'average_price':(a.price - a.stoploss)}))
                elif  (a.price + a.stoploss) < high and a.STATE == "SELL":
                    a.profit = a.profit - (a.price + a.stoploss)
                    logTrade("SO BUY",a)
                    #logger.info("SO B:"+a.STATE+":"+str('%.2f'%a.profit)+","+str('%.2f'%(a.price + a.stoploss)))
                    #a.STATE = "WAIT"
                    a.update_order(pd.DataFrame.from_dict(orient='index', data={'order_id':173215,'tradingsymbol':a.symbol,'parent_order_id':a.order_id,'order_id':54321, 'status':'COMPLETE', 'filled_quantity':1,'average_price':(a.price + a.stoploss)}))
                elif  (a.price + a.target) < high and a.STATE == "BUY":
                    a.profit = a.profit + (a.price + a.target)
                    logTrade("PB SELL",a)
                    #logger.info("PB S:"+a.STATE+":"+str('%.2f'%a.profit)+","+str('%.2f'%(a.price + a.target)))
                    #a.STATE = "WAIT"
                    a.update_order(pd.DataFrame.from_dict(orient='index', data={'order_id':173215,'tradingsymbol':a.symbol,'parent_order_id':a.order_id,'order_id':54321, 'status':'COMPLETE', 'filled_quantity':1,'average_price':(a.price + a.target)}))
                elif  (a.price - a.target) > low and a.STATE == "SELL":
                    a.profit = a.profit - (a.price - a.target)
                    logTrade("PB BUY",a)
                    #logger.info("PB B:"+a.STATE+":"+str('%.2f'%a.profit)+","+str('%.2f'%(a.price + a.target)))
                    #a.STATE = "WAIT"
                    a.update_order(pd.DataFrame.from_dict(orient='index', data={'order_id':173215,'tradingsymbol':a.symbol,'parent_order_id':a.order_id,'order_id':54321, 'status':'COMPLETE', 'filled_quantity':1,'average_price':(a.price - a.target)}))
            
            #print(low, high)

            
            a.zone = a.checkZone(index)
            
            if mode != "backtest":
                    #logger.debug('Simulator')
                    a.trade_manager(y[0], temp_data.iloc[l:r])
                    a.order_id = 173215
            else:
                    #logger.debug('Backtest')
                # Lunch Time
                    if(a.zone=="zone3" and a.checkZone(previndex)=="zone2"):
                        temp_price = temp_data.loc[index.replace(minute=31),'close']
                        a.trade_setup(temp_price)
                        a.place_order("SQUAREOFF")
                        previndex = index
                        continue

                    # Too close to closing bell
                    #if(zone=="zone5" and s.checkZone(s.CLOSE.index[-2])=="zone4"):
                    #    return(s.place_order("SQUAREOFF"))

                    if(a.zone == "zone3"):
                        previndex = index
                        continue  



                    temp_price = buysell.loc[index].values

                    if type(temp_price[0]).__name__ == 'ndarray':
                        temp_price = -1 * abs(temp_price[0]) #if both buy and sell then do buy

                    temp_price = temp_price[0]

                    if temp_price < 0:
                        a.trade_setup(-1*temp_price)
                    else:
                        a.trade_setup(temp_price)

                    if temp_price < 0:
                        a.place_order("BUY")
                        a.order_id = 173215
                    elif temp_price > 0:
                        a.place_order("SELL")
                        a.order_id = 173215
                    
                    
            previndex = index
            #logger.info(a.profit)
            
        if a.STATE == "BUY":
            a.profit = a.profit + temp_data.close[-1]
            
        elif a.STATE == "SELL":
            a.profit = a.profit - temp_data.close[-1]
        
        if prn:
            print('%10s'%y[0],"\t",'%.2f'%a.profit, "\t",str(round(a.profit/a.CLOSE[-1]*100,2))+"%" )
        logger.info('%10s'%y[0]+": "+str('%.2f'%a.profit)+" ("+str(round(a.profit/a.CLOSE[-1]*100,2))+"%)")
    
        cumsum = cumsum + round(a.profit/a.CLOSE[-1]*100,2)
        if portfolio.shape[0] == 1 and plot and mode=="backtest":
            a.plot(a.symbol, 360, ha=ha)
            pass
        
    return cumsum

#%% [markdown]
# ##  <span style="color:green">Trading Strategy - Base Class</span>

#%%
# ====== Tradescript Wrapper =======
# Methods
REF = lambda df, i: df.shift(i)
TREND_UP = lambda a,b: ROC(a, b) >= 0.1
TREND_DOWN = lambda a,b: ROC(a, b) <= -0.1

import traceback
#TREND_UP = lambda a,b: a > MAX(REF(a,1),b)
#TREND_DOWN = lambda a,b: a < MIN(REF(a,1),b)

CROSSOVER = lambda a, b: (REF(a,1)<=REF(b,1)) & (a > b)
blackoutEnabled = True

blackoutFrom = '11:30:00'
blackoutTo = '12:30:00'

blackout = lambda a,x,y=blackoutFrom,z=blackoutTo: a[blackoutEnabled & (a.index >= x) & (a.index < x+' '+y) | (a.index > x+' '+z)]

logTrade = lambda prefix, a: logger.debug('%25s' %prefix+a.timestamp.strftime("[%H:%M]: ")+"\t"+str('%0.2f'%a.profit)+"\t"+str('%0.2f'%a.price)+"\t | "+ re.sub('<br>',', ',a.getHoverText(a.timestamp),0))

class algoTrade_base:  
    def __init__(s, symbol):
        logger.debug("algoTrade_base: "+symbol)
        
        #s.symbol = symbol
        s.STATE = "WAIT"
        s.stoploss = 0
        s.target = 0
        s.price = 0
        s.trigger = 0
        s.quantity = 0
        s.tag = "bot"
        s.profit = 0
        
        s.order_id = None
        s.symbol = symbol

    
    def long_candle(s):
        temp_df = False
        temp_df = temp_df | (CDLMORNINGSTAR(s.OPEN, s.HIGH, s.LOW, s.CLOSE) == 100)
        temp_df = temp_df | (CDLMORNINGDOJISTAR(s.OPEN, s.HIGH, s.LOW, s.CLOSE) == 100)
        temp_df = temp_df | (CDL3WHITESOLDIERS(s.OPEN, s.HIGH, s.LOW, s.CLOSE) == 100)
        
        temp_df = temp_df | (CDLENGULFING(s.OPEN, s.HIGH, s.LOW, s.CLOSE) == 100)
        temp_df = temp_df | (CDL3LINESTRIKE(s.OPEN, s.HIGH, s.LOW, s.CLOSE) == 100)
        temp_df = temp_df | (CDL3OUTSIDE(s.OPEN, s.HIGH, s.LOW, s.CLOSE) == 100)
        temp_df = temp_df | (CDLABANDONEDBABY(s.OPEN, s.HIGH, s.LOW, s.CLOSE) == 100)
        temp_df = temp_df | (CDLBELTHOLD(s.OPEN, s.HIGH, s.LOW, s.CLOSE) == 100)
        
        temp_df = temp_df | CDL3INSIDE(s.OPEN, s.HIGH, s.LOW, s.CLOSE) == 100
        temp_df = temp_df | (CDLHARAMI(s.OPEN, s.HIGH, s.LOW, s.CLOSE) == 100)
        temp_df = temp_df | (CDLPIERCING(s.OPEN, s.HIGH, s.LOW, s.CLOSE) == 100)
        temp_df = temp_df | (CDLHAMMER(s.OPEN, s.HIGH, s.LOW, s.CLOSE) == 100)
        temp_df = temp_df | (CDLDOJISTAR(s.OPEN, s.HIGH, s.LOW, s.CLOSE) != 0)
        
        temp_df = temp_df | (CDLMARUBOZU(s.OPEN, s.HIGH, s.LOW, s.CLOSE) == 100) 
        temp_df = temp_df | (CDLCLOSINGMARUBOZU(s.OPEN, s.HIGH, s.LOW, s.CLOSE) ==100)
    
        
        return pd.DataFrame(temp_df, columns=["buy"])

    
    def short_candle(s):
        temp_df = False
        
        temp_df = temp_df | (CDL3BLACKCROWS(s.OPEN, s.HIGH, s.LOW, s.CLOSE) == -100)
        temp_df = temp_df | (CDLIDENTICAL3CROWS(s.OPEN, s.HIGH, s.LOW, s.CLOSE) == -100)
        temp_df = temp_df | (CDLEVENINGSTAR(s.OPEN, s.HIGH, s.LOW, s.CLOSE) == -100)
        temp_df = temp_df | (CDL3LINESTRIKE(s.OPEN, s.HIGH, s.LOW, s.CLOSE) == -100)
        
        temp_df = temp_df | (CDLENGULFING(s.OPEN, s.HIGH, s.LOW, s.CLOSE) == -100)
        temp_df = temp_df | (CDLEVENINGDOJISTAR(s.OPEN, s.HIGH, s.LOW, s.CLOSE) == -100)
        temp_df = temp_df | (CDL3OUTSIDE(s.OPEN, s.HIGH, s.LOW, s.CLOSE) == -100)
        temp_df = temp_df | (CDLBELTHOLD(s.OPEN, s.HIGH, s.LOW, s.CLOSE) == -100)
        temp_df = temp_df | (CDLABANDONEDBABY(s.OPEN, s.HIGH, s.LOW, s.CLOSE) == -100)
        
        temp_df = temp_df | (CDLHARAMI(s.OPEN, s.HIGH, s.LOW, s.CLOSE) == -100)
        temp_df = temp_df | (CDLDARKCLOUDCOVER(s.OPEN, s.HIGH, s.LOW, s.CLOSE) == -100)
        temp_df = temp_df | (CDLDOJISTAR(s.OPEN, s.HIGH, s.LOW, s.CLOSE) != 0)
        temp_df = temp_df | (CDLHANGINGMAN(s.OPEN, s.HIGH, s.LOW, s.CLOSE) == -100)
        temp_df = temp_df | (CDLMARUBOZU(s.OPEN, s.HIGH, s.LOW, s.CLOSE) == -100)
        temp_df = temp_df | (CDLCLOSINGMARUBOZU(s.OPEN, s.HIGH, s.LOW, s.CLOSE) == -100)
        
        return pd.DataFrame( temp_df , columns=["sell"])

    def postprocess(s):
        s.buy['low'] = s.LOW
        s.sell['low'] = s.LOW
        s.buy['high'] = s.HIGH
        s.sell['high'] = s.HIGH
        s.buy['close'] = -1 * s.CLOSE
        s.sell['close'] = s.CLOSE
        
        #if s.buy['buy'].any():
        #    s.buy = blackout(s.buy,s.buy.index[-1].date().strftime("%Y-%m-%d"))
        #if s.sell['sell'].any():
        #    s.sell = blackout(s.sell, s.sell.index[-1].date().strftime("%Y-%m-%d"))
        
        s.buy = s.buy[s.buy['buy']]
        s.sell = s.sell[s.sell['sell']]

    def tradeDecision(s, price_ohlc_df):     
        #logger.info(price_ohlc_df.index[-1])
        s.price_df = price_ohlc_df
        s.OPEN = price_ohlc_df['open']
        s.CLOSE = price_ohlc_df['close']
        s.HIGH = price_ohlc_df['high']
        s.LOW = price_ohlc_df['low']
    
        s.hovertextarray = pd.DataFrame(np.full(price_ohlc_df.index.shape[0],""))
        
        #Heikin Asi
        s.haOPEN  = (s.OPEN.shift(1) + s.CLOSE.shift(1))/2
        s.haHIGH  = pd.DataFrame([s.HIGH,s.OPEN,s.CLOSE]).max(axis = 0, skipna = True)
        s.haLOW   = pd.DataFrame([s.HIGH,s.LOW,s.CLOSE]).min(axis = 0, skipna = True)
        s.haCLOSE = (s.OPEN+s.HIGH+s.LOW+s.CLOSE)/4
        
        s.BBT, s.BBM, s.BBB = BBANDS( s.CLOSE, 20,2,2,0)
        #s.BBT2, s.BBM2, s.BBB2 = BBANDS( s.CLOSE, 20,3,3,0)
        s.macd, s.macdsignal, s.macdhist = MACDEXT(s.CLOSE, fastperiod=12, slowperiod=26, signalperiod=9,  fastmatype=1, slowmatype=1,signalmatype=1)
        s.SD = STDDEV(s.CLOSE)
        s.fastk, s.fastd = STOCHF(s.HIGH, s.LOW, s.CLOSE)
        s.rsi = RSI(s.CLOSE, timeperiod=20)
        s.min = MIN(s.CLOSE, timeperiod=30)
        s.max = MAX(s.CLOSE, timeperiod=30)

        
    def update_order(s,order_df):
        order_id = order_df.loc['order_id'][0]
        parent_order_id = order_df.loc['parent_order_id'][0]
        status = order_df.loc['status'][0]
        tradingsymbol = order_df.loc['tradingsymbol'][0]
        filled_quantity = order_df.loc['filled_quantity'][0]
        average_price = order_df.loc['average_price'][0]

        prev_order_id = s.order_id

        #New order
        if status == 'COMPLETE' and parent_order_id == None:
            s.order_id = order_id

        #Square off: Stoploss or Target reached
        if status == 'COMPLETE' and parent_order_id != None and prev_order_id == parent_order_id:
            logger.info("Auto Squareoff: "+str(tradingsymbol)+":"+str(average_price))
            s.order_id = None
            s.STATE = "WAIT"
  

    def checkZone(s, a):
        s.zone1 = a.replace( hour=9, minute=15, second=0,microsecond=0)
        s.zone2 = a.replace(hour=11, minute=30, second=0,microsecond=0)
        s.zone3 = a.replace(hour=12, minute=30, second=0,microsecond=0)
        s.zone4 = a.replace(hour=14, minute=45, second=0,microsecond=0)

        text = ""
        if a < s.zone1:
            text = "zone1"
        elif a <= s.zone2:
            text = "zone2"
        elif a <= s.zone3:
            text = "zone3"
        elif a <= s.zone4:
            text = "zone4"
        else:
            text = "zone5"

        return text

    def getHoverText(s,index):
        iloc = s.CLOSE.index.get_loc(index)
        return s.hovertextarray.iloc[iloc][0]
        
    def appendHoverText(s, index,text):
        #logger.debug(str(index)+": "+text)
        iloc = s.CLOSE.index.get_loc(index)
        s.hovertextarray.iloc[iloc] = s.hovertextarray.iloc[iloc] + text

    
    def trade_manager(s, symbol, price_ohlc_df): 
        #logger.info(price_ohlc_df.tail())
            
        s.symbol = symbol
        s.tradeDecision(price_ohlc_df)
        
        #logger.debug(s.buy)
        #logger.debug(s.sell)
        
        s.timestamp = price_ohlc_df.index[-1].replace(second=0, microsecond=0)
        
        try:
            buy  = (s.timestamp - s.buy.index[-1]).seconds
        except:
            buy = 360
            
        try:
            sell = (s.timestamp - s.sell.index[-1]).seconds
        except:
            sell = 360
        
        #logger.info(str(s.timestamp)+":"+str(buy)+", "+str(sell))
        s.trade_setup(s.CLOSE.iloc[-1])
        
        # Lunch Time
        s.zone = s.checkZone(s.timestamp)
        
        if(s.zone=="zone3" and s.checkZone(s.CLOSE.index[-2])=="zone2"):
            return(s.place_order("SQUAREOFF"))
        
        # Too close to closing bell
        #if(zone=="zone5" and s.checkZone(s.CLOSE.index[-2])=="zone4"):
        #    return(s.place_order("SQUAREOFF"))
        
        if(s.zone == "zone3"):
            return 0            
        
        if buy < 60:
            return(s.place_order("BUY"))
        elif sell <60:
            return(s.place_order("SELL"))
        
        
        return 0

    
    
    def trade_setup(s, price):
        s.price = price
        s.trigger = price
        s.stoploss = toTick(price * 0.1/100)
        s.target = toTick(price * 1/100)
        s.quantity = 1
        s.tag = "bot"
        
        #logger.debug(s.price)
    
    def place_order(s, action):
        
        #trade_setup()
        #zone = s.checkZone(s.CLOSE.index[-1])
        
        fbuy = lambda:   buy_bo(symbol=s.symbol, price=s.price, trigger=s.trigger, stoploss=s.stoploss, squareoff=s.target, quantity=s.quantity, tag=s.tag)
        fsell = lambda: sell_bo(symbol=s.symbol, price=s.price, trigger=s.trigger, stoploss=s.stoploss, squareoff=s.target, quantity=s.quantity, tag=s.tag)
        fsquareoff = lambda: squareoff(symbol=s.symbol, tag="bot")
        margin = s.quantity * s.price
        
        if s.zone == "zone5":
            action = "SQUAREOFF"
         
        logTrade("New Order["+s.symbol+"]: "+s.STATE+"=>"+action, s)    
        #logger.info("Place Order["+s.timestamp.strftime("%H:%M")+"]: "+s.STATE+"=>"+action)
                
        if s.STATE == "WAIT":
            if action == "BUY":
                fbuy()
                s.profit = s.profit - margin
                s.STATE = "BUY"
            elif action == "SELL":
                fsell()
                s.profit = s.profit + margin
                s.STATE = "SELL"
        elif s.STATE == "BUY":
            if action == "SELL":
                fsquareoff()
                fsell()
                s.profit = s.profit + 2 * margin
                s.STATE = "SELL"
            elif action == "SQUAREOFF":
                fsquareoff()
                s.profit = s.profit + margin
                s.STATE = "WAIT"
        elif s.STATE == "SELL":
            if action == "BUY":
                fsquareoff()
                fbuy()
                s.profit = s.profit - 2 * margin
                s.STATE = "BUY"
            elif action == "SQUAREOFF":
                fsquareoff()
                s.profit = s.profit - margin
                s.STATE = "WAIT"
                
        return s.profit

        
    def plot(s, symbol, noofcandles=120, ha=False):
        init_notebook_mode(connected=True)
        fig = tools.make_subplots(rows=4, cols=1, shared_xaxes=True, row_width=[1,1,1,3], vertical_spacing = 0.05)
        fig['layout']['xaxis'] = dict(rangeslider = dict(visible=False), side="bottom") #, range=[xMin,xMax])
        fig['layout'].update(height=750, plot_bgcolor='rgba(0,0,0,0)', title="Charts for "+symbol)
        
        fig['layout']['yaxis']['anchor'] = 'x'
        fig['layout']['yaxis']['side'] = 'right'

        fig['layout']['xaxis']['rangeselector'] = dict(
                    buttons=list([dict(count=1, label='1h', step='hour', stepmode='backward'),
                                  dict(count=2, label='2h', step='hour', stepmode='backward'),
                                  dict(count=3, label='3h', step='hour', stepmode='backward'),
                                  dict(count=6, label='4h', step='hour', stepmode='backward'),
                                  dict(count=6, label='1d', step='hour', stepmode='backward'),
                                  dict(step='all')]))
        
        yMin = s.LOW.iloc[-1*noofcandles:-1].min()*0.99
        yMax = s.HIGH.iloc[-1*noofcandles:-1].max()*1.001

        xMin = s.CLOSE.index[-1*noofcandles]
        xMax = s.CLOSE.index[-1]
        fig['layout']['yaxis']['range'] = [yMin, yMax]
        fig['layout']['xaxis']['range'] = [xMin, xMax]
        
        #print(s.buy.shape)
        #print(s.sell.shape)
        
        
        #print("plot")
        
        traceBuy=go.Scatter(x=s.buy.index.astype('str'), y=-1 * s.buy.close, name='BUY', mode="markers",marker = dict(color = 'rgba(119, 221, 119, 0.2)', size = 25, symbol='circle', line = dict(color = 'rgb(119, 221, 119)', width = 3)),showlegend=False)
        
        traceSell=go.Scatter(x=s.sell.index.astype('str'), y=s.sell.close, name='SELL', mode="markers",marker = dict(color = 'rgba(255, 0, 0,0.2)',size = 25, symbol='circle', line = dict(color = 'rgb(255, 0, 0)', width = 3)),showlegend=False)
       
        if ha:
            trace = go.Candlestick(x=s.CLOSE.index.astype('str'), open=s.haOPEN, high=s.haHIGH, low=s.haLOW, close=s.haCLOSE, name="Heikin Ashi", showlegend=False, hoverinfo = 'x+y+text', hovertext=s.hovertextarray)        
        else:
            trace = go.Candlestick(x=s.CLOSE.index.astype('str'), open=s.OPEN, high=s.HIGH, low=s.LOW, close=s.CLOSE, name="Candlestick", showlegend=False, hoverinfo = 'x+y+text', hovertext=s.hovertextarray)      
            
        traceMACD = go.Scatter(x=s.CLOSE.index.astype('str'), y=s.macd, name='MACD', line=dict(color='black'),showlegend=False)
        traceMACDSignal = go.Scatter(x=s.CLOSE.index.astype('str'), y=s.macdsignal, name='MACD signal', line=dict(color='red'),showlegend=False)
        traceMACDHist = go.Bar(x=s.CLOSE.index.astype('str'), y=s.macdhist, name='MACD Hist', marker=dict(color="grey"),showlegend=False)
        traceSK  = go.Scatter(x=s.CLOSE.index.astype('str'), y=s.fastk, name='%K', line=dict(color='black'), yaxis='y3',showlegend=False)
        traceSD  = go.Scatter(x=s.CLOSE.index.astype('str'), y=s.fastd, name='%D', line=dict(color='red'),showlegend=False)
        traceBBT = go.Scatter(x=s.BBT.index.astype('str'), y=s.BBT, name='BB_up',  line=dict(color='lightgrey'),showlegend=False, hoverinfo = 'none')
        traceBBB = go.Scatter(x=s.BBB.index.astype('str'), y=s.BBB, name='BB_low',  line=dict(color='lightgrey'), fill = 'tonexty', fillcolor="rgba(0,40,100,0.02)",showlegend=False, hoverinfo = 'none')
        traceBBM = go.Scatter(x=s.BBM.index.astype('str'), y=s.BBM, name='BB_mid',  line=dict(color='lightgrey'), fill = 'tonexty', fillcolor="rgba(0,40,100,0.02)",showlegend=False, hoverinfo = 'none')
        
        traceMIN = go.Scatter(x=s.min.index.astype('str'), y=s.min, name='MIN',  line=dict(color='darkblue'), showlegend=False, hoverinfo = 'none')
        traceMAX = go.Scatter(x=s.max.index.astype('str'), y=s.max, name='MAX',  line=dict(color='darkcyan'), showlegend=False, hoverinfo = 'none')
        
        traceRSI = go.Scatter(x=s.rsi.index.astype('str'), y=s.rsi, name='RSI',  line=dict(color='darkcyan'), showlegend=False)
        
        fig.append_trace(traceBBT, 1, 1)
        fig.append_trace(traceBBB, 1, 1)
        fig.append_trace(traceBBM, 1, 1)
        fig.append_trace(traceBuy, 1, 1)
        fig.append_trace(traceSell, 1, 1)
        fig.append_trace(traceMIN, 1, 1)
        fig.append_trace(traceMAX, 1, 1)
    
        fig.append_trace(traceSK, 2, 1)
        fig.append_trace(traceSD, 2, 1)
        
        
        
        fig['layout']['yaxis2']['title']="Stochastics"
        fig['layout']['yaxis2']['side']="right"
        fig['layout']['yaxis2']['anchor']="x"
        fig['layout']['yaxis2']['tickvals']=[20,80,30,70]
        
        fig.append_trace(traceMACD, 3, 1)
        fig.append_trace(traceMACDSignal, 3, 1)
        fig.append_trace(traceMACDHist, 3, 1)
        fig.append_trace(trace, 1, 1)
        
        fig['layout']['yaxis3']['anchor']="x"
        fig['layout']['yaxis3']['side']="right"
        fig['layout']['yaxis3']['title']="MACD"
        fig['layout']['yaxis1']['title']="Candlestick"
        
        
        fig.append_trace(traceRSI, 4, 1)
        fig['layout']['yaxis4']['title']="RSI"
        fig['layout']['yaxis4']['side']="right"
        fig['layout']['yaxis4']['anchor']="x"
        fig['layout']['yaxis4']['tickvals']=[30,50,70]
        
        iplot(fig, filename="plot/"+symbol+".html")


class algoTrade(algoTrade_base):
   
    def __init__(s, symbol):
        logger.debug("AlgoTrade Called: "+symbol)
        super(algoTrade, s).__init__(symbol)
        
    
    # Long Strategies
    def long_indicators(s):
        temp_df = 0
        #temp_df = temp_df | (REF(s.fastk,1) <=20)&(s.fastk>20)
        #temp_df = temp_df | CROSSOVER(s.fastk, s.fastd)
        #temp_df = temp_df | (REF(s.macd,1) <=0)&(s.macd>0)
        #temp_df = temp_df | CROSSOVER(s.macd, s.macdsignal)
        temp_df = temp_df | (REF(s.rsi,1) <=40)
        #temp_df = temp_df | (s.CLOSE.shift(1) == s.min.shift(1)) & (s.CLOSE > s.OPEN)
        #temp_df = temp_df & CROSSOVER(s.BBB, s.CLOSE)
        
        return pd.DataFrame(  temp_df, columns=["buy"] )
     
    # Short Strategies
    def short_indicators(s):
        temp_df = 0
        #temp_df = temp_df | (REF(s.fastk,1) >=80)&(s.fastk<80)
        #temp_df = temp_df | CROSSOVER(s.fastd, s.fastk)
        #temp_df = temp_df | (REF(s.macd,1) >=0)&(s.macd<0)
        #temp_df = temp_df | CROSSOVER(s.macdsignal, s.macd)
        temp_df = temp_df | (REF(s.rsi,1) >=60)
        #temp_df = temp_df & CROSSOVER(s.CLOSE, s.BBT)
        return pd.DataFrame( temp_df, columns=["sell"])
    

    
    def long_breakout(s):
        temp_df = s.haCLOSE >= s.BBT.shift(1)
        temp_df = temp_df | (s.haCLOSE >= s.haOPEN.shift(2))
        temp_df = temp_df & ( CROSSOVER(s.OPEN, s.BBT) | CROSSOVER(s.OPEN, s.BBB) | CROSSOVER(s.OPEN, s.BBM))
        return pd.DataFrame( temp_df , columns=["buy"])
    
    def short_breakout(s):
        temp_df = s.haOPEN <= s.BBB.shift(1)
        temp_df = temp_df | (s.haCLOSE <= s.haOPEN.shift(2))
        temp_df = temp_df & ( CROSSOVER(s.BBT, s.OPEN) | CROSSOVER(s.BBB, s.OPEN) | CROSSOVER(s.BBM, s.OPEN))
        return pd.DataFrame( temp_df , columns=["sell"])
    
    def long_ha(s):
        temp_df = (REF(s.haCLOSE,3) < REF(s.haOPEN,3)) & (REF(s.haCLOSE,2) < REF(s.haOPEN,2)) & (REF(s.haCLOSE,1) > REF(s.haOPEN,1)) & (s.haCLOSE > s.haOPEN)  
        #temp_df = temp_df & (s.rsi < 40 )
        return pd.DataFrame( temp_df , columns=["buy"])
    
    def short_ha(s):
        temp_df = (REF(s.haCLOSE,3) > REF(s.haOPEN,3)) & (REF(s.haCLOSE,2) > REF(s.haOPEN,2)) & (REF(s.haCLOSE,1) < REF(s.haOPEN,1)) & (s.haCLOSE < s.haOPEN)
        #temp_df = temp_df & (s.rsi > 60 )
        return pd.DataFrame( temp_df , columns=["sell"])

    def tradeDecision(s, price_ohlc_df):
        global blackoutEnabled

        super(algoTrade, s).tradeDecision(price_ohlc_df)
        
        #s.rsi = RSI(s.CLOSE, timeperiod=20)
        #s.min = MIN(s.CLOSE, timeperiod=30)
        #s.max = MAX(s.CLOSE, timeperiod=30)
        
        s.buy  = s.long_ha() & s.long_indicators()
        s.sell = s.short_ha() & s.short_indicators()
        
        s.postprocess()
        
        return (s.buy,s.sell)
    
      
    def trade_setup(s, price):
        s.price = price
        s.trigger = price
        s.stoploss = toTick(price * 0.1/100)
        s.target = toTick(price * 1/100)
        s.quantity = 1
        s.tag = "bot"

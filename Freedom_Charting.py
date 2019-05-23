
#%%
fig = ""
noofcandles = 60
def candlestick(price, pos=1, plot=False):
    #print(price.index)
    global fig
    # Candlestick
    trace = go.Candlestick(x=price.index.astype('str'), open=price['open'], high=price['high'], low=price['low'], close=price['close'], name="Candlestick", showlegend=False)

    if plot:
        fig.append_trace(trace, pos, 1)
        fig['layout']['yaxis'+str(pos)]['title']="Candlestick"
    return price

def macd(price, pos=1, plot=False):
    global fig
    #price['macd'], price['macdsignal'], price['macdhist'] = MACD(price.close, fastperiod=12, slowperiod=26, signalperiod=9)
    price['macd'], price['macdsignal'], price['macdhist'] = MACDEXT(price.close, fastperiod=12, slowperiod=26, signalperiod=9, fastmatype=1, slowmatype=1,signalmatype=1)
        
    # list of values for the Moving Average Type:  
    #0: SMA (simple)  
    #1: EMA (exponential)  
    #2: WMA (weighted)  
    #3: DEMA (double exponential)  
    #4: TEMA (triple exponential)  
    #5: TRIMA (triangular)  
    #6: KAMA (Kaufman adaptive)  
    #7: MAMA (Mesa adaptive)  
    #8: T3 (triple exponential T3)
    
    # MACD plots
    traceMACD = go.Scatter(x=price.index.astype('str'), y=price.macd, name='MACD', line=dict(color='black'),showlegend=False)
    traceMACDSignal = go.Scatter(x=price.index.astype('str'), y=price.macdsignal, name='MACD signal', line=dict(color='red'),showlegend=False)
    traceMACDHist = go.Bar(x=price.index.astype('str'), y=price.macdhist, name='MACD Hist', marker=dict(color="grey"),showlegend=False)
        
    if plot:
        fig.append_trace(traceMACD, pos, 1)
        fig.append_trace(traceMACDSignal, pos, 1)
        fig.append_trace(traceMACDHist, pos, 1)
        fig['layout']['yaxis'+str(pos)]['anchor']="x"
        fig['layout']['yaxis'+str(pos)]['side']="right"
        fig['layout']['yaxis'+str(pos)]['title']="MACD"
    
    return price

def rsi(price, pos=1, plot=False):
    global fig
    price['RSI'] = RSI(price.close, timeperiod=14)
    
    traceRSI = go.Scatter(x=price.index.astype('str'), y=price['RSI'],mode='lines', line=dict(color='rgb(63, 72, 204)'), name='RSI',showlegend=False)
    
    if plot:
        fig.append_trace(traceRSI, pos, 1)
        fig['layout']['yaxis'+str(pos)]['anchor']="x"
        fig['layout']['yaxis'+str(pos)]['tickvals']=[20,30,70,80]
        fig['layout']['yaxis'+str(pos)]['side']="right"
        fig['layout']['yaxis'+str(pos)]['title']="RSI"
        
    return price

def stoch(price, pos=1, plot=False):
    global fig
    price['slowk'], price['slowd'] = STOCHF(price.high, price.low, price.close)
    
    traceSK  = go.Scatter(x=price.index.astype('str'), y=price['slowk'], name='%K', line=dict(color='black'), yaxis='y3',showlegend=False)
    traceSD  = go.Scatter(x=price.index.astype('str'), y=price['slowd'], name='%D', line=dict(color='red'),showlegend=False)
    
    if plot:
        fig.append_trace(traceSK, pos, 1)
        fig.append_trace(traceSD, pos, 1)
        fig['layout']['yaxis'+str(pos)]['title']="Stochastics"
        fig['layout']['yaxis'+str(pos)]['side']="right"
        fig['layout']['yaxis'+str(pos)]['anchor']="x"
        fig['layout']['yaxis'+str(pos)]['tickvals']=[20,80,30,70]
    return price

def aroon(price, pos=1, plot=False):
    global fig
    price['aroondown'], price['aroonup'] = AROON(price.high, price.low, 25)
    
    traceARD  = go.Scatter(x=price.index.astype('str'), y=price['aroondown'], name='AROON Down', line=dict(color='red'), yaxis='y4',showlegend=False)
    traceARU  = go.Scatter(x=price.index.astype('str'), y=price['aroonup'], name='AROON UP', line=dict(color='green'),showlegend=False)
    
    if plot:
        fig.append_trace(traceARD, pos, 1)
        fig.append_trace(traceARU, pos, 1)
        fig['layout']['yaxis'+str(pos)]['side']="right"
        fig['layout']['yaxis'+str(pos)]['title']="Aroon"
        fig['layout']['yaxis'+str(pos)]['anchor']="x"
    return price

def atr(price, pos=1, plot=False, multiplier=2):
    global fig
    price['atr'] = ATR(price.high, price.low, price.close, timeperiod=22)
    price['chandlierLong'] = price.close-multiplier*price.atr
    price['chandlierShort'] = price.close+multiplier*price.atr
    
    traceATR = go.Scatter(x=price.index.astype('str'), y=price['atr'],mode='lines', line=dict(color='lightgrey'), name='ATR',showlegend=False)
    
    traceStopLoss = go.Scatter(x=price.index, y=price['chandlierLong'].shift(1), mode='lines', line=dict(color='lightgrey'), name='SL',showlegend=False)
    if plot:
        fig.append_trace(traceStopLoss, pos, 1)
        
    return price

def bbands(price, pos=1, plot=False, plotPrice=False):
    global fig
    price['bbt'], price['bbm'], price['bbb'] = BBANDS(price.close, timeperiod=20, nbdevup=1.6, nbdevdn=1.6, matype=0)
    price['bbt2'], price['bbm2'], price['bbb2'] = BBANDS(price.close, timeperiod=20, nbdevup=2.4, nbdevdn=2.4, matype=0)
    
    tracePrice = go.Scatter(x=price.index.astype('str'), y=price.close, marker = dict(color='grey', size=2), mode='lines', name="Close Price", yaxis='y1', showlegend=False)
    traceBBT = go.Scatter(x=price.index.astype('str'), y=price['bbt'], name='BB_up',  line=dict(color='lightgrey'),showlegend=False)
    traceBBB = go.Scatter(x=price.index.astype('str'), y=price['bbb'], name='BB_low',  line=dict(color='lightgrey'), fill = 'tonexty', fillcolor="rgba(0,40,100,0.02)",showlegend=False)
    traceBBM = go.Scatter(x=price.index.astype('str'), y=price['bbm'], name='BB_mid',  line=dict(color='lightgrey'), fill = 'tonexty', fillcolor="rgba(0,40,100,0.02)",showlegend=False)
    
    traceBBT2 = go.Scatter(x=price.index.astype('str'), y=price['bbt2'], name='BB_up2',  line=dict(color='blue'),showlegend=False)
    traceBBB2 = go.Scatter(x=price.index.astype('str'), y=price['bbb2'], name='BB_low2',  line=dict(color='blue'), fill = 'tonexty', fillcolor="rgba(0,40,100,0.02)",showlegend=False)
    #traceBBM2 = go.Scatter(x=price.index.astype('str'), y=price['bbm2'], name='BB_mid2',  line=dict(color='grey'), fill = 'tonexty', fillcolor="rgba(0,40,100,0.02)",showlegend=False)
    
    
    if plot:
        if plotPrice:
            fig.append_trace(tracePrice, pos, 1)
            
        fig.append_trace(traceBBT, pos, 1)
        fig.append_trace(traceBBB, pos, 1)
        fig.append_trace(traceBBM, pos, 1)
        
        fig.append_trace(traceBBT2, pos, 1)
        fig.append_trace(traceBBB2, pos, 1)
        #fig.append_trace(traceBBM2, pos, 1)
    
    return price

def sma(price, pos=1, plot=False, plotPrice=False):
    global fig
    price['sma20'] = SMA(price.close, timeperiod=50)
    price['sma50'] = SMA(price.close, timeperiod=200)
    
    tracePrice = go.Scatter(x=price.index.astype('str'), y=price.close, marker = dict(color='grey', size=2), mode='lines', name="Close Price", yaxis='y1', showlegend=False)
    traceSMA20 = go.Scatter(x=price.index.astype('str'), y=price['sma20'], name='SMA20',  line=dict(color='black'),showlegend=False)
    traceSMA50 = go.Scatter(x=price.index.astype('str'), y=price['sma50'], name='SMA50',  line=dict(color='red'),showlegend=False)
    
    
    if plot:
        if plotPrice:
            fig.append_trace(tracePrice, pos, 1)
            
        fig.append_trace(traceSMA20, pos, 1)
        fig.append_trace(traceSMA50, pos, 1)
    
    return price

def obv(price, pos=1, plot=False):
    global fig
    price['obv'] = OBV(np.linspace(1,100,price.index.shape[0]), price.volume)
    
    traceV = go.Bar(x=price.index.astype('str'), y=price.volume, name='Volume', marker=dict(color='blue'), showlegend=False)
    traceOBV = go.Scatter(x=price.index.astype('str'), y=price['obv'], name='OBV', line=dict(color='black'),showlegend=False)
    if plot:
        fig.append_trace(traceV, pos, 1)
        
        #fig['layout']['yaxis2']['overlaying']='y'
        fig['layout']['yaxis'+str(pos)]['anchor']='x'
        fig['layout']['yaxis'+str(pos)]['showgrid']=True
        fig['layout']['yaxis'+str(pos)]['visible']=True
        
        maxV = round(price.iloc[-3*noofcandles:-1]['volume'].max()/2,0)*2
        fig['layout']['yaxis'+str(pos)]['tickvals']=[maxV/2,maxV]
        fig['layout']['yaxis'+str(pos)]['range']=[0,maxV*2]
        fig['layout']['yaxis'+str(pos)]['side']="right"
        fig['layout']['yaxis'+str(pos)]['title']="volume"
    return price

def adosc(price, pos=1, plot=False):
    global fig
    
    price['adosc'] = ADOSC(price.high, price.low, price.close, price.volume)
    return price

def adx(price, pos=1, plot=False):
    global fig
    price['ADX'] = ADX(price.high, price.low, price.close)
    price['DI_P'] = PLUS_DI(price.high, price.low, price.close)
    price['DI_M'] = MINUS_DI(price.high, price.low, price.close)
    
    traceADX  = go.Scatter(x=price.index.astype('str'), y=price['ADX'], name='ADX', line=dict(color='blue') ,showlegend=False)
    traceDIP  = go.Scatter(x=price.index.astype('str'), y=price['DI_P'], name='DI+', line=dict(color='green'),showlegend=False)
    traceDIM  = go.Scatter(x=price.index.astype('str'), y=price['DI_M'], name='DI-', line=dict(color='red'),showlegend=False)
    
    
    if plot:
        fig.append_trace(traceADX, pos, 1)
        fig.append_trace(traceDIP, pos, 1)
        fig.append_trace(traceDIM, pos, 1)
        fig['layout']['yaxis'+str(pos)]['title']="ADX"
        fig['layout']['yaxis'+str(pos)]['side']="right"
        fig['layout']['yaxis'+str(pos)]['anchor']="x"
    return price

# Heikin-Ashi
def ha(price, pos=1, plot=False):
    global fig
    price['HAopen'] = (price.open.shift(1) + price.close.shift(1))/2
    price['HAhigh'] = price[['high','open','close']].max(axis = 1, skipna = True)
    price['HAlow'] = price[['low','open','close']].min(axis = 1, skipna = True)
    price['HAclose'] = (price.open+price.high+price.low+price.close)/4
    
    traceHA = go.Candlestick(x=price.index.astype('str'), open=price['HAopen'], high=price['HAhigh'], low=price['HAlow'], close=price['HAclose'], name="Heikin Ashi", showlegend=False)
    
    if plot:
        fig.append_trace(traceHA, pos, 1)
    return price

def pivotPoint(price, pos=1,fibo=True, noofcandles=60, plot=False):
    global fig
    prev = raw_data_day.loc[prevday]

    PP = (prev.high+prev.low+prev.close)/3
    S1 = PP - 0.382 * (prev.high - prev.low)
    S2 = PP - 0.618 * (prev.high - prev.low)
    S3 = PP -         (prev.high - prev.low)

    R1 = PP + 0.382 * (prev.high - prev.low)
    R2 = PP + 0.618 * (prev.high - prev.low)
    R3 = PP +         (prev.high - prev.low)
    
    tracePP  = go.Scatter(x=price.index.astype('str'), y=np.full(raw_data.index.shape[0],PP), line=dict(color='red'),showlegend=False)
    traceS1  = go.Scatter(x=price.index.astype('str'), y=np.full(raw_data.index.shape[0],S1), line=dict(color='red'),showlegend=False)
    traceS2  = go.Scatter(x=price.index.astype('str'), y=np.full(raw_data.index.shape[0],S2), line=dict(color='red'),showlegend=False)
    traceS3  = go.Scatter(x=price.index.astype('str'), y=np.full(raw_data.index.shape[0],S3), line=dict(color='red'),showlegend=False)
    traceR1  = go.Scatter(x=price.index.astype('str'), y=np.full(raw_data.index.shape[0],R1), line=dict(color='red'),showlegend=False)
    traceR2  = go.Scatter(x=price.index.astype('str'), y=np.full(raw_data.index.shape[0],R2), line=dict(color='red'),showlegend=False)
    traceR3  = go.Scatter(x=price.index.astype('str'), y=np.full(raw_data.index.shape[0],R3), line=dict(color='red'),showlegend=False)
    
    
    
    if plot:
        fig.append_trace(tracePP, pos, 1)
        fig.append_trace(traceS1, pos, 1)
        fig.append_trace(traceS2, pos, 1)
        fig.append_trace(traceS3, pos, 1)
        fig.append_trace(traceR1, pos, 1)
        fig.append_trace(traceR2, pos, 1)
        fig.append_trace(traceR3, pos, 1)
    return price

def emasma(price, plot=False):
    global fig
    price['ema21'] = EMA(price['close'], 21)
    price['ema9'] = EMA(price['close'], 9)
    price['sma200'] = SMA(price['close'], 200)
    price['sma50'] = SMA(price['close'], 50)
    
    return price

def detectCDPattern(prices, plot=True, noofcandles=15, strPlot="engulfing"):
    global fig
    #https://github.com/mrjbq7/ta-lib
    
    annotateText = []
    annotateIndex = []
    
    for index, curr in prices.iloc[noofcandles:].iterrows():
        text = ""
        
        index_r = raw_data.index.get_loc(index)+1
        index_l = max(0, index_r - noofcandles)
        #rice = pd.DataFrame()
        popen = prices.iloc[index_l:index_r]['open']
        phigh = prices.iloc[index_l:index_r]['high']
        pclose = prices.iloc[index_l:index_r]['close']
        plow = prices.iloc[index_l:index_r]['low']
        
        
        if (CDLDOJI(popen, phigh, plow, pclose)[-1] != 0) and (strPlot.find("Doji") >= 0):
            text += "Doji,"
        
        if CDLENGULFING(popen, phigh, plow, pclose)[-1] == -100 and strPlot.find("engulfing")>=0:
            text += "Bearish Engulfing,"
            
        if CDLENGULFING(popen, phigh, plow, pclose)[-1] == 100 and strPlot.find("engulfing")>=0:
            #print(CDLENGULFING(popen, phigh, plow, pclose))
            #print(index.strftime("%d-%m-%y %H:%M"))
            text += "Bullish Engulfing,"
            
        
        if CDLABANDONEDBABY(popen, phigh, plow, pclose)[-1] != 0 and strPlot.find("abandonedbaby")>=0:
            text += "Abandoned Baby,"
    
        if CDL3BLACKCROWS(popen, phigh, plow, pclose)[-1] != 0 and strPlot.find("3blackcrows")>=0:
            text += "3 Black Crows,"
    
    
        if CDLDOJISTAR(popen, phigh, plow, pclose)[-1] != 0 and strPlot.find("dojistar")>=0:
            text += "Doji Star,"
    
        if CDLDRAGONFLYDOJI(popen, phigh, plow, pclose)[-1] != 0 and strPlot.find("dragonflydoji")>=0:
            text += "Dragon Fly Doji,"
    
        if CDLEVENINGDOJISTAR(popen, phigh, plow, pclose)[-1] != 0 and strPlot.find("eveningdojistar")>=0:
            text += "Evening Doji Star,"
    
        if CDLEVENINGSTAR(popen, phigh, plow, pclose)[-1] != 0 and strPlot.find("eveningstar")>=0:
            text += "Evening Star,"
    
        if CDLGRAVESTONEDOJI(popen, phigh, plow, pclose)[-1] != 0 and strPlot.find("gravestonedoji")>=0:
            text += "Gravestone Doji,"
    
        if CDLHAMMER(popen, phigh, plow, pclose)[-1] != 0 and strPlot.find("hammer")>=0:
            text += "Hammer,"
    
        if CDLHANGINGMAN(popen, phigh, plow, pclose)[-1] != 0 and strPlot.find("hangingman")>=0:
            text += "Hanging Man,"
    
        if CDLHARAMI(popen, phigh, plow, pclose)[-1] == 100 and strPlot.find("Harami")>=0:
            text += "Bullish Harami,"
            
        if CDLHARAMI(popen, phigh, plow, pclose)[-1] == -100 and strPlot.find("Harami")>=0:
            text += "Bearish Harami,"
    
        if CDLHARAMICROSS(popen, phigh, plow, pclose)[-1] != 0 and strPlot.find("haramicross")>=0:
            text += "Harami Cross,"
    
        if CDLINVERTEDHAMMER(popen, phigh, plow, pclose)[-1] != 0 and strPlot.find("invertedhammer")>=0:
            text += "Inverted Hammer,"
    
        if CDLMARUBOZU(popen, phigh, plow, pclose)[-1] != 0 and strPlot.find("marubozu")>=0:
            text += "Marubozu,"
    
        if CDLMORNINGDOJISTAR(popen, phigh, plow, pclose)[-1] != 0 and strPlot.find("morningdojistar")>=0:
            text += "Morning Doji Star,"
    
        if CDLMORNINGSTAR(popen, phigh, plow, pclose)[-1] != 0 and strPlot.find("morningstar")>=0:
            text += "Morning Star,"
    
        if CDLSHOOTINGSTAR(popen, phigh, plow, pclose)[-1] != 0 and strPlot.find("shootingstar")>=0:
            text += "Shooting Star,"
    
    
        if CDLSPINNINGTOP(popen, phigh, plow, pclose)[-1] and strPlot.find("spinningtop")>=0:
            text += "Spinning Top,"
    
        if CDL3LINESTRIKE(popen, phigh, plow, pclose)[-1] and strPlot.find("3linestrike")>=0:
            text += "3 Line Strike,"
        
        
        if text != "":
            annotateText.append(text)
            annotateIndex.append(index)
            
            #print(index.strftime("%d-%m-%y %H:%M")+": "+text)
    

    data = pd.DataFrame({'open':prices.loc[annotateIndex, 'open'],
                         'close':prices.loc[annotateIndex, 'close'],
                         'high':prices.loc[annotateIndex, 'high'],
                         'low':prices.loc[annotateIndex, 'low'],
                         'text':annotateText})
    prices['text'] =""
    
    prices['text'] = data['text']
    
    if plot:
        annotateBuySell(data)
    return prices

def calculateStats(price):
    global fig
    price['priceSD'] = STDDEV(price.close)
    price['priceVAR'] = VAR(price.close)

    price['delBBT'] = (price.bbt - price.close)/(price.bbt-price.bbm)
    price['delBBB'] = (price.close - price.bbb)/(price.bbm-price.bbb)

    price['priceTSF'] = TSF(price.close)

    price['priceROC'] = ROC(price.close)
    price['macdROC'] = ROC(price.macd)
    price['macdsigROC'] = ROC(price.macd)
    price['bbROC'] = ROC(price.bbm)

    price['sdPriceRange'] = STDDEV((price.high - price.close))
    price['priceRange'] = (price.high - price.close)

    return price


def annotateBuySell(price, AnnotateType=""):
    global fig
    arr = []
    for index, row in price.iterrows():
        if AnnotateType == "Buy":
            arr.append(dict(x=index,y=row['low']-1, xref='x',yref='y', ax=0, ay=35,
                                       showarrow=True, arrowhead=1,arrowsize=0.5, arrowwidth=7,arrowcolor='green'))
        elif AnnotateType == "Sell":
            arr.append(dict(x=index,y=row['high']+1, xref='x',yref='y', ax=0, ay=-35,
                                       showarrow=True, arrowhead=1,arrowsize=0.5, arrowwidth=7,arrowcolor='red'))
        else:
            arr.append(dict(x=index,y=row['low']-1, xref='x',yref='y', ax=0, ay=50, text=row["text"],
                                       showarrow=True, arrowhead=1,arrowsize=1, arrowwidth=1,arrowcolor='black'))
            
    
    arr.extend(fig['layout']['annotations'])
    fig['layout']['annotations']=arr
    return price

def createPlot(symbol):
    global fig
    fig = tools.make_subplots(rows=5, cols=1, shared_xaxes=True, row_width=[1,1,3,1,5], vertical_spacing = 0.01)
    fig['layout']['xaxis'] = dict(rangeslider = dict(visible=False), side="bottom") #, range=[xMin,xMax])
    fig['layout'].update(height=950, plot_bgcolor='rgba(0,0,0,0)', title="Charts for "+symbol)
    #fig['layout']['yaxis']['range'] = [yMin, yMax]
    fig['layout']['yaxis']['anchor'] = 'x'
    fig['layout']['yaxis']['side'] = 'right'
    
    fig['layout']['xaxis']['rangeselector'] = dict(
                buttons=list([dict(count=1, label='1h', step='hour', stepmode='backward'),
                              dict(count=3, label='3h', step='hour', stepmode='backward'),
                              dict(count=6, label='1d', step='hour', stepmode='backward'),
                              dict(step='all')]))
    return fig

def plotData(filename="analysis", auto_open=True, inline = True):
    global fig
    init_notebook_mode(connected=True)
    if inline:
        iplot(fig, filename="plot/"+filename+toDate.strftime("_%d_%m_%y")+".html")
    else:
        plot(fig, filename="plot/"+filename+toDate.strftime("_%d_%m_%y")+".html", auto_open=auto_open)


#%%
def plot_set1(temp_data): 
    global fig, noofcandles
    noofcandles = 180
    yMin = temp_data.iloc[-1*noofcandles:-1]['low'].min()-10
    yMax = temp_data.iloc[-1*noofcandles:-1]['high'].max()

    xMin = temp_data.index[-1*noofcandles]
    xMax = temp_data.index[-1]
    fig['layout']['yaxis']['range'] = [yMin, yMax]
    fig['layout']['xaxis']['range'] = [xMin, xMax]

    temp_data = candlestick(temp_data,1,True)
    temp_data = bbands(temp_data,1, True)

    temp_data = macd(temp_data,3,True)
    temp_data = rsi(temp_data,4, True)
    #temp_data = aroon(temp_data,5, True)
    temp_data = stoch(temp_data,5, True)
    temp_data = sma(temp_data,1,True)
    obv(temp_data,2, True)



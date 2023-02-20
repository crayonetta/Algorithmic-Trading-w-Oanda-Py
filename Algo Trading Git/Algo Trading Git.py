#!/usr/bin/env python
# coding: utf-8

# In[1]:


#using Yahoo Finaance as our free API
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


#Importing today's date
from datetime import datetime
today = datetime.today().strftime('%Y-%m-%d')
print(today)


# In[3]:


Apple = yf.download("AAPL", start = "2018-01-01", end = today )


# In[4]:


Apple


# In[5]:


#downlaod ticker stocks from yahoo finance 
ticker=["SPY","AAPL","MSFT","GOOGL"]
stocks = yf.download(ticker, start = "2018-01-01", end = today)


# In[6]:


#exploring data and organizing data from yahoo finance


# In[7]:


stocks.head()


# In[8]:


stocks.tail()


# In[9]:


stocks.info()


# In[10]:


stocks.to_csv("stocksALGOTRADING.csv")


# In[11]:


#parsing our data into a csv file 
stocks = pd.read_csv("stocksALGOTRADING.csv",header=[0,1],index_col=[0],parse_dates=[0])
stocks


# In[12]:


stocks.columns


# In[13]:


#convert mult index to one tuple
stocks.columns=stocks.columns.to_flat_index()
stocks.columns
stocks.columns=pd.MultiIndex.from_tuples(stocks.columns)
stocks


# In[14]:


stocks.describe()


# In[15]:


close=stocks.loc[:,"Close"].copy()
close


# In[16]:


#plotting closing price of Apple, Google, Microsoft and SNP500
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn")
close.plot(figsize=(15,8),fontsize=12)
plt.legend(fontsize=12)
plt.show()


# In[17]:


# Load the necessary packages and modules
import matplotlib.pyplot as plt

# Simple Moving Average 
def SMA(data, ndays): 
    SMA = pd.Series(data['Close'].rolling(ndays).mean(), name = 'SMA') 
    data = data.join(SMA) 
    return data

# Exponentially-weighted Moving Average 
def EWMA(data, ndays): 
    EMA = pd.Series(data['Close'].ewm(span = ndays, min_periods = ndays - 1).mean(), 
                 name = 'EWMA_' + str(ndays)) 
    data = data.join(EMA) 
    return data

# Retrieve the Goolge stock data from Yahoo finance
data = yf.download('GOOGL', start="2018-01-01", end = today )
close = data['Close']

# Compute the 50-day SMA
n = 50
SMA = SMA(data,n)
SMA = SMA.dropna()
SMA = SMA['SMA']

# Compute the 200-day EWMA
ew = 200
EWMA = EWMA(data,ew)
EWMA = EWMA.dropna()
EWMA = EWMA['EWMA_200']

# Plotting the Google stock Price Series chart and Moving Averages below
plt.figure(figsize=(10,7))

# Set the title and axis labels
plt.title('Moving Average')
plt.xlabel('Date')
plt.ylabel('Price')

# Plot close price and moving averages
plt.plot(data['Close'],lw=1, label='Close Price')
plt.plot(SMA,'g',lw=1, label='50-day SMA')
plt.plot(EWMA,'r', lw=1, label='200-day EMA')

# Add a legend to the axis
plt.legend()

plt.show()


# In[18]:


# Load the necessary packages and modules
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Compute the Bollinger Bands 
def BBANDS(data, window=n):
    MA = data.Close.rolling(window=n).mean()
    SD = data.Close.rolling(window=n).std()
    data['MiddleBand'] = MA
    data['UpperBand'] = MA + (2 * SD) 
    data['LowerBand'] = MA - (2 * SD)
    return data
 
# Retrieve the Goolge stock data from Yahoo finance
data = yf.download('GOOGL', start="2018-01-01", end = today )

# Compute the Bollinger Bands for Google using the 50-day Moving average
n = 50
BBANDS = BBANDS(data, n)

# Create the plot
# pd.concat([BBANDS.Close, BBANDS.UpperBB, BBANDS.LowerBB],axis=1).plot(figsize=(9,5),)

plt.figure(figsize=(10,7))

# Set the title and axis labels
plt.title('Bollinger Bands')
plt.xlabel('Date')
plt.ylabel('Price')

plt.plot(BBANDS.Close,lw=1, label='Close Price')
plt.plot(data['UpperBand'],'g',lw=1, label='Upper band')
plt.plot(data['MiddleBand'],'r',lw=1, label='Middle band')
plt.plot(data['LowerBand'],'g', lw=1, label='Lower band')

# Add a legend to the axis
plt.legend()

plt.show()


# In[19]:


data.iloc[:,:]
#dataF.Open.iloc


# In[20]:


#defining signal function
#engulfing method: stock must open at a lower price on Day 2 than it closed at on Day 1 [hence iloc(-1),(-2)]
#we can change the signal generator based on the factors that we find important such as volume
def signal_generator(df):
    open = df.Open.iloc[-1]
    close = df.Close.iloc[-1]
    previous_open = df.Open.iloc[-2]
    previous_close = df.Close.iloc[-2]
    
    # Bearish Pattern (can be defined by us)
    if (open>close and 
    previous_open<previous_close and 
    close<previous_open and
    open>=previous_close):
        return 1

    # Bullish Pattern
    elif (open<close and 
        previous_open>previous_close and 
        close>previous_open and
        open<=previous_close):
        return 2
    
    # No clear pattern
    else:
        return 0

signal = []
signal.append(0)
for i in range(1,len(data)):
    df = data[i-1:i+1]
    signal.append(signal_generator(df))
#signal_generator(data)
data["signal"] = signal


# In[21]:


data.signal.value_counts()
#how to make it sort into 0,1,2
#dataF.iloc[:, :]


# In[22]:


from apscheduler.schedulers.blocking import BlockingScheduler
from oandapyV20 import API
import oandapyV20.endpoints.orders as orders
from oandapyV20.contrib.requests import MarketOrderRequest
from oanda_candles import Pair, Gran, CandleClient
from oandapyV20.contrib.requests import TakeProfitDetails, StopLossDetails


# In[23]:


# from config import access_token, accountID, OBTAIN OANDA V20 TOKEN AND ALLOW ACCESS TO OANDA PUBLIC ACCOUNT 
access_token='12345678900987654321-abc34135acde13f13530' #a sample token has been input

def get_candles(n):
    #you need token here generated from OANDA account
    client = CandleClient(access_token, real=False)
    collector = client.get_collector(Pair.EUR_USD, Gran.M15)
    candles = collector.grab(n)
    return candles

candles = get_candles(3)
for candle in candles:
    print(float(str(candle.bid.o))>1)


# In[24]:


def trading_job():
    candles = get_candles(3)
    dfstream = pd.DataFrame(columns=['Open','Close','High','Low'])
    
    i=0
    for candle in candles:
        dfstream.loc[i, ['Open']] = float(str(candle.bid.o))
        dfstream.loc[i, ['Close']] = float(str(candle.bid.c))
        dfstream.loc[i, ['High']] = float(str(candle.bid.h))
        dfstream.loc[i, ['Low']] = float(str(candle.bid.l))
        i=i+1

    dfstream['Open'] = dfstream['Open'].astype(float)
    dfstream['Close'] = dfstream['Close'].astype(float)
    dfstream['High'] = dfstream['High'].astype(float)
    dfstream['Low'] = dfstream['Low'].astype(float)

    signal = signal_generator(dfstream.iloc[:-1,:])#
    
    # EXECUTING ORDERS
    accountID = "101-003-24710842-001" 
    client = API(access_token)
         
    SLTPRatio = 2.
    previous_candleR = abs(dfstream['High'].iloc[-2]-dfstream['Low'].iloc[-2])
    
    SLBuy = float(str(candle.bid.o))-previous_candleR
    SLSell = float(str(candle.bid.o))+previous_candleR

    TPBuy = float(str(candle.bid.o))+previous_candleR*SLTPRatio
    TPSell = float(str(candle.bid.o))-previous_candleR*SLTPRatio
    
    print(dfstream.iloc[:-1,:])
    print(TPBuy, "  ", SLBuy, "  ", TPSell, "  ", SLSell)
    signal = 2
    
    #Sell at the peak before the bear drop
    if signal == 1:
        mo = MarketOrderRequest(instrument="EUR_USD", units=-1000, takeProfitOnFill=TakeProfitDetails(price=TPSell).data, stopLossOnFill=StopLossDetails(price=SLSell).data)
        r = orders.OrderCreate(accountID, data=mo.data)
        rv = client.request(r)
        print(rv)
        
    #Buy at the through before the bull increase
    elif signal == 2:
        mo = MarketOrderRequest(instrument="EUR_USD", units=1000, takeProfitOnFill=TakeProfitDetails(price=TPBuy).data, stopLossOnFill=StopLossDetails(price=SLBuy).data)
        r = orders.OrderCreate(accountID, data=mo.data)
        rv = client.request(r)
        print(rv)


# In[25]:


trading_job()

#scheduler = BlockingScheduler()
#scheduler.add_job(trading_job, 'cron', day_of_week='mon-fri', hour='00-23', minute='1,16,31,46', start_date='2022-01-12 12:00:00', timezone='America/Chicago')
#scheduler.start()


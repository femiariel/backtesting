# Import necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import datetime 
from pandas_datareader import data as pdr
import pandas_ta as ta
import seaborn as sns
import yfinance as yf

# data = dataframe in pandas format to calculate sma
# sma_value = is an array to calcuate the lenth of sma 
# variable = variable used to calculate sma, it can be a "Close", or "Adj Close" in string format
def sma(data,sma_value,variable):
     # auto assign a variable name based on length of sma 
    variable_name = 'SMA'+ str(sma_value)
    # calculate sma 
    data[variable_name] =ta.sma(data[variable], length=sma_value)
    # return sma variable calculated 
    return data[variable_name]

# data = dataframe in pandas format to calculate ema
# ema_value = is an array to calcuate the lenth of ema 
# variable = variable used to calculate sma, it can be a "Close", or "Adj Close" in string format
def ema(data,ema_value,variable):
    # auto assign a variable name based on length of ema 
    variable_name = 'EMA'+ str(ema_value)
    # calculate ema 
    data[variable_name] = data[variable].ewm(span=ema_value, adjust=False).mean()
     # return ema variable calculated 
    return data[variable_name]

def getdata(sym,sma_value,ema_value,close):
    yf.pdr_override()
    today = datetime.datetime.now()
    df = pdr.get_data_yahoo(sym, start=datetime.datetime(today.year-5,today.month, 1),
                                    end=datetime.datetime(today.year, today.month, today.day)).reset_index()
    df['ticker'] =sym
        
    for i in sma_value:
        variable_name = 'SMA'+ str(i)
        df[variable_name] = sma(df,i,close)
    
    for i in ema_value:
        variable_name = 'EMA'+ str(i)
        df[variable_name] = ema(df,i,close)
        
    df['Bearish_engulfing'] = np.nan
    df['Bearish swing'] = np.nan
    df['Bearish pinbar'] = np.nan
    for i in range(2,df.shape[0]):
        current = df.iloc[i,:]
        prev = df.iloc[i-1,:]
        prev_2 = df.iloc[i-2,:]
        realbody = abs(current['Open'] - current['Close'])
        candle_range = current['High'] - current['Low']
        idx = df.index[i]
        df.loc[idx,'Bearish_engulfing'] = current['High'] > prev['High'] and current['Low'] < prev['Low'] and realbody >= 0.8 * candle_range and current['Close'] < current['Open']
        df.loc[idx,'Bearish swing'] = current['High'] < prev['High'] and prev['High'] > prev_2['High']
        df.loc[idx,'Bearish pinbar'] = realbody <= candle_range/3 and max(current['Open'] , current['Close']) < (current['High'] + current['Low'])/2 and current['High'] > prev['High']        
    return df


# buy when price > SMA120 and SMA10>EMA10, annual return> 20 and (RSI <70 or RSI >30)
# sell when sma10 < ema10 or bearish engulfing candles or annual return < 20 or price< sma120
############input###########
#data = dataframe with variable needed
#sma_selected, cross over sma selected
#ema_selected, cross over ema selected
def Generate_signal(data, sma_selected,ema_selected):    
    #table start
    df_init =  data.copy()
    
    '''To make sure row arrange in an ascending order'''
    df.sort_values(by = 'Date',inplace=True)
    
    '''Process data by remove Null value in moving average varaible that your want to used for triggered''' 
    df_init.dropna(subset=[ema_selected,sma_selected],inplace=True)
    '''generate as 1 if Adj close is higher than variable you want to test, generate as -1 if Adj close is less than the
    variable you want to test in a signal variable  
    '''
    # create a signal variable
    df_init['signal'] = np.nan 
    # trigger a buy only if it is a up trend 
    df_init.loc[(df_init[sma_selected]>=df_init[ema_selected]) &(df_init['Price']>df_init['SMA120'])&(df_init['Return']>20)&((df_init['RSI']<70)|(df_init['RSI']>30)),'signal'] = 1
    
    # just triggered sell as close is less than variable triggered
    df_init.loc[((df_init[sma_selected]<df_init[ema_selected]) )| (df_init['Bearish_engulfing']==True)|(df_init['Return']<20)|(df_init['Price']<=df_init['SMA120']),'signal'] = -1
    
    # if close > variable triggered but sma20<= sma 200, let it be do nothing
    df_init['signal'].fillna(0,inplace=True) 
    
    ''' move the signal of today to tmr, thus, we need to define a shift(1), as the signal buy is based on yesterday'''
    df_init['signal'] = df_init['signal'].shift(1)
    
    '''after we have a buy and sell signal, lets create a hold signal which is equal to 2 after a buy signal and
    a do nothing signal which is equal to 0 after a sell signal'''
    
    # In order to make a decision on a hold or do nothing signal based on yesterday signal we need to create a for loop 
    # a calendar date should be created to used for a a for loop 
    date_list = df_init.Date.unique()
    
    # declare a first day, the first date is a null in signal as there is no yesterday data for first data
    first_date = date_list[0]
    
    # declare a previous day 
    prev = first_date 
    
    for i in date_list:
        if i == first_date:
            df_init.loc[df_init.Date == i,'signal'] = 0 
        else:
            # if you have a sell signal yesterday, you should have a do nothing signal today
            if (df_init.loc[df_init.Date == prev,'signal'].values[0]==-1):
                df_init.loc[df_init.Date == i,'signal']=0 
            # if you have a buy signal yesterday, you should have a hold signal today
            elif (df_init.loc[df_init.Date == prev,'signal'].values[0]==1):
                df_init.loc[df_init.Date == i,'signal'] = 2
              
            # if you have a hold signal yesterday and you do not have a sell signal today, you should have a hold signal today
            elif ((df_init.loc[df_init.Date == prev,'signal'].values[0]==2)&(df_init.loc[df_init.Date == i,'signal'].values[0]!=-1) ):
                df_init.loc[df_init.Date == i,'signal'] = 2
                
            # if you have a do nothing signal yesterday and today is a sell signal, you should equal to have a do nothing signal
            elif ((df_init.loc[df_init.Date == prev,'signal'].values[0]==0)&(df_init.loc[df_init.Date == i,'signal'].values[0]==-1) ):
                df_init.loc[df_init.Date == i,'signal'] = 0
                
        # redeclare your previous date before to next day in for loop
        prev = i
    
    return df_init


# data with signal for all ticker
# capital is your starting capital
def backtest_strategy_portfolio(data,capital):   
    
    #table start
    df_init =  data.copy()
    df_init.set_index('Date',inplace=True) 
    #stock_pick_df = stock_pick_data.copy()
    
    
    #assign dummy row - day before the trade
    start_date = pd.DataFrame(columns=data.columns,index=[df_init.index.min()- datetime.timedelta(days=1)])
    df_init = pd.concat([df_init, start_date])

    
    #initiat 2 variable: cash and units which represent each status of cash and units of apple holding for each day
    df_init = df_init.assign(cash=np.nan,units = 0)
    
    #assign capital for first dummy day assigned to be capital defined in function
    df_init.loc[pd.Series(df_init.index.min()), 'cash'] = capital
    
    # obtain list of calendar 
    calendar = pd.Series(df_init[df_init['ticker']!='spy'].index.sort_values().unique()).iloc[1:]
    #sp_calendar = pd.Series(stock_pick_df.index.sort_values().unique())
    i=0

    for date in calendar:
        
        #emp = stock_pick_df[(stock_pick_df['Return_without_trailing']>=stock_pick_df['hold_without_sell_return (%)'])&(stock_pick_df['Return_trailing_12m']>100)&(stock_pick_df['Return_without_trailing']>120)&(stock_pick_df.index==date)].sort_values(by='Return_without_trailing',ascending=False)[:10]
        #temp = stock_pick_df.sort_values(by='RSI',ascending=False)[:10]
        #temp = stock_pick_df[(stock_pick_df['Return_trailing_12m']>120)&(stock_pick_df.index==date)].sort_values(by='Return_trailing_12m_hold',ascending=False)[:5]
        stock_pick = df_init[(df_init.index==date)].ticker.unique() 


        #get yesterday data
        prev_date = df_init.index[df_init.index<date].unique().sort_values()[-1]
        
        
        # calculate total stock value of yesterday 
        total_stock_holding=[]
        cash = []
        for stock in  df_init.loc[(df_init.index==prev_date) &((df_init.signal==1)|(df_init.signal==2))].ticker.unique():
            stock_holding = df_init.loc[(df_init.index==prev_date)&(df_init['ticker']==stock), 'units'].values[0]*df_init.loc[(df_init.index==date)&(df_init['ticker']==stock),'Price'].values[0]
            total_stock_holding.append(stock_holding)
            cash_value = df_init.loc[(df_init.index==prev_date)&(df_init['ticker']==stock), 'cash'].values[0]
            cash.append(cash_value)
        
        # total portfolio value by add cash and stock value of yesterday 
        port_value = sum(total_stock_holding) + df_init.loc[prev_date, 'cash'].sum()

        # reallocation to each stock
        if len(df_init.loc[(df_init.index==date) &((df_init.signal==1)|(df_init.signal==2))].ticker.unique())==0:
            number_stock =1 
            df_init.loc[((df_init.index==date)&(df_init['ticker']=='spy')), 'signal'] = 1
            stock_pick = np.append(stock_pick,'spy')
            
        else: 
            number_stock = len(df_init.loc[(df_init.index==date) &((df_init.signal==1)|(df_init.signal==2))& (df_init.ticker.isin(stock_pick))].ticker.unique())

        allocation_each = port_value/number_stock
        print(date)
        print(port_value)
        print(df_init.loc[(df_init.index==date) &((df_init.signal==1)|(df_init.signal==2))].ticker.unique())
        
        
        for stock in stock_pick:
            # if signal is do nothing or sell, mean our cash = portfolio value and units=0
            
            if ((df_init.loc[(df_init.index==date)&(df_init['ticker']==stock), 'signal'].values[0] == 0)|(df_init.loc[(df_init.index==date)&(df_init['ticker']==stock), 'signal'].values[0] == -1)):          
                df_init.loc[(df_init.index==date)&(df_init['ticker']==stock), 'units'] = 0
                df_init.loc[(df_init.index==date)&(df_init['ticker']==stock), 'cash'] = 0
            #if we have a buy signal 
            #start to calculate the trade
            #we start to calculate start_cap which represent the starting capital for each trade
            #unit_buy is total unit buy based on port_value available
            elif ((df_init.loc[(df_init.index==date)&(df_init['ticker']==stock), 'signal'].values[0] == 1)|(df_init.loc[(df_init.index==date)&(df_init['ticker']==stock), 'signal'].values[0] == 2)):
                unit_buy = allocation_each /df_init.loc[((df_init.index==date)&(df_init['ticker']==stock)), 'Price'].values[0]
                df_init.loc[(df_init.index==date)&(df_init['ticker']==stock), 'units'] = unit_buy
                df_init.loc[(df_init.index==date)&(df_init['ticker']==stock), 'cash'] = allocation_each - unit_buy*df_init.loc[((df_init.index==date)&(df_init['ticker']==stock)), 'Price'].values[0]
        
        i=i+1
    # calculate current value of the strategy, the formula = unit holding * Price + cash available
    df_init['Total_value_todate'] = df_init['units']*df_init['Price'] + df_init['cash'] 
    
    #remove dummy rows
    df_init.drop(df_init[df_init.index == df_init.index.min()].index,axis=0,inplace=True)
  
    # get summarize of total portfolio value, return by date, benchmark_index
    total_port_value = pd.DataFrame(df_init.groupby([df_init.index])['Total_value_todate'].sum())
    total_port_value['Return_without_trailing'] = total_port_value['Total_value_todate']/total_port_value['Total_value_todate'].iloc[0] *100
    total_port_value['Return_trailing_12m'] = total_port_value['Total_value_todate']/total_port_value['Total_value_todate'].shift(12)*100

    return total_port_value, df_init


# annualized return 
# df =data with return 
# variable = variable name of return
def annualized(df,variable):
    days_held =df.shape[0]
    Return = (df.iloc[-1][variable] - df.iloc[0][variable])/df.iloc[0][variable]
    ar = ((1+Return) ** (365/days_held))-1
    # get annualized in %
    return ar*100

# maximum drawdown 
# df =data with return 
# variable = variable name of return
def MDD(df,variable):
    window = 252
    Roll_Max = df[variable].rolling(window, min_periods=1).max()
    Drawdown = df[variable]/Roll_Max - 1.0
    mdd = Drawdown.min()
    # get drawdown in %
    return mdd*100


# Function to calculate RSI
def RSI(prices, n=14):
  # Get the difference in price from previous step
    delta = prices.diff()

  # Get rid of the first row, which is NaN since it did not have a previous
  # row to calculate the differences
    delta = delta[1:]

  # Make the positive gains (up) and negative gains (down) Series
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

  # Calculate the SMA
    roll_up = up.rolling(n).mean()
    roll_down = down.abs().rolling(n).mean()

  # Calculate the RSI based on SMA
    RS = roll_up / roll_down
    RSI = 100.0 - (100.0 / (1.0 + RS))

    return RSI

#Function to calculate return
def Calculate_Return (df): 
    
    df['12_months_date'] = pd.to_datetime(df.Date) +  pd.DateOffset(months=12) 
    
    df_12_ago = df[['ticker','Date','12_months_date','Close']].rename(
    columns={'Close':'df_ago','Date':'Date_12_ago','12_months_date':'Date'})
    
    df_uptodate = df.rename(
    columns={'Close':'Price'})
    
    df_12 = df_12_ago.merge(df_uptodate,how = 'left', on=['ticker','Date'])
    df_12.drop(df_12[df_12.Price.isna()].index,axis=0)
    df_12.set_index('Date',inplace=True) 
    
    df_12['Return'] = ((df_12['Price'] - df_12['df_ago']) /  df_12['df_ago'])*100
    
    return df_12


table=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
df = table[0]
ticker = df['Symbol'].values
result_data = pd.DataFrame()

ticker=["AAPL","AMZN","AMD"]

summary = pd.DataFrame()
all_signal = pd.DataFrame()
for i in ticker:
    # Read in the data
    try:
        print(i)
        df = getdata(i,sma_value = [5,10,15,20,50,60,120,150,200],ema_value = [5,10,15,20,50,60,120,150,200],close = "Close")
        if df is not None:
            df['RSI'] = RSI(df['Close'])
            df =Calculate_Return (df)
            df.reset_index(inplace=True)
            df_signal = Generate_signal(df,'SMA10','EMA10')
            df_signal['Price']=df_signal['Open']
            df_signal['ticker']=i
            all_signal = pd.concat([all_signal,df_signal])
            #summary = pd.concat([result,summary])
            df= None
    except:
        pass # doing nothing on exception


spy = getdata('spy',sma_value = [20,50,200],ema_value =[20,50,200],close = "Close")
spy['signal'] = 0 
spy['ticker'] = 'spy'
spy['Price']=spy['Open']
spy =spy[spy['Date']>=all_signal.Date.min()]
spy =spy[spy['Date'].isin(all_signal.Date.unique())]
signal_etf_stock = pd.concat([spy,all_signal])

result, hold_detail = backtest_strategy_portfolio(signal_etf_stock[signal_etf_stock.Date>'2014-11-12'],10000)

spy['benchmark-spy(%)'] = spy['Close']/spy['Close'].iloc[0]*100
result['benchmark-spy(%)'] =  spy.set_index('Date')['benchmark-spy(%)']


# plot the return and calculate the annualized return and MDD (maximum draw down) for both strategy
plt.rcParams["figure.figsize"] = (15,6)
plt.plot(result.Return_without_trailing, label='Return without Annualized_return of Portfolio based on EMA cross-over strategy')
plt.plot(result['benchmark-spy(%)'], label='benchmark-spy(%)')
plt.legend()
plt.show()
print('Annualized_return of Portfolio based on EMA cross-over strategy:',annualized (result,'Return_without_trailing'))
print('MDD of Portfolio based on EMA cross-over strategy:', MDD(result,'Return_without_trailing'))

print('Annualized_return of benchmark-spy:',annualized (result,'benchmark-spy(%)'))
print('MDD of benchmark-spy:', MDD(result,'benchmark-spy(%)'))
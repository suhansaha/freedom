#%% [markdown]
# ## <span style="color:grey">Initialization</span>

#%%
#get_ipython().run_line_magic('run', '"KiteConnect_AlgoBase.ipynb"')

from Freedom_AlgoBase import *

#%% [markdown]
# ##  <span style="color:green">Trading Strategy </span>

#%%


if mode == "Algo":
    x = 'ICICIBANK'
    gc.collect()
    days=1
    portfolio = pd.DataFrame([x])
    toDate = dt.datetime(2019,5,2,15,15)
    #portfolio = pd.DataFrame(downloadlist)    
    print("\nPortfolio return: "+ '%.2f'%trade_simulator(portfolio, toDate, cd=False, plot=True)+"%")
    


#%%
# Backtesting for no of days

if mode == "Algo":
    x = 'YESBANK'
    gc.collect()
    days=2
    portfolio = pd.DataFrame([x])
    
    noofdays = 180
    for i in np.linspace(noofdays,1,noofdays):
        toDate = dt.datetime(2019,5,3,15,15) - dt.timedelta(days=i)
        #print(toDate)
        if isholiday(toDate) == True:
            continue
        print(toDate)
        #portfolio = pd.DataFrame(downloadlist)    
        trade_simulator(portfolio, toDate, cd=False, plot=False)
    
    


#%%
# Return on a portfolio of stocks
if mode == "Algo":
    gc.collect()
    days =2
    toDate = dt.datetime(2019,5,3,15,15)
    portfolio = pd.DataFrame(downloadlist.drop(24))    
    print("\nPortfolio return: "+ '%.2f'%trade_simulator(portfolio, toDate, cd=False, plot=False)+"%")




#%%
# Backtesting of a portfolio of stocks for 180 days
if mode == "Algo":
    gc.collect()
    days =2
    toDate = dt.datetime(2019,5,3,15,15)
    #portfolio = pd.DataFrame(downloadlist.drop(24))    

    cumsum = 0
    for x in downloadlist.drop(24):
        #print(x)
        portfolio = pd.DataFrame([x])

        noofdays = 180

        cumsum2 = 0
        for i in np.linspace(noofdays,1,noofdays):
            toDate = dt.datetime(2019,5,3,15,15) - dt.timedelta(days=i)
            #print(toDate)
            if isholiday(toDate) == True:
                continue
            #print(toDate)
            #portfolio = pd.DataFrame(downloadlist)    
            cumsum2 = cumsum2 + trade_simulator(portfolio, toDate, cd=False, plot=False, prn=False)

        print("\n"+x+" return: "+ '%.2f'%cumsum2+"%")
        cumsum = cumsum + cumsum2
        
    
    print("\nPortfolio return: "+ '%.2f'%cumsum+"%")

    
    




#%%
#Using real simulator 
if mode == "Algo":
    x = 'INFRATEL'
    gc.collect()
    portfolio = pd.DataFrame([x])
    toDate = dt.datetime(2019,4,30,15,15)
    #portfolio = pd.DataFrame(downloadlist)    
    print("\nPortfolio return: "+ '%.2f'%trade_simulator(portfolio, toDate, mode="simulator")+"%")






#%%

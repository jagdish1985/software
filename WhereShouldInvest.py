# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 14:44:29 2020

@author: yh10
"""

###Where should invest?
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import quandl
import scipy.optimize as sco

plt.style.use('fivethirtyeight')
np.random.seed(777)

%matplotlib inline
%config InlineBackend.figure_format = 'retina'


quandl.ApiConfig.api_key = 'your_api_key_here'
#stocks = ['AAPL','AMZN','GOOGL','FB']
#stocks = ['ITC.NS','IRCTC.NS','SPICEJET.NS','BEML.NS']

stocks = ['0P0000XVUH.BO','0P0000XVTR.BO','0P0001BURC.BO','0P0001J414.BO']
          '0P0000XVTL.BO','0P0000XVU7.BO','0P00011MAV.BO','0P0001EP9Q.BO']
#0P0000XVUH.BO, 0P0000XVTR.BO, 0P0001BURC.BO,0P0001J414.BO,0P0000XVTL.BO,0P0000XVU7.BO,0P00011MAV.BO,0P0001EP9Q.BO

data = quandl.get_table('WIKI/PRICES', ticker = stocks,
                        qopts = { 'columns': ['date', 'ticker', 'adj_close'] },
                        date = { 'gte': '2019-11-1', 'lte': '2020-04-30' }, 
                        paginate=True)
data.head()

data.info()
df =data.set_index('date')
table = df.pivot(columns = 'ticker')
#By specifying col[1] in below list comprehension
#you can select the stock names under multi-level column
table.columns = [col[1]] for col in table.columns]
table.head()
#####
import ffn
df = ffn.get('ITC.NS,IRCTC.NS,SPICEJET.NS,BEML.NS',start = '2019-11-01', end = '2020-04-30')
df = df.reset_index()
df = df.set_index('Date')
df.info()
df.head()

##Ploting 
plt.figure(figsize=(14, 7))
for c in df.columns.values:
    plt.plot(df.index, df[c], lw=3, alpha=0.8,label=c)
plt.legend(loc='upper left', fontsize=12)
plt.ylabel('price in $')


##Plotting volatility plot
returns = df.pct_change()

plt.figure(figsize = (14,7))
for c in returns.columns.values:
    plt.plot(df.index, returns[c], lw=3, alpha =0.8, label = c)
plt.legend(loc = 'upper left', fontsize =12)
plt.ylabel('Daily returns')

###Random Portfolio 
def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns*weights ) *252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns
  
def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    results = np.zeros((3,num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(4)
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
        results[0,i] = portfolio_std_dev
        results[1,i] = portfolio_return
        results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return results, weights_record

table = df.copy()
returns = table.pct_change()
mean_returns = returns.mean()
cov_matrix = returns.cov()
num_portfolios = 25000
risk_free_rate = 0.0178

def display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate):
    results, weights = random_portfolios(num_portfolios,mean_returns, cov_matrix, risk_free_rate)
    
    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[0,max_sharpe_idx], results[1,max_sharpe_idx]
    max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx],index=table.columns,columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T
    
    min_vol_idx = np.argmin(results[0])
    sdp_min, rp_min = results[0,min_vol_idx], results[1,min_vol_idx]
    min_vol_allocation = pd.DataFrame(weights[min_vol_idx],index=table.columns,columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T
    
    #print "-"*80
    print("Maximum Sharpe Ratio Portfolio Allocation")
    print('Annualised Return:%.3f' % round(rp,2))
    print('Annualised Volatility:%.3f' % round(sdp,2))
    #print "\n"
    print(max_sharpe_allocation)
    #print "-"*80
    print("Minimum Volatility Portfolio Allocation")
    print('Annualised Return:%.3f' % round(rp_min,2))
    print('Annualised Volatility: %.3f' % round(sdp_min,2))
    #print "\n"
    print(min_vol_allocation)
    
    plt.figure(figsize=(10, 7))
    plt.scatter(results[0,:],results[1,:],c=results[2,:],cmap='YlGnBu', marker='o', s=10, alpha=0.3)
    plt.colorbar()
    plt.scatter(sdp,rp,marker='*',color='r',s=500, label='Maximum Sharpe ratio')
    plt.scatter(sdp_min,rp_min,marker='*',color='g',s=500, label='Minimum volatility')
    plt.title('Simulated Portfolio Optimization based on Efficient Frontier')
    plt.xlabel('annualised volatility')
    plt.ylabel('annualised returns')
    plt.legend(labelspacing=0.8)
    
display_simulated_ef_with_random(mean_returns, cov_matrix,num_portfolios,risk_free_rate)
    
##Efficient Frontier
constraints = ({'type' : 'eq','fun' : lambda x:np.sum(x)-1})

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_var, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_var

def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(neg_sharpe_ratio, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result
def portfolio_volatility(weights, mean_returns, cov_matrix):
    return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[0]

def min_variance(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type' : 'eq', 'fun' : lambda x:np.sum(x)-1})
    bound =(0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    
    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,],args = args,
                          method = 'SLSQP', bounds = bounds, constraints = constraints)
    return result


def efficient_return(mean_returns, cov_matrix, target):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)

    def portfolio_return(weights):
        return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[1]

    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x) - target},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0,1) for asset in range(num_assets))
    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def efficient_frontier(mean_returns, cov_matrix, returns_range):
    efficients = []
    for ret in returns_range:
        efficients.append(efficient_return(mean_returns, cov_matrix, ret))
    return efficients


def display_calculated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate):
    results,_ = random_portfolios(num_portfolios,mean_returns, cov_matrix, risk_free_rate)
    
    max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
    sdp, rp = portfolio_annualised_performance(max_sharpe['x'], mean_returns, cov_matrix)
    max_sharpe_allocation = pd.DataFrame(max_sharpe.x,index=table.columns,columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T

    min_vol = min_variance(mean_returns, cov_matrix)
    sdp_min, rp_min = portfolio_annualised_performance(min_vol['x'], mean_returns, cov_matrix)
    min_vol_allocation = pd.DataFrame(min_vol.x,index=table.columns,columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T
    #print "-"*80
    print("Maximum Sharpe Ratio Portfolio Allocation/n")
    print('Annualised Return: %.3f' % round(rp,2))
    print("Annualised Volatility: %.3f" % round(sdp,2))
    print("\n")
    print(max_sharpe_allocation)
    #print "-"*80
    print("Minimum Volatility Portfolio Allocation/n")
    print("Annualised Return: %.3f" % round(rp_min,2))
    print("Annualised Volatility: %.3f" % round(sdp_min,2))
    print("\n")
    print(min_vol_allocation)
    
    plt.figure(figsize=(10, 7))
    plt.scatter(results[0,:],results[1,:],c=results[2,:],cmap='YlGnBu', marker='o', s=10, alpha=0.3)
    plt.colorbar()
    plt.scatter(sdp,rp,marker='*',color='r',s=500, label='Maximum Sharpe ratio')
    plt.scatter(sdp_min,rp_min,marker='*',color='g',s=500, label='Minimum volatility')

    target = np.linspace(rp_min, 0.32, 50)
    efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, target)
    plt.plot([p['fun'] for p in efficient_portfolios], target, linestyle='-.', color='black', label='efficient frontier')
    plt.title('Calculated Portfolio Optimization based on Efficient Frontier')
    plt.xlabel('annualised volatility')
    plt.ylabel('annualised returns')
    plt.legend(labelspacing=0.8)
    
display_calculated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate)

def display_ef_with_selected(mean_returns, cov_matrix, risk_free_rate):
    max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
    sdp, rp = portfolio_annualised_performance(max_sharpe['x'], mean_returns, cov_matrix)
    max_sharpe_allocation = pd.DataFrame(max_sharpe.x,index=table.columns,columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T

    min_vol = min_variance(mean_returns, cov_matrix)
    sdp_min, rp_min = portfolio_annualised_performance(min_vol['x'], mean_returns, cov_matrix)
    min_vol_allocation = pd.DataFrame(min_vol.x,index=table.columns,columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T
    
    an_vol = np.std(returns) * np.sqrt(252)
    an_rt = mean_returns * 252
    
    #print("-"*80)
    print("Maximum Sharpe Ratio Portfolio Allocation\n")
    print('Annualised Return: %.3f' % round(rp,2))
    print("Annualised Volatility: %.3f" % round(sdp,2))
    #print("\n")
    print(max_sharpe_allocation)
    #print "-"*80
    print("Minimum Volatility Portfolio Allocation\n")
    print("Annualised Return: %.3f" % round(rp_min,2))
    print("Annualised Volatility: %.3f" % round(sdp_min,2))
    #print("\n")
    print(min_vol_allocation)
    #print("-"*80)
    print("Individual Stock Returns and Volatility\n")
    for i, txt in enumerate(table.columns):
        print(txt,":","annuaised return %.3f" % round(an_rt[i],2),",annualised volatility: %.3f" % round(an_vol[i],2))
    print("-"*80)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(an_vol,an_rt,marker='o',s=200)

    for i, txt in enumerate(table.columns):
        ax.annotate(txt, (an_vol[i],an_rt[i]), xytext=(10,0), textcoords='offset points')
    ax.scatter(sdp,rp,marker='*',color='r',s=500, label='Maximum Sharpe ratio')
    ax.scatter(sdp_min,rp_min,marker='*',color='g',s=500, label='Minimum volatility')

    target = np.linspace(rp_min, 0.34, 50)
    efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, target)
    ax.plot([p['fun'] for p in efficient_portfolios], target, linestyle='-.', color='black', label='efficient frontier')
    ax.set_title('Portfolio Optimization with Individual Stocks')
    ax.set_xlabel('annualised volatility')
    ax.set_ylabel('annualised returns')
    ax.legend(labelspacing=0.8)

display_ef_with_selected(mean_returns, cov_matrix, risk_free_rate)



###########################################################################
## Case 6 Monto Carlo model
###https://medium.com/python-data/efficient-frontier-portfolio-optimization-with-python-part-2-2-2fe23413ad94
###########################################################################
#import relevant libraries
import pandas as pd
import numpy as np
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import os
os.chdir("C:\\Users\\yh10\\Desktop\\Jagdish\\optimizationtechniq")

##Fetch data from yahoo and save under dataframe
##select = ['CNP', 'F', 'WMT', 'GE','TSLA']
select = ['SBIN.BO','ITC.NS','ICICIBANK.NS','HDFCBANK.NS','IRCTC.NS','SPICEJET.NS','BEML.NS']

data = web.DataReader(select, data_source = "yahoo", 
                      start = '2019-11-1', end = '2020-04-30')['Adj Close']

table = data

#Calculate daily and annual returns of the stocks
returns_daily = data.pct_change()
returns_annual = returns_daily.mean() * 250

##Get daily and covraince of returns of the stock
cov_daily = returns_daily.cov()
cov_annual = cov_daily * 250

##Empty lists to store returns, volatility and weights of imiginary portfolios
port_returns = []
port_volatility = []
sharpe_ratio = []
stock_weights = []

##set the number of combinations for imaginary portfolios
num_assets = len(select)
num_portfolios = 50000

##set random seed for reproduction's sake
np.random.seed(101)

#Populate the empty lists with each portfolios returns, risk and weights
for single_portfolio in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    returns = np.dot(weights, returns_annual)
    volatility = np.sqrt(np.dot(weights.T,np.dot(cov_annual, weights)))
    sharpe = returns / volatility
    sharpe_ratio.append(sharpe)
    port_returns.append(returns)
    port_volatility.append(volatility)
    stock_weights.append(weights)

##a dictionary for returns and Risk values of each portfolio
portfolio = {'Returns' : port_returns,
             'Volatility' : port_volatility,
             'Sharpe Ratio' : sharpe_ratio}

##extend original dictionary to accomdate each ticker and weights in in the portfolio
for counter, symbol in enumerate(select):
    portfolio[symbol + ' Weight'] = [Weight[counter] for Weight in stock_weights]

# make a nice dataframe of the extended dictionary
df = pd.DataFrame(portfolio)
    
##Get better labels for desired arrangement of columns
column_order = ['Returns','Volatility','Sharpe Ratio'] + [stock + ' Weight' for stock in select] 
               
##reorder dataframe columns 
df = df[column_order]

#Plot frontier max sharpe * min volatility values with a scatterplot
plt.style.use('seaborn-dark')
df.plot.scatter(x='Volatility', y = 'Returns', c = 'Sharpe Ratio',
                cmap = 'RdYlGn', edgecolors = 'black', figsize = (10,8), grid = True)
plt.xlabel('Volatility (Std. Deviation)')
plt.ylabel('Expected Returns')
plt.title('Efficient Frontier')
plt.show()
    
##find min Volatility and max sharpe value in the dataframe
min_volatility = df['Volatility'].min()
max_sharpe = df['Sharpe Ratio'].max()

##Use the min, max values to locate and create the two special portfolio
sharpe_portfolio = df.loc[df['Sharpe Ratio'] == max_sharpe]
min_variance_port = df.loc[df['Volatility'] == min_volatility]

##Plot frontier max sharpe & min volatility values with a scatterplot
plt.style.use('seaborn-dark')
df.plot.scatter(x='Volatility', y='Returns', c='Sharpe Ratio',
                cmap='RdYlGn', edgecolors='black', figsize=(10, 8), grid=True)
plt.scatter(x= sharpe_portfolio['Volatility'], y = sharpe_portfolio['Returns'], c = 'red', marker = 'D', s=200)
plt.scatter(x = min_variance_port['Volatility'], y = min_variance_port['Returns', c ='blue', markder = 'D', s= 200)
plt.xlabel('Volatility (Std. Deviation)') 
plt.ylabel('Expected Returns')
plt.title('Efficient Frontier')
plt.show()

###
print(min_variance_port.T)
print(sharpe_portfolio.T)


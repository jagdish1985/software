# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 19:46:56 2019

@author: yh10

"""
###############Case study - 1 https://www.kdnuggets.com/2019/06/optimization-python-money-risk.html
import ffn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cvxpy import *

##mp = ffn.get('msft,v,wmt', start = '2016-01-01')
mp = pd.read_csv("C:\\Users\\yh10\\Desktop\\Jagdish\\optimizationtechniq\\mp.csv",index_col =0)

##plot the data
#mp.index
#mp.dtypes

# get columns to plot
import matplotlib.pyplot as plt
columns = mp.columns
# create x data
x_data = range(0, mp.shape[0])
# create figure and axis
fig, ax = plt.subplots()
# plot each column
for column in columns:
    ax.plot(x_data, mp[column], label=column)
# set title and legend
ax.set_title('Stock Market Price')
ax.xlabel('Date')
ax.ylabel('Stock Price')
ax.legend()

############Or

fig, ax = plt.subplots(figsize=(10, 10))

# Add the x-axis and the y-axis to the plot
ax.plot(mp['Date'],
        mp['msft'],
        mp['v'],
        mp['wmt'],
        color='purple')

# Set title and labels for axes
ax.set(xlabel="Date",
       ylabel="Stock price",
       title="Stock Market Price")
ax.legend()
ax.plot()

##Compute monthly returns
mr = pd.DataFrame()
for s in mp.columns:
    date = mp.index[0]
    pr0 = mp[s][date]
    for t in range(1,len(mp.index)):
        date = mp.index[t]
        pr1 = mp[s][date]
        ret = (pr1-pr0)/pr0
        mr.set_value(date,s,ret)
        pr0=pr1
        
mr.head()

#Plot of returns
columns = mr.columns
x_data = range(0,mr.shape[0])
fig,ax = plt.subplots()
for column in columns:
    ax.plot(x_data,mr[column],label = column)
ax.set_title("Stock price return")
ax.xlabel('Date')
ax.ylabel('Returns')
ax.legend()

#Get symbol names
symbols = mr.columns
return_data = mr.as_matrix().T

#Mean Return
r = np.asarray(np.mean(return_data,axis=1))

#Covariance Matrix

C = np.asmatrix(np.cov(return_data))

##Print expected Returns and risk
for j in range(len(symbols)):
    print('%s: Exp ret = %f,Risk = %f' %(symbols[j], r[j],C[j,j]**0.5))

####Set up the optimization model
#Number of variables
  n = len(symbols)

#The variables vector
  x = Variable(n)
  
#The minimum return
  req_return = 0.02

#The Return
  ret = r.T*x

#The risk in xT.Q.x format
  risk = quad_form(x,C)
  
#The core problem definition with the problem class from CVXPY
  prob = Problem(Minimize(risk), [sum(x)==1, ret >= req_return, x >=0])
  
##Try solving the problem within a try/except loop
  try:
      prob.solve()
      print("Optimal portfolio")
      print("----------------------")
      for s in range(len(symbols)):
          print("Investment in {} : {}% of the portfolio".format(symbols[s],round(100*x.value[s],2)))
      print("-------------------")
      print("Exp ret = {}%".format(round(100*ret.value,2)))
      print("Expected Risk = {}%".format(round(100*risk.value**0.5,2)))
    except:
        print("Error")
        
prob.status

x.value
######################################Case study -2 
import ffn ##Finanical functions
#import pyfinance as ffn
from empyrical import alpha_beta
from empyrical import alpha_beta	
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import discrete_allocation
import matplotlib as pyplot
import numpy as np
import pandas as pd

prices = ffn.get('msft,aapl,amzn,fb,brk-b,jnj',start = '2016-01-01')

benchmark = ffn.get('spy',start ='2016-01-01')
msft = ffn.get('msft', start='2016-01-01')
ax = prices.rebase().plot()

returns = prices.to_returns().dropna()
ax = returns.hist(figsize=(10,10))


##Expected annualized return: 0.25
##Volatility: 0.18

##histrogram
figsize=(10,5)
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

#data = np.random.randn(1000)
plt.hist(returns['msft']);
plt.title('msft')

plt.hist(returns['amzn'])
plt.title('amzn')

plt.hist(returns['brkb'])
plt.title('brkb')

plt.hist(returns['fb'])
plt.title('fb')

plt.hist(returns['jnj'])
plt.title('jnj')

plt.hist(returns['msft'])
plt.title('msft')

##Interesting stats
stats = prices.calc_stats()
stats.display()

alpha,beta = alpha_beta(msft,benchmark)
print(beta)

##Efficient Frontier & Portfolio Optimization
returns = prices.pct_change()

#Mean daily return and covariance of daily returns
mean_daily_returns = returns.mean()
cov_matrix = returns.cov()

#Portfolio weigths
weigths = np.asarray([0.4,0.2,0.1,0.1,0.1,0.1])

portfolio_returns = round(np.sum(mean_daily_returns * weights) * 252,2)
portfolio_std_dev = round(np.sqrt(np.dot(weights.T,np.dot(cov_matrix,weights))) * np.sqrt(252),2)

print("Expected annualised return:" + str(portfolio_return))
print("Volatility:" + str(portfolio_std_dev))
#Expected annualised return:0.25
#Volatility:0.18

##Expected returns and sample covariance
mu = expected_returns.mean_historical_return(prices)
S = risk_models.sample_cov(prices)

#Optimise portfolio for maximum sharpe Ratio
ef = EfficientFrontier(mu,S)
raw_weights = ef.max_sharpe()
cleaned_weights =ef.clean_weights()
print(cleaned_weights)
ef.portfolio_performance(verbose = True)

##
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
latest_prices = get_latest_prices(prices)
da = DiscreteAllocation(cleaned_weights, latest_prices,total_portfolio_value = 20000)
allocation, leftover = da.lp_portfolio()
print(allocation)
print("Funds remaining : ${:.2f}".format(leftover))

##Efficiency
ef = EfficientFrontier(mu, S, weight_bounds=(-1, 1))
ef.efficient_return(target_return=0.2, market_neutral=True)


####################Case study 3 https://towardsdatascience.com/efficient-frontier-portfolio-optimisation-in-python-e7844051e7f
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
stocks = ['AAPL','AMZN','GOOGL','FB']
data = quandl.get_table('WIKI/PRICES', ticker = stocks,
                        qopts = { 'columns': ['date', 'ticker', 'adj_close'] },
                        date = { 'gte': '2016-1-1', 'lte': '2017-12-31' }, 
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
df = ffn.get('aapl,amzn,googl,fb',start = '2017-01-01', end = '2019-11-15')
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

############################################################Case - 4 
##Modeling and optimization of a weekly workforce with Python and Pyomo
#https://towardsdatascience.com/modeling-and-optimization-of-a-weekly-workforce-with-python-and-pyomo-29484ba065bb

from pyomo.environ import *
from pyomo.opt import SolverFactory

#Define days (1 week)
days = ['Mon','Tue', 'Wed','Thu','Fri', 'Sat','Sun']

#Enter shifts of each day
shifts = ['morning','evening', 'night'] # 3 shits for 8hours
days_shifts = {day: shifts for day in days} # dict with day as key and list of its shits as values

#Enter workers ids (name, number)
workers = ['w' + str(i) for i in range(1,11)] # 10 workers available more than needed

#Initialize model
model = ConcreteModel()

#Binary variables representing if a worker is scheduled somewhere
model.works = Var(((worker, day, shift) for worker in workers for day in days for shift in days_shifts[day]),within=Binary, initialize =0)

#Objective function
#Define an objective fucntion with model as input to pass later
def obj_rule(m):
    c = len(workers)
    return sum(m.no_pref[worker] for worker in workers) + sum(c*m.needed[worker] for worker in workers)
# we multiply the second term by a constant to make sure that it is the primary objective
# since sum(m.no_prefer) is at most len(workers), len(workers) + 1 is a valid constant.

# add objective function to the model. rule (pass function) or expr (pass expression directly)
model.obj = Objective(rule = obj_rule, sense = minimize)

##Model constraints
model.constraints = ConstraintList()  # Create a set of constraints

# Constraint: all shifts are assigned
for day in days:
    for shift in days_shifts[day]:
        if day in days[:-1] and shift in ['morning', 'evening']:
            # weekdays' and Saturdays' day shifts have exactly two workers
            model.constraints.add(  # to add a constraint to model.constraints set
                2 == sum(model.works[worker, day, shift] for worker in workers)
            )
        else:
            # Sundays' and nights' shifts have exactly one worker
            model.constraints.add(
                1 == sum(model.works[worker, day, shift] for worker in workers)
            )

# Constraint: no more than 40 hours worked
for worker in workers:
    model.constraints.add(
        40 >= sum(8 * model.works[worker, day, shift] for day in days for shift in days_shifts[day])
    )

# Constraint: rest between two shifts is of 12 hours (i.e., at least two shifts)
for worker in workers:
    for j in range(len(days)):
        # if working in morning, cannot work again that day
        model.constraints.add(
            1 >= sum(model.works[worker, days[j], shift] for shift in days_shifts[days[j]])
        )
        # if working in evening, until next evening (note that after sunday comes next monday)
        model.constraints.add(
            1 >= sum(model.works[worker, days[j], shift] for shift in ['evening', 'night']) +
            model.works[worker, days[(j + 1) % 7], 'morning']
        )
        # if working in night, until next night
        model.constraints.add(
            1 >= model.works[worker, days[j], 'night'] +
            sum(model.works[worker, days[(j + 1) % 7], shift] for shift in ['morning', 'evening'])
        )

# Constraint (def of model.needed)
for worker in workers:
    model.constraints.add(
        10000 * model.needed[worker] >= sum(model.works[worker, day, shift] for day in days for shift in days_shifts[day])
    )  # if any model.works[worker, ??, ??] non-zero, model.needed[worker] must be one; else is zero to reduce the obj function
    # 10000 is to remark, but 5 was enough since max of 40 hours yields max of 5 shifts, the maximum possible sum

# Constraint (def of model.no_pref)
for worker in workers:
    model.constraints.add(
        model.no_pref[worker] >= sum(model.works[worker, 'Sat', shift] for shift in days_shifts['Sat'])
        - sum(model.works[worker, 'Sun', shift] for shift in days_shifts['Sun'])
    )  # if not working on sunday but working saturday model.needed must be 1; else will be zero to reduce the obj function



###########################################################################
## Case 5 Monto Carlo model
###https://blog.quantinsti.com/portfolio-optimization-maximum-return-risk-ratio-python/
###########################################################################
#import relevant libraries
import pandas as pd
import numpy as np
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import os
os.chdir("C:\\Users\\yh10\\Desktop\\Jagdish\\optimizationtechniq")

##Fetch data from yahoo and save under dataframe
#stock = ['BAC', 'GS', 'JPM', 'MS']

stock = ['SBIN.BO','ITC.NS','ICICIBANK.NS','HDFCBANK.NS','IRCTC.NS']

data = web.DataReader(stock, data_source = "yahoo", 
                      start = '2019-11-01', end = '2020-03-31')['Adj Close']

##Arrange the data in ascending order
data = data.iloc[::-1]
print(data.round(2))

##compute stock returns and print the reutns in percentage
stock_ret = data.pct_change()
print(stock_ret.round(4)*100)

##Covariance matrix
mean_returns = stock_ret.mean()
cov_matrix = stock_ret.cov()
print(mean_returns)
print(cov_matrix)

##Set the number of iterations to 10000 and define an array to hold the simulation
num_iterations = 10000
simulation_res = np.zeros((4+len(stock)-1, num_iterations))

for i in range(num_iterations):
    #select random weights and normalize to set the sum to a
    weights = np.array(np.random.random(5)) ##Need to change parameter basis on matrix
    weights /= np.sum(weights)
    
    #calculate the return and standard deviation for every step
    portfolio_return = np.sum(mean_returns * weights)
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    #store all results in defined array
    simulation_res[0,i] = simulation_res[0,i]/simulation_res[1,i]
    
    #Save the weights in the array
    for j in range(len(weights)):
        simulation_res[j+3,i] = weights[j]
        
##save
sim_frame = pd.DataFrame(simulation_res.T, columns = ['ret','stdev','sharpe',
                         stock[0],stock[1],stock[2],stock[3],stock[4]])  ##Need to change column basis on matrix
print(sim_frame.head(5))
print(sim_frame.tail(5))    

#Spot the position of the portfolio with highest sharpe ratio       
max_sharpe = sim_frame.iloc[sim_frame['sharpe'].idxmax()] 

#Spot the position of the portfolio with minimu standard deviation
min_std = sim_frame.iloc[sim_frame['stdev'].idxmin()]

print("The portfolio for max Sharpe Ratio:\n", max_sharpe) 
print("The portfolio for min_std risk:\n", min_std)                                

##SCatter plot coloured by various sharpe ratio with stddev
plt.scatter(sim_frame.stdev, sim_frame.ret, c = sim_frame.sharpe, cmap = 'RdYlBu')                                                
plt.xlabel('Standard Deviation')
plt.ylabel('Returns')
plt.ylim(0,0.003)
plt.xlim(0.0075,0.03)        

#Plot a red star to highlight position of the portfolio with highest sharpe Ratio
plt.scatter(max_sharpe[1],max_sharpe[1],marker=(5,1,0),color='r',s=600)

#Plot a blue star to highlight position of the portfolio with minimum Variance
plt.scatter(min_std[1],min_std[1],marker=(5,1,0),color='b',s=600)
plt.show()

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
select = ['CNP', 'F', 'WMT', 'GE','TSLA']
##select = ['SBIN.BO','ITC.NS','ICICIBANK.NS','HDFCBANK.NS','IRCTC.NS']

data = web.DataReader(select, data_source = "yahoo", 
                      start = '2014-1-1', end = '2016-12-31')['Adj Close']

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
                                                                    
    

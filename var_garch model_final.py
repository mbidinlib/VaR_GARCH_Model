# -*- coding: utf-8 -*-
"""*******************************************
Date: March , 2023
Purpose:Value-at-Risk forecasting of MNT/USD 
        from Jan 2021 to Feb 2023 (GARCH-Model)
Author:
*******************************************"""

# Import libraries and packages
import pandas as pd
import numpy as np
import re
import quandl
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import arch
from arch import *
from arch.__future__ import reindexing
from scipy.stats import norm
#arch_model,ConstantMean, GARCH, Normal


'''********************************************************************
# Import Open-source data Jan 21 - Feb 23
 -Souce : https://www.investing.com/currencies/usd-mnt-historical-data
*********************************************************************'''
# Impoirt data and set date column as index
df =  pd.read_csv("C:/Users/Public/USD_MNT Historical Data_jan16-dec22.csv",
                  index_col=0, parse_dates=True)
# Print the head of the dataset
print(df.head())
'''**************************************
-----------CLEAN DATA--------------------
Transforms data from USD/MNT to MNT/USD*
---Data was obtained in USD/MNT format and
---Must be inverted to MNT/USD
**************************************'''
for j in ['exchange', 'open', 'high', 'low']:
    df[j] = (1/ df[j])
print(df.head())

#Keep only the rate and inflation columns
df =  df[["exchange", "inflation"]]
print(df.head())

# Calculate the log returns of the exchange rate
df['log_exchange'] = np.log(df['exchange']).diff()          # Use Numpy to calculate the log returns of the exchange rate
keep = ~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)      # Remove any nan or inf values after calculation of log returns
df = df[keep].astype(np.float64)
df =  df*1000                                               # Multiply data by 1000 to avoid warnings when the variance is extremely small 
inflation = df['inflation']

'''*********************************************************
----------INTRODUCE TRATMENT------------
I chose January 1 2020 as the treatment start
---Data before 2020 would be treated as pre-treatment 
---Data from 2020 would be treated as post-treatment 
*************************************************************'''
treatment_date = '2020-01-01'
pre_treat_df = df[df.index < treatment_date]
post_treat_df = df[df.index >= treatment_date]

'''*********************************************
----------FIT THE GARCH MODEL------------
GARCH model with Inflation as the exogenous 
variable in the mean process
*************************************************'''
# Define the model with exogenous variable in mean process
garch_model = arch_model(df['log_exchange'], 
                   vol='GARCH', p=1, o=0, q=1, 
                   mean='ARX', lags=1, x=inflation)
# Fit the model
garch_fit = garch_model.fit(disp='off')


# Forecast Value-at-Risk at 5% confidence level
forecast_horizon = 1
forecast = garch_fit.forecast(horizon=forecast_horizon,
                            x=inflation.iloc[-1:], method='simulation')
volatility = np.sqrt(forecast.variance)
VaR = -volatility * np.percentile(df["log_exchange"], 10)
var_10 = np.percentile(forecast.variance[-1:], 10)
print(forecast)
VaR = -volatility * np.percentile(df["log_exchange"], 10)


# Print the summary of the model
print(garch_fit.summary())

print(VaR, "===Var")
print("Nvars ==",var_10) 

'''*********************************************
--------Conditional Districution Curve----------
GARCH model with Inflation as the exogenous 
variable in the mean process
*************************************************'''

cond_var = garch_fit.conditional_volatility
# Calculate the 10% VaR
VaR = -1.645 * np.sqrt(cond_var['2021-01-04'])

print(VaR, "===Var")
print("cond_var ==",cond_var) 


fig, ax = plt.subplots()
df['log_exchange']['2021-01-04':].plot(ax=ax)
ax.axhline(y=-VaR, color='r', linestyle='--')
ax.axhline(y=VaR, color='r', linestyle='--')
ax.set_title('Conditional Distribution on 2021-01-04')
ax.set_xlabel('Date')
ax.set_ylabel('Log Returns')
#plt.show()

forecast_var = forecast.variance.iloc[-1,0]
last_return = df['log_exchange'].iloc[-1]




exog = df['inflation'].values.reshape(-1,1)
# Define GARCH model with exogenous variable
model = arch.arch_model(df['log_exchange'], vol='GARCH', mean='ARX', lags=1, x=exog)
# Fit the model
res = model.fit()
# Forecast VaR for a specific date
specific_date = '2021-01-04'
forecast_horizon = 1
#forecast = res.forecast(start=specific_date, horizon=forecast_horizon, method='simulation')
# Calculate 10% VaR for both tails
var_10 = np.percentile(forecast.simulations.values[0][-1, :], 10)

# Plot conditional distribution curve for the specific date
fig, ax = plt.subplots(figsize=(8,6))
x = np.linspace(-5, 5, 100)
y = res.conditional_volatility['2021-01-04']* np.sqrt(252)
print(y)
density = np.exp(-(x + res.params['x0'] *df['inflation'][-1]**2)**2 / (2*y**2)) / (np.sqrt(2*np.pi)*y)
ax.plot(x, density, linewidth=2)
ax.axvline(-var_10, linestyle='--', color='red')
ax.axvline(var_10, linestyle='--', color='red')
ax.set_xlabel('Log Returns')
ax.set_ylabel('Density')
ax.set_title(f'Conditional Distribution on {specific_date}')
plt.show()














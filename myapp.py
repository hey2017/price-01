import numpy as np
from numpy import loadtxt
import pandas as pd
import yfinance as yf
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from datetime import *


from plotly import graph_objs as go
from fbprophet.plot import plot_plotly
from fbprophet import Prophet
#%%
#from prophet import Prophet
#%%

st.title("Stock Price Forecasting App")


#%%

today = date.today().strftime("%Y-%m-%d")
tickers = ['msft', 'aapl', 'TSLA', 'GOOG', 'amzn', 'fb', 'NFLX','GM', 'DS', 'BTBT']
df = pdr.DataReader(tickers, data_source='yahoo', start='2015-01-01', end=today)
df2 = pdr.DataReader(tickers, data_source='yahoo', start='2015-01-01', end=today)['Close']
df3 = pdr.DataReader(tickers, data_source='yahoo', start='2015-01-01', end=today)['Adj Close']
#%%
Stocks = ['Microsoft','Apple','Tesla','Google','Amazon','Facebook', 'Netflix', 'General Motors', 'Disney','Bit Digital']
df2.columns = Stocks
df3.columns = Stocks

#%%
import altair as alt
#%%
#selected_stock = st.selectbox("Please selet the stock", Stocks)
selected_stock = st.sidebar.selectbox("Please selet the stock", Stocks)
st.subheader('Selected Stock')
st.line_chart(df2[selected_stock])
n_test = st.sidebar.slider("Months of prediction:", 1,40)
period = n_test*30

#%%
st.subheader(selected_stock + ' Monthly Return Data')
monthly_return = df3[selected_stock].resample('M').ffill().pct_change()
df6 = pd.DataFrame(monthly_return)
st.bar_chart( df6 )

#%%
st.header('The results of the forecast will be ready in a minute')

df2 = df2.reset_index()
#install prophet by: conda install -c conda-forge prophet
#from plotly import graph_objs as go
#from fbprophet.plot import plot_plotly

#forecasting
df_train = df2[['Date',selected_stock]]
df_train = df_train.rename(columns = {"Date":"ds" , selected_stock:"y"})


m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods = period)
forecast = m.predict(future)
st.header('\nForecast data')
st.write(forecast.tail())

#plotting the forecast
st.header('\nPlot of forecast data and the stock value')
fig1 = plot_plotly(m,forecast)
st.plotly_chart(fig1)

#st.write('forecast components')
#fig2 = m.plot_components(forecast)
#st.write(fig2)
#%%



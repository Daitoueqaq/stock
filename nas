import pandas as pd
from datetime import datetime
import ta
from  myTT import *   

nas = pd.read_csv('nas.csv')
nas['Date'] = pd.to_datetime(nas['Date'])
nas['Day of Week'] = nas['Date'].dt.day_name()
day_map = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
nas['Day Number'] = nas['Day of Week'].map(day_map)
nas['Diff'] = nas['Close'].pct_change()
nas['macd'] = ta.trend.macd_diff(nas['Close'])
nas['Next_Diff'] = nas['Diff'].shift(-1)
K,D,J = KDJ(nas['Close'],nas['High'],nas['Low'], N=9,M1=3,M2=3)
nas['kdj_k']=K; nas['kdj_d']=D; nas['kdj_j']=J

nas['MA5'] = nas['Close'].rolling(window=5).mean()
nas['MA8'] = nas['Close'].rolling(window=8).mean()
nas['MA13'] = nas['Close'].rolling(window=13).mean()
nas['MA21'] = nas['Close'].rolling(window=21).mean()
nas['MA34'] = nas['Close'].rolling(window=34).mean()
nas['MA55'] = nas['Close'].rolling(window=55).mean()
nas['MA89'] = nas['Close'].rolling(window=89).mean()
nas['MA144'] = nas['Close'].rolling(window=144).mean()
nas['MA233'] = nas['Close'].rolling(window=233).mean()
nas['MA377'] = nas['Close'].rolling(window=377).mean()

rsi_indicator6 = ta.momentum.RSIIndicator(nas['Close'], window=6)
rsi_indicator12 = ta.momentum.RSIIndicator(nas['Close'], window=12)
rsi_indicator24 = ta.momentum.RSIIndicator(nas['Close'], window=24)
nas['RSI6'] = rsi_indicator6.rsi()
nas['RSI12'] = rsi_indicator12.rsi()
nas['RSI24'] = rsi_indicator24.rsi()
williamsr = ta.momentum.WilliamsRIndicator(high=nas['High'], low=nas['Low'], close=nas['Close'], lbp=14)
nas['Williams %R'] = williamsr.williams_r()
bb = ta.volatility.BollingerBands(nas['Close'], window=20, window_dev=2)
nas['Upper_BB'] = bb.bollinger_hband()
nas['Lower_BB'] = bb.bollinger_lband()

pd.options.display.max_columns = None
print(nas)


nas.dropna(inplace=True)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
RandomForestRegressor(random_state=42)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Mean Squared Error: ', mse)
print('R2 Score: ', r2)



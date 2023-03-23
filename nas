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



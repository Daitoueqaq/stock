import pandas as pd
from datetime import datetime
import ta

nas = pd.read_csv('nas.csv')
nas['Date'] = pd.to_datetime(nas['Date'])
nas['Day of Week'] = nas['Date'].dt.day_name()
day_map = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
nas['Day Number'] = nas['Day of Week'].map(day_map)
nas['Diff'] = nas['Close'].pct_change()
nas['macd'] = ta.trend.macd_diff(nas['Close'])
nas['Next_Diff'] = nas['Diff'].shift(-1)
period1 = 9
period2 = 3
period3 = 3
low_min = nas['Low'].rolling(window=period1).min()
high_max = nas['High'].rolling(window=period1).max()
nas['RSV'] =  ((nas['Close'] - low_min) / (high_max - low_min)) * 100
nas['kdj_k'] = nas['RSV'].rolling(window=period2).mean()
nas['kdj_d'] = nas['kdj_k'].rolling(window=period3).mean()
nas['kdj_j'] = nas['%J'] = (3 * nas['kdj_k']) - (2 * nas['kdj_d'])

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


def kdj_k(x):
    y = nas['RSV']
    rsv_today = y[-1]
    kdj_k_yesterday = x[:-1][-1]
    if np.isnan(kdj_k_yesterday):
        kdj_k_yesterday = 50
    kdj_k_today = (1/3) * rsv_today + (2/3) * kdj_k_yesterday
    return kdj_k_today

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

nas.dropna(inplace=True)

print(nas)

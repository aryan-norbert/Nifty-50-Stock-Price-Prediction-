import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Fetch historical stock price data for Nifty 50 index
nifty50 = yf.download('^NSEI', start='2013-01-01', end='2023-01-01')
nifty50.to_csv('nifty50_historical_stock_prices.csv')

# Load historical stock price data
data = pd.read_csv('nifty50_historical_stock_prices.csv')

# Convert 'Date' to datetime and set as index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Handle missing values by forward filling
data.fillna(method='ffill', inplace=True)

# Create additional features (e.g., moving averages, RSI)
data['SMA_10'] = data['Close'].rolling(window=10).mean()
data['SMA_50'] = data['Close'].rolling(window=50).mean()

# Function to calculate Relative Strength Index (RSI)
def compute_rsi(data, window):
    delta = data.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

data['RSI'] = compute_rsi(data['Close'], 14)

# Drop rows with NaN values
data.dropna(inplace=True)

# Creating lagged features
for lag in range(1, 6):
    data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)

data.dropna(inplace=True)

# Save preprocessed data for future use
data.to_csv('nifty50_preprocessed_data.csv')

# Load preprocessed data
data = pd.read_csv('nifty50_preprocessed_data.csv', index_col='Date', parse_dates=True)

# Define features and target
X = data[['SMA_10', 'SMA_50', 'RSI', 'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_4', 'Close_Lag_5']]
y = data['Close']

# Add constant term for statsmodels
X = sm.add_constant(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train the model using statsmodels
model = sm.OLS(y_train, X_train).fit()

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Visualize the predicted prices vs. actual prices
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label='Actual Prices')
plt.plot(X_test.index, y_pred, label='Predicted Prices', linestyle='--', color='red')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Nifty 50 Stock Price Prediction')
plt.legend()
plt.show()

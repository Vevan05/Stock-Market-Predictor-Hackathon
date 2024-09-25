import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def fetch_stock_data(stock_symbol, start_date, end_date):
    # Fetch historical stock data
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    return data

def preprocess_data(data):
    # Reset index to make 'Date' a column
    data.reset_index(inplace=True)

    # Use 'Date' as a feature
    data['Date'] = pd.to_datetime(data['Date'])
    data['Date'] = data['Date'].map(pd.Timestamp.timestamp)  # Convert to timestamp

    # Prepare features (X) and target (y)
    X = data[['Date']]  # Features
    y = data['Close']   # Target variable (Close price)

    return X, y

def train_model(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate mean squared error
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse:.2f}')

    return model

def predict_future_price(model, future_date):
    # Convert future date to timestamp
    future_date_timestamp = pd.Timestamp(future_date).timestamp()
    future_price = model.predict([[future_date_timestamp]])
    return future_price[0]

def plot_results(data, model):
    # Plot the actual vs predicted prices
    plt.figure(figsize=(14, 7))
    plt.scatter(data['Date'], data['Close'], color='blue', label='Actual Prices')

    # Create predictions for plotting
    data['Predicted'] = model.predict(data[['Date']])
    plt.plot(data['Date'], data['Predicted'], color='red', label='Predicted Prices')

    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # Ask user for stock symbol and number of years to predict
    stock_symbol = input("Enter the stock symbol (e.g., AAPL, GOOGL): ")
    years_ahead = int(input("Enter the number of years into the future to predict: "))

    # Define date range
    start_date = '2020-01-01'
    end_date = '2023-09-25'  # Use today's date

    # Fetch stock data
    data = fetch_stock_data(stock_symbol, start_date, end_date)

    # Preprocess data
    X, y = preprocess_data(data)

    # Train model
    model = train_model(X, y)

    # Predict future price
    future_date = pd.Timestamp.now() + pd.DateOffset(years=years_ahead)
    future_price = predict_future_price(model, future_date)
    print(f'Predicted price for {stock_symbol} on {future_date.date()}: ${future_price:.2f}')

    # Plot results
    plot_results(data, model)

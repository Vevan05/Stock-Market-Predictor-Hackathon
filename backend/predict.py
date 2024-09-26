import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import plotly.graph_objs as go  # Import Plotly for interactive plots

def fetch_stock_data(stock_symbol, start_date, end_date):
    # Fetch historical stock data
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    return data

def preprocess_data(data):
    if data.empty:
        raise ValueError("The fetched data is empty. Please check the stock symbol and date range.")

    # Reset index to make 'Date' a column
    data.reset_index(inplace=True)

    # Sort by date
    data = data.sort_values('Date')

    # Use 'Close' prices as the target
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']])

    # Create training and testing data
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])  # Previous 60 days data
        y.append(scaled_data[i, 0])  # Target: the next day’s price

    X, y = np.array(X), np.array(y)

    # Reshape the data to 3D (samples, timesteps, features) for LSTM
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler

def build_lstm_model(input_shape):
    # Build the LSTM model
    model = Sequential()

    # Add LSTM layers with Dropout to prevent overfitting
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))

    # Add a Dense layer for output
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

def train_model(model, X_train, y_train):
    # Train the LSTM model
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    return model

def predict_future_price(model, data, scaler, future_days):
    # Prepare the data for prediction
    last_60_days = data[-60:].values
    last_60_days_scaled = scaler.transform(last_60_days)

    X_future = []
    X_future.append(last_60_days_scaled)
    X_future = np.array(X_future)
    X_future = np.reshape(X_future, (X_future.shape[0], X_future.shape[1], 1))

    # Predict the price
    predicted_price = model.predict(X_future)
    predicted_price = scaler.inverse_transform(predicted_price)

    return predicted_price[0, 0]  # Return the predicted price

if __name__ == '__main__':
    # Ask user for stock symbol and number of years to predict
    stock_symbol = input("Enter the stock symbol (e.g., AAPL, GOOGL): ")
    years_ahead = int(input("Enter the number of years into the future to predict: "))

    # Call Gemini API to fetch latest stock activity
    start_date = '2017-01-01'
    end_date = pd.Timestamp.now().strftime('%Y-%m-%d')

    # Fetch stock data
    data = fetch_stock_data(stock_symbol, start_date, end_date)

    try:
        X, y, scaler = preprocess_data(data)

        # Build and train LSTM model
        model = build_lstm_model((X.shape[1], 1))
        model = train_model(model, X, y)

        # Predict future price
        future_date = pd.Timestamp.now() + pd.DateOffset(years=years_ahead)
        future_price = predict_future_price(model, data[['Close']], scaler, future_days=years_ahead * 365)
        print(f'Predicted price for {stock_symbol} on {future_date.date()}: ${future_price:.2f}')

        # Plot results

    except ValueError as e:
        print(e)
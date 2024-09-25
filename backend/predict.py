import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import google.generativeai as genai  # type: ignore
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

def plot_results(data, model, scaler):
    # Prepare the data for plotting
    scaled_data = scaler.fit_transform(data[['Close']])

    # Predict the data for the full range
    X_test, y_test = [], []
    for i in range(60, len(scaled_data)):
        X_test.append(scaled_data[i-60:i, 0])
        y_test.append(scaled_data[i, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # Find the maximum prices at specific intervals (e.g., every 30 days)
    interval = 30  # Days for interval
    max_prices_dates = []
    max_prices_values = []

    for i in range(0, len(data), interval):
        max_prices_dates.append(data['Date'].iloc[i:i + interval].max())
        max_prices_values.append(data['High'].iloc[i:i + interval].max())

    # Create a Plotly figure
    fig = go.Figure()

    # Actual Prices
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Actual Prices', line=dict(color='blue')))

    # Predicted Prices
    fig.add_trace(go.Scatter(x=data['Date'][60:], y=predicted_prices.flatten(), mode='lines', name='Predicted Prices', line=dict(color='red')))

    # Max Prices as Points
    fig.add_trace(go.Scatter(x=max_prices_dates, y=max_prices_values, mode='markers', name='Max Prices', marker=dict(color='green', size=10)))

    # Update layout
    fig.update_layout(title='Stock Price Prediction with LSTM',
                      xaxis_title='Date',
                      yaxis_title='Price in $',
                      hovermode='x unified')  # Set hover mode to show details

    # Show the figure
    fig.show()

def info(stock_symbol):
    api_key = "AIzaSyCFi0S-ftUqPKDUsR8OJ8aThdwp-_MM1aw"  # Directly pass the API key
    genai.configure(api_key=api_key)  # Explicitly configure the API key

    a = f"What is the latest activity and performance of {stock_symbol} and also provide links if you dont have information about it?"
    b = f"What is the investor sentiment of {stock_symbol} and also provide links if you dont have information about it?"
    c = f"What are the funding details of {stock_symbol} and also provide links if you dont have information about it?"

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(a)
        print(f"Stock Activity: {response.text}")
    except Exception as e:
        print(f"Error fetching activity data: {e}")

    print()

    try:
        response = model.generate_content(b)
        print(f"Investor Sentiment: {response.text}")
    except Exception as e:
        print(f"Error fetching sentiment data: {e}")

    print()

    try:
        response = model.generate_content(c)
        print(f"Funding: {response.text}")
    except Exception as e:
        print(f"Error fetching funding data: {e}")

if __name__ == '__main__':
    # Ask user for stock symbol and number of years to predict
    stock_symbol = input("Enter the stock symbol (e.g., AAPL, GOOGL): ")
    years_ahead = int(input("Enter the number of years into the future to predict: "))

    # Call Gemini API to fetch latest stock activity
    info(stock_symbol)

    # Define date range
    start_date = '2017-01-01'
    end_date = pd.Timestamp.now().strftime('%Y-%m-%d')

    # Fetch stock data
    data = fetch_stock_data(stock_symbol, start_date, end_date)

    try:
        # Preprocess data
        X, y, scaler = preprocess_data(data)

        # Build and train LSTM model
        model = build_lstm_model((X.shape[1], 1))
        model = train_model(model, X, y)

        # Predict future price
        future_date = pd.Timestamp.now() + pd.DateOffset(years=years_ahead)
        future_price = predict_future_price(model, data[['Close']], scaler, future_days=years_ahead * 365)
        print(f'Predicted price for {stock_symbol} on {future_date.date()}: ${future_price:.2f}')

        # Plot results
        plot_results(data, model, scaler)

    except ValueError as e:
        print(e)
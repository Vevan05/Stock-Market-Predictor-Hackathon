from flask import Flask, jsonify, request
import requests
import json
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import plotly.graph_objects as go

app = Flask(__name__)

# Fetch historical stock data
def fetch_stock_data(stock_symbol, start_date, end_date):
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    return data

# Preprocess the stock data
def preprocess_data(data):
    if data.empty:
        return None, None, None

    data.reset_index(inplace=True)
    data = data.sort_values('Date')
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']])

    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler

# Build the LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train the model
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train, epochs=7, batch_size=32)
    return model

# Predict the future price
def predict_future_price(model, data, scaler, future_days):
    last_60_days = data[-60:].values
    last_60_days_scaled = scaler.transform(last_60_days)

    X_future = []
    X_future.append(last_60_days_scaled)
    X_future = np.array(X_future)
    X_future = np.reshape(X_future, (X_future.shape[0], X_future.shape[1], 1))

    predicted_price = model.predict(X_future)
    predicted_price = scaler.inverse_transform(predicted_price)

    return predicted_price[0, 0]

@app.route('/predict', methods=['POST'])
def predict_stock():
    data = request.json
    stock_symbol = data['symbol']
    years_ahead = data['years']

    start_date = '2017-01-01'
    end_date = pd.Timestamp.now().strftime('%Y-%m-%d')

    # Fetch stock data
    stock_data = fetch_stock_data(stock_symbol, start_date, end_date)
    if stock_data.empty:
        return jsonify({'error': 'Stock data not found. Please check the symbol.'}), 400

    # Preprocess data
    X, y, scaler = preprocess_data(stock_data)
    if X is None:
        return jsonify({'error': 'Error in preprocessing the data.'}), 400

    # Build and train the model
    model = build_lstm_model((X.shape[1], 1))
    model = train_model(model, X, y)

    # Predict future price
    future_price = predict_future_price(model, stock_data[['Close']], scaler, years_ahead * 365)

    # Convert future_price to Python native float
    return jsonify({'predicted_price': round(float(future_price), 2)})



@app.route('/plot', methods=['POST'])
def plot_stock():
    data = request.json
    stock_symbol = data['symbol']

    start_date = '2017-01-01'
    end_date = pd.Timestamp.now().strftime('%Y-%m-%d')

    stock_data = fetch_stock_data(stock_symbol, start_date, end_date)

    if stock_data.empty:
        return jsonify({'error': 'Stock data not found.'}), 400

    try:
        X, y, scaler = preprocess_data(stock_data)
        model = build_lstm_model((X.shape[1], 1))
        model = train_model(model, X, y)

        # Generate the prediction plot
        scaled_data = scaler.fit_transform(stock_data[['Close']])
        X_test, y_test = [], []
        for i in range(60, len(scaled_data)):
            X_test.append(scaled_data[i-60:i, 0])
            y_test.append(scaled_data[i, 0])

        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        predicted_prices = model.predict(X_test)
        predicted_prices = scaler.inverse_transform(predicted_prices)

        # Create a Plotly figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Close'], mode='lines', name='Actual Prices'))
        fig.add_trace(go.Scatter(x=stock_data['Date'][60:], y=predicted_prices.flatten(), mode='lines', name='Predicted Prices'))

        fig.update_layout(title='Stock Price Prediction with LSTM',
                          xaxis_title='Date',
                          yaxis_title='Price in $',
                          hovermode='x unified')

        return fig.to_json()

    except ValueError as e:
        return jsonify({'error': str(e)}), 400



CORS(app)  # To handle CORS between Flask and React

# Replace with your actual API key and endpoint
API_KEY = '3bc255f066af4f0781a2df90d96258b3'  # Get this from a service like NewsAPI
NEWS_API_URL = 'https://newsapi.org/v2/top-headlines'

@app.route('/news', methods=['GET'])
def get_stock_market_news():
    try:
        # Make a request to the news API for stock market news
        response = requests.get(
            NEWS_API_URL,
            params={
                'category': 'business',
                'q': 'stock market',     # Or search specific keywords
                'apiKey': '3bc255f066af4f0781a2df90d96258b3',       # Your API key
                'language': 'en',
                'sortBy': 'publishedAt'
            }
        )
        news_data = response.json()

        # Check for errors
        if response.status_code != 200 or 'articles' not in news_data:
            return jsonify({'error': 'Could not fetch stock market news'}), 500

        return jsonify(news_data['articles']), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

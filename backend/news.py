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

app = Flask(__name__)

# Fetch historical stock data
def fetch_stock_data(stock_symbol, start_date, end_date):
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    return data

# Preprocess the stock data
def preprocess_data(data):
    data.reset_index(inplace=True)
    data = data.sort_values('Date')
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']])
    
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])  # Previous 60 days data
        y.append(scaled_data[i, 0])  # Target: the next day’s price
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler

# Build and compile the LSTM model
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
    model.fit(X_train, y_train, epochs=5, batch_size=32)
    return model

# Predict future stock prices
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

# API route for stock prediction
@app.route('/predict', methods=['POST'])
def predict_stock_price():
    data = request.json
    usedata = json.dumps((data))
    stock_symbol = usedata[0]
    years_ahead = usedata[1]

    # Fetch stock data
    stock_data = fetch_stock_data(stock_symbol, '2017-01-01', pd.Timestamp.now().strftime('%Y-%m-%d'))

    if stock_data.empty:
        return jsonify({'error': 'Failed to fetch stock data. Please check the symbol.'}), 400

    # Preprocess the data
    X, y, scaler = preprocess_data(stock_data)

    # Build and train the model
    model = build_lstm_model((X.shape[1], 1))
    model = train_model(model, X, y)

    # Predict the future price
    future_price = predict_future_price(model, stock_data[['Close']], scaler, future_days=years_ahead * 365)

    return jsonify({'predicted_price': round(future_price, 2)})


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

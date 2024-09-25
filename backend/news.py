from flask import Flask, jsonify, request
import requests

app = Flask(__name__)

@app.route('/stock-news', methods=['GET'])
def fetch_stock_news():
    api_key = 'YOUR_API_KEY'  # Replace with your actual News API key
    query = request.args.get('query', 'stock market')  # Get query from URL parameters
    language = request.args.get('language', 'en')  # Default to English

    url = 'https://newsapi.org/v2/everything'
    parameters = {
        'q': query,
        'language': language,
        'sortBy': 'relevancy',  # Sort by relevance
        'apiKey': api_key
    }

    try:
        response = requests.get(url, params=parameters)
        response.raise_for_status()  # Raise an error for bad responses
        news_data = response.json()

        # Check if the response contains articles
        if news_data['status'] == 'ok' and news_data['totalResults'] > 0:
            return jsonify(news_data['articles']), 200
        else:
            return jsonify({'message': 'No articles found.'}), 404
    
    except requests.exceptions.RequestException as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
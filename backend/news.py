import React, { useEffect, useState } from 'react';
import { 
  SafeAreaView, 
  Text, 
  TextInput, 
  Button, 
  View, 
  ScrollView, 
  StyleSheet, 
  TouchableOpacity
} from 'react-native';

import axios from 'axios';


function App() {
  const [stockSymbol, setStockSymbol] = useState([])
  const [news, setNews] = useState([]);
  const [error, setError] = useState('');

  const [symbol, setSymbol] = useState('');
  const [years, setYears] = useState(1); 
  const [predictedPrice, setPredictedPrice] = useState(null);
  const [errorMessage, setErrorMessage] = useState('');

  const handlePredict = async (e) => {
    e.preventDefault();
    setErrorMessage('');  
    setPredictedPrice(null); 

    try {
      const response = await axios.post('http://127.0.0.1:5000/predict', {
        symbol: symbol,
        years: years
      });

      if (response.data && response.data.predicted_price) {
        setPredictedPrice(response.data.predicted_price);
      } else {
        setErrorMessage('No prediction data available.');
      }
    } catch (err) {
      setErrorMessage('Error fetching the prediction. Please check the stock symbol.');
    }
  };


  useEffect(() => {
    const fetchNews = async () => {
      try {
        const response = await axios.get('http://127.0.0.1:5000/news');
        setNews(response.data);
      } catch (err) {
        setError('Failed to fetch stock market news');
      }
    };

    fetchNews();
  }, []);
  return (
    <SafeAreaView style={styles.container}>
      <TouchableOpacity style = {styles.header}>
      <Text style={styles.heading}>Trade Forecast</Text>
      </TouchableOpacity>


      <View style = {{flex: 1, flexDirection: 'row', justifyContent: 'center'}}>
        <View>
        <div 
        style = {{width: 600, height: 533, justifyContent: 'center'}}>
        <View>
        <View>
          <img src = 'https://cdn.pixabay.com/photo/2024/01/06/02/44/ai-generated-8490532_640.png' style = {{width: 600, height: 350, marginLeft: 20, marginTop: 30}}/></View>
        <View style = {{marginTop: 10, marginLeft: 50}}>
            <div>
            <form onSubmit={handlePredict} style = {{marginLeft: 100, justifyContent: 'center'}}>
              <div style = {{marginBottom: 10, marginTop: 10}}>
                <label style = {{fontSize: 20, fontFamily: 'Verdana'}}>Stock Symbol: </label>
                <input
                  type="text"
                  value={symbol}
                  onChange={(e) => setSymbol(e.target.value)}
                  placeholder="AAPL, TSLA, etc."
                  required
                />
              </div>
              <div style = {{marginBottom: 10}}>
                <label style = {{fontSize: 20, fontFamily: 'Verdana'}}>Years Ahead: </label>
                <input
                  type="number"
                  value={years}
                  onChange={(e) => setYears(e.target.value)}
                  min="1"
                  required
                />
              </div>
              <button type="submit" style = {{height: 50,width: 300 ,justifyContent: 'center', borderRadius: 5, backgroundColor: '#C4DAD2', marginLeft: 10}}>
                <Text  style = {{alignSelf: 'center', fontFamily: 'Verdana', fontSize: 28}}>Predict</Text>
                </button>
            </form>

            <div style={{ marginLeft: 50 }}>
            {predictedPrice ? (
              <h2>Predicted Price after {years} years: ${predictedPrice}</h2>
              ) : (
              <h5>(ETA: 15s)</h5>
            )}
  </div>

            {errorMessage && (
              <div style={{ color: 'red' }}>
                <p>{errorMessage}</p>
              </div>
            )}
          </div>
        </View>
      </View>

        </div>
        </View>

      <View style = {{width: 100}}></View>

    <View style = {{flex: 1, flexDirection: 'column', justifyContent: 'center', marginTop: 20}}>
      <div>
        <View style = {{flex: 1, marginTop: 10}}>
        <TouchableOpacity style = {{height: 40}}><Text style = {{fontSize: 32, textAlign: 'center', fontFamily: 'Verdana'}}><u>Stock Market News</u></Text></TouchableOpacity>
        </View>
       <ScrollView style = {{height: 310}}> 
    
    <View>


      {error && <p>{error}</p>}

      <div style = {{marginTop: 10}}>
        {news.map((article, index) => (
          <div key={index} style={{ marginBottom: 20 }}>
            <p style = {{fontFamily: 'Verdana'}}>{index+ 1}) {article.title}</p>
            <p style = {{fontFamily: 'Verdana'}}>{article.description}</p>
            <a href={article.url} target="_blank" rel="noopener noreferrer" style = {{fontFamily: 'Verdana'}}>
              Read more
            </a>
          </div>
        ))}
      </div>
    </View>
    </ScrollView>
    </div>
    <View style = {{height: 30, marginBottom: 20, borderBottomWidth: 1, width: 510}}></View>
    <View>
    <TouchableOpacity style = {{height: 40}}><Text style = {{fontSize: 32, textAlign: 'center', fontFamily: 'Verdana'}}><u>Stock Market Apps</u></Text></TouchableOpacity>
  <View style = {{flex: 1, flexDirection: 'row', marginTop: 50}}>
    <View>      
  <a href="https://www.webull.com/" target='_blank'><img src="https://play-lh.googleusercontent.com/WXbQNRz-G6P16seMX9vtBhzYPWPIbmyQRsn-RZOoQ5mI3WmqRcZxwUkPX15lwWpvaaY" style = {{width: 70, height: 70, padding: 9, marginLeft: 45}}/></a>
  <Text style = {{padding: 5, textAlign: 'center', marginLeft: 43}}>Webull</Text>
  </View>
  <View>
  <a href="https://www.5paisa.com/" target='_blank'><img src="https://play-lh.googleusercontent.com/-MNspIooJf9VXXApEo8Z1eaokA5k5Be7TcgKeWSwBZGTRcsKZTVWkEcCzHq5ntAcsI0" style = {{width: 70, height: 70, padding: 9}}/></a>
  <Text style = {{padding: 5, textAlign: 'center'}}>5 Paisa</Text>
  </View>
  <View>
  <a href="https://www.angelone.in/" target='_blank'><img src="https://cdn.brandfetch.io/angelbroking.com/fallback/transparent/theme/dark/h/512/w/512/icon?t=1717818058774" style = {{width: 70, height: 70, padding: 9}} /></a>
  <Text style = {{padding: 5, textAlign: 'center'}}>Angel One</Text>
  </View>
  <View>
  <a href="https://groww.in/" target='_blank'><img src="https://logowik.com/content/uploads/images/groww1643.logowik.com.webp" style = {{width: 70, height: 70, padding: 9}}/></a>
  <Text style = {{padding: 5, textAlign: 'center'}}>Groww</Text>
  </View>
  <View>
  <a href="https://upstox.com/" target='_blank'><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTWPSWjViJCWKR3opI6fnHvUVLAls14aN50NQ&s" style = {{width: 70, height: 70, padding: 9}}/></a>
  <Text style = {{padding: 5, textAlign: 'center'}}>Upstox</Text>
  </View>
  </View>
    </View>
    </View>
    </View>

    <View>
    <a href="https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjRs7L_5d2IAxX84TgGHWjFMIUQFnoECBcQAQ&url=https%3A%2F%2Fkb.icai.org%2Fpdfs%2FPDFFile5b28c9ce64e524.54675199.pdf&usg=AOvVaw00rsDfJ9_kGvSsG_ZIQepK&opi=89978449 " target="_blank" rel="Rules and Clauses for investing in stock market">
        Government Policy
      </a>
    </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    padding: 20,
    backgroundColor: '#E9EFEC',
    width: 1224.4
  },
  heading: {
    fontSize: 28,  
    fontWeight: 'bold',
    color: '#fff',  
    textAlign: 'center', 
    textShadowColor: 'rgba(0, 0, 0, 0.25)', 
    fontFamily: 'Verdana',
    fontSize: 40
  },
  header:{
    backgroundColor: '#16423C',
    height: 80,
    justifyContent: 'center',
  },
  inputContainer: {
    marginBottom: 20,
    marginTop: 20,
    alignSelf: 'center',
    width: 250,
  },
  input: {
    borderBottomWidth: 2,
    borderColor: '#00000',
    padding: 10,
    marginVertical: 10,
    width: 250,
    alignSelf: 'center',
    justifyContent: 'center'
  },
  subHeading: {
    fontSize: 18,
    marginVertical: 10,
    fontWeight: 'bold',
  },
  predictedPrice: {
    fontSize: 20,
    color: '#3178bf',
    marginBottom: 20,
  },
  noPrediction: {
    fontSize: 16,
    color: '#555',
    fontFamily: 'Verdana'
  },
  newsItem: {
    borderBottomWidth: 1,
    borderBottomColor: '#ddd',
    paddingVertical: 10,
  },
  newsTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    fontFamily: 'Verdana'
  },
  link: {
    color: '#007bff',
    marginTop: 5,
  },
});

export default App

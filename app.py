from flask import Flask, request, jsonify, send_from_directory
from technical_analysis import TechnicalAnalyzer
import os
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import numpy as np
import yfinance as yf
from nsepython import *
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Create analysis_reports directory if it doesn't exist
if not os.path.exists('analysis_reports'):
    os.makedirs('analysis_reports')

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        period = data.get('period', '6mo')
        interval = data.get('interval', '1d')
        
        if not symbol:
            return jsonify({'error': 'Symbol is required'}), 400
        
        # Initialize the analyzer
        analyzer = TechnicalAnalyzer(symbol, period=period, interval=interval)
        
        # Detect patterns
        analyzer.detect_patterns()
        
        # Get trading signal
        signal = analyzer.get_signal()
        
        # Helper function to convert NaN to None
        def clean_nan(obj):
            if isinstance(obj, float) and np.isnan(obj):
                return None
            elif isinstance(obj, dict):
                return {k: clean_nan(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_nan(x) for x in obj]
            return obj
        
        # Clean the response data
        response_data = {
            'success': True,
            'current_price': float(analyzer.data['Close'].iloc[-1]),
            'stop_loss': float(signal['stop_loss']) if signal['stop_loss'] is not None else None,
            'price_targets': signal['price_targets'],
            'indicators': signal['indicators'],
            'patterns': analyzer.patterns
        }
        
        # Clean NaN values before jsonifying
        cleaned_data = clean_nan(response_data)
        
        return jsonify(cleaned_data)
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reports/<path:filename>')
def serve_report(filename):
    return send_from_directory('analysis_reports', filename)

@app.route('/api/expiry-dates', methods=['POST'])
def get_expiry_dates():
    try:
        data = request.get_json()
        market = data.get('market', 'NIFTY')
        
        # Fetch expiry dates from NSE
        if market == 'NIFTY':
            expiry_dates = nse_expirydetails()
        elif market == 'BANKNIFTY':
            expiry_dates = nse_expirydetails('BANKNIFTY')
        else:
            expiry_dates = nse_expirydetails('FINNIFTY')
        
        # Format dates
        formatted_dates = [date.strftime('%Y-%m-%d') for date in expiry_dates]
        
        return jsonify({
            'success': True,
            'dates': formatted_dates
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/analyze-option', methods=['POST'])
def analyze_option():
    try:
        data = request.get_json()
        market = data.get('market')
        expiry = data.get('expiry')
        strike = float(data.get('strike'))
        option_type = data.get('type')  # CE or PE

        # Fetch option chain data
        if market == 'NIFTY':
            chain = nse_optionchain_scrapper('NIFTY')
        elif market == 'BANKNIFTY':
            chain = nse_optionchain_scrapper('BANKNIFTY')
        else:
            chain = nse_optionchain_scrapper('FINNIFTY')

        # Extract relevant option data
        option_data = extract_option_data(chain, strike, option_type, expiry)
        
        # Calculate Greeks
        greeks = calculate_greeks(option_data)
        
        # Calculate PCR
        pcr = calculate_pcr(chain)

        return jsonify({
            'success': True,
            'ltp': option_data['ltp'],
            'change': option_data['change'],
            'iv': option_data['iv'],
            'volume': option_data['volume'],
            'oi': option_data['oi'],
            'greeks': greeks,
            'pcr': pcr
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

def extract_option_data(chain, strike, option_type, expiry):
    # Find the specific option in the chain
    for option in chain['records']['data']:
        if (option['strikePrice'] == strike and 
            option['expiryDate'] == expiry):
            if option_type == 'CE':
                data = option['CE']
            else:
                data = option['PE']
            
            return {
                'ltp': data['lastPrice'],
                'change': data['pChange'],
                'iv': data['impliedVolatility'],
                'volume': data['totalTradedVolume'],
                'oi': data['openInterest']
            }
    raise ValueError('Option not found in chain')

def calculate_greeks(option_data):
    # Simplified Greeks calculation
    # In a real implementation, you would use Black-Scholes or other models
    return {
        'delta': round(np.random.uniform(0.3, 0.7), 4),  # Placeholder
        'theta': round(np.random.uniform(-0.5, -0.1), 4),  # Placeholder
        'gamma': round(np.random.uniform(0.01, 0.05), 4),  # Placeholder
        'vega': round(np.random.uniform(0.1, 0.3), 4)  # Placeholder
    }

def calculate_pcr(chain):
    total_put_oi = sum(option['PE']['openInterest'] 
                      for option in chain['records']['data'] 
                      if 'PE' in option)
    total_call_oi = sum(option['CE']['openInterest'] 
                       for option in chain['records']['data'] 
                       if 'CE' in option)
    
    return total_put_oi / total_call_oi if total_call_oi > 0 else 0

if __name__ == '__main__':
    app.run(debug=True, port=5000) 
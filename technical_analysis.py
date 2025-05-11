import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volume import VolumeWeightedAveragePrice
from datetime import datetime, timedelta
from pattern_detection import PatternDetector

class TechnicalAnalyzer:
    def __init__(self, symbol, period="1y", interval="1d"):
        """
        Initialize the technical analyzer with stock data
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL')
            period (str): Data period (e.g., '1y', '6mo', '1mo')
            interval (str): Data interval (e.g., '1d', '1h', '15m')
        """
        self.symbol = symbol
        self.period = period
        self.interval = interval
        self.data = None
        self.indicators = {}
        self.patterns = []
        self.pattern_detector = None
        self.fetch_data()
        
    def fetch_data(self):
        """Fetch historical data from Yahoo Finance"""
        try:
            print(f"Fetching data for {self.symbol}...")
            stock = yf.Ticker(self.symbol)
            
            # Try to get info first to validate the symbol
            try:
                info = stock.info
                if not info or 'regularMarketPrice' not in info:
                    raise ValueError(f"Invalid symbol or no data available for {self.symbol}")
            except Exception as e:
                raise ValueError(f"Error validating symbol {self.symbol}: {str(e)}")
            
            # Fetch historical data
            self.data = stock.history(period=self.period, interval=self.interval)
            
            if self.data is None or len(self.data) == 0:
                raise ValueError(f"No historical data found for symbol {self.symbol}")
            
            # Verify we have enough data points
            if len(self.data) < 20:  # Minimum required for indicators
                raise ValueError(f"Insufficient data points for {self.symbol}. Got {len(self.data)} points, need at least 20.")
                
            # Verify data quality
            if self.data['Close'].isnull().any() or self.data['Volume'].isnull().any():
                raise ValueError(f"Data quality issues detected for {self.symbol}. Missing values found.")
            
            print(f"Successfully fetched {len(self.data)} data points")
            self.calculate_indicators()
            self.pattern_detector = PatternDetector(self.data)
            
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            self.data = None
            raise
        
    def calculate_indicators(self):
        """Calculate technical indicators"""
        if self.data is None or len(self.data) == 0:
            return
            
        # RSI
        rsi = RSIIndicator(close=self.data['Close'], window=14)
        self.indicators['RSI'] = rsi.rsi()
        
        # MACD
        macd = MACD(close=self.data['Close'])
        self.indicators['MACD'] = macd.macd()
        self.indicators['MACD_Signal'] = macd.macd_signal()
        self.indicators['MACD_Hist'] = macd.macd_diff()
        
        # Moving Averages
        sma_20 = SMAIndicator(close=self.data['Close'], window=20)
        sma_50 = SMAIndicator(close=self.data['Close'], window=50)
        sma_200 = SMAIndicator(close=self.data['Close'], window=200)
        ema_20 = EMAIndicator(close=self.data['Close'], window=20)
        
        self.indicators['SMA_20'] = sma_20.sma_indicator()
        self.indicators['SMA_50'] = sma_50.sma_indicator()
        self.indicators['SMA_200'] = sma_200.sma_indicator()
        self.indicators['EMA_20'] = ema_20.ema_indicator()
        
        # Volume Analysis
        vwap = VolumeWeightedAveragePrice(
            high=self.data['High'],
            low=self.data['Low'],
            close=self.data['Close'],
            volume=self.data['Volume']
        )
        self.indicators['VWAP'] = vwap.volume_weighted_average_price()
        
        # Calculate Fibonacci levels
        self.calculate_fibonacci_levels()
        
    def calculate_fibonacci_levels(self):
        """Calculate Fibonacci retracement levels"""
        if len(self.data) < 2:
            return
            
        # Find recent high and low
        recent_high = self.data['High'].max()
        recent_low = self.data['Low'].min()
        diff = recent_high - recent_low
        
        # Calculate Fibonacci levels
        self.indicators['Fib_0'] = recent_low
        self.indicators['Fib_0.236'] = recent_low + 0.236 * diff
        self.indicators['Fib_0.382'] = recent_low + 0.382 * diff
        self.indicators['Fib_0.5'] = recent_low + 0.5 * diff
        self.indicators['Fib_0.618'] = recent_low + 0.618 * diff
        self.indicators['Fib_0.786'] = recent_low + 0.786 * diff
        self.indicators['Fib_1'] = recent_high
        
    def detect_patterns(self):
        """Detect chart patterns"""
        if self.data is None or len(self.data) == 0:
            print("No data available for pattern detection")
            return
            
        try:
            # Detect Head and Shoulders
            hs_pattern = self.pattern_detector.detect_head_and_shoulders()
            if hs_pattern['detected']:
                points = self.pattern_detector.get_pattern_points('head_and_shoulders')
                if points:  # Only add if points were found
                    self.patterns.append({
                        'type': 'Head and Shoulders',
                        'confidence': hs_pattern['confidence'],
                        'description': f"Head and Shoulders pattern detected with {hs_pattern['confidence']:.2%} confidence. Target: {hs_pattern['pattern_height']:.2f} points below neckline.",
                        'points': points
                    })
                
            # Detect Double Top
            dt_pattern = self.pattern_detector.detect_double_top()
            if dt_pattern['detected']:
                points = self.pattern_detector.get_pattern_points('double_top')
                if points:  # Only add if points were found
                    self.patterns.append({
                        'type': 'Double Top',
                        'confidence': dt_pattern['confidence'],
                        'description': f"Double Top pattern detected with {dt_pattern['confidence']:.2%} confidence. Target: {dt_pattern['pattern_height']:.2f} points below support.",
                        'points': points
                    })
                
            # Detect Double Bottom
            db_pattern = self.pattern_detector.detect_double_bottom()
            if db_pattern['detected']:
                points = self.pattern_detector.get_pattern_points('double_bottom')
                if points:  # Only add if points were found
                    self.patterns.append({
                        'type': 'Double Bottom',
                        'confidence': db_pattern['confidence'],
                        'description': f"Double Bottom pattern detected with {db_pattern['confidence']:.2%} confidence. Target: {db_pattern['pattern_height']:.2f} points above resistance.",
                        'points': points
                    })
                
            # Detect Bullish Flag
            bf_pattern = self.pattern_detector.detect_bullish_flag()
            if bf_pattern['detected']:
                points = self.pattern_detector.get_pattern_points('bullish_flag')
                if points:  # Only add if points were found
                    self.patterns.append({
                        'type': 'Bullish Flag',
                        'confidence': bf_pattern['confidence'],
                        'description': f"Bullish Flag pattern detected with {bf_pattern['confidence']:.2%} confidence. Pole rise: {bf_pattern['pole_rise']:.2%}, Flag duration: {bf_pattern['flag_duration']} periods.",
                        'points': points
                    })
                
            # Detect Bearish Flag
            bf_pattern = self.pattern_detector.detect_bearish_flag()
            if bf_pattern['detected']:
                points = self.pattern_detector.get_pattern_points('bearish_flag')
                if points:  # Only add if points were found
                    self.patterns.append({
                        'type': 'Bearish Flag',
                        'confidence': bf_pattern['confidence'],
                        'description': f"Bearish Flag pattern detected with {bf_pattern['confidence']:.2%} confidence. Pole fall: {bf_pattern['pole_fall']:.2%}, Flag duration: {bf_pattern['flag_duration']} periods.",
                        'points': points
                    })
                    
        except Exception as e:
            print(f"Error in pattern detection: {str(e)}")
        
    def get_signal(self):
        """Generate trading signal based on patterns and indicators"""
        if self.data is None or len(self.data) == 0:
            return {
                'signal': 'ERROR',
                'confidence': 0,
                'indicators': {}
            }
            
        signal = 'WAIT'
        confidence = 0
        indicator_analysis = {}
        buy_signals = 0  # Initialize counter
        sell_signals = 0  # Initialize counter
        
        # Get current price and recent price action
        last_close = self.data['Close'].iloc[-1]
        recent_high = self.data['High'].iloc[-20:].max()
        recent_low = self.data['Low'].iloc[-20:].min()
        atr = self.calculate_atr()  # Calculate Average True Range for stop loss
        
        # RSI Analysis
        last_rsi = self.indicators['RSI'].iloc[-1]
        rsi_strength = 'NEUTRAL'
        rsi_bias = 'NEUTRAL'
        if last_rsi > 70:
            rsi_strength = 'OVERBOUGHT'
            rsi_bias = 'SELL'
        elif last_rsi < 30:
            rsi_strength = 'OVERSOLD'
            rsi_bias = 'BUY'
        elif last_rsi > 60:
            rsi_strength = 'HIGH'
            rsi_bias = 'SELL'
        elif last_rsi < 40:
            rsi_strength = 'LOW'
            rsi_bias = 'BUY'
            
        indicator_analysis['RSI'] = {
            'value': last_rsi,
            'strength': rsi_strength,
            'bias': rsi_bias,
            'range': '0-100',
            'interpretation': f'RSI is {rsi_strength.lower()} at {last_rsi:.2f}, suggesting {rsi_bias.lower()} bias'
        }
        
        # MACD Analysis
        last_macd = self.indicators['MACD'].iloc[-1]
        last_signal = self.indicators['MACD_Signal'].iloc[-1]
        macd_strength = 'NEUTRAL'
        macd_bias = 'NEUTRAL'
        
        if last_macd > last_signal:
            macd_strength = 'BULLISH'
            macd_bias = 'BUY'
        else:
            macd_strength = 'BEARISH'
            macd_bias = 'SELL'
            
        indicator_analysis['MACD'] = {
            'value': last_macd,
            'signal': last_signal,
            'strength': macd_strength,
            'bias': macd_bias,
            'interpretation': f'MACD is {macd_strength.lower()} (MACD: {last_macd:.2f}, Signal: {last_signal:.2f})'
        }
        
        # Moving Averages Analysis
        sma_20 = self.indicators['SMA_20'].iloc[-1]
        sma_50 = self.indicators['SMA_50'].iloc[-1]
        sma_200 = self.indicators['SMA_200'].iloc[-1]
        
        ma_strength = 'NEUTRAL'
        ma_bias = 'NEUTRAL'
        
        if last_close > sma_20 > sma_50 > sma_200:
            ma_strength = 'STRONG BULLISH'
            ma_bias = 'BUY'
        elif last_close < sma_20 < sma_50 < sma_200:
            ma_strength = 'STRONG BEARISH'
            ma_bias = 'SELL'
        elif last_close > sma_20 and sma_20 > sma_50:
            ma_strength = 'BULLISH'
            ma_bias = 'BUY'
        elif last_close < sma_20 and sma_20 < sma_50:
            ma_strength = 'BEARISH'
            ma_bias = 'SELL'
            
        indicator_analysis['Moving_Averages'] = {
            'close': last_close,
            'sma_20': sma_20,
            'sma_50': sma_50,
            'sma_200': sma_200,
            'strength': ma_strength,
            'bias': ma_bias,
            'interpretation': f'Price is {ma_strength.lower()} relative to moving averages'
        }
        
        # VWAP Analysis
        last_vwap = self.indicators['VWAP'].iloc[-1]
        vwap_strength = 'NEUTRAL'
        vwap_bias = 'NEUTRAL'
        
        if last_close > last_vwap:
            vwap_strength = 'BULLISH'
            vwap_bias = 'BUY'
        else:
            vwap_strength = 'BEARISH'
            vwap_bias = 'SELL'
            
        indicator_analysis['VWAP'] = {
            'value': last_vwap,
            'strength': vwap_strength,
            'bias': vwap_bias,
            'interpretation': f'Price is {vwap_strength.lower()} relative to VWAP'
        }
        
        # Enhanced pattern analysis in signal generation
        pattern_signals = []
        pattern_confidence = 0
        
        for pattern in self.patterns:
            pattern_type = pattern['type']
            confidence = pattern['confidence']
            
            if pattern_type in ['Head and Shoulders', 'Double Top', 'Bearish Flag']:
                pattern_signals.append({
                    'type': pattern_type,
                    'bias': 'SELL',
                    'confidence': confidence,
                    'details': pattern.get('details', {})
                })
                sell_signals += confidence
            elif pattern_type in ['Double Bottom', 'Bullish Flag']:
                pattern_signals.append({
                    'type': pattern_type,
                    'bias': 'BUY',
                    'confidence': confidence,
                    'details': pattern.get('details', {})
                })
                buy_signals += confidence
            
            pattern_confidence = max(pattern_confidence, confidence)
        
        # Add indicator signals
        buy_signals += sum(1 for ind in indicator_analysis.values() if ind['bias'] == 'BUY')
        sell_signals += sum(1 for ind in indicator_analysis.values() if ind['bias'] == 'SELL')
        
        # Calculate overall signal and confidence
        total_signals = len(indicator_analysis) + (1 if pattern_signals else 0)
        
        if buy_signals > sell_signals:
            signal = 'BUY'
            confidence = buy_signals / total_signals
        elif sell_signals > buy_signals:
            signal = 'SELL'
            confidence = sell_signals / total_signals
        else:
            signal = 'WAIT'
            confidence = 0
            
        # Calculate price targets and stop loss
        price_targets = []
        stop_loss = None
        
        if signal == 'BUY':
            # Calculate price targets based on Fibonacci levels
            for level, value in self.indicators.items():
                if level.startswith('Fib_') and value > last_close:
                    price_targets.append({
                        'level': level.replace('Fib_', ''),
                        'price': value,
                        'distance': ((value - last_close) / last_close) * 100
                    })
            
            # Add pattern-based targets if available
            for pattern in self.patterns:
                if pattern['type'] in ['Double Bottom', 'Bullish Flag']:
                    target_price = pattern.get('resistance', 0) + pattern.get('pattern_height', 0)
                    if target_price > last_close:
                        price_targets.append({
                            'level': f"{pattern['type']} Target",
                            'price': target_price,
                            'distance': ((target_price - last_close) / last_close) * 100
                        })
            
            # Calculate stop loss
            if atr is not None:
                stop_loss = last_close - (2 * atr)  # 2 ATR below current price
            else:
                stop_loss = recent_low  # Use recent low as stop loss
            
        elif signal == 'SELL':
            # Calculate price targets based on Fibonacci levels
            for level, value in self.indicators.items():
                if level.startswith('Fib_') and value < last_close:
                    price_targets.append({
                        'level': level.replace('Fib_', ''),
                        'price': value,
                        'distance': ((last_close - value) / last_close) * 100
                    })
            
            # Add pattern-based targets if available
            for pattern in self.patterns:
                if pattern['type'] in ['Head and Shoulders', 'Double Top', 'Bearish Flag']:
                    target_price = pattern.get('support', 0) - pattern.get('pattern_height', 0)
                    if target_price < last_close:
                        price_targets.append({
                            'level': f"{pattern['type']} Target",
                            'price': target_price,
                            'distance': ((last_close - target_price) / last_close) * 100
                        })
            
            # Calculate stop loss
            if atr is not None:
                stop_loss = last_close + (2 * atr)  # 2 ATR above current price
            else:
                stop_loss = recent_high  # Use recent high as stop loss
        
        # Sort price targets by distance
        price_targets.sort(key=lambda x: x['distance'])
            
        return {
            'signal': signal,
            'confidence': confidence,
            'indicators': indicator_analysis,
            'patterns': pattern_signals,
            'current_price': last_close,
            'price_targets': price_targets,
            'stop_loss': stop_loss
        }
    
    def calculate_atr(self, period=14):
        """Calculate Average True Range"""
        try:
            high = self.data['High']
            low = self.data['Low']
            close = self.data['Close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            
            return atr.iloc[-1]
        except Exception as e:
            print(f"Error calculating ATR: {str(e)}")
            return None
    
    def plot_chart(self, use_plotly=True):
        """Plot the chart with indicators"""
        if self.data is None or len(self.data) == 0:
            print("No data available to plot")
            return
            
        if use_plotly:
            self._plot_plotly()
        else:
            self._plot_matplotlib()
            
    def _plot_plotly(self):
        """Create interactive plot using plotly"""
        # Create figure with secondary y-axis
        fig = go.Figure()
        
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=self.data.index,
            open=self.data['Open'],
            high=self.data['High'],
            low=self.data['Low'],
            close=self.data['Close'],
            name='OHLC'
        ))
        
        # Add moving averages
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=self.indicators['SMA_20'],
            name='SMA 20',
            line=dict(color='blue', width=1)
        ))
        
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=self.indicators['SMA_50'],
            name='SMA 50',
            line=dict(color='red', width=1)
        ))
        
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=self.indicators['SMA_200'],
            name='SMA 200',
            line=dict(color='green', width=1)
        ))
        
        # Add VWAP
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=self.indicators['VWAP'],
            name='VWAP',
            line=dict(color='purple', width=1)
        ))
        
        # Add Fibonacci levels
        for level, value in self.indicators.items():
            if level.startswith('Fib_'):
                fig.add_hline(
                    y=value,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text=level.replace('Fib_', ''),
                    annotation_position="right"
                )
        
        # Add RSI subplot
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=self.indicators['RSI'],
            name='RSI',
            line=dict(color='orange'),
            yaxis='y2'
        ))
        
        # Add MACD subplot
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=self.indicators['MACD'],
            name='MACD',
            line=dict(color='blue'),
            yaxis='y3'
        ))
        
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=self.indicators['MACD_Signal'],
            name='MACD Signal',
            line=dict(color='red'),
            yaxis='y3'
        ))
        
        # Add MACD Histogram
        fig.add_trace(go.Bar(
            x=self.data.index,
            y=self.indicators['MACD_Hist'],
            name='MACD Histogram',
            yaxis='y3'
        ))
        
        # Update layout with subplots
        fig.update_layout(
            title=f'{self.symbol} Technical Analysis',
            yaxis_title='Price',
            xaxis_title='Date',
            template='plotly_dark',
            height=1000,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Add secondary y-axis for RSI
        fig.update_layout(
            yaxis2=dict(
                title="RSI",
                overlaying="y",
                side="right",
                range=[0, 100],
                showgrid=False
            )
        )
        
        # Add tertiary y-axis for MACD
        fig.update_layout(
            yaxis3=dict(
                title="MACD",
                overlaying="y",
                side="right",
                anchor="free",
                position=0.95,
                showgrid=False
            )
        )
        
        # Add horizontal lines for RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", yaxis="y2")
        fig.add_hline(y=30, line_dash="dash", line_color="green", yaxis="y2")
        
        fig.show()
        
    def _plot_matplotlib(self):
        """Create static plot using matplotlib"""
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1])
        
        # Price and Moving Averages
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(self.data.index, self.data['Close'], label='Close Price', alpha=0.7)
        ax1.plot(self.data.index, self.indicators['SMA_20'], label='SMA 20', alpha=0.7)
        ax1.plot(self.data.index, self.indicators['SMA_50'], label='SMA 50', alpha=0.7)
        ax1.plot(self.data.index, self.indicators['SMA_200'], label='SMA 200', alpha=0.7)
        ax1.plot(self.data.index, self.indicators['VWAP'], label='VWAP', alpha=0.7)
        
        # Add Fibonacci levels
        for level, value in self.indicators.items():
            if level.startswith('Fib_'):
                ax1.axhline(y=value, color='gray', linestyle='--', alpha=0.5)
                ax1.text(self.data.index[-1], value, level.replace('Fib_', ''), 
                        verticalalignment='bottom')
        
        ax1.set_title(f'{self.symbol} Technical Analysis')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True)
        
        # RSI
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.plot(self.data.index, self.indicators['RSI'], label='RSI', color='purple')
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
        ax2.set_ylabel('RSI')
        ax2.set_ylim(0, 100)
        ax2.legend()
        ax2.grid(True)
        
        # MACD
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        ax3.plot(self.data.index, self.indicators['MACD'], label='MACD', color='blue')
        ax3.plot(self.data.index, self.indicators['MACD_Signal'], label='Signal', color='red')
        ax3.bar(self.data.index, self.indicators['MACD_Hist'], label='Histogram', alpha=0.5)
        ax3.set_ylabel('MACD')
        ax3.legend()
        ax3.grid(True)
        
        # Volume
        ax4 = fig.add_subplot(gs[3], sharex=ax1)
        ax4.bar(self.data.index, self.data['Volume'], label='Volume', alpha=0.5)
        ax4.set_ylabel('Volume')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Example usage
    analyzer = TechnicalAnalyzer('AAPL', period='6mo', interval='1d')
    analyzer.detect_patterns()
    signal = analyzer.get_signal()
    print(f"Signal: {signal['signal']}")
    print(f"Confidence: {signal['confidence']:.2f}")
    print("\nIndicator Values:")
    for indicator, value in signal['indicators'].items():
        print(f"{indicator}: {value['value']:.2f}")
        print(f"  Strength: {value['strength']}")
        print(f"  Bias: {value['bias']}")
        print(f"  Range: {value['range']}")
        print(f"  Interpretation: {value['interpretation']}")
    
    # Plot the chart
    analyzer.plot_chart(use_plotly=True) 
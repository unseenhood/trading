from technical_analysis import TechnicalAnalyzer
import argparse
import plotly.graph_objects as go
from datetime import datetime
import os
import sys

def save_analysis_report(analyzer, signal, output_dir='analysis_reports'):
    """Save analysis results to HTML and image files"""
    try:    
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        symbol = analyzer.symbol.replace('.', '_')
        
        # Create HTML report
        html_content = f"""
        <html>
        <head>
            <title>Technical Analysis Report - {analyzer.symbol}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ background-color: #f5f5f5; padding: 20px; border-radius: 5px; }}
                .results {{ margin-top: 20px; }}
                .indicator {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .signal {{ font-size: 24px; font-weight: bold; margin: 20px 0; }}
                .confidence {{ color: #666; }}
                .chart-container {{ margin-top: 30px; }}
                .indicator-name {{ font-size: 18px; font-weight: bold; color: #333; }}
                .indicator-value {{ font-size: 16px; color: #666; }}
                .indicator-strength {{ font-weight: bold; }}
                .indicator-bias {{ font-weight: bold; }}
                .indicator-interpretation {{ margin-top: 10px; color: #444; }}
                .buy {{ color: #28a745; }}
                .sell {{ color: #dc3545; }}
                .neutral {{ color: #6c757d; }}
                .patterns-container {{ margin: 20px 0; }}
                .pattern {{ margin: 15px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9; }}
                .pattern-name {{ font-size: 18px; font-weight: bold; color: #333; }}
                .pattern-confidence {{ color: #666; margin: 5px 0; }}
                .pattern-description {{ margin-top: 10px; color: #444; }}
                .price-info {{ margin-top: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .current-price {{ font-size: 18px; font-weight: bold; margin-bottom: 10px; }}
                .stop-loss {{ font-size: 16px; color: #666; margin-bottom: 10px; }}
                .price-targets {{ margin-top: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .price-targets h3 {{ font-size: 18px; font-weight: bold; margin-bottom: 10px; }}
                .price-targets table {{ width: 100%; border-collapse: collapse; }}
                .price-targets th, .price-targets td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                .price-targets th {{ background-color: #f5f5f5; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Technical Analysis Report</h1>
                    <p>Symbol: {analyzer.symbol}</p>
                    <p>Period: {analyzer.period}</p>
                    <p>Interval: {analyzer.interval}</p>
                    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="results">
                    <div class="signal">
                        Signal: <span class="{signal['signal'].lower()}">{signal['signal']}</span>
                    </div>
                    <div class="confidence">
                        Confidence: {signal['confidence']:.2f}
                    </div>
                    
                    <div class="price-info">
                        <div class="current-price">
                            Current Price: {signal['current_price']:.2f}
                        </div>
                        <div class="stop-loss">
                            Stop Loss: {f"{signal['stop_loss']:.2f}" if signal['stop_loss'] is not None else 'N/A'}
                        </div>
                    </div>
                    
                    <div class="price-targets">
                        <h3>Price Targets</h3>
                        <table>
                            <tr>
                                <th>Level</th>
                                <th>Price</th>
                                <th>Distance</th>
                            </tr>
                            {''.join(f'''
                            <tr>
                                <td>{target['level']}</td>
                                <td>{target['price']:.2f}</td>
                                <td>{target['distance']:.2f}%</td>
                            </tr>
                            ''' for target in signal['price_targets'])}
                        </table>
                    </div>
                    
                    <h2>Technical Indicators Analysis</h2>
        """
        
        for indicator_name, analysis in signal['indicators'].items():
            if not isinstance(analysis, dict):
                continue
                
            bias_class = analysis.get('bias', 'neutral').lower()
            
            # Special handling for Moving Averages
            if indicator_name == 'Moving_Averages':
                html_content += f"""
                    <div class="indicator">
                        <div class="indicator-name">{indicator_name}</div>
                        <div class="indicator-value">
                            Current Price: {analysis.get('close', 'N/A'):.2f}<br>
                            SMA 20: {analysis.get('sma_20', 'N/A'):.2f}<br>
                            SMA 50: {analysis.get('sma_50', 'N/A'):.2f}<br>
                            SMA 200: {analysis.get('sma_200', 'N/A'):.2f}
                        </div>
                        <div class="indicator-strength">Strength: <span class="{bias_class}">{analysis.get('strength', 'NEUTRAL')}</span></div>
                        <div class="indicator-bias">Bias: <span class="{bias_class}">{analysis.get('bias', 'NEUTRAL')}</span></div>
                        <div class="indicator-interpretation">{analysis.get('interpretation', 'No interpretation available')}</div>
                    </div>
                """
            else:
                value = analysis.get('value', 'N/A')
                value_str = f"{value:.2f}" if isinstance(value, (int, float)) else str(value)
                
                html_content += f"""
                    <div class="indicator">
                        <div class="indicator-name">{indicator_name}</div>
                        <div class="indicator-value">Value: {value_str}</div>
                        <div class="indicator-strength">Strength: <span class="{bias_class}">{analysis.get('strength', 'NEUTRAL')}</span></div>
                        <div class="indicator-bias">Bias: <span class="{bias_class}">{analysis.get('bias', 'NEUTRAL')}</span></div>
                        <div class="indicator-interpretation">{analysis.get('interpretation', 'No interpretation available')}</div>
                    </div>
                """
        
        html_content += """
                    <h2>Chart Patterns</h2>
                    <div class="patterns-container">
        """
        
        if analyzer.patterns:
            for pattern in analyzer.patterns:
                html_content += f"""
                    <div class="pattern">
                        <div class="pattern-name">{pattern['type']}</div>
                        <div class="pattern-confidence">Confidence: {pattern['confidence']:.2%}</div>
                        <div class="pattern-description">{pattern['description']}</div>
                    </div>
                """
        else:
            html_content += """
                    <div class="pattern">
                        <div class="pattern-name">No Patterns Detected</div>
                        <div class="pattern-description">No significant chart patterns were detected in the current timeframe.</div>
                    </div>
            """
        
        html_content += """
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save HTML report
        html_filename = os.path.join(output_dir, f'{symbol}_analysis_{timestamp}.html')
        with open(html_filename, 'w') as f:
            f.write(html_content)
        
        # Create and save the interactive chart
        fig = go.Figure()
        
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=analyzer.data.index,
            open=analyzer.data['Open'],
            high=analyzer.data['High'],
            low=analyzer.data['Low'],
            close=analyzer.data['Close'],
            name='OHLC'
        ))
        
        # Add moving averages
        for ma_name in ['SMA_20', 'SMA_50', 'SMA_200']:
            if ma_name in analyzer.indicators:
                fig.add_trace(go.Scatter(
                    x=analyzer.data.index,
                    y=analyzer.indicators[ma_name],
                    name=ma_name,
                    line=dict(color='blue' if ma_name == 'SMA_20' else 'red' if ma_name == 'SMA_50' else 'green', width=1)
                ))
        
        # Add VWAP
        if 'VWAP' in analyzer.indicators:
            fig.add_trace(go.Scatter(
                x=analyzer.data.index,
                y=analyzer.indicators['VWAP'],
                name='VWAP',
                line=dict(color='purple', width=1)
            ))
        
        # Add Fibonacci levels
        for level, value in analyzer.indicators.items():
            if level.startswith('Fib_'):
                fig.add_hline(
                    y=value,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text=level.replace('Fib_', ''),
                    annotation_position="right"
                )
        
        # Add RSI subplot
        if 'RSI' in analyzer.indicators:
            fig.add_trace(go.Scatter(
                x=analyzer.data.index,
                y=analyzer.indicators['RSI'],
                name='RSI',
                line=dict(color='orange'),
                yaxis='y2'
            ))
        
        # Add MACD subplot
        if all(key in analyzer.indicators for key in ['MACD', 'MACD_Signal', 'MACD_Hist']):
            fig.add_trace(go.Scatter(
                x=analyzer.data.index,
                y=analyzer.indicators['MACD'],
                name='MACD',
                line=dict(color='blue'),
                yaxis='y3'
            ))
            
            fig.add_trace(go.Scatter(
                x=analyzer.data.index,
                y=analyzer.indicators['MACD_Signal'],
                name='MACD Signal',
                line=dict(color='red'),
                yaxis='y3'
            ))
            
            fig.add_trace(go.Bar(
                x=analyzer.data.index,
                y=analyzer.indicators['MACD_Hist'],
                name='MACD Histogram',
                yaxis='y3'
            ))
        
        # Update layout with subplots
        fig.update_layout(
            title=f'{analyzer.symbol} Technical Analysis',
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
            ),
            yaxis2=dict(
                title="RSI",
                overlaying="y",
                side="right",
                range=[0, 100],
                showgrid=False
            ),
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
        fig.add_hline(y=70, line_dash="dash", line_color="red", yref="y2")
        fig.add_hline(y=30, line_dash="dash", line_color="green", yref="y2")
        
        # Save chart
        chart_filename = os.path.join(output_dir, f'{symbol}_chart_{timestamp}.html')
        fig.write_html(chart_filename)
        
        print(f"\nAnalysis report saved to: {html_filename}")
        print(f"Interactive chart saved to: {chart_filename}")
        
    except Exception as e:
        print(f"Error saving analysis report: {str(e)}")

def main():
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Technical Analysis System')
    parser.add_argument('symbol', help='Stock symbol (e.g., AAPL, TATASTEEL.NS)')
    parser.add_argument('--period', default='6mo', help='Data period (e.g., 1d, 5d, 1mo, 3mo, 6mo, 1y)')
    parser.add_argument('--interval', default='1d', help='Data interval (e.g., 1m, 2m, 5m, 15m, 30m, 1h, 1d)')
    
    args = parser.parse_args()
    
    try:
        # Initialize the analyzer with command line arguments
        print("Initializing Technical Analysis System...")
        
        # Try different symbol formats for Indian stocks
        symbol = args.symbol
        if symbol.endswith('.BO') or symbol.endswith('.NS'):
            # Try alternative formats for Indian stocks
            base_symbol = symbol[:-3]
            alternative_symbols = [
                f"{base_symbol}.NS",  # Try NSE format
                f"{base_symbol}.BO",  # Try BSE format
                f"{base_symbol}"      # Try without suffix
            ]
            
            analyzer = None
            last_error = None
            
            for alt_symbol in alternative_symbols:
                try:
                    print(f"\nTrying symbol: {alt_symbol}")
                    analyzer = TechnicalAnalyzer(alt_symbol, period=args.period, interval=args.interval)
                    if analyzer.data is not None and len(analyzer.data) > 0:
                        symbol = alt_symbol
                        print(f"Successfully found data for {alt_symbol}")
                        break
                except Exception as e:
                    last_error = str(e)
                    print(f"Failed with {alt_symbol}: {str(e)}")
                    continue
            
            if analyzer is None or analyzer.data is None or len(analyzer.data) == 0:
                raise ValueError(f"No data found for symbol {symbol}. Last error: {last_error}")
        else:
            analyzer = TechnicalAnalyzer(symbol, period=args.period, interval=args.interval)
        
        # Check if we got any data
        if analyzer.data is None or len(analyzer.data) == 0:
            print(f"\nError: No data found for symbol {symbol}")
            print("\nCommon issues and solutions:")
            print("1. For Indian stocks:")
            print("   - NSE stocks: Add .NS suffix (e.g., TATASTEEL.NS)")
            print("   - BSE stocks: Add .BO suffix (e.g., TATASTEEL.BO)")
            print("2. For US stocks:")
            print("   - NYSE stocks: Use symbol directly (e.g., AAPL)")
            print("   - NASDAQ stocks: Use symbol directly (e.g., GOOGL)")
            print("3. For other markets:")
            print("   - London: Add .L suffix")
            print("   - Tokyo: Add .T suffix")
            print("\nAdditional troubleshooting:")
            print("1. Check your internet connection")
            print("2. Verify the stock symbol is correct")
            print("3. Try a different period (e.g., --period 1mo)")
            print("4. Try a different interval (e.g., --interval 1d)")
            sys.exit(1)
        
        # Detect patterns
        print("\nDetecting patterns...")
        analyzer.detect_patterns()
        
        # Get trading signal
        print("\nGenerating trading signal...")
        signal = analyzer.get_signal()
        
        # Print results
        print("\n=== Analysis Results ===")
        print(f"Signal: {signal['signal']}")
        print(f"Confidence: {signal['confidence']:.2f}")
        print("\nIndicator Values:")
        for indicator, value in signal['indicators'].items():
            if isinstance(value, dict):
                print(f"{indicator}: {value.get('value', 'N/A')}")
            else:
                print(f"{indicator}: {value}")
        
        # Save analysis report and chart
        save_analysis_report(analyzer, signal)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nPlease check:")
        print("1. Your internet connection")
        print("2. The stock symbol is correct")
        print("3. The period and interval are valid")
        sys.exit(1)

if __name__ == "__main__":
    main() 
# Technical Analysis Trading System

A Python-based technical analysis system that automatically detects trading patterns and provides trading signals based on multiple technical indicators.

## Features

- Automatic pattern detection for:
  - Head and Shoulders
  - Double Top/Bottom
  - Bullish/Bearish Flags
  - Wedges
- Technical Indicators:
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Volume Analysis
  - Moving Averages (SMA/EMA)
  - Fibonacci Retracement Levels
- Interactive visualization using Plotly
- Static visualization using Matplotlib
- Confidence scoring for pattern detection
- Trading signals with multiple confirmations

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd technical-analysis-system
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Basic usage example:

```python
from technical_analysis import TechnicalAnalyzer

# Initialize analyzer with stock symbol
analyzer = TechnicalAnalyzer('AAPL', period='6mo', interval='1d')

# Detect patterns
analyzer.detect_patterns()

# Get trading signal
signal = analyzer.get_signal()
print(f"Signal: {signal['signal']}")
print(f"Confidence: {signal['confidence']:.2f}")

# Plot the chart
analyzer.plot_chart(use_plotly=True)  # or False for matplotlib
```

## Pattern Detection

The system uses various algorithms to detect common chart patterns:

1. Head and Shoulders:
   - Identifies three peaks with the middle peak higher than the others
   - Calculates neckline and pattern height
   - Provides confidence based on pattern symmetry

2. Double Top/Bottom:
   - Detects two similar price levels with a valley/peak between them
   - Calculates pattern height and support/resistance levels
   - Provides confidence based on pattern symmetry

3. Bullish/Bearish Flags:
   - Identifies strong price movements followed by consolidation
   - Analyzes flag slope and duration
   - Provides confidence based on pattern characteristics

## Technical Indicators

The system uses multiple technical indicators to confirm trading signals:

1. RSI:
   - Overbought (>70) and oversold (<30) conditions
   - Divergence analysis

2. MACD:
   - Signal line crossovers
   - Histogram analysis
   - Divergence detection

3. Moving Averages:
   - SMA/EMA crossovers
   - Support/Resistance levels
   - Trend direction

4. Volume Analysis:
   - Volume trend confirmation
   - Volume spikes
   - VWAP analysis

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
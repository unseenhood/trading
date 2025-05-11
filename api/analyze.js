// Import required packages
const yfinance = require('yfinance');
const { TechnicalAnalyzer } = require('../technical_analysis');

export default async function handler(req, res) {
  // Only allow POST requests
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const data = req.body;
    const symbol = data.symbol;
    const period = data.period || '6mo';
    const interval = data.interval || '1d';

    if (!symbol) {
      return res.status(400).json({ error: 'Symbol is required' });
    }

    // Initialize the analyzer
    const analyzer = new TechnicalAnalyzer(symbol, period, interval);
    
    // Detect patterns
    await analyzer.detectPatterns();
    
    // Get trading signal
    const signal = await analyzer.getSignal();

    // Clean NaN values
    const cleanNaN = (obj) => {
      if (typeof obj === 'number' && isNaN(obj)) {
        return null;
      }
      if (Array.isArray(obj)) {
        return obj.map(cleanNaN);
      }
      if (obj && typeof obj === 'object') {
        return Object.fromEntries(
          Object.entries(obj).map(([k, v]) => [k, cleanNaN(v)])
        );
      }
      return obj;
    };

    // Prepare response data
    const responseData = {
      success: true,
      current_price: analyzer.data.Close[analyzer.data.Close.length - 1],
      stop_loss: signal.stop_loss,
      price_targets: signal.price_targets,
      indicators: signal.indicators,
      patterns: analyzer.patterns
    };

    // Clean NaN values before sending response
    const cleanedData = cleanNaN(responseData);

    return res.status(200).json(cleanedData);

  } catch (error) {
    return res.status(500).json({
      success: false,
      error: error.message
    });
  }
}

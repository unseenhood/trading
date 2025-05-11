import numpy as np
from scipy.signal import argrelextrema

class PatternDetector:
    def __init__(self, data):
        """
        Initialize pattern detector with price data
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data
        """
        try:
            if data is None or len(data) == 0:
                raise ValueError("No data provided")
                
            if not all(col in data.columns for col in ['High', 'Low', 'Close']):
                raise ValueError("Data must contain High, Low, and Close columns")
                
            self.data = data
            self.highs = None
            self.lows = None
            self.find_extrema()
            
        except Exception as e:
            print(f"Error initializing PatternDetector: {str(e)}")
            self.data = None
            self.highs = np.array([0])
            self.lows = np.array([0])
        
    def find_extrema(self, order=5):
        """Find local maxima and minima in price data"""
        try:
            if self.data is None or len(self.data) == 0:
                print("No data available for finding extrema")
                self.highs = np.array([0])
                self.lows = np.array([0])
                return
                
            if len(self.data) < order * 2:
                print(f"Insufficient data points for order {order}. Need at least {order * 2} points.")
                self.highs = np.array([0])
                self.lows = np.array([0])
                return
                
            self.highs = argrelextrema(self.data['High'].values, np.greater, order=order)[0]
            self.lows = argrelextrema(self.data['Low'].values, np.less, order=order)[0]
            
            # Ensure we have at least one high and one low
            if len(self.highs) == 0:
                self.highs = np.array([0])
            if len(self.lows) == 0:
                self.lows = np.array([0])
                
        except Exception as e:
            print(f"Error in find_extrema: {str(e)}")
            self.highs = np.array([0])
            self.lows = np.array([0])
        
    def detect_head_and_shoulders(self, tolerance=0.02):
        """
        Detect head and shoulders pattern with detailed analysis
        """
        if len(self.highs) < 3:
            return {'detected': False, 'confidence': 0}
        
        for i in range(len(self.highs) - 2):
            left_shoulder = self.data['High'].iloc[self.highs[i]]
            head = self.data['High'].iloc[self.highs[i + 1]]
            right_shoulder = self.data['High'].iloc[self.highs[i + 2]]
            
            # Check pattern criteria
            if head > left_shoulder and head > right_shoulder:
                # Calculate neckline
                neckline = min(self.data['Low'].iloc[self.highs[i]:self.highs[i + 2]])
                pattern_height = head - neckline
                
                # Calculate confidence based on multiple factors
                confidence_factors = {
                    'shoulder_symmetry': 1 - abs(left_shoulder - right_shoulder) / left_shoulder,
                    'head_prominence': (head - max(left_shoulder, right_shoulder)) / head,
                    'volume_confirmation': self._check_volume_confirmation(i),
                    'trend_alignment': self._check_trend_alignment(i)
                }
                
                # Weight and combine confidence factors
                confidence = (
                    confidence_factors['shoulder_symmetry'] * 0.3 +
                    confidence_factors['head_prominence'] * 0.3 +
                    confidence_factors['volume_confirmation'] * 0.2 +
                    confidence_factors['trend_alignment'] * 0.2
                )
                
                return {
                    'detected': True,
                    'confidence': confidence,
                    'pattern_height': pattern_height,
                    'neckline': neckline,
                    'details': {
                        'left_shoulder': left_shoulder,
                        'head': head,
                        'right_shoulder': right_shoulder,
                        'confidence_factors': confidence_factors
                    }
                }
        
        return {'detected': False, 'confidence': 0}

    def _check_volume_confirmation(self, start_idx):
        """Check if volume confirms the pattern"""
        try:
            pattern_volume = self.data['Volume'].iloc[start_idx:self.highs[start_idx + 2]]
            avg_volume = pattern_volume.mean()
            head_volume = pattern_volume.iloc[len(pattern_volume)//2]
            
            # Higher volume on shoulders than head is bearish confirmation
            return min(1.0, head_volume / avg_volume)
        except:
            return 0.5

    def _check_trend_alignment(self, start_idx):
        """Check if pattern aligns with overall trend"""
        try:
            pre_pattern_trend = self.data['Close'].iloc[max(0, start_idx-20):start_idx].mean()
            pattern_price = self.data['Close'].iloc[start_idx]
            
            # Pattern should form at end of uptrend
            return min(1.0, pattern_price / pre_pattern_trend)
        except:
            return 0.5
        
    def detect_double_top(self, tolerance=0.02):
        """
        Detect double top pattern
        
        Args:
            tolerance (float): Tolerance for price level comparison
            
        Returns:
            dict: Pattern detection results
        """
        if len(self.highs) < 2:
            return {'detected': False, 'confidence': 0}
            
        for i in range(len(self.highs) - 1):
            first_top = self.data['High'].iloc[self.highs[i]]
            second_top = self.data['High'].iloc[self.highs[i + 1]]
            
            # Check if tops are at similar levels
            if abs(first_top - second_top) / first_top < tolerance:
                # Find the valley between tops
                valley_idx = self.lows[(self.lows > self.highs[i]) & (self.lows < self.highs[i + 1])]
                if len(valley_idx) > 0:
                    valley = self.data['Low'].iloc[valley_idx[0]]
                    
                    # Calculate pattern height
                    pattern_height = first_top - valley
                    
                    # Calculate confidence based on pattern symmetry
                    symmetry = 1 - abs(first_top - second_top) / first_top
                    
                    return {
                        'detected': True,
                        'confidence': symmetry,
                        'pattern_height': pattern_height,
                        'support': valley
                    }
                    
        return {'detected': False, 'confidence': 0}
        
    def detect_double_bottom(self, tolerance=0.02):
        """
        Detect double bottom pattern
        
        Args:
            tolerance (float): Tolerance for price level comparison
            
        Returns:
            dict: Pattern detection results
        """
        if len(self.lows) < 2:
            return {'detected': False, 'confidence': 0}
            
        for i in range(len(self.lows) - 1):
            first_bottom = self.data['Low'].iloc[self.lows[i]]
            second_bottom = self.data['Low'].iloc[self.lows[i + 1]]
            
            # Check if bottoms are at similar levels
            if abs(first_bottom - second_bottom) / first_bottom < tolerance:
                # Find the peak between bottoms
                peak_idx = self.highs[(self.highs > self.lows[i]) & (self.highs < self.lows[i + 1])]
                if len(peak_idx) > 0:
                    peak = self.data['High'].iloc[peak_idx[0]]
                    
                    # Calculate pattern height
                    pattern_height = peak - first_bottom
                    
                    # Calculate confidence based on pattern symmetry
                    symmetry = 1 - abs(first_bottom - second_bottom) / first_bottom
                    
                    return {
                        'detected': True,
                        'confidence': symmetry,
                        'pattern_height': pattern_height,
                        'resistance': peak
                    }
                    
        return {'detected': False, 'confidence': 0}
        
    def detect_bullish_flag(self, min_rise=0.1, max_rise=0.3, min_duration=5, max_duration=20):
        """
        Detect bullish flag pattern
        
        Args:
            min_rise (float): Minimum price rise for pole
            max_rise (float): Maximum price rise for pole
            min_duration (int): Minimum duration for flag
            max_duration (int): Maximum duration for flag
            
        Returns:
            dict: Pattern detection results
        """
        try:
            if len(self.highs) < 2 or len(self.lows) < 1:
                return {'detected': False, 'confidence': 0}
                
            for i in range(len(self.highs) - 1):
                # Find potential pole
                pole_start_idx = self.lows[self.lows < self.highs[i]]
                if len(pole_start_idx) == 0:
                    continue
                    
                pole_start = pole_start_idx[-1]
                pole_end = self.highs[i]
                
                try:
                    pole_rise = (self.data['High'].iloc[pole_end] - self.data['Low'].iloc[pole_start]) / self.data['Low'].iloc[pole_start]
                    
                    if min_rise <= pole_rise <= max_rise:
                        # Look for flag
                        flag_duration = self.highs[i + 1] - self.highs[i]
                        
                        if min_duration <= flag_duration <= max_duration:
                            # Check if flag is sloping downward
                            flag_highs = self.data['High'].iloc[self.highs[i]:self.highs[i + 1]]
                            flag_lows = self.data['Low'].iloc[self.lows[(self.lows > self.highs[i]) & (self.lows < self.highs[i + 1])]]
                            
                            if len(flag_highs) > 1 and len(flag_lows) > 1:
                                high_slope = np.polyfit(range(len(flag_highs)), flag_highs, 1)[0]
                                low_slope = np.polyfit(range(len(flag_lows)), flag_lows, 1)[0]
                                
                                if high_slope < 0 and low_slope < 0:
                                    return {
                                        'detected': True,
                                        'confidence': 0.7,
                                        'pole_rise': pole_rise,
                                        'flag_duration': flag_duration
                                    }
                except Exception as e:
                    print(f"Error processing bullish flag at index {i}: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"Error in detect_bullish_flag: {str(e)}")
            
        return {'detected': False, 'confidence': 0}
        
    def detect_bearish_flag(self, min_fall=0.1, max_fall=0.3, min_duration=5, max_duration=20):
        """
        Detect bearish flag pattern
        
        Args:
            min_fall (float): Minimum price fall for pole
            max_fall (float): Maximum price fall for pole
            min_duration (int): Minimum duration for flag
            max_duration (int): Maximum duration for flag
            
        Returns:
            dict: Pattern detection results
        """
        try:
            if len(self.lows) < 2 or len(self.highs) < 1:
                return {'detected': False, 'confidence': 0}
                
            for i in range(len(self.lows) - 1):
                # Find potential pole
                pole_start_idx = self.highs[self.highs > self.lows[i]]  # Changed from < to >
                if len(pole_start_idx) == 0:
                    continue
                    
                pole_start = pole_start_idx[0]  # Changed from -1 to 0
                pole_end = self.lows[i]
                
                try:
                    pole_fall = (self.data['High'].iloc[pole_start] - self.data['Low'].iloc[pole_end]) / self.data['High'].iloc[pole_start]
                    
                    if min_fall <= pole_fall <= max_fall:
                        # Look for flag
                        flag_duration = self.lows[i + 1] - self.lows[i]
                        
                        if min_duration <= flag_duration <= max_duration:
                            # Check if flag is sloping upward
                            flag_highs = self.data['High'].iloc[self.highs[(self.highs > self.lows[i]) & (self.highs < self.lows[i + 1])]]
                            flag_lows = self.data['Low'].iloc[self.lows[i]:self.lows[i + 1]]
                            
                            if len(flag_highs) > 1 and len(flag_lows) > 1:
                                high_slope = np.polyfit(range(len(flag_highs)), flag_highs, 1)[0]
                                low_slope = np.polyfit(range(len(flag_lows)), flag_lows, 1)[0]
                                
                                if high_slope > 0 and low_slope > 0:
                                    return {
                                        'detected': True,
                                        'confidence': 0.7,
                                        'pole_fall': pole_fall,
                                        'flag_duration': flag_duration
                                    }
                except Exception as e:
                    print(f"Error processing bearish flag at index {i}: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"Error in detect_bearish_flag: {str(e)}")
            
        return {'detected': False, 'confidence': 0}
        
    def get_pattern_points(self, pattern_type):
        """Get points for pattern visualization"""
        try:
            if pattern_type == 'head_and_shoulders':
                for i in range(len(self.highs) - 2):
                    if i + 2 >= len(self.highs):
                        continue
                    left_shoulder = self.data['High'].iloc[self.highs[i]]
                    head = self.data['High'].iloc[self.highs[i + 1]]
                    right_shoulder = self.data['High'].iloc[self.highs[i + 2]]
                    
                    if head > left_shoulder and head > right_shoulder:
                        return [
                            (self.data.index[self.highs[i]], left_shoulder),
                            (self.data.index[self.highs[i + 1]], head),
                            (self.data.index[self.highs[i + 2]], right_shoulder)
                        ]
                    
            elif pattern_type == 'double_top':
                for i in range(len(self.highs) - 1):
                    if i + 1 >= len(self.highs):
                        continue
                    first_top = self.data['High'].iloc[self.highs[i]]
                    second_top = self.data['High'].iloc[self.highs[i + 1]]
                    
                    if abs(first_top - second_top) / first_top < 0.02:
                        return [
                            (self.data.index[self.highs[i]], first_top),
                            (self.data.index[self.highs[i + 1]], second_top)
                        ]
                    
            elif pattern_type == 'double_bottom':
                for i in range(len(self.lows) - 1):
                    if i + 1 >= len(self.lows):
                        continue
                    first_bottom = self.data['Low'].iloc[self.lows[i]]
                    second_bottom = self.data['Low'].iloc[self.lows[i + 1]]
                    
                    if abs(first_bottom - second_bottom) / first_bottom < 0.02:
                        return [
                            (self.data.index[self.lows[i]], first_bottom),
                            (self.data.index[self.lows[i + 1]], second_bottom)
                        ]
                    
            elif pattern_type == 'bullish_flag':
                for i in range(len(self.highs) - 1):
                    if i + 1 >= len(self.highs):
                        continue
                    pole_start_idx = self.lows[self.lows < self.highs[i]]
                    if len(pole_start_idx) == 0:
                        continue
                    pole_start = pole_start_idx[-1]
                    pole_end = self.highs[i]
                    
                    if 0.1 <= (self.data['High'].iloc[pole_end] - self.data['Low'].iloc[pole_start]) / self.data['Low'].iloc[pole_start] <= 0.3:
                        return [
                            (self.data.index[pole_start], self.data['Low'].iloc[pole_start]),
                            (self.data.index[pole_end], self.data['High'].iloc[pole_end])
                        ]
                    
            elif pattern_type == 'bearish_flag':
                for i in range(len(self.lows) - 1):
                    if i + 1 >= len(self.lows):
                        continue
                    pole_start_idx = self.highs[self.highs < self.lows[i]]
                    if len(pole_start_idx) == 0:
                        continue
                    pole_start = pole_start_idx[-1]
                    pole_end = self.lows[i]
                    
                    if 0.1 <= (self.data['High'].iloc[pole_start] - self.data['Low'].iloc[pole_end]) / self.data['High'].iloc[pole_start] <= 0.3:
                        return [
                            (self.data.index[pole_start], self.data['High'].iloc[pole_start]),
                            (self.data.index[pole_end], self.data['Low'].iloc[pole_end])
                        ]
                    
            return []
        except Exception as e:
            print(f"Error in get_pattern_points: {str(e)}")
            return [] 
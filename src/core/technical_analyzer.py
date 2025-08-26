"""
High-performance technical analysis engine optimized for crypto and AI consumption
Focus on high-ROI indicators that maximize trading profitability
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime

from .config import Config

logger = logging.getLogger(__name__)

class SignalStrength(Enum):
    """Signal strength classification"""
    VERY_STRONG = 5
    STRONG = 4  
    MODERATE = 3
    WEAK = 2
    VERY_WEAK = 1

class TrendDirection(Enum):
    """Trend direction classification"""
    BULLISH = 1
    NEUTRAL = 0
    BEARISH = -1

@dataclass
class TechnicalSignal:
    """Structured technical signal for AI analysis"""
    indicator: str
    signal_type: str  # 'buy', 'sell', 'hold', 'warning'
    strength: SignalStrength
    confidence: float  # 0-1
    timeframe: str
    price_target: Optional[float]
    stop_loss: Optional[float]
    risk_reward_ratio: Optional[float]
    reasoning: str
    timestamp: str

class CryptoIndicators:
    """Crypto-optimized technical indicators beyond standard TA-Lib"""
    
    @staticmethod
    def ichimoku_cloud(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate Ichimoku Cloud - excellent for crypto trend analysis"""
        
        # Tenkan-sen (Conversion Line): (9-period high + low)/2
        period_9_high = pd.Series(high).rolling(9).max()
        period_9_low = pd.Series(low).rolling(9).min()
        tenkan_sen = (period_9_high + period_9_low) / 2
        
        # Kijun-sen (Base Line): (26-period high + low)/2  
        period_26_high = pd.Series(high).rolling(26).max()
        period_26_low = pd.Series(low).rolling(26).min()
        kijun_sen = (period_26_high + period_26_low) / 2
        
        # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 projected 26 periods forward
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        # Senkou Span B (Leading Span B): (52-period high + low)/2 projected 26 periods forward
        period_52_high = pd.Series(high).rolling(52).max()
        period_52_low = pd.Series(low).rolling(52).min()
        senkou_span_b = ((period_52_high + period_52_low) / 2).shift(26)
        
        # Chikou Span (Lagging Span): Close shifted back 26 periods
        chikou_span = pd.Series(close).shift(-26)
        
        return {
            'tenkan_sen': tenkan_sen.values,
            'kijun_sen': kijun_sen.values, 
            'senkou_span_a': senkou_span_a.values,
            'senkou_span_b': senkou_span_b.values,
            'chikou_span': chikou_span.values
        }
    
    @staticmethod
    def supertrend(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                   period: int = 10, multiplier: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """SuperTrend indicator - excellent for crypto trend following"""
        
        hl2 = (high + low) / 2
        atr = talib.ATR(high, low, close, timeperiod=period)
        
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        supertrend = np.full_like(close, np.nan)
        direction = np.full_like(close, 1, dtype=int)
        
        for i in range(period, len(close)):
            # Determine trend direction
            if close[i] <= lower_band[i-1]:
                direction[i] = -1
            elif close[i] >= upper_band[i-1]:
                direction[i] = 1
            else:
                direction[i] = direction[i-1]
            
            # Set SuperTrend value
            if direction[i] == 1:
                supertrend[i] = lower_band[i]
            else:
                supertrend[i] = upper_band[i]
        
        return supertrend, direction
    
    @staticmethod
    def vwap(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Volume Weighted Average Price - critical for crypto entry/exit"""
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()
    
    @staticmethod
    def crypto_volatility_bands(close: np.ndarray, period: int = 20, std_dev: float = 2.5) -> Dict[str, np.ndarray]:
        """Crypto-optimized volatility bands (wider than traditional Bollinger)"""
        sma = talib.SMA(close, timeperiod=period)
        std = pd.Series(close).rolling(period).std()
        
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        
        return {
            'upper_band': upper_band,
            'middle_band': sma,
            'lower_band': lower_band,
            'width': (upper_band - lower_band) / sma * 100
        }

class TechnicalAnalyzer:
    """World-class technical analysis engine optimized for crypto ROI"""
    
    def __init__(self):
        self.config = Config.THRESHOLDS
        self.crypto_indicators = CryptoIndicators()
        
    def analyze(self, data: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Complete technical analysis returning AI-optimized insights"""
        
        if len(data) < 50:
            logger.warning(f"Insufficient data for {symbol} {timeframe}: {len(data)} candles")
            return self._create_empty_analysis(symbol, timeframe)
        
        try:
            close = data['close'].values
            high = data['high'].values
            low = data['low'].values
            volume = data['volume'].values
            
            analysis = {
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': datetime.now().isoformat(),
                'data_quality': self._assess_data_quality(data),
                'indicators': {},
                'signals': [],
                'trend_analysis': {},
                'support_resistance': {},
                'volume_analysis': {},
                'risk_metrics': {},
                'ai_summary': {}
            }
            
            # Calculate all indicators
            analysis['indicators'] = self._calculate_all_indicators(high, low, close, volume)
            
            # Generate trading signals
            analysis['signals'] = self._generate_signals(analysis['indicators'], symbol, timeframe)
            
            # Trend analysis
            analysis['trend_analysis'] = self._analyze_trend(analysis['indicators'])
            
            # Support/Resistance levels
            analysis['support_resistance'] = self._find_support_resistance(high, low, close)
            
            # Volume analysis  
            analysis['volume_analysis'] = self._analyze_volume(close, volume)
            
            # Risk metrics
            analysis['risk_metrics'] = self._calculate_risk_metrics(close, analysis['indicators'])
            
            # AI summary for decision making
            analysis['ai_summary'] = self._generate_ai_summary(analysis, symbol, timeframe)
            
            logger.info(f"✅ Technical analysis complete for {symbol} {timeframe}")
            return analysis
            
        except Exception as e:
            logger.error(f"Technical analysis failed for {symbol} {timeframe}: {e}")
            return self._create_empty_analysis(symbol, timeframe)
    
    def _calculate_all_indicators(self, high: np.ndarray, low: np.ndarray, 
                                 close: np.ndarray, volume: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive set of crypto-optimized indicators"""
        
        indicators = {}
        
        # === TREND INDICATORS ===
        # Moving Averages
        indicators['sma_20'] = talib.SMA(close, timeperiod=20)[-1]
        indicators['sma_50'] = talib.SMA(close, timeperiod=50)[-1]
        indicators['sma_200'] = talib.SMA(close, timeperiod=200)[-1] if len(close) >= 200 else np.nan
        
        indicators['ema_12'] = talib.EMA(close, timeperiod=12)[-1]
        indicators['ema_26'] = talib.EMA(close, timeperiod=26)[-1]
        indicators['ema_50'] = talib.EMA(close, timeperiod=50)[-1]
        
        # Ichimoku Cloud
        ichimoku = self.crypto_indicators.ichimoku_cloud(high, low, close)
        for key, values in ichimoku.items():
            indicators[f'ichimoku_{key}'] = values[-1] if len(values) > 0 and not np.isnan(values[-1]) else np.nan
        
        # SuperTrend
        supertrend, direction = self.crypto_indicators.supertrend(high, low, close)
        indicators['supertrend'] = supertrend[-1]
        indicators['supertrend_direction'] = direction[-1]
        
        # === MOMENTUM INDICATORS ===
        # RSI variants (crypto-optimized periods)
        indicators['rsi_14'] = talib.RSI(close, timeperiod=14)[-1]
        indicators['rsi_7'] = talib.RSI(close, timeperiod=7)[-1]  # More responsive for crypto
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        indicators['macd'] = macd[-1]
        indicators['macd_signal'] = macd_signal[-1]
        indicators['macd_histogram'] = macd_hist[-1]
        
        # Stochastic
        stoch_k, stoch_d = talib.STOCH(high, low, close, 
                                       fastk_period=14, slowk_period=3, slowd_period=3)
        indicators['stoch_k'] = stoch_k[-1]
        indicators['stoch_d'] = stoch_d[-1]
        
        # Williams %R
        indicators['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)[-1]
        
        # === VOLATILITY INDICATORS ===
        # Crypto-optimized Bollinger Bands
        crypto_bands = self.crypto_indicators.crypto_volatility_bands(close)
        indicators['bb_upper'] = crypto_bands['upper_band'].iloc[-1]
        indicators['bb_middle'] = crypto_bands['middle_band'][-1]
        indicators['bb_lower'] = crypto_bands['lower_band'].iloc[-1]
        indicators['bb_width'] = crypto_bands['width'].iloc[-1]
        indicators['bb_position'] = (close[-1] - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
        
        # ATR (Average True Range)
        indicators['atr'] = talib.ATR(high, low, close, timeperiod=14)[-1]
        indicators['atr_percentage'] = indicators['atr'] / close[-1] * 100
        
        # === VOLUME INDICATORS ===
        # Volume analysis
        indicators['volume_sma'] = talib.SMA(volume, timeperiod=20)[-1]
        indicators['volume_ratio'] = volume[-1] / indicators['volume_sma']
        
        # VWAP
        indicators['vwap'] = self.crypto_indicators.vwap(high, low, close, volume)[-1]
        indicators['vwap_distance'] = (close[-1] - indicators['vwap']) / indicators['vwap'] * 100
        
        # On Balance Volume
        indicators['obv'] = talib.OBV(close, volume)[-1]
        
        # Money Flow Index
        indicators['mfi'] = talib.MFI(high, low, close, volume, timeperiod=14)[-1]
        
        # === CURRENT PRICE METRICS ===
        indicators['current_price'] = close[-1]
        indicators['price_change_24h'] = (close[-1] - close[-24]) / close[-24] * 100 if len(close) >= 24 else 0
        indicators['price_change_7d'] = (close[-1] - close[-168]) / close[-168] * 100 if len(close) >= 168 else 0
        
        return {k: v for k, v in indicators.items() if not (isinstance(v, float) and np.isnan(v))}
    
    def _generate_signals(self, indicators: Dict[str, Any], symbol: str, timeframe: str) -> List[TechnicalSignal]:
        """Generate high-ROI trading signals optimized for crypto"""
        
        signals = []
        timestamp = datetime.now().isoformat()
        
        # RSI Signals (crypto-optimized thresholds)
        if 'rsi_14' in indicators:
            rsi = indicators['rsi_14']
            
            if rsi <= self.config['rsi_extreme_oversold']:
                signals.append(TechnicalSignal(
                    indicator='rsi_14',
                    signal_type='strong_buy',
                    strength=SignalStrength.VERY_STRONG,
                    confidence=min((self.config['rsi_oversold'] - rsi) / 10, 1.0),
                    timeframe=timeframe,
                    price_target=indicators['current_price'] * 1.15,  # 15% target
                    stop_loss=indicators['current_price'] * 0.95,     # 5% stop
                    risk_reward_ratio=3.0,
                    reasoning=f"Extreme oversold RSI: {rsi:.1f}",
                    timestamp=timestamp
                ))
            elif rsi <= self.config['rsi_oversold']:
                signals.append(TechnicalSignal(
                    indicator='rsi_14',
                    signal_type='buy',
                    strength=SignalStrength.STRONG,
                    confidence=(self.config['rsi_oversold'] - rsi) / 5,
                    timeframe=timeframe,
                    price_target=indicators['current_price'] * 1.10,
                    stop_loss=indicators['current_price'] * 0.97,
                    risk_reward_ratio=2.5,
                    reasoning=f"Oversold RSI: {rsi:.1f}",
                    timestamp=timestamp
                ))
            elif rsi >= self.config['rsi_extreme_overbought']:
                signals.append(TechnicalSignal(
                    indicator='rsi_14',
                    signal_type='strong_sell',
                    strength=SignalStrength.VERY_STRONG,
                    confidence=min((rsi - self.config['rsi_overbought']) / 10, 1.0),
                    timeframe=timeframe,
                    price_target=indicators['current_price'] * 0.90,
                    stop_loss=indicators['current_price'] * 1.03,
                    risk_reward_ratio=3.5,
                    reasoning=f"Extreme overbought RSI: {rsi:.1f}",
                    timestamp=timestamp
                ))
            elif rsi >= self.config['rsi_overbought']:
                signals.append(TechnicalSignal(
                    indicator='rsi_14',
                    signal_type='sell',
                    strength=SignalStrength.STRONG,
                    confidence=(rsi - self.config['rsi_overbought']) / 5,
                    timeframe=timeframe,
                    price_target=indicators['current_price'] * 0.93,
                    stop_loss=indicators['current_price'] * 1.02,
                    risk_reward_ratio=2.5,
                    reasoning=f"Overbought RSI: {rsi:.1f}",
                    timestamp=timestamp
                ))
        
        # MACD Signals
        if all(k in indicators for k in ['macd', 'macd_signal', 'macd_histogram']):
            macd = indicators['macd']
            signal = indicators['macd_signal']
            histogram = indicators['macd_histogram']
            
            if macd > signal and histogram > 0:
                strength = SignalStrength.STRONG if abs(macd - signal) > 0.5 else SignalStrength.MODERATE
                signals.append(TechnicalSignal(
                    indicator='macd',
                    signal_type='buy',
                    strength=strength,
                    confidence=min(abs(macd - signal) * 2, 1.0),
                    timeframe=timeframe,
                    price_target=indicators['current_price'] * 1.08,
                    stop_loss=indicators['current_price'] * 0.96,
                    risk_reward_ratio=2.0,
                    reasoning=f"MACD bullish crossover",
                    timestamp=timestamp
                ))
            elif macd < signal and histogram < 0:
                strength = SignalStrength.STRONG if abs(macd - signal) > 0.5 else SignalStrength.MODERATE
                signals.append(TechnicalSignal(
                    indicator='macd',
                    signal_type='sell',
                    strength=strength,
                    confidence=min(abs(macd - signal) * 2, 1.0),
                    timeframe=timeframe,
                    price_target=indicators['current_price'] * 0.92,
                    stop_loss=indicators['current_price'] * 1.04,
                    risk_reward_ratio=2.0,
                    reasoning=f"MACD bearish crossover",
                    timestamp=timestamp
                ))
        
        # Bollinger Bands Signals
        if 'bb_position' in indicators:
            bb_pos = indicators['bb_position']
            bb_width = indicators.get('bb_width', 0)
            
            if bb_pos >= 0.95:  # Near upper band
                signals.append(TechnicalSignal(
                    indicator='bollinger_bands',
                    signal_type='sell',
                    strength=SignalStrength.STRONG,
                    confidence=min((bb_pos - 0.8) / 0.2, 1.0),
                    timeframe=timeframe,
                    price_target=indicators['bb_middle'],
                    stop_loss=indicators['current_price'] * 1.02,
                    risk_reward_ratio=2.5,
                    reasoning=f"Price at upper Bollinger Band ({bb_pos:.2f})",
                    timestamp=timestamp
                ))
            elif bb_pos <= 0.05:  # Near lower band
                signals.append(TechnicalSignal(
                    indicator='bollinger_bands',
                    signal_type='buy',
                    strength=SignalStrength.STRONG,
                    confidence=min((0.2 - bb_pos) / 0.2, 1.0),
                    timeframe=timeframe,
                    price_target=indicators['bb_middle'],
                    stop_loss=indicators['current_price'] * 0.98,
                    risk_reward_ratio=2.5,
                    reasoning=f"Price at lower Bollinger Band ({bb_pos:.2f})",
                    timestamp=timestamp
                ))
            
            # Bollinger Band squeeze (low volatility before breakout)
            if bb_width < self.config['bb_squeeze']:
                signals.append(TechnicalSignal(
                    indicator='bollinger_squeeze',
                    signal_type='warning',
                    strength=SignalStrength.MODERATE,
                    confidence=0.7,
                    timeframe=timeframe,
                    price_target=None,
                    stop_loss=None,
                    risk_reward_ratio=None,
                    reasoning=f"Bollinger Band squeeze - breakout imminent",
                    timestamp=timestamp
                ))
        
        # Volume Signals
        if 'volume_ratio' in indicators:
            vol_ratio = indicators['volume_ratio']
            
            if vol_ratio >= self.config['volume_surge']:
                signals.append(TechnicalSignal(
                    indicator='volume',
                    signal_type='warning',
                    strength=SignalStrength.STRONG,
                    confidence=min(vol_ratio / 5, 1.0),
                    timeframe=timeframe,
                    price_target=None,
                    stop_loss=None,
                    risk_reward_ratio=None,
                    reasoning=f"Volume surge: {vol_ratio:.1f}x average",
                    timestamp=timestamp
                ))
        
        # SuperTrend Signals
        if 'supertrend_direction' in indicators and 'current_price' in indicators:
            direction = indicators['supertrend_direction']
            supertrend_value = indicators.get('supertrend', 0)
            
            if direction == 1 and indicators['current_price'] > supertrend_value:
                signals.append(TechnicalSignal(
                    indicator='supertrend',
                    signal_type='buy',
                    strength=SignalStrength.STRONG,
                    confidence=0.8,
                    timeframe=timeframe,
                    price_target=indicators['current_price'] * 1.12,
                    stop_loss=supertrend_value,
                    risk_reward_ratio=3.0,
                    reasoning="SuperTrend bullish",
                    timestamp=timestamp
                ))
            elif direction == -1 and indicators['current_price'] < supertrend_value:
                signals.append(TechnicalSignal(
                    indicator='supertrend',
                    signal_type='sell',
                    strength=SignalStrength.STRONG,
                    confidence=0.8,
                    timeframe=timeframe,
                    price_target=indicators['current_price'] * 0.88,
                    stop_loss=supertrend_value,
                    risk_reward_ratio=3.0,
                    reasoning="SuperTrend bearish",
                    timestamp=timestamp
                ))
        
        return signals
    
    def _analyze_trend(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive trend analysis"""
        
        trend_analysis = {
            'direction': TrendDirection.NEUTRAL,
            'strength': 0.5,
            'confidence': 0.5,
            'key_levels': {},
            'trend_factors': []
        }
        
        if 'current_price' not in indicators:
            return trend_analysis
        
        price = indicators['current_price']
        trend_score = 0
        factors = []
        
        # Moving average analysis
        if all(k in indicators for k in ['sma_20', 'sma_50']):
            sma_20 = indicators['sma_20']
            sma_50 = indicators['sma_50']
            
            if price > sma_20 > sma_50:
                trend_score += 0.3
                factors.append("Price above both MA20 and MA50")
            elif price < sma_20 < sma_50:
                trend_score -= 0.3
                factors.append("Price below both MA20 and MA50")
            
            trend_analysis['key_levels']['sma_20'] = sma_20
            trend_analysis['key_levels']['sma_50'] = sma_50
        
        # SuperTrend analysis
        if 'supertrend_direction' in indicators:
            direction = indicators['supertrend_direction']
            if direction == 1:
                trend_score += 0.25
                factors.append("SuperTrend bullish")
            elif direction == -1:
                trend_score -= 0.25
                factors.append("SuperTrend bearish")
        
        # MACD trend
        if 'macd' in indicators and 'macd_signal' in indicators:
            macd_diff = indicators['macd'] - indicators['macd_signal']
            if macd_diff > 0:
                trend_score += 0.2
                factors.append("MACD above signal line")
            else:
                trend_score -= 0.2
                factors.append("MACD below signal line")
        
        # Determine final trend
        trend_analysis['strength'] = abs(trend_score)
        trend_analysis['confidence'] = min(abs(trend_score) * 2, 1.0)
        
        if trend_score > 0.3:
            trend_analysis['direction'] = TrendDirection.BULLISH
        elif trend_score < -0.3:
            trend_analysis['direction'] = TrendDirection.BEARISH
        else:
            trend_analysis['direction'] = TrendDirection.NEUTRAL
        
        trend_analysis['trend_factors'] = factors
        
        return trend_analysis
    
    def _find_support_resistance(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, Any]:
        """Find key support and resistance levels"""
        
        lookback = min(50, len(high))
        
        support_resistance = {
            'immediate_support': np.min(low[-20:]) if len(low) >= 20 else low[-1],
            'immediate_resistance': np.max(high[-20:]) if len(high) >= 20 else high[-1],
            'strong_support': np.min(low[-lookback:]),
            'strong_resistance': np.max(high[-lookback:]),
            'pivot_levels': []
        }
        
        # Calculate pivot points
        if len(high) >= 3:
            high_prev = high[-2]
            low_prev = low[-2]
            close_prev = close[-2]
            
            pivot = (high_prev + low_prev + close_prev) / 3
            
            support_resistance['pivot_levels'] = {
                'pivot_point': pivot,
                'resistance_1': 2 * pivot - low_prev,
                'support_1': 2 * pivot - high_prev,
                'resistance_2': pivot + (high_prev - low_prev),
                'support_2': pivot - (high_prev - low_prev)
            }
        
        return support_resistance
    
    def _analyze_volume(self, close: np.ndarray, volume: np.ndarray) -> Dict[str, Any]:
        """Analyze volume patterns and trends"""
        
        volume_analysis = {
            'trend': 'neutral',
            'strength': 0.5,
            'patterns': [],
            'volume_price_trend': 'neutral'
        }
        
        if len(volume) < 20:
            return volume_analysis
        
        # Volume trend analysis
        recent_volume = np.mean(volume[-10:])
        older_volume = np.mean(volume[-30:-10]) if len(volume) >= 30 else np.mean(volume[:-10])
        
        if recent_volume > older_volume * 1.2:
            volume_analysis['trend'] = 'increasing'
            volume_analysis['strength'] = min((recent_volume / older_volume - 1) * 2, 1.0)
            volume_analysis['patterns'].append('Volume increasing')
        elif recent_volume < older_volume * 0.8:
            volume_analysis['trend'] = 'decreasing'
            volume_analysis['strength'] = min((1 - recent_volume / older_volume) * 2, 1.0)
            volume_analysis['patterns'].append('Volume decreasing')
        
        # Volume-Price Trend
        price_change = (close[-1] - close[-10]) / close[-10]
        volume_change = (recent_volume - older_volume) / older_volume
        
        if price_change > 0 and volume_change > 0:
            volume_analysis['volume_price_trend'] = 'bullish_confirmation'
        elif price_change < 0 and volume_change > 0:
            volume_analysis['volume_price_trend'] = 'bearish_confirmation'
        elif price_change > 0 and volume_change < 0:
            volume_analysis['volume_price_trend'] = 'bullish_divergence'
        elif price_change < 0 and volume_change < 0:
            volume_analysis['volume_price_trend'] = 'bearish_divergence'
        
        return volume_analysis
    
    def _calculate_risk_metrics(self, close: np.ndarray, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics"""
        
        risk_metrics = {
            'volatility_score': 0.5,
            'risk_level': 'medium',
            'sharpe_estimate': 0.0,
            'max_drawdown_estimate': 0.0,
            'risk_factors': []
        }
        
        if len(close) < 30:
            return risk_metrics
        
        # Calculate returns
        returns = np.diff(close) / close[:-1]
        
        # Volatility (annualized)
        volatility = np.std(returns) * np.sqrt(365) * 100
        risk_metrics['volatility_score'] = min(volatility / 100, 2.0)  # Normalize
        
        # Risk level classification
        if volatility > self.config['high_volatility']:
            risk_metrics['risk_level'] = 'very_high'
            risk_metrics['risk_factors'].append(f'High volatility: {volatility:.1f}%')
        elif volatility > 75:
            risk_metrics['risk_level'] = 'high'
        elif volatility < 25:
            risk_metrics['risk_level'] = 'low'
        
        # Sharpe ratio estimate (assuming 0% risk-free rate)
        if np.std(returns) > 0:
            risk_metrics['sharpe_estimate'] = np.mean(returns) / np.std(returns) * np.sqrt(365)
        
        # Maximum drawdown estimate
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        risk_metrics['max_drawdown_estimate'] = np.min(drawdown) * 100
        
        # ATR-based risk
        if 'atr_percentage' in indicators:
            atr_pct = indicators['atr_percentage']
            if atr_pct > 5:
                risk_metrics['risk_factors'].append(f'High ATR: {atr_pct:.1f}%')
        
        return risk_metrics
    
    def _generate_ai_summary(self, analysis: Dict[str, Any], symbol: str, timeframe: str) -> Dict[str, Any]:
        """Generate AI-consumable summary for decision making"""
        
        signals = analysis['signals']
        trend = analysis['trend_analysis']
        risk = analysis['risk_metrics']
        
        # Count signal types
        buy_signals = len([s for s in signals if s.signal_type in ['buy', 'strong_buy']])
        sell_signals = len([s for s in signals if s.signal_type in ['sell', 'strong_sell']])
        
        # Overall recommendation
        if buy_signals > sell_signals and buy_signals >= 2:
            recommendation = 'BUY'
            confidence = min(buy_signals / 3, 1.0)
        elif sell_signals > buy_signals and sell_signals >= 2:
            recommendation = 'SELL'
            confidence = min(sell_signals / 3, 1.0)
        else:
            recommendation = 'HOLD'
            confidence = 0.5
        
        # Risk-adjusted recommendation
        if risk['risk_level'] in ['very_high', 'high'] and recommendation == 'BUY':
            recommendation = 'HOLD'
            confidence *= 0.7
        
        ai_summary = {
            'recommendation': recommendation,
            'confidence': confidence,
            'risk_level': risk['risk_level'],
            'trend_direction': trend['direction'].name,
            'trend_strength': trend['strength'],
            'signal_count': {'buy': buy_signals, 'sell': sell_signals, 'warning': len(signals) - buy_signals - sell_signals},
            'key_insights': self._extract_key_insights(analysis),
            'action_priority': self._calculate_action_priority(signals, risk),
            'optimal_entry': self._find_optimal_entry(analysis),
            'risk_management': self._suggest_risk_management(analysis)
        }
        
        return ai_summary
    
    def _extract_key_insights(self, analysis: Dict[str, Any]) -> List[str]:
        """Extract top 3 key insights for AI decision making"""
        insights = []
        
        indicators = analysis['indicators']
        signals = analysis['signals']
        
        # RSI insights
        if 'rsi_14' in indicators:
            rsi = indicators['rsi_14']
            if rsi <= 20:
                insights.append(f"Extremely oversold (RSI: {rsi:.1f})")
            elif rsi >= 80:
                insights.append(f"Extremely overbought (RSI: {rsi:.1f})")
        
        # Volume insights
        if 'volume_ratio' in indicators:
            vol_ratio = indicators['volume_ratio']
            if vol_ratio >= 3:
                insights.append(f"Volume surge: {vol_ratio:.1f}x average")
        
        # Trend insights
        trend = analysis['trend_analysis']
        if trend['strength'] > 0.7:
            insights.append(f"Strong {trend['direction'].name.lower()} trend")
        
        # Price action insights
        if 'price_change_24h' in indicators:
            change = indicators['price_change_24h']
            if abs(change) > 10:
                insights.append(f"24h price change: {change:+.1f}%")
        
        return insights[:3]  # Top 3 most important
    
    def _calculate_action_priority(self, signals: List[TechnicalSignal], risk: Dict[str, Any]) -> int:
        """Calculate action priority (1-10, 10 = most urgent)"""
        priority = 5  # Base priority
        
        # High strength signals increase priority
        strong_signals = [s for s in signals if s.strength.value >= 4]
        priority += len(strong_signals)
        
        # High risk situations increase priority
        if risk['risk_level'] in ['very_high', 'high']:
            priority += 2
        
        # Multiple converging signals increase priority
        if len(signals) >= 3:
            priority += 2
        
        return min(priority, 10)
    
    def _find_optimal_entry(self, analysis: Dict[str, Any]) -> Dict[str, float]:
        """Find optimal entry points based on support/resistance"""
        indicators = analysis['indicators']
        sr = analysis['support_resistance']
        
        current_price = indicators.get('current_price', 0)
        
        entry_points = {
            'aggressive_entry': current_price,
            'conservative_entry': sr.get('immediate_support', current_price * 0.98),
            'breakout_entry': sr.get('immediate_resistance', current_price * 1.02)
        }
        
        return entry_points
    
    def _suggest_risk_management(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest risk management parameters"""
        indicators = analysis['indicators']
        risk = analysis['risk_metrics']
        sr = analysis['support_resistance']
        
        current_price = indicators.get('current_price', 0)
        atr_pct = indicators.get('atr_percentage', 3)
        
        # Dynamic stop loss based on ATR and support levels
        atr_stop = current_price * (1 - atr_pct / 100 * 2)
        support_stop = sr.get('immediate_support', current_price * 0.95)
        
        risk_management = {
            'position_size': min(1.0 / max(risk['volatility_score'], 0.1), 0.1),  # Max 10% position
            'stop_loss': max(atr_stop, support_stop),
            'take_profit_1': current_price * 1.08,  # 8% target
            'take_profit_2': current_price * 1.15,  # 15% target
            'max_risk_per_trade': 0.02,  # 2% of portfolio
            'hold_period_days': 7 if risk['risk_level'] == 'high' else 14
        }
        
        return risk_management
    
    def _assess_data_quality(self, data: pd.DataFrame) -> float:
        """Assess quality of market data (0-1)"""
        quality_score = 1.0
        
        # Penalize missing data
        if len(data) < 400:  # Less than 80% of expected 500 candles
            quality_score *= len(data) / 500
        
        # Check for data gaps
        time_diffs = data.index.to_series().diff().dropna()
        if len(time_diffs) > 0:
            median_diff = time_diffs.median()
            large_gaps = (time_diffs > median_diff * 3).sum()
            if large_gaps > 0:
                quality_score *= max(0.8, 1 - large_gaps / len(data))
        
        # Check for zero volume
        zero_volume = (data['volume'] == 0).sum()
        if zero_volume > 0:
            quality_score *= max(0.9, 1 - zero_volume / len(data))
        
        return max(0.1, quality_score)
    
    def _create_empty_analysis(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Create empty analysis structure for failed analyses"""
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamp': datetime.now().isoformat(),
            'data_quality': 0.0,
            'indicators': {},
            'signals': [],
            'trend_analysis': {'direction': TrendDirection.NEUTRAL, 'strength': 0, 'confidence': 0},
            'support_resistance': {},
            'volume_analysis': {},
            'risk_metrics': {'risk_level': 'unknown'},
            'ai_summary': {'recommendation': 'HOLD', 'confidence': 0, 'risk_level': 'unknown'}
        }

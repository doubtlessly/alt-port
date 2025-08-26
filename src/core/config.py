"""
Configuration management optimized for GitHub workflows and public APIs
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

@dataclass
class ExchangeEndpoint:
    """Public API endpoint configuration"""
    name: str
    base_url: str
    rate_limit_per_minute: int
    kline_endpoint: str
    symbol_format: str  # Template for symbol formatting
    timeframe_mapping: Dict[str, str]
    requires_auth: bool = False
    priority: int = 1  # Lower = higher priority
    
class Config:
    """Centralized configuration for maximum performance and reliability"""
    
    # Project paths
    ROOT_DIR = Path(__file__).parent.parent.parent
    SRC_DIR = ROOT_DIR / "src"
    DOCS_DIR = ROOT_DIR / "docs"
    CACHE_DIR = ROOT_DIR / "cache"
    LOGS_DIR = ROOT_DIR / "logs"
    CONFIG_DIR = ROOT_DIR / "config"
    
    # Data collection settings
    TIMEFRAMES = ["1d", "4h", "1h"]  # Optimized for trend + momentum analysis
    CANDLE_LIMIT = 500  # Balance between data richness and performance
    
    # Public exchange endpoints (NO API KEYS NEEDED)
    EXCHANGES = {
        "binance": ExchangeEndpoint(
            name="binance",
            base_url="https://api.binance.com/api/v3",
            rate_limit_per_minute=1200,  # Conservative rate limit
            kline_endpoint="/klines",
            symbol_format="{symbol}USDT",
            timeframe_mapping={
                "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
                "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w"
            },
            priority=1
        ),
        "kucoin": ExchangeEndpoint(
            name="kucoin", 
            base_url="https://api.kucoin.com/api/v1",
            rate_limit_per_minute=600,
            kline_endpoint="/market/candles",
            symbol_format="{symbol}-USDT",
            timeframe_mapping={
                "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
                "1h": "1hour", "4h": "4hour", "1d": "1day", "1w": "1week"
            },
            priority=2
        ),
        "gate": ExchangeEndpoint(
            name="gate",
            base_url="https://api.gateio.ws/api/v4",
            rate_limit_per_minute=600,
            kline_endpoint="/spot/candlesticks",
            symbol_format="{symbol}_USDT",
            timeframe_mapping={
                "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", 
                "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w"
            },
            priority=3
        )
    }
    
    # High-ROI Technical Indicators
    INDICATORS = {
        # Trend indicators (primary for direction)
        "trend": [
            "sma_20", "sma_50", "sma_200",
            "ema_12", "ema_26", "ema_50", 
            "supertrend", "ichimoku_conversion", "ichimoku_base"
        ],
        # Momentum indicators (entry/exit timing)
        "momentum": [
            "rsi_14", "rsi_7", 
            "macd", "macd_signal", "macd_histogram",
            "stoch_k", "stoch_d", "williams_r"
        ],
        # Volatility indicators (risk assessment)
        "volatility": [
            "bb_upper", "bb_middle", "bb_lower", "bb_width",
            "atr", "keltner_upper", "keltner_lower"
        ],
        # Volume indicators (conviction confirmation)
        "volume": [
            "volume_sma", "volume_ratio", "obv", "mfi", "vwap"
        ]
    }
    
    # AI Decision Thresholds (optimized for crypto volatility)
    THRESHOLDS = {
        "rsi_oversold": 25,      # More extreme for crypto
        "rsi_overbought": 75,    # More extreme for crypto  
        "rsi_extreme_oversold": 15,
        "rsi_extreme_overbought": 85,
        "bb_squeeze": 0.1,       # Bollinger Band width threshold
        "volume_surge": 3.0,     # Volume spike threshold
        "trend_strength_strong": 0.75,
        "trend_strength_weak": 0.25,
        "stop_loss_threshold": -0.08,    # 8% stop loss
        "take_profit_threshold": 0.20,   # 20% take profit
        "high_volatility": 100,          # Annualized volatility %
    }
    
    # Portfolio Management Rules
    PORTFOLIO_RULES = {
        "max_position_size": 0.10,      # Max 10% per position
        "min_position_size": 0.005,     # Min 0.5% per position  
        "rebalancing_threshold": 0.05,   # Rebalance if 5% drift
        "correlation_threshold": 0.8,    # High correlation warning
        "max_portfolio_risk": 0.25,     # Max 25% portfolio risk
    }
    
    # GitHub Actions optimization
    GITHUB_SETTINGS = {
        "max_runtime_minutes": 25,      # Stay under 30min limit
        "concurrent_requests": 50,       # Balance speed vs rate limits
        "retry_attempts": 3,
        "cache_duration_minutes": 240,   # 4 hour cache
        "output_compression": True,
    }
    
    @classmethod
    def setup_directories(cls) -> None:
        """Create necessary directories"""
        for directory in [cls.DOCS_DIR, cls.CACHE_DIR, cls.LOGS_DIR, cls.CONFIG_DIR]:
            directory.mkdir(exist_ok=True, parents=True)
            
    @classmethod
    def load_portfolio(cls, path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load portfolio configuration"""
        if path is None:
            path = cls.CONFIG_DIR / "portfolio.json"
        
        try:
            with open(path, 'r') as f:
                portfolio = json.load(f)
                logger.info(f"Loaded {len(portfolio)} holdings from {path}")
                return portfolio
        except FileNotFoundError:
            logger.warning(f"Portfolio file not found: {path}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in portfolio file: {e}")
            return []
    
    @classmethod
    def get_exchange_by_priority(cls) -> List[ExchangeEndpoint]:
        """Get exchanges sorted by priority (best first)"""
        return sorted(cls.EXCHANGES.values(), key=lambda x: x.priority)
    
    @classmethod
    def validate_environment(cls) -> Dict[str, bool]:
        """Validate environment for GitHub Actions"""
        checks = {
            "python_version": True,  # Will be checked by workflow
            "directories_created": all(d.exists() for d in [cls.CACHE_DIR, cls.LOGS_DIR]),
            "portfolio_exists": (cls.CONFIG_DIR / "portfolio.json").exists(),
            "internet_access": True,  # Assumed in GitHub Actions
        }
        
        return checks
    
    @classmethod
    def get_runtime_config(cls) -> Dict[str, Any]:
        """Get runtime configuration optimized for current environment"""
        is_github_actions = os.getenv("GITHUB_ACTIONS") == "true"
        
        config = {
            "max_concurrent_requests": cls.GITHUB_SETTINGS["concurrent_requests"] if is_github_actions else 20,
            "request_timeout": 30 if is_github_actions else 60,
            "max_retries": cls.GITHUB_SETTINGS["retry_attempts"],
            "cache_enabled": True,
            "cache_duration": cls.GITHUB_SETTINGS["cache_duration_minutes"] * 60,
            "output_compression": cls.GITHUB_SETTINGS["output_compression"] if is_github_actions else False,
            "verbose_logging": not is_github_actions,
        }
        
        return config

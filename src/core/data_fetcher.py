"""
High-performance data fetcher with intelligent failover and caching
Optimized for GitHub Actions and public APIs (no authentication required)
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
import json
import time
import hashlib
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass, asdict
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from asyncio_throttle import Throttler

from .config import Config, ExchangeEndpoint

logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Structured market data response"""
    symbol: str
    timeframe: str
    exchange: str
    timestamp: datetime
    data: pd.DataFrame
    quality_score: float
    cache_hit: bool = False

class SmartCache:
    """Intelligent caching system optimized for GitHub Actions"""
    
    def __init__(self, cache_dir: Path = Config.CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.runtime_config = Config.get_runtime_config()
        
    def _get_cache_key(self, symbol: str, timeframe: str, exchange: str) -> str:
        """Generate deterministic cache key"""
        data_string = f"{symbol}_{timeframe}_{exchange}_{datetime.now().strftime('%Y%m%d%H')}"
        return hashlib.md5(data_string.encode()).hexdigest()
    
    def get(self, symbol: str, timeframe: str, exchange: str) -> Optional[pd.DataFrame]:
        """Retrieve cached data if fresh"""
        if not self.runtime_config["cache_enabled"]:
            return None
            
        cache_key = self._get_cache_key(symbol, timeframe, exchange)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if not cache_file.exists():
            return None
            
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                
            # Check if cache is still fresh
            cache_age = time.time() - cache_data['timestamp']
            if cache_age < self.runtime_config["cache_duration"]:
                logger.debug(f"Cache hit: {symbol} {timeframe} on {exchange}")
                return cache_data['data']
            else:
                # Remove stale cache
                cache_file.unlink()
                return None
                
        except Exception as e:
            logger.warning(f"Cache read error for {symbol}: {e}")
            return None
    
    def set(self, symbol: str, timeframe: str, exchange: str, data: pd.DataFrame) -> None:
        """Cache data with metadata"""
        if not self.runtime_config["cache_enabled"] or data.empty:
            return
            
        cache_key = self._get_cache_key(symbol, timeframe, exchange)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            cache_data = {
                'timestamp': time.time(),
                'data': data,
                'symbol': symbol,
                'timeframe': timeframe,
                'exchange': exchange
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
                
            logger.debug(f"Cached: {symbol} {timeframe} on {exchange}")
            
        except Exception as e:
            logger.warning(f"Cache write error for {symbol}: {e}")

class ExchangeConnector:
    """Connector for individual exchange with rate limiting and error handling"""
    
    def __init__(self, endpoint: ExchangeEndpoint, session: aiohttp.ClientSession):
        self.endpoint = endpoint
        self.session = session
        self.cache = SmartCache()
        
        # Rate limiter: convert per-minute to per-second
        requests_per_second = endpoint.rate_limit_per_minute / 60
        self.throttler = Throttler(rate_limit=requests_per_second)
        
        self.runtime_config = Config.get_runtime_config()
        
    def format_symbol(self, symbol: str) -> str:
        """Format symbol for this exchange"""
        # Handle special cases
        if symbol.upper() == "1000CAT":
            formatted = "1000CATUSDT" if self.endpoint.name == "binance" else "1000CAT-USDT"
        else:
            formatted = self.endpoint.symbol_format.format(symbol=symbol.upper())
            
        return formatted
    
    def format_timeframe(self, timeframe: str) -> str:
        """Format timeframe for this exchange"""
        return self.endpoint.timeframe_mapping.get(timeframe, timeframe)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
    )
    async def fetch_klines(self, symbol: str, timeframe: str, limit: int = 500) -> Optional[pd.DataFrame]:
        """Fetch kline/candlestick data with retry logic"""
        
        # Check cache first
        cached_data = self.cache.get(symbol, timeframe, self.endpoint.name)
        if cached_data is not None:
            return cached_data
        
        # Rate limiting
        async with self.throttler:
            try:
                formatted_symbol = self.format_symbol(symbol)
                formatted_timeframe = self.format_timeframe(timeframe)
                
                url, params = self._build_request(formatted_symbol, formatted_timeframe, limit)
                
                timeout = aiohttp.ClientTimeout(total=self.runtime_config["request_timeout"])
                async with self.session.get(url, params=params, timeout=timeout) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        df = self._parse_response(data)
                        
                        if df is not None and not df.empty:
                            # Cache successful result
                            self.cache.set(symbol, timeframe, self.endpoint.name, df)
                            logger.info(f"✅ Fetched {len(df)} candles for {symbol} from {self.endpoint.name}")
                            return df
                    
                    elif response.status == 429:  # Rate limited
                        logger.warning(f"Rate limited by {self.endpoint.name} for {symbol}")
                        await asyncio.sleep(5)  # Wait longer for rate limits
                        return None
                    
                    else:
                        logger.warning(f"HTTP {response.status} from {self.endpoint.name} for {symbol}")
                        return None
                        
            except asyncio.TimeoutError:
                logger.warning(f"Timeout fetching {symbol} from {self.endpoint.name}")
                return None
            except Exception as e:
                logger.warning(f"Error fetching {symbol} from {self.endpoint.name}: {e}")
                return None
        
        return None
    
    def _build_request(self, symbol: str, timeframe: str, limit: int) -> Tuple[str, Dict[str, Any]]:
        """Build request URL and parameters for specific exchange"""
        
        if self.endpoint.name == "binance":
            url = f"{self.endpoint.base_url}{self.endpoint.kline_endpoint}"
            params = {
                "symbol": symbol,
                "interval": timeframe,
                "limit": limit
            }
            
        elif self.endpoint.name == "kucoin":
            # KuCoin uses different parameter structure
            url = f"{self.endpoint.base_url}{self.endpoint.kline_endpoint}"
            
            # Calculate time range for KuCoin
            end_time = int(time.time())
            timeframe_seconds = {
                "1min": 60, "5min": 300, "15min": 900, "30min": 1800,
                "1hour": 3600, "4hour": 14400, "1day": 86400, "1week": 604800
            }
            interval = timeframe_seconds.get(timeframe, 3600)
            start_time = end_time - (limit * interval)
            
            params = {
                "symbol": symbol,
                "type": timeframe,
                "startAt": start_time,
                "endAt": end_time
            }
            
        elif self.endpoint.name == "gate":
            url = f"{self.endpoint.base_url}{self.endpoint.kline_endpoint}"
            params = {
                "currency_pair": symbol,
                "interval": timeframe,
                "limit": limit
            }
            
        else:
            raise ValueError(f"Unsupported exchange: {self.endpoint.name}")
        
        return url, params
    
    def _parse_response(self, data: Any) -> Optional[pd.DataFrame]:
        """Parse exchange-specific response format"""
        
        if not data:
            return None
        
        try:
            if self.endpoint.name == "binance":
                # Binance format: [timestamp, open, high, low, close, volume, ...]
                df = pd.DataFrame(data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore'
                ])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
            elif self.endpoint.name == "kucoin":
                # KuCoin format: [timestamp, open, close, high, low, volume, turnover]
                if 'data' in data:
                    klines = data['data']
                    df = pd.DataFrame(klines, columns=[
                        'timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'
                    ])
                    df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp']), unit='s')
                    # Reorder columns to match standard format
                    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                else:
                    return None
                    
            elif self.endpoint.name == "gate":
                # Gate.io format: [timestamp, volume, close, high, low, open]
                df = pd.DataFrame(data, columns=[
                    'timestamp', 'volume', 'close', 'high', 'low', 'open'
                ])
                df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp']), unit='s')
                # Reorder columns to match standard format
                df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                
            else:
                return None
            
            # Standardize data types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Set timestamp as index and sort
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            # Remove any NaN rows
            df.dropna(inplace=True)
            
            return df if len(df) > 0 else None
            
        except Exception as e:
            logger.error(f"Error parsing {self.endpoint.name} response: {e}")
            return None

class DataFetcher:
    """Main data fetching engine with intelligent failover"""
    
    def __init__(self):
        self.exchanges = Config.get_exchange_by_priority()
        self.runtime_config = Config.get_runtime_config()
        self.session: Optional[aiohttp.ClientSession] = None
        self.connectors: List[ExchangeConnector] = []
        
    async def __aenter__(self):
        """Async context manager entry"""
        # Create session optimized for GitHub Actions
        connector = aiohttp.TCPConnector(
            limit=self.runtime_config["max_concurrent_requests"],
            limit_per_host=20,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=self.runtime_config["request_timeout"]),
            headers={
                'User-Agent': 'CryptoPortfolioAnalytics/1.0'
            }
        )
        
        # Initialize exchange connectors
        self.connectors = [
            ExchangeConnector(exchange, self.session) 
            for exchange in self.exchanges
        ]
        
        logger.info(f"Initialized {len(self.connectors)} exchange connectors")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def fetch_symbol_data(self, symbol: str, timeframe: str) -> Optional[MarketData]:
        """Fetch data for a single symbol with intelligent failover"""
        
        for connector in self.connectors:
            try:
                df = await connector.fetch_klines(symbol, timeframe)
                
                if df is not None and len(df) > 10:  # Minimum data threshold
                    quality_score = self._calculate_quality_score(df)
                    
                    return MarketData(
                        symbol=symbol,
                        timeframe=timeframe,
                        exchange=connector.endpoint.name,
                        timestamp=datetime.now(),
                        data=df,
                        quality_score=quality_score,
                        cache_hit=False  # Would be set by cache logic
                    )
                    
            except Exception as e:
                logger.warning(f"Connector {connector.endpoint.name} failed for {symbol}: {e}")
                continue
        
        logger.error(f"❌ Failed to fetch {symbol} {timeframe} from all exchanges")
        return None
    
    async def fetch_portfolio_data(self, portfolio: List[Dict[str, Any]], timeframes: List[str]) -> Dict[str, Dict[str, MarketData]]:
        """Fetch data for entire portfolio efficiently"""
        
        logger.info(f"Fetching data for {len(portfolio)} symbols across {len(timeframes)} timeframes")
        
        # Create all fetch tasks
        tasks = []
        task_metadata = []
        
        for holding in portfolio:
            symbol = holding['symbol']
            for timeframe in timeframes:
                task = asyncio.create_task(self.fetch_symbol_data(symbol, timeframe))
                tasks.append(task)
                task_metadata.append((symbol, timeframe))
        
        # Execute all tasks with progress tracking
        logger.info(f"Executing {len(tasks)} concurrent requests...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Organize results
        portfolio_data = {}
        successful_fetches = 0
        
        for i, (symbol, timeframe) in enumerate(task_metadata):
            result = results[i]
            
            if isinstance(result, Exception):
                logger.warning(f"Task failed for {symbol} {timeframe}: {result}")
                continue
            
            if result is not None:
                if symbol not in portfolio_data:
                    portfolio_data[symbol] = {}
                
                portfolio_data[symbol][timeframe] = result
                successful_fetches += 1
        
        success_rate = successful_fetches / len(tasks) * 100
        logger.info(f"✅ Data fetching complete: {successful_fetches}/{len(tasks)} successful ({success_rate:.1f}%)")
        
        return portfolio_data
    
    def _calculate_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate data quality score (0-1)"""
        score = 1.0
        
        # Penalize for missing data
        if len(df) < Config.CANDLE_LIMIT * 0.8:  # Less than 80% of expected data
            score *= len(df) / Config.CANDLE_LIMIT
        
        # Penalize for gaps in data
        time_diffs = df.index.to_series().diff().dropna()
        expected_diff = time_diffs.median()
        large_gaps = (time_diffs > expected_diff * 2).sum()
        if large_gaps > 0:
            score *= max(0.7, 1 - (large_gaps / len(df)))
        
        # Penalize for zero volume periods
        zero_volume = (df['volume'] == 0).sum()
        if zero_volume > 0:
            score *= max(0.8, 1 - (zero_volume / len(df)))
        
        return max(0.1, score)  # Minimum score of 0.1
    
    def get_fetch_summary(self, portfolio_data: Dict[str, Dict[str, MarketData]]) -> Dict[str, Any]:
        """Generate summary of data fetching results"""
        
        total_symbols = len(portfolio_data)
        timeframe_counts = {}
        exchange_counts = {}
        quality_scores = []
        
        for symbol, timeframe_data in portfolio_data.items():
            for timeframe, market_data in timeframe_data.items():
                # Count timeframes
                timeframe_counts[timeframe] = timeframe_counts.get(timeframe, 0) + 1
                
                # Count exchanges
                exchange = market_data.exchange
                exchange_counts[exchange] = exchange_counts.get(exchange, 0) + 1
                
                # Collect quality scores
                quality_scores.append(market_data.quality_score)
        
        return {
            'total_symbols_fetched': total_symbols,
            'timeframe_coverage': timeframe_counts,
            'exchange_distribution': exchange_counts,
            'average_quality_score': np.mean(quality_scores) if quality_scores else 0,
            'min_quality_score': np.min(quality_scores) if quality_scores else 0,
            'data_completeness': total_symbols / len(Config.load_portfolio()) if Config.load_portfolio() else 0
        }

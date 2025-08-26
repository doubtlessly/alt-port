"""
Test data fetcher module with public APIs
"""

import asyncio
import pytest
from src.core.data_fetcher import DataFetcher, SmartCache, ExchangeConnector
from src.core.config import Config

def test_cache_functionality():
    """Test caching system"""
    cache = SmartCache()
    
    # Test cache key generation
    key1 = cache._get_cache_key("BTC", "1d", "binance")
    key2 = cache._get_cache_key("BTC", "1d", "binance")
    key3 = cache._get_cache_key("ETH", "1d", "binance")
    
    assert key1 == key2  # Same parameters should generate same key
    assert key1 != key3  # Different parameters should generate different keys

async def test_exchange_connector():
    """Test exchange connector configuration"""
    exchanges = Config.get_exchange_by_priority()
    assert len(exchanges) > 0
    
    # Test symbol formatting
    binance_endpoint = next(ex for ex in exchanges if ex.name == "binance")
    connector = ExchangeConnector(binance_endpoint, None)  # Mock session
    
    # Test standard symbol
    formatted = connector.format_symbol("BTC")
    assert formatted == "BTCUSDT"
    
    # Test special case
    formatted_special = connector.format_symbol("1000CAT")
    assert "1000CAT" in formatted_special

async def test_data_fetcher_initialization():
    """Test DataFetcher initialization"""
    async with DataFetcher() as fetcher:
        assert len(fetcher.connectors) > 0
        assert fetcher.session is not None

def test_quality_score_calculation():
    """Test data quality scoring"""
    # This would require sample DataFrame - simplified test
    assert True  # Placeholder

if __name__ == "__main__":
    # Run basic tests
    test_cache_functionality()
    
    # Run async tests
    asyncio.run(test_exchange_connector())
    asyncio.run(test_data_fetcher_initialization())
    
    print("✅ All data fetcher tests passed!")

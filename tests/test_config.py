"""
Test configuration module
"""

import pytest
from pathlib import Path
from src.core.config import Config, ExchangeEndpoint

def test_config_directories():
    """Test directory setup"""
    Config.setup_directories()
    
    assert Config.DOCS_DIR.exists()
    assert Config.CACHE_DIR.exists()  
    assert Config.LOGS_DIR.exists()
    assert Config.CONFIG_DIR.exists()

def test_load_portfolio():
    """Test portfolio loading"""
    portfolio = Config.load_portfolio()
    
    assert isinstance(portfolio, list)
    if portfolio:  # If portfolio exists
        assert all("symbol" in holding for holding in portfolio)
        assert all("amount" in holding for holding in portfolio)

def test_exchange_configuration():
    """Test exchange endpoints"""
    exchanges = Config.get_exchange_by_priority()
    
    assert len(exchanges) > 0
    assert all(isinstance(ex, ExchangeEndpoint) for ex in exchanges)
    assert all(not ex.requires_auth for ex in exchanges)  # All should be public
    
    # Test priority ordering
    priorities = [ex.priority for ex in exchanges]
    assert priorities == sorted(priorities)

def test_timeframes():
    """Test timeframe configuration"""
    assert len(Config.TIMEFRAMES) == 3
    assert "1d" in Config.TIMEFRAMES
    assert "4h" in Config.TIMEFRAMES  
    assert "1h" in Config.TIMEFRAMES

def test_thresholds():
    """Test trading thresholds"""
    thresholds = Config.THRESHOLDS
    
    assert thresholds["rsi_oversold"] < thresholds["rsi_overbought"]
    assert thresholds["stop_loss_threshold"] < 0
    assert thresholds["take_profit_threshold"] > 0

def test_environment_validation():
    """Test environment validation"""
    checks = Config.validate_environment()
    
    assert isinstance(checks, dict)
    assert "directories_created" in checks

if __name__ == "__main__":
    pytest.main([__file__])

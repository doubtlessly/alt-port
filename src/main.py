#!/usr/bin/env python3
"""
Main entry point for Crypto Portfolio Analytics System
Optimized for GitHub Actions and maximum ROI insights

This module handles multiple execution environments:
- Direct script execution (python src/main.py)
- Module execution (python -m src.main)
- GitHub Actions (python src/main.py from repo root)
- Local development (various Python path configurations)
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# === ROBUST IMPORT SYSTEM ===
# Handles all execution environments with multiple fallback strategies

def setup_imports():
    """
    World-class import setup that works in all environments:
    1. GitHub Actions (script execution from repo root)
    2. Local development (various Python paths)
    3. Module execution (python -m)
    4. Direct script execution
    """
    # Get the absolute path of this file
    current_file = Path(__file__).resolve()
    src_dir = current_file.parent
    repo_root = src_dir.parent
    
    # Strategy 1: Add src directory to Python path
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    
    # Strategy 2: Add repo root to Python path (for GitHub Actions)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    
    # Strategy 3: Ensure core module is discoverable
    core_dir = src_dir / "core"
    if str(core_dir) not in sys.path:
        sys.path.insert(0, str(core_dir))

# Setup imports before any module imports
setup_imports()

# === IMPORT WITH MULTIPLE FALLBACK STRATEGIES ===
def import_modules():
    """Import required modules with multiple fallback strategies"""
    
    # Strategy 1: Try relative imports (when run as module)
    try:
        from .core.config import Config
        from .core.data_fetcher import DataFetcher
        from .core.technical_analyzer import TechnicalAnalyzer
        from .core.ai_engine import AIEngine
        return Config, DataFetcher, TechnicalAnalyzer, AIEngine
    except ImportError as e1:
        pass
    
    # Strategy 2: Try absolute imports from src
    try:
        from core.config import Config
        from core.data_fetcher import DataFetcher
        from core.technical_analyzer import TechnicalAnalyzer
        from core.ai_engine import AIEngine
        return Config, DataFetcher, TechnicalAnalyzer, AIEngine
    except ImportError as e2:
        pass
    
    # Strategy 3: Try src.core imports (from repo root)
    try:
        from src.core.config import Config
        from src.core.data_fetcher import DataFetcher
        from src.core.technical_analyzer import TechnicalAnalyzer
        from src.core.ai_engine import AIEngine
        return Config, DataFetcher, TechnicalAnalyzer, AIEngine
    except ImportError as e3:
        pass
    
    # Strategy 4: Try direct imports with explicit path manipulation
    try:
        current_dir = Path(__file__).parent
        core_path = current_dir / "core"
        
        # Import each module individually
        import importlib.util
        
        def load_module(module_name, file_path):
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        
        config_module = load_module("config", core_path / "config.py")
        data_fetcher_module = load_module("data_fetcher", core_path / "data_fetcher.py")
        technical_analyzer_module = load_module("technical_analyzer", core_path / "technical_analyzer.py")
        ai_engine_module = load_module("ai_engine", core_path / "ai_engine.py")
        
        return (config_module.Config, 
                data_fetcher_module.DataFetcher,
                technical_analyzer_module.TechnicalAnalyzer,
                ai_engine_module.AIEngine)
    except Exception as e4:
        # Final fallback with detailed error reporting
        print("🚨 CRITICAL: Failed to import required modules!")
        print("=" * 60)
        print(f"Current working directory: {Path.cwd()}")
        print(f"Script location: {Path(__file__).resolve()}")
        print(f"Python path: {sys.path[:5]}...")  # Show first 5 entries
        print("=" * 60)
        print("Import attempts failed:")
        print(f"1. Relative imports: {e1}")
        print(f"2. Core imports: {e2}")
        print(f"3. Src.core imports: {e3}")
        print(f"4. Direct imports: {e4}")
        print("=" * 60)
        
        # Check if core files exist
        current_dir = Path(__file__).parent
        core_dir = current_dir / "core"
        print(f"Core directory exists: {core_dir.exists()}")
        if core_dir.exists():
            core_files = list(core_dir.glob("*.py"))
            print(f"Core files found: {[f.name for f in core_files]}")
        
        raise ImportError(
            "Failed to import core modules. Please ensure all files are present and "
            "the script is run from the correct directory. See detailed error above."
        )

# Import all required modules
Config, DataFetcher, TechnicalAnalyzer, AIEngine = import_modules()

# === LOGGING SETUP ===
# Configure logging after successful imports
def setup_logging():
    """Setup robust logging for all environments"""
    try:
        # Ensure logs directory exists
        Config.setup_directories()
        log_file = Config.LOGS_DIR / 'portfolio_analyzer.log'
    except:
        # Fallback if Config is not available
        log_file = Path("portfolio_analyzer.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode='a')
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class CryptoPortfolioSystem:
    """Main portfolio analytics system"""
    
    def __init__(self):
        # Setup directories
        Config.setup_directories()
        
        # Load configuration
        self.portfolio = Config.load_portfolio()
        self.runtime_config = Config.get_runtime_config()
        
        # Initialize components
        self.technical_analyzer = TechnicalAnalyzer()
        self.ai_engine = AIEngine()
        
        # Validate environment
        self.validation_results = Config.validate_environment()
        
        logger.info(f"🚀 System initialized with {len(self.portfolio)} holdings")
        logger.info(f"📊 Runtime config: {self.runtime_config}")
        
    def validate_system(self) -> bool:
        """Validate system readiness"""
        
        if not self.portfolio:
            logger.error("❌ No portfolio holdings found")
            return False
        
        if not all(self.validation_results.values()):
            logger.error(f"❌ Environment validation failed: {self.validation_results}")
            return False
        
        logger.info("✅ System validation passed")
        return True
    
    async def run_complete_analysis(self) -> Dict[str, Any]:
        """Run complete portfolio analysis"""
        
        start_time = time.time()
        logger.info("🧠 Starting comprehensive portfolio analysis...")
        
        # Fetch market data
        portfolio_data = await self._fetch_portfolio_data()
        
        if not portfolio_data:
            raise RuntimeError("Failed to fetch any market data")
        
        # Perform technical analysis
        analyzed_data = await self._perform_technical_analysis(portfolio_data)
        
        # Generate AI insights
        ai_insights = self._generate_ai_insights(analyzed_data)
        
        # Compile results
        analysis_duration = time.time() - start_time
        
        results = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'analysis_duration': analysis_duration,
                'total_holdings': len(self.portfolio),
                'successful_analyses': len(analyzed_data),
                'data_completeness': len(analyzed_data) / len(self.portfolio),
                'timeframes': Config.TIMEFRAMES,
                'system_version': '1.0.0',
                'runtime_environment': 'github_actions' if self.runtime_config.get('max_concurrent_requests', 0) > 30 else 'local'
            },
            'portfolio_data': analyzed_data,
            'ai_insights': ai_insights,
            'market_summary': self._generate_market_summary(analyzed_data),
            'performance_metrics': self._calculate_performance_metrics(analyzed_data)
        }
        
        logger.info(f"✅ Analysis completed in {analysis_duration:.2f}s")
        logger.info(f"📊 Success rate: {len(analyzed_data)}/{len(self.portfolio)} ({len(analyzed_data)/len(self.portfolio)*100:.1f}%)")
        
        return results
    
    async def _fetch_portfolio_data(self) -> Dict[str, Dict[str, Any]]:
        """Fetch market data for all portfolio holdings"""
        
        logger.info(f"📡 Fetching data for {len(self.portfolio)} holdings...")
        
        portfolio_data = {}
        
        async with DataFetcher() as fetcher:
            # Process holdings in batches to manage memory and rate limits
            batch_size = 10
            
            for i in range(0, len(self.portfolio), batch_size):
                batch = self.portfolio[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(self.portfolio)-1)//batch_size + 1}")
                
                # Fetch data for current batch
                batch_data = await fetcher.fetch_portfolio_data(batch, Config.TIMEFRAMES)
                portfolio_data.update(batch_data)
                
                # Brief pause between batches for rate limiting
                if i + batch_size < len(self.portfolio):
                    await asyncio.sleep(1)
        
        logger.info(f"✅ Data fetched for {len(portfolio_data)} symbols")
        return portfolio_data
    
    async def _perform_technical_analysis(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform technical analysis on portfolio data"""
        
        logger.info(f"🔬 Performing technical analysis on {len(portfolio_data)} symbols...")
        
        analyzed_data = {}
        
        # Create analysis tasks for parallel execution
        tasks = []
        symbols = []
        
        for symbol, timeframe_data in portfolio_data.items():
            symbols.append(symbol)
            tasks.append(self._analyze_symbol_complete(symbol, timeframe_data))
        
        # Execute analysis tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, symbol in enumerate(symbols):
            result = results[i]
            if isinstance(result, Exception):
                logger.error(f"❌ Analysis failed for {symbol}: {result}")
            else:
                analyzed_data[symbol] = result
                logger.debug(f"✅ Analyzed {symbol}")
        
        logger.info(f"✅ Technical analysis complete for {len(analyzed_data)} symbols")
        return analyzed_data
    
    async def _analyze_symbol_complete(self, symbol: str, timeframe_data: Dict[str, Any]) -> Dict[str, Any]:
        """Complete analysis for a single symbol"""
        
        symbol_analysis = {
            'symbol': symbol,
            'timeframes': {},
            'summary': {},
            'last_updated': datetime.now().isoformat()
        }
        
        # Analyze each timeframe
        for timeframe, market_data in timeframe_data.items():
            try:
                analysis = self.technical_analyzer.analyze(
                    market_data.data, symbol, timeframe
                )
                symbol_analysis['timeframes'][timeframe] = analysis
                
            except Exception as e:
                logger.warning(f"Analysis failed for {symbol} {timeframe}: {e}")
        
        # Generate symbol summary
        symbol_analysis['summary'] = self._generate_symbol_summary(symbol_analysis)
        
        return symbol_analysis
    
    def _generate_ai_insights(self, analyzed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI insights for portfolio optimization"""
        
        logger.info("🤖 Generating AI insights...")
        
        # Convert analyzed data to format expected by AI engine
        portfolio_data_for_ai = {}
        
        for symbol, symbol_data in analyzed_data.items():
            portfolio_data_for_ai[symbol] = symbol_data['timeframes']
        
        # Generate insights
        insights = self.ai_engine.analyze_portfolio(portfolio_data_for_ai, self.portfolio)
        
        # Convert to serializable format
        ai_insights = {
            'portfolio_health': {
                'total_value': insights.total_portfolio_value,
                'risk_score': insights.risk_score,
                'diversification_score': insights.diversification_score,
                'momentum_score': insights.momentum_score,
                'overall_score': (insights.diversification_score + (1 - insights.risk_score) + insights.momentum_score) / 3
            },
            'market_overview': {
                'regime': insights.market_regime,
                'confidence': self._calculate_market_regime_confidence(insights),
                'characteristics': self._generate_market_characteristics(insights)
            },
            'individual_recommendations': [
                {
                    'symbol': rec.symbol,
                    'action': rec.action.value,
                    'confidence': rec.confidence,
                    'urgency': rec.urgency,
                    'target_allocation': rec.target_allocation,
                    'expected_roi': rec.expected_roi,
                    'reasoning': rec.reasoning,
                    'risk_level': rec.risk_metrics.get('risk_category', {}).value if hasattr(rec.risk_metrics.get('risk_category', {}), 'value') else 'unknown'
                }
                for rec in insights.recommendations
            ],
            'rebalancing_suggestions': {
                'needed': insights.rebalancing_needed,
                'opportunities': [
                    {
                        'symbol': opp['symbol'],
                        'action': opp['action'],
                        'reasoning': opp['reasoning'],
                        'expected_profit': opp.get('expected_profit', 0)
                    }
                    for opp in insights.profit_taking_opportunities
                ]
            },
            'risk_assessment': {
                'portfolio_risk': insights.risk_score,
                'high_risk_holdings': insights.underperforming_assets,
                'diversification_needed': insights.diversification_score < 0.6
            },
            'opportunity_scanner': [
                {
                    'symbol': symbol,
                    'type': 'buy_opportunity',
                    'score': self._calculate_opportunity_score(symbol, insights),
                    'reasoning': self._generate_opportunity_reasoning(symbol, insights)
                }
                for symbol in insights.top_opportunities
            ],
            'exit_alerts': [
                {
                    'symbol': alert['symbol'],
                    'type': 'stop_loss',
                    'urgency': alert['urgency'],
                    'message': alert['reasoning']
                }
                for alert in insights.stop_loss_alerts
            ]
        }
        
        # Generate executive summary
        executive_summary = self.ai_engine.generate_executive_summary(insights)
        ai_insights['executive_summary'] = executive_summary
        
        logger.info("✅ AI insights generated")
        return ai_insights
    
    def _calculate_market_regime_confidence(self, insights) -> float:
        """Calculate dynamic confidence for market regime based on data quality and consistency"""
        # Base confidence from momentum and risk consistency
        momentum_confidence = insights.momentum_score
        risk_stability = 1 - insights.risk_score  # Lower risk = higher confidence
        diversification_factor = insights.diversification_score
        
        # Calculate overall confidence (weighted average)
        confidence = (
            momentum_confidence * 0.4 +
            risk_stability * 0.3 +
            diversification_factor * 0.3
        )
        
        # Boost confidence if we have many recommendations (more data = higher confidence)
        recommendation_count = len(insights.recommendations)
        if recommendation_count > 30:
            confidence *= 1.1  # 10% boost for large portfolios
        elif recommendation_count > 20:
            confidence *= 1.05  # 5% boost for medium portfolios
        
        return min(confidence, 1.0)  # Cap at 1.0
    
    def _generate_market_characteristics(self, insights) -> list:
        """Generate dynamic market characteristics based on portfolio analysis"""
        characteristics = []
        
        # Momentum-based characteristics
        if insights.momentum_score > 0.7:
            characteristics.append("strong_bullish_momentum")
        elif insights.momentum_score > 0.6:
            characteristics.append("moderate_bullish_trend")
        elif insights.momentum_score < 0.3:
            characteristics.append("bearish_sentiment")
        elif insights.momentum_score < 0.4:
            characteristics.append("weak_momentum")
        else:
            characteristics.append("neutral_trend")
        
        # Risk-based characteristics
        if insights.risk_score > 0.8:
            characteristics.append("high_volatility_environment")
        elif insights.risk_score > 0.6:
            characteristics.append("elevated_risk_levels")
        elif insights.risk_score < 0.3:
            characteristics.append("low_volatility_environment")
        
        # Diversification characteristics
        if insights.diversification_score < 0.4:
            characteristics.append("concentrated_positions")
        elif insights.diversification_score > 0.8:
            characteristics.append("well_diversified_portfolio")
        
        # Rebalancing characteristics
        if insights.rebalancing_needed:
            characteristics.append("rebalancing_recommended")
        
        # Opportunity characteristics
        if len(insights.profit_taking_opportunities) > 3:
            characteristics.append("multiple_profit_opportunities")
        
        if len(insights.stop_loss_alerts) > 2:
            characteristics.append("risk_management_required")
        
        return characteristics
    
    def _calculate_opportunity_score(self, symbol: str, insights) -> float:
        """Calculate dynamic opportunity score based on multiple factors"""
        # Find the recommendation for this symbol
        recommendation = next(
            (rec for rec in insights.recommendations if rec.symbol == symbol),
            None
        )
        
        if not recommendation:
            return 0.5  # Default neutral score
        
        # Base score from expected ROI and confidence
        base_score = recommendation.expected_roi * recommendation.confidence
        
        # Adjust based on urgency (higher urgency = higher opportunity)
        urgency_factor = recommendation.urgency / 10  # Normalize to 0-1
        
        # Adjust based on risk level
        risk_factor = 1.0
        if hasattr(recommendation.risk_metrics.get('risk_category'), 'value'):
            risk_category = recommendation.risk_metrics['risk_category'].value
            if risk_category == 'conservative':
                risk_factor = 1.2  # Boost conservative opportunities
            elif risk_category == 'speculative':
                risk_factor = 0.8  # Reduce speculative opportunities
        
        # Calculate final score
        opportunity_score = (
            base_score * 0.5 +
            urgency_factor * 0.3 +
            recommendation.confidence * 0.2
        ) * risk_factor
        
        return min(max(opportunity_score, 0.1), 1.0)  # Clamp between 0.1 and 1.0
    
    def _generate_opportunity_reasoning(self, symbol: str, insights) -> str:
        """Generate dynamic reasoning for opportunity based on analysis"""
        # Find the recommendation for this symbol
        recommendation = next(
            (rec for rec in insights.recommendations if rec.symbol == symbol),
            None
        )
        
        if not recommendation:
            return f"Symbol {symbol} identified as potential opportunity"
        
        # Build reasoning based on multiple factors
        reasons = []
        
        # Action-based reasoning
        if recommendation.action.value in ['strong_buy', 'buy']:
            reasons.append(f"{recommendation.action.value.replace('_', ' ').title()} signal")
        
        # ROI-based reasoning
        if recommendation.expected_roi > 0.1:
            reasons.append(f"High ROI potential ({recommendation.expected_roi:.1%})")
        elif recommendation.expected_roi > 0.05:
            reasons.append(f"Moderate ROI potential ({recommendation.expected_roi:.1%})")
        
        # Confidence-based reasoning
        if recommendation.confidence > 0.8:
            reasons.append("High confidence analysis")
        elif recommendation.confidence > 0.6:
            reasons.append("Good confidence level")
        
        # Urgency-based reasoning
        if recommendation.urgency >= 8:
            reasons.append("Urgent action recommended")
        elif recommendation.urgency >= 6:
            reasons.append("Timely action suggested")
        
        # Risk-based reasoning
        if hasattr(recommendation.risk_metrics.get('risk_category'), 'value'):
            risk_category = recommendation.risk_metrics['risk_category'].value
            if risk_category == 'conservative':
                reasons.append("Low-risk opportunity")
            elif risk_category == 'moderate':
                reasons.append("Balanced risk-reward")
        
        # Use specific reasoning if available, otherwise construct from factors
        if recommendation.reasoning:
            primary_reason = recommendation.reasoning[0]
            if len(reasons) > 0:
                return f"{primary_reason} - {', '.join(reasons[:2])}"
            else:
                return primary_reason
        else:
            return ', '.join(reasons[:3]) if reasons else f"Opportunity identified for {symbol}"
    
    def _generate_symbol_summary(self, symbol_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary for individual symbol"""
        
        symbol = symbol_analysis['symbol']
        timeframes = symbol_analysis['timeframes']
        
        if not timeframes:
            return {'status': 'no_data'}
        
        # Get primary analysis from 1d timeframe
        primary_analysis = timeframes.get('1d', list(timeframes.values())[0])
        
        summary = {
            'status': 'analyzed',
            'data_quality': primary_analysis.get('data_quality', 0),
            'signal_count': len(primary_analysis.get('signals', [])),
            'recommendation': primary_analysis.get('ai_summary', {}).get('recommendation', 'HOLD'),
            'confidence': primary_analysis.get('ai_summary', {}).get('confidence', 0.5),
            'risk_level': primary_analysis.get('ai_summary', {}).get('risk_level', 'unknown'),
            'trend_direction': primary_analysis.get('ai_summary', {}).get('trend_direction', 'NEUTRAL'),
            'key_insights': primary_analysis.get('ai_summary', {}).get('key_insights', [])
        }
        
        return summary
    
    def _generate_market_summary(self, analyzed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate market-wide summary"""
        
        recommendations = {}
        risk_levels = {}
        signal_counts = {'buy': 0, 'sell': 0, 'hold': 0}
        
        for symbol, data in analyzed_data.items():
            summary = data.get('summary', {})
            
            # Count recommendations
            rec = summary.get('recommendation', 'HOLD')
            if rec in ['BUY', 'STRONG_BUY']:
                signal_counts['buy'] += 1
            elif rec in ['SELL', 'STRONG_SELL']:
                signal_counts['sell'] += 1
            else:
                signal_counts['hold'] += 1
            
            # Count risk levels
            risk = summary.get('risk_level', 'unknown')
            risk_levels[risk] = risk_levels.get(risk, 0) + 1
        
        return {
            'total_analyzed': len(analyzed_data),
            'market_sentiment': {
                'bullish': signal_counts['buy'],
                'bearish': signal_counts['sell'],
                'neutral': signal_counts['hold']
            },
            'risk_distribution': risk_levels,
            'data_quality_average': sum(
                data.get('summary', {}).get('data_quality', 0) 
                for data in analyzed_data.values()
            ) / len(analyzed_data) if analyzed_data else 0
        }
    
    def _calculate_performance_metrics(self, analyzed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate system performance metrics"""
        
        return {
            'total_symbols_processed': len(analyzed_data),
            'analysis_success_rate': len(analyzed_data) / len(self.portfolio) if self.portfolio else 0,
            'average_data_quality': sum(
                data.get('summary', {}).get('data_quality', 0)
                for data in analyzed_data.values()
            ) / len(analyzed_data) if analyzed_data else 0,
            'average_signal_count': sum(
                data.get('summary', {}).get('signal_count', 0)
                for data in analyzed_data.values()
            ) / len(analyzed_data) if analyzed_data else 0
        }
    
    def save_results(self, results: Dict[str, Any]) -> None:
        """Save analysis results to files"""
        
        logger.info("💾 Saving analysis results...")
        
        timestamp = datetime.now().isoformat()
        
        # Save complete analysis
        analysis_file = Config.DOCS_DIR / "portfolio_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary for quick access
        summary = {
            'generated_at': timestamp,
            'summary': results['market_summary'],
            'ai_insights': {
                'portfolio_health': results['ai_insights']['portfolio_health'],
                'top_recommendations': results['ai_insights']['individual_recommendations'][:5],
                'executive_summary': results['ai_insights']['executive_summary']
            },
            'metadata': results['metadata']
        }
        
        summary_file = Config.DOCS_DIR / "portfolio_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save individual symbol files for API access
        symbols_dir = Config.DOCS_DIR / "symbols"
        symbols_dir.mkdir(exist_ok=True)
        
        for symbol, data in results['portfolio_data'].items():
            symbol_file = symbols_dir / f"{symbol.lower()}.json"
            symbol_data = {
                'symbol': symbol,
                'generated_at': timestamp,
                'analysis': data,
                'ai_recommendation': next(
                    (rec for rec in results['ai_insights']['individual_recommendations'] 
                     if rec['symbol'] == symbol), 
                    None
                )
            }
            
            with open(symbol_file, 'w') as f:
                json.dump(symbol_data, f, indent=2, default=str)
        
        logger.info(f"✅ Results saved to {Config.DOCS_DIR}")
        logger.info(f"📊 Analysis file: {analysis_file}")
        logger.info(f"📋 Summary file: {summary_file}")
        logger.info(f"🗂️  Symbol files: {len(results['portfolio_data'])} files in {symbols_dir}")

async def main():
    """Main execution function"""
    
    try:
        # Initialize system
        system = CryptoPortfolioSystem()
        
        # Validate system readiness
        if not system.validate_system():
            logger.error("❌ System validation failed")
            sys.exit(1)
        
        # Run analysis
        results = await system.run_complete_analysis()
        
        # Save results
        system.save_results(results)
        
        # Success summary
        metadata = results['metadata']
        ai_insights = results['ai_insights']
        
        print(f"\n🎉 Portfolio Analysis Complete!")
        print(f"📊 Analyzed {metadata['successful_analyses']}/{metadata['total_holdings']} holdings in {metadata['analysis_duration']:.1f}s")
        print(f"🏥 Portfolio Health: {ai_insights['portfolio_health']['overall_score']:.2f}/1.0")
        print(f"📈 Market Sentiment: {results['market_summary']['market_sentiment']}")
        print(f"⚠️  High Priority Actions: {sum(1 for r in ai_insights['individual_recommendations'] if r['urgency'] >= 7)}")
        print(f"💰 Profit Opportunities: {len(ai_insights['rebalancing_suggestions']['opportunities'])}")
        print(f"🌐 Data available at GitHub Pages")
        
        logger.info("🎉 Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ System error: {e}")
        # Print stack trace for debugging
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # Additional environment diagnostics for debugging
    print(f"🔍 Execution Environment:")
    print(f"   Python version: {sys.version}")
    print(f"   Current directory: {Path.cwd()}")
    print(f"   Script location: {Path(__file__).resolve()}")
    print(f"   Python path entries: {len(sys.path)}")
    print()
    
    # Run the main function
    asyncio.run(main())
"""
AI-driven insights engine for maximum ROI portfolio decisions
Focuses on sell timing, capital reallocation, and profit maximization
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass, asdict
from enum import Enum

from .config import Config
from .technical_analyzer import TechnicalSignal, SignalStrength, TrendDirection

logger = logging.getLogger(__name__)

class ActionType(Enum):
    """Portfolio action types optimized for ROI"""
    STRONG_BUY = "strong_buy"          # High conviction buy
    BUY = "buy"                        # Regular buy signal
    ACCUMULATE = "accumulate"          # DCA/gradual buying
    HOLD = "hold"                      # Maintain position
    REDUCE = "reduce"                  # Partial sell (profit taking)
    SELL = "sell"                      # Full sell signal
    STOP_LOSS = "stop_loss"           # Emergency exit

class RiskCategory(Enum):
    """Risk categories for portfolio allocation"""
    CONSERVATIVE = "conservative"      # Low risk, stable assets
    MODERATE = "moderate"             # Balanced risk/reward
    AGGRESSIVE = "aggressive"         # High risk, high reward
    SPECULATIVE = "speculative"       # Very high risk, moonshot potential

@dataclass
class AIRecommendation:
    """Comprehensive AI recommendation for a holding"""
    symbol: str
    action: ActionType
    confidence: float  # 0-1
    urgency: int      # 1-10 (10 = act immediately)
    target_allocation: float  # Percentage of portfolio
    reasoning: List[str]
    price_targets: Dict[str, float]
    risk_metrics: Dict[str, Any]
    timeframe: str
    expected_roi: float  # Expected return on investment
    max_loss: float     # Maximum acceptable loss
    rebalance_trigger: bool

@dataclass
class PortfolioInsights:
    """Portfolio-level insights and recommendations"""
    total_portfolio_value: float
    risk_score: float  # 0-1 (1 = very high risk)
    diversification_score: float  # 0-1 (1 = well diversified)
    momentum_score: float  # 0-1 (1 = strong positive momentum)
    market_regime: str
    recommendations: List[AIRecommendation]
    rebalancing_needed: bool
    profit_taking_opportunities: List[Dict[str, Any]]
    stop_loss_alerts: List[Dict[str, Any]]
    underperforming_assets: List[str]
    top_opportunities: List[str]

class AIEngine:
    """Advanced AI engine for portfolio optimization and ROI maximization"""
    
    def __init__(self):
        self.config = Config.PORTFOLIO_RULES
        self.thresholds = Config.THRESHOLDS
        
        # ROI optimization parameters
        self.profit_taking_threshold = 0.15    # Take profits at 15%+
        self.stop_loss_threshold = -0.08       # Stop loss at -8%
        self.momentum_lookback = 14            # Days for momentum calculation
        self.correlation_threshold = 0.75      # High correlation warning
        self.max_position_concentration = 0.15  # Max 15% in any single asset
        
    def analyze_portfolio(self, portfolio_data: Dict[str, Dict[str, Any]], 
                         holdings: List[Dict[str, Any]]) -> PortfolioInsights:
        """Generate comprehensive portfolio insights for maximum ROI"""
        
        logger.info(f"Analyzing portfolio with {len(holdings)} holdings")
        
        # Calculate portfolio metrics
        total_value = sum(holding['amount'] for holding in holdings)
        
        # Individual recommendations
        recommendations = []
        for holding in holdings:
            symbol = holding['symbol']
            if symbol in portfolio_data:
                recommendation = self._analyze_holding(
                    symbol, holding, portfolio_data[symbol], total_value
                )
                recommendations.append(recommendation)
        
        # Portfolio-level analysis
        risk_score = self._calculate_portfolio_risk(recommendations, portfolio_data)
        diversification_score = self._calculate_diversification(holdings)
        momentum_score = self._calculate_portfolio_momentum(portfolio_data, holdings)
        market_regime = self._determine_market_regime(portfolio_data)
        
        # Identify opportunities and risks
        profit_opportunities = self._identify_profit_taking_opportunities(recommendations)
        stop_loss_alerts = self._identify_stop_loss_candidates(recommendations)
        underperforming = self._identify_underperforming_assets(recommendations)
        top_opportunities = self._identify_top_opportunities(recommendations)
        
        # Determine if rebalancing is needed
        rebalancing_needed = self._assess_rebalancing_need(recommendations, holdings)
        
        insights = PortfolioInsights(
            total_portfolio_value=total_value,
            risk_score=risk_score,
            diversification_score=diversification_score,
            momentum_score=momentum_score,
            market_regime=market_regime,
            recommendations=recommendations,
            rebalancing_needed=rebalancing_needed,
            profit_taking_opportunities=profit_opportunities,
            stop_loss_alerts=stop_loss_alerts,
            underperforming_assets=underperforming,
            top_opportunities=top_opportunities
        )
        
        logger.info(f"✅ Portfolio analysis complete. Risk: {risk_score:.2f}, "
                   f"Diversification: {diversification_score:.2f}, "
                   f"Momentum: {momentum_score:.2f}")
        
        return insights
    
    def _analyze_holding(self, symbol: str, holding: Dict[str, Any], 
                        symbol_data: Dict[str, Any], total_portfolio_value: float) -> AIRecommendation:
        """Analyze individual holding and generate AI recommendation"""
        
        current_weight = holding['amount'] / total_portfolio_value
        
        # Aggregate signals across timeframes
        all_signals = []
        price_performance = {}
        risk_indicators = {}
        
        for timeframe, analysis in symbol_data.items():
            if 'signals' in analysis:
                all_signals.extend(analysis['signals'])
            
            if 'indicators' in analysis:
                indicators = analysis['indicators']
                
                # Price performance metrics
                if 'price_change_24h' in indicators:
                    price_performance[f'{timeframe}_24h'] = indicators['price_change_24h']
                if 'price_change_7d' in indicators:
                    price_performance[f'{timeframe}_7d'] = indicators['price_change_7d']
                
                # Risk indicators
                if 'atr_percentage' in indicators:
                    risk_indicators[f'{timeframe}_volatility'] = indicators['atr_percentage']
                if 'rsi_14' in indicators:
                    risk_indicators[f'{timeframe}_rsi'] = indicators['rsi_14']
        
        # AI decision making
        action, confidence, reasoning = self._make_ai_decision(
            symbol, all_signals, price_performance, risk_indicators, current_weight
        )
        
        # Calculate targets and risk metrics
        price_targets = self._calculate_price_targets(symbol_data, action)
        risk_metrics = self._calculate_holding_risk_metrics(symbol_data, price_performance)
        expected_roi = self._estimate_roi(action, price_targets, risk_metrics)
        max_loss = self._calculate_max_acceptable_loss(current_weight, risk_metrics)
        
        # Determine optimal allocation
        target_allocation = self._calculate_optimal_allocation(
            symbol, action, risk_metrics, current_weight
        )
        
        # Urgency scoring
        urgency = self._calculate_urgency(action, confidence, risk_metrics, price_performance)
        
        # Rebalancing trigger
        rebalance_trigger = abs(target_allocation - current_weight) > self.config['rebalancing_threshold']
        
        return AIRecommendation(
            symbol=symbol,
            action=action,
            confidence=confidence,
            urgency=urgency,
            target_allocation=target_allocation,
            reasoning=reasoning,
            price_targets=price_targets,
            risk_metrics=risk_metrics,
            timeframe="1d",  # Primary timeframe
            expected_roi=expected_roi,
            max_loss=max_loss,
            rebalance_trigger=rebalance_trigger
        )
    
    def _make_ai_decision(self, symbol: str, signals: List[Any], 
                         performance: Dict[str, float], risk: Dict[str, Any], 
                         current_weight: float) -> Tuple[ActionType, float, List[str]]:
        """Core AI decision making algorithm"""
        
        # Score signals by strength and confidence
        buy_score = 0
        sell_score = 0
        reasoning = []
        
        for signal in signals:
            # Handle both TechnicalSignal objects and dictionaries
            if hasattr(signal, 'strength'):
                signal_strength = signal.strength.value if hasattr(signal.strength, 'value') else 3
                signal_confidence = signal.confidence
                signal_type = signal.signal_type
                signal_indicator = signal.indicator
                signal_reasoning = signal.reasoning
            else:
                signal_strength = signal.get('strength', {}).get('value', 3)
                signal_confidence = signal.get('confidence', 0.5)
                signal_type = signal.get('signal_type', 'hold')
                signal_indicator = signal.get('indicator', 'signal')
                signal_reasoning = signal.get('reasoning', '')
            
            weight = signal_strength * signal_confidence
            
            if signal_type in ['buy', 'strong_buy']:
                buy_score += weight
                if signal_confidence > 0.7:
                    reasoning.append(f"Strong {signal_indicator}: {signal_reasoning}")
            elif signal_type in ['sell', 'strong_sell']:
                sell_score += weight
                if signal_confidence > 0.7:
                    reasoning.append(f"Sell {signal_indicator}: {signal_reasoning}")
        
        # Performance-based adjustments
        recent_performance = performance.get('1d_24h', 0)
        weekly_performance = performance.get('1d_7d', 0)
        
        # Profit taking logic
        if recent_performance > self.profit_taking_threshold * 100:  # 15%+ gain
            sell_score += 3
            reasoning.append(f"Profit taking opportunity: +{recent_performance:.1f}% in 24h")
        
        # Stop loss logic
        if weekly_performance < self.stop_loss_threshold * 100:  # -8% loss
            sell_score += 5
            reasoning.append(f"Stop loss triggered: {weekly_performance:.1f}% weekly loss")
        
        # Risk-based adjustments
        avg_volatility = np.mean([v for k, v in risk.items() if 'volatility' in k])
        if avg_volatility > 8:  # High volatility
            if current_weight > self.max_position_concentration:
                sell_score += 2
                reasoning.append(f"High volatility position ({avg_volatility:.1f}%) - reduce exposure")
        
        # Position size considerations
        if current_weight > self.max_position_concentration:
            sell_score += 1
            reasoning.append(f"Overweight position ({current_weight:.1%}) - rebalance needed")
        elif current_weight < self.config['min_position_size']:
            buy_score += 1
            reasoning.append(f"Underweight position ({current_weight:.1%}) - consider accumulating")
        
        # Make final decision
        net_score = buy_score - sell_score
        total_signals = buy_score + sell_score
        confidence = min(total_signals / 10, 1.0) if total_signals > 0 else 0.5
        
        if net_score > 3:
            action = ActionType.STRONG_BUY
        elif net_score > 1:
            action = ActionType.BUY
        elif net_score > 0:
            action = ActionType.ACCUMULATE
        elif net_score < -3:
            action = ActionType.SELL
        elif net_score < -1:
            action = ActionType.REDUCE
        elif weekly_performance < self.stop_loss_threshold * 100:
            action = ActionType.STOP_LOSS
        else:
            action = ActionType.HOLD
        
        return action, confidence, reasoning[:3]  # Top 3 reasons
    
    def _calculate_price_targets(self, symbol_data: Dict[str, Any], action: ActionType) -> Dict[str, float]:
        """Calculate price targets based on technical analysis"""
        
        # Get current price and key levels from 1d timeframe
        analysis_1d = symbol_data.get('1d', {})
        indicators = analysis_1d.get('indicators', {})
        
        current_price = indicators.get('current_price', 0)
        resistance = indicators.get('resistance_20', current_price * 1.1)
        support = indicators.get('support_20', current_price * 0.9)
        
        targets = {'current': current_price}
        
        if action in [ActionType.STRONG_BUY, ActionType.BUY, ActionType.ACCUMULATE]:
            targets.update({
                'target_1': resistance,
                'target_2': resistance * 1.1,
                'stop_loss': support
            })
        elif action in [ActionType.SELL, ActionType.REDUCE, ActionType.STOP_LOSS]:
            targets.update({
                'target_1': support,
                'target_2': support * 0.9,
                'stop_loss': resistance
            })
        
        return targets
    
    def _calculate_holding_risk_metrics(self, symbol_data: Dict[str, Any], 
                                      performance: Dict[str, float]) -> Dict[str, Any]:
        """Calculate risk metrics for individual holding"""
        
        risk_metrics = {
            'volatility_score': 0.5,
            'momentum_score': 0.5,
            'risk_category': RiskCategory.MODERATE,
            'max_drawdown_estimate': 0.0,
            'correlation_risk': 0.5
        }
        
        # Get volatility from multiple timeframes
        volatilities = []
        for timeframe, analysis in symbol_data.items():
            if 'indicators' in analysis:
                atr_pct = analysis['indicators'].get('atr_percentage', 0)
                if atr_pct > 0:
                    volatilities.append(atr_pct)
        
        if volatilities:
            avg_volatility = np.mean(volatilities)
            risk_metrics['volatility_score'] = min(avg_volatility / 10, 2.0)  # Normalize
            
            # Risk categorization
            if avg_volatility > 15:
                risk_metrics['risk_category'] = RiskCategory.SPECULATIVE
            elif avg_volatility > 8:
                risk_metrics['risk_category'] = RiskCategory.AGGRESSIVE
            elif avg_volatility > 4:
                risk_metrics['risk_category'] = RiskCategory.MODERATE
            else:
                risk_metrics['risk_category'] = RiskCategory.CONSERVATIVE
        
        # Momentum scoring
        recent_perf = performance.get('1d_24h', 0)
        weekly_perf = performance.get('1d_7d', 0)
        
        momentum = (recent_perf * 0.3 + weekly_perf * 0.7) / 100
        risk_metrics['momentum_score'] = max(0, min(1, 0.5 + momentum))
        
        # Estimate maximum drawdown
        if volatilities:
            risk_metrics['max_drawdown_estimate'] = -np.mean(volatilities) * 0.5
        
        return risk_metrics
    
    def _estimate_roi(self, action: ActionType, price_targets: Dict[str, float], 
                     risk_metrics: Dict[str, Any]) -> float:
        """Estimate expected ROI for the recommendation"""
        
        current = price_targets.get('current', 1)
        target_1 = price_targets.get('target_1', current)
        
        if action in [ActionType.STRONG_BUY, ActionType.BUY]:
            base_roi = (target_1 - current) / current
            # Adjust for risk
            risk_adjustment = 1 - (risk_metrics['volatility_score'] * 0.2)
            return base_roi * risk_adjustment
        elif action in [ActionType.SELL, ActionType.REDUCE]:
            # ROI from avoiding losses
            return 0.05  # Assume 5% gain from avoiding potential losses
        else:
            return 0.0
    
    def _calculate_max_acceptable_loss(self, current_weight: float, 
                                     risk_metrics: Dict[str, Any]) -> float:
        """Calculate maximum acceptable loss for position"""
        
        base_loss = current_weight * 0.02  # 2% of portfolio base risk
        
        # Adjust for asset risk
        risk_multiplier = risk_metrics['volatility_score']
        
        return min(base_loss * risk_multiplier, current_weight * 0.1)  # Max 10% of position
    
    def _calculate_optimal_allocation(self, symbol: str, action: ActionType, 
                                    risk_metrics: Dict[str, Any], current_weight: float) -> float:
        """Calculate optimal portfolio allocation for the holding"""
        
        risk_category = risk_metrics['risk_category']
        
        # Base allocation by risk category
        base_allocations = {
            RiskCategory.CONSERVATIVE: 0.08,    # 8%
            RiskCategory.MODERATE: 0.05,        # 5%
            RiskCategory.AGGRESSIVE: 0.03,      # 3%
            RiskCategory.SPECULATIVE: 0.01      # 1%
        }
        
        base_allocation = base_allocations.get(risk_category, 0.05)
        
        # Adjust based on action
        if action == ActionType.STRONG_BUY:
            target = min(base_allocation * 2, self.max_position_concentration)
        elif action == ActionType.BUY:
            target = min(base_allocation * 1.5, self.max_position_concentration)
        elif action == ActionType.ACCUMULATE:
            target = base_allocation
        elif action == ActionType.REDUCE:
            target = base_allocation * 0.5
        elif action in [ActionType.SELL, ActionType.STOP_LOSS]:
            target = 0.0
        else:  # HOLD
            target = current_weight
        
        # Ensure within bounds
        return max(0, min(target, self.max_position_concentration))
    
    def _calculate_urgency(self, action: ActionType, confidence: float, 
                         risk_metrics: Dict[str, Any], performance: Dict[str, float]) -> int:
        """Calculate urgency score (1-10)"""
        
        urgency = 5  # Base urgency
        
        # Action-based urgency
        if action == ActionType.STOP_LOSS:
            urgency = 10
        elif action == ActionType.STRONG_BUY:
            urgency = 8
        elif action == ActionType.SELL:
            urgency = 7
        elif action == ActionType.BUY:
            urgency = 6
        
        # Confidence adjustment
        urgency = int(urgency * confidence)
        
        # Risk adjustment
        if risk_metrics['volatility_score'] > 1.5:
            urgency += 1
        
        # Performance adjustment
        weekly_perf = performance.get('1d_7d', 0)
        if weekly_perf < -15:  # Major loss
            urgency += 2
        elif weekly_perf > 20:  # Major gain
            urgency += 1
        
        return max(1, min(10, urgency))
    
    def _calculate_portfolio_risk(self, recommendations: List[AIRecommendation], 
                                portfolio_data: Dict[str, Dict[str, Any]]) -> float:
        """Calculate overall portfolio risk score"""
        
        if not recommendations:
            return 0.5
        
        # Weighted average of individual risk scores
        total_allocation = sum(rec.target_allocation for rec in recommendations)
        if total_allocation == 0:
            return 0.5
        
        weighted_risk = sum(
            rec.risk_metrics['volatility_score'] * rec.target_allocation 
            for rec in recommendations
        ) / total_allocation
        
        # Concentration risk
        max_allocation = max(rec.target_allocation for rec in recommendations)
        concentration_penalty = max_allocation if max_allocation > 0.2 else 0
        
        portfolio_risk = min(weighted_risk + concentration_penalty, 1.0)
        
        return portfolio_risk
    
    def _calculate_diversification(self, holdings: List[Dict[str, Any]]) -> float:
        """Calculate portfolio diversification score"""
        
        num_holdings = len(holdings)
        
        # Base diversification from number of holdings
        if num_holdings >= 20:
            base_score = 1.0
        elif num_holdings >= 10:
            base_score = 0.8
        elif num_holdings >= 5:
            base_score = 0.6
        else:
            base_score = 0.4
        
        # Adjust for concentration
        total_value = sum(holding['amount'] for holding in holdings)
        if total_value > 0:
            weights = [holding['amount'] / total_value for holding in holdings]
            max_weight = max(weights)
            
            if max_weight > 0.5:  # Single holding > 50%
                base_score *= 0.5
            elif max_weight > 0.3:  # Single holding > 30%
                base_score *= 0.7
            elif max_weight > 0.2:  # Single holding > 20%
                base_score *= 0.9
        
        return base_score
    
    def _calculate_portfolio_momentum(self, portfolio_data: Dict[str, Dict[str, Any]], 
                                    holdings: List[Dict[str, Any]]) -> float:
        """Calculate portfolio momentum score"""
        
        momentum_scores = []
        
        for holding in holdings:
            symbol = holding['symbol']
            if symbol in portfolio_data and '1d' in portfolio_data[symbol]:
                indicators = portfolio_data[symbol]['1d'].get('indicators', {})
                
                # Calculate momentum from price changes
                change_24h = indicators.get('price_change_24h', 0)
                change_7d = indicators.get('price_change_7d', 0)
                
                # Weight recent performance more heavily
                momentum = (change_24h * 0.4 + change_7d * 0.6) / 100
                momentum_scores.append(momentum)
        
        if not momentum_scores:
            return 0.5
        
        # Average momentum, normalized to 0-1 scale
        avg_momentum = np.mean(momentum_scores)
        return max(0, min(1, 0.5 + avg_momentum))
    
    def _determine_market_regime(self, portfolio_data: Dict[str, Dict[str, Any]]) -> str:
        """Determine current market regime"""
        
        # Analyze major market movers
        major_symbols = ['BTC', 'ETH', 'SOL']
        
        trends = []
        volatilities = []
        
        for symbol in major_symbols:
            if symbol in portfolio_data and '1d' in portfolio_data[symbol]:
                indicators = portfolio_data[symbol]['1d'].get('indicators', {})
                
                change_7d = indicators.get('price_change_7d', 0)
                volatility = indicators.get('atr_percentage', 5)
                
                trends.append(change_7d)
                volatilities.append(volatility)
        
        if not trends:
            return "unknown"
        
        avg_trend = np.mean(trends)
        avg_volatility = np.mean(volatilities)
        
        # Classify regime
        if avg_volatility > 8:
            return "high_volatility"
        elif avg_trend > 10:
            return "bull_market"
        elif avg_trend < -10:
            return "bear_market"
        elif abs(avg_trend) < 3:
            return "sideways"
        else:
            return "transitional"
    
    def _identify_profit_taking_opportunities(self, recommendations: List[AIRecommendation]) -> List[Dict[str, Any]]:
        """Identify immediate profit taking opportunities"""
        
        opportunities = []
        
        for rec in recommendations:
            if rec.action in [ActionType.SELL, ActionType.REDUCE] and rec.confidence > 0.6:
                if rec.expected_roi > 0 or any('profit' in reason.lower() for reason in rec.reasoning):
                    opportunities.append({
                        'symbol': rec.symbol,
                        'action': rec.action.value,
                        'confidence': rec.confidence,
                        'expected_profit': rec.expected_roi,
                        'urgency': rec.urgency,
                        'reasoning': rec.reasoning[0] if rec.reasoning else "Profit taking opportunity"
                    })
        
        # Sort by urgency and confidence
        opportunities.sort(key=lambda x: x['urgency'] * x['confidence'], reverse=True)
        
        return opportunities[:5]  # Top 5
    
    def _identify_stop_loss_candidates(self, recommendations: List[AIRecommendation]) -> List[Dict[str, Any]]:
        """Identify holdings that need immediate stop loss"""
        
        alerts = []
        
        for rec in recommendations:
            if rec.action == ActionType.STOP_LOSS or rec.urgency >= 8:
                if any('loss' in reason.lower() or 'stop' in reason.lower() for reason in rec.reasoning):
                    alerts.append({
                        'symbol': rec.symbol,
                        'urgency': rec.urgency,
                        'max_loss': rec.max_loss,
                        'reasoning': rec.reasoning[0] if rec.reasoning else "Stop loss triggered",
                        'recommended_exit': rec.price_targets.get('target_1', 0)
                    })
        
        # Sort by urgency
        alerts.sort(key=lambda x: x['urgency'], reverse=True)
        
        return alerts
    
    def _identify_underperforming_assets(self, recommendations: List[AIRecommendation]) -> List[str]:
        """Identify consistently underperforming assets"""
        
        underperformers = []
        
        for rec in recommendations:
            if rec.expected_roi < -0.05 and rec.confidence > 0.5:  # Expected loss > 5%
                underperformers.append(rec.symbol)
        
        return underperformers
    
    def _identify_top_opportunities(self, recommendations: List[AIRecommendation]) -> List[str]:
        """Identify top investment opportunities"""
        
        opportunities = []
        
        for rec in recommendations:
            if rec.action in [ActionType.STRONG_BUY, ActionType.BUY] and rec.expected_roi > 0.1:
                opportunities.append((rec.symbol, rec.expected_roi * rec.confidence))
        
        # Sort by risk-adjusted expected return
        opportunities.sort(key=lambda x: x[1], reverse=True)
        
        return [symbol for symbol, _ in opportunities[:5]]
    
    def _assess_rebalancing_need(self, recommendations: List[AIRecommendation], 
                               holdings: List[Dict[str, Any]]) -> bool:
        """Assess if portfolio rebalancing is needed"""
        
        # Count recommendations requiring rebalancing
        rebalance_count = sum(1 for rec in recommendations if rec.rebalance_trigger)
        
        # Check for high urgency situations
        high_urgency_count = sum(1 for rec in recommendations if rec.urgency >= 8)
        
        return rebalance_count >= 3 or high_urgency_count >= 1
    
    def generate_executive_summary(self, insights: PortfolioInsights) -> Dict[str, Any]:
        """Generate executive summary for quick decision making"""
        
        summary = {
            'overall_health': 'good',
            'immediate_actions_needed': 0,
            'profit_opportunities': len(insights.profit_taking_opportunities),
            'risk_alerts': len(insights.stop_loss_alerts),
            'portfolio_balance': 'balanced',
            'key_recommendations': [],
            'performance_outlook': 'neutral'
        }
        
        # Assess overall health
        if insights.risk_score > 0.7:
            summary['overall_health'] = 'risky'
        elif insights.risk_score < 0.3 and insights.diversification_score > 0.7:
            summary['overall_health'] = 'excellent'
        
        # Count immediate actions
        summary['immediate_actions_needed'] = len([
            rec for rec in insights.recommendations 
            if rec.urgency >= 8
        ])
        
        # Portfolio balance assessment
        if insights.diversification_score < 0.5:
            summary['portfolio_balance'] = 'concentrated'
        elif insights.diversification_score > 0.8:
            summary['portfolio_balance'] = 'well_diversified'
        
        # Key recommendations (top 3)
        high_priority_recs = sorted(
            insights.recommendations,
            key=lambda x: x.urgency * x.confidence,
            reverse=True
        )[:3]
        
        summary['key_recommendations'] = [
            {
                'symbol': rec.symbol,
                'action': rec.action.value,
                'reasoning': rec.reasoning[0] if rec.reasoning else "AI recommendation"
            }
            for rec in high_priority_recs
        ]
        
        # Performance outlook
        if insights.momentum_score > 0.7:
            summary['performance_outlook'] = 'positive'
        elif insights.momentum_score < 0.3:
            summary['performance_outlook'] = 'negative'
        
        return summary

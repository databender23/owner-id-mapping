"""
Specialized Subagents for AI Optimization

This module contains specialized agents that work under the main orchestrator
to perform specific optimization tasks.
"""

from .pattern_discovery import PatternDiscoveryAgent
# These will be implemented next
# from .validation import ValidationAgent
# from .strategy_optimizer import StrategyOptimizationAgent
# from .threshold_tuner import ThresholdTuningAgent
# from .temporal_analyzer import TemporalAnalyzerAgent
# from .meta_learner import MetaLearningAgent

__all__ = [
    'PatternDiscoveryAgent',
    # 'ValidationAgent',
    # 'StrategyOptimizationAgent',
    # 'ThresholdTuningAgent',
    # 'TemporalAnalyzerAgent',
    # 'MetaLearningAgent'
]
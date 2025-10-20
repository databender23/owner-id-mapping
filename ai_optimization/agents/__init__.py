"""
AI Optimization Agents Module

This module contains the orchestrator and specialized subagents for
owner ID matching optimization.
"""

from .orchestrator import EnhancedOrchestrator, IterationResult
from .context_manager import (
    ContextManager,
    LearningContext,
    Pattern,
    StrategyPerformance
)

__all__ = [
    'EnhancedOrchestrator',
    'IterationResult',
    'ContextManager',
    'LearningContext',
    'Pattern',
    'StrategyPerformance'
]

__version__ = '1.0.0'
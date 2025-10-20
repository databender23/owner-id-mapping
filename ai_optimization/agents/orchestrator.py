"""
Enhanced Orchestrator for AI-Powered Owner ID Matching Optimization

This module implements the main orchestrator that coordinates multiple
specialized subagents working in parallel to optimize the matching process.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import yaml
import pandas as pd
from tqdm.asyncio import tqdm

from .context_manager import ContextManager, LearningContext, Pattern, StrategyPerformance

# Import subagents (to be implemented)
from .subagents.pattern_discovery import PatternDiscoveryAgent
from .subagents.validation import ValidationAgent  # Now implemented!
# from .subagents.strategy_optimizer import StrategyOptimizationAgent  # Not implemented yet
# from .subagents.threshold_tuner import ThresholdTuningAgent  # Not implemented yet
from .subagents.temporal_analyzer import TemporalAnalyzerAgent
# from .subagents.meta_learner import MetaLearningAgent  # Not implemented yet

logger = logging.getLogger(__name__)


@dataclass
class IterationResult:
    """Results from a single optimization iteration."""
    iteration_num: int
    match_rate: float
    total_matches: int
    new_patterns: List[Pattern]
    strategy_updates: Dict[str, Any]
    threshold_changes: Dict[str, float]
    validation_results: Dict[str, Any]
    meta_insights: List[str]
    convergence_status: str
    execution_time: float
    timestamp: str


@dataclass
class SubagentTask:
    """Represents a task for a subagent."""
    agent_name: str
    task_type: str
    input_data: Any
    priority: str  # "high", "medium", "low"
    timeout: int
    dependencies: List[str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class EnhancedOrchestrator:
    """
    Main orchestrator that coordinates multiple subagents for optimization.

    Features:
    - Hierarchical task delegation
    - Parallel subagent execution
    - Context-aware decision making
    - Continuous learning integration
    - Meta-level optimization insights
    """

    def __init__(self, config_path: str = "ai_optimization/config.yaml"):
        """
        Initialize the orchestrator with configuration.

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.context_manager = ContextManager(
            self.config['database']['context_store_path']
        )

        # Initialize subagents
        self.agents = self._initialize_agents()

        # Tracking
        self.current_iteration = 0
        self.iteration_history = []
        self.active_tasks = {}
        self.task_queue = asyncio.Queue()

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize all subagents based on configuration."""
        agents = {}

        if self.config['agents']['pattern_discovery']['enabled']:
            agents['pattern_discovery'] = PatternDiscoveryAgent(
                self.config['agents']['pattern_discovery']
            )

        if self.config['agents']['validation']['enabled']:
            agents['validation'] = ValidationAgent(
                self.config['agents']['validation']
            )

        # Strategy optimization agent - not implemented yet
        # if self.config['agents']['strategy_optimization']['enabled']:
        #     agents['strategy_optimization'] = StrategyOptimizationAgent(
        #         self.config['agents']['strategy_optimization']
        #     )

        # Threshold tuning agent - not implemented yet
        # if self.config['agents']['threshold_tuning']['enabled']:
        #     agents['threshold_tuning'] = ThresholdTuningAgent(
        #         self.config['agents']['threshold_tuning']
        #     )

        if self.config['agents']['temporal_analyzer']['enabled']:
            agents['temporal_analyzer'] = TemporalAnalyzerAgent(
                self.config['agents']['temporal_analyzer']
            )

        # Meta-learner agent - not implemented yet
        # if self.config['agents']['meta_learner']['enabled']:
        #     agents['meta_learner'] = MetaLearningAgent(
        #         self.config['agents']['meta_learner']
        #     )

        logger.info(f"Initialized {len(agents)} subagents")
        return agents

    async def run_iteration(
        self,
        iteration_num: int,
        match_results_df: pd.DataFrame,
        continue_from_checkpoint: bool = True
    ) -> IterationResult:
        """
        Run a complete optimization iteration.

        Args:
            iteration_num: Current iteration number
            match_results_df: Results from current matching run
            continue_from_checkpoint: Whether to load previous context

        Returns:
            IterationResult with all findings and updates
        """
        start_time = datetime.now()
        self.current_iteration = iteration_num

        logger.info(f"Starting iteration {iteration_num}")

        # Load context if continuing
        context = None
        if continue_from_checkpoint:
            context = await self.context_manager.load_checkpoint()
            if context:
                logger.info(f"Loaded context from iteration {context.current_iteration}")

        # Prepare data for subagents
        unmatched_records = match_results_df[match_results_df['new_id'].isna()]
        matched_records = match_results_df[~match_results_df['new_id'].isna()]

        # Create tasks for subagents
        tasks = self._create_subagent_tasks(
            matched_records,
            unmatched_records,
            context
        )

        # Execute tasks in parallel with priority handling
        results = await self._execute_parallel_tasks(tasks)

        # Synthesize results from all agents
        synthesis = await self._synthesize_results(results, context)

        # Apply learnings and update context
        await self._apply_learnings(synthesis, iteration_num)

        # Save checkpoint
        new_context = await self._create_context_snapshot(
            iteration_num,
            matched_records,
            match_results_df,
            synthesis
        )
        await self.context_manager.save_checkpoint(new_context)

        # Create iteration result
        execution_time = (datetime.now() - start_time).total_seconds()

        result = IterationResult(
            iteration_num=iteration_num,
            match_rate=len(matched_records) / len(match_results_df),
            total_matches=len(matched_records),
            new_patterns=synthesis.get('patterns', []),
            strategy_updates=synthesis.get('strategy_updates', {}),
            threshold_changes=synthesis.get('threshold_changes', {}),
            validation_results=synthesis.get('validation', {}),
            meta_insights=synthesis.get('meta_insights', []),
            convergence_status=synthesis.get('convergence_status', 'improving'),
            execution_time=execution_time,
            timestamp=datetime.now().isoformat()
        )

        self.iteration_history.append(result)
        logger.info(f"Iteration {iteration_num} completed in {execution_time:.2f}s")

        return result

    def _create_subagent_tasks(
        self,
        matched_records: pd.DataFrame,
        unmatched_records: pd.DataFrame,
        context: Optional[LearningContext]
    ) -> List[SubagentTask]:
        """
        Create tasks for each enabled subagent.

        Args:
            matched_records: Successfully matched records
            unmatched_records: Records that didn't match
            context: Previous learning context

        Returns:
            List of tasks to execute
        """
        tasks = []

        # Pattern Discovery Task
        if 'pattern_discovery' in self.agents:
            tasks.append(SubagentTask(
                agent_name='pattern_discovery',
                task_type='analyze_unmatched',
                input_data={
                    'records': unmatched_records,
                    'previous_patterns': context.patterns if context else [],
                    'iteration': self.current_iteration
                },
                priority='high',
                timeout=300
            ))

        # Validation Task
        if 'validation' in self.agents and len(matched_records) > 0:
            tasks.append(SubagentTask(
                agent_name='validation',
                task_type='validate_matches',
                input_data={
                    'matches_df': matched_records,  # Changed from 'matches' to 'matches_df'
                    'old_owners_df': None,  # Will be populated if needed
                    'new_owners_df': None,  # Will be populated if needed
                    'iteration': self.current_iteration
                },
                priority='high',
                timeout=300
            ))

        # Strategy Optimization Task
        if 'strategy_optimization' in self.agents:
            tasks.append(SubagentTask(
                agent_name='strategy_optimization',
                task_type='optimize_strategies',
                input_data={
                    'all_records': pd.concat([matched_records, unmatched_records]),
                    'current_performance': context.strategy_performances if context else {},
                    'iteration': self.current_iteration
                },
                priority='medium',
                timeout=300,
                dependencies=['pattern_discovery']
            ))

        # Threshold Tuning Task
        if 'threshold_tuning' in self.agents:
            tasks.append(SubagentTask(
                agent_name='threshold_tuning',
                task_type='tune_thresholds',
                input_data={
                    'matched': matched_records,
                    'unmatched': unmatched_records,
                    'current_thresholds': self._get_current_thresholds(),
                    'threshold_history': context.threshold_history if context else []
                },
                priority='medium',
                timeout=300,
                dependencies=['validation']
            ))

        # Temporal Analysis Task
        if 'temporal_analyzer' in self.agents:
            tasks.append(SubagentTask(
                agent_name='temporal_analyzer',
                task_type='analyze_temporal_changes',
                input_data={
                    'unmatched': unmatched_records,
                    'separators': self.config['agents']['temporal_analyzer']['separators'],
                    'iteration': self.current_iteration
                },
                priority='high',
                timeout=300
            ))

        # Meta Learning Task (depends on all others)
        if 'meta_learner' in self.agents:
            tasks.append(SubagentTask(
                agent_name='meta_learner',
                task_type='analyze_optimization',
                input_data={
                    'iteration_history': self.iteration_history,
                    'current_iteration': self.current_iteration
                },
                priority='low',
                timeout=300,
                dependencies=['pattern_discovery', 'validation', 'strategy_optimization']
            ))

        return tasks

    async def _execute_parallel_tasks(
        self,
        tasks: List[SubagentTask]
    ) -> Dict[str, Any]:
        """
        Execute subagent tasks in parallel with dependency handling.

        Args:
            tasks: List of tasks to execute

        Returns:
            Dictionary of results from each agent
        """
        results = {}
        completed_tasks = set()

        # Sort tasks by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        tasks.sort(key=lambda t: priority_order[t.priority])

        # Create task groups by dependency level
        task_groups = self._organize_by_dependencies(tasks)

        # Execute each group in sequence, tasks within group in parallel
        for group_level, group_tasks in enumerate(task_groups):
            logger.info(f"Executing task group {group_level + 1}/{len(task_groups)}")

            # Create coroutines for this group
            coroutines = []
            for task in group_tasks:
                if self._dependencies_met(task, completed_tasks):
                    coroutines.append(self._execute_single_task(task))

            # Execute group in parallel
            if coroutines:
                group_results = await asyncio.gather(*coroutines, return_exceptions=True)

                # Process results
                for task, result in zip(group_tasks, group_results):
                    if isinstance(result, Exception):
                        logger.error(f"Task {task.agent_name} failed: {result}")
                        results[task.agent_name] = {'error': str(result)}
                    else:
                        results[task.agent_name] = result
                        completed_tasks.add(task.agent_name)

        return results

    def _organize_by_dependencies(
        self,
        tasks: List[SubagentTask]
    ) -> List[List[SubagentTask]]:
        """
        Organize tasks into groups based on dependencies.

        Args:
            tasks: List of all tasks

        Returns:
            List of task groups, where each group can run in parallel
        """
        groups = []
        remaining = tasks.copy()
        completed = set()

        while remaining:
            current_group = []

            for task in remaining[:]:
                if self._dependencies_met(task, completed):
                    current_group.append(task)
                    remaining.remove(task)

            if not current_group:
                # Circular dependency or error
                logger.error("Circular dependency detected in tasks")
                break

            groups.append(current_group)
            completed.update(t.agent_name for t in current_group)

        return groups

    def _dependencies_met(
        self,
        task: SubagentTask,
        completed: set
    ) -> bool:
        """Check if all dependencies for a task are met."""
        return all(dep in completed for dep in task.dependencies)

    async def _execute_single_task(self, task: SubagentTask) -> Dict[str, Any]:
        """
        Execute a single subagent task with timeout.

        Args:
            task: Task to execute

        Returns:
            Result from the agent
        """
        logger.info(f"Executing task: {task.agent_name}")

        try:
            agent = self.agents[task.agent_name]

            # Execute with timeout
            result = await asyncio.wait_for(
                agent.execute(task.task_type, task.input_data),
                timeout=task.timeout
            )

            logger.info(f"Task {task.agent_name} completed successfully")
            return result

        except asyncio.TimeoutError:
            logger.error(f"Task {task.agent_name} timed out after {task.timeout}s")
            raise
        except Exception as e:
            logger.error(f"Task {task.agent_name} failed: {e}")
            raise

    async def _synthesize_results(
        self,
        results: Dict[str, Any],
        context: Optional[LearningContext]
    ) -> Dict[str, Any]:
        """
        Synthesize results from all agents into actionable insights.

        Args:
            results: Results from each agent
            context: Previous learning context

        Returns:
            Synthesized insights and recommendations
        """
        synthesis = {
            'patterns': [],
            'strategy_updates': {},
            'threshold_changes': {},
            'validation': {},
            'meta_insights': [],
            'convergence_status': 'unknown'
        }

        # Extract patterns
        if 'pattern_discovery' in results and not results['pattern_discovery'].get('error'):
            synthesis['patterns'] = results['pattern_discovery'].get('patterns', [])

        # Extract validation results
        if 'validation' in results and not results['validation'].get('error'):
            synthesis['validation'] = results['validation']

        # Extract strategy recommendations
        if 'strategy_optimization' in results and not results['strategy_optimization'].get('error'):
            synthesis['strategy_updates'] = results['strategy_optimization'].get('recommendations', {})

        # Extract threshold recommendations
        if 'threshold_tuning' in results and not results['threshold_tuning'].get('error'):
            synthesis['threshold_changes'] = results['threshold_tuning'].get('recommended_changes', {})

        # Extract temporal patterns
        if 'temporal_analyzer' in results and not results['temporal_analyzer'].get('error'):
            temporal_patterns = results['temporal_analyzer'].get('temporal_patterns', [])
            synthesis['patterns'].extend(temporal_patterns)

        # Extract meta insights
        if 'meta_learner' in results and not results['meta_learner'].get('error'):
            synthesis['meta_insights'] = results['meta_learner'].get('insights', [])
            synthesis['convergence_status'] = results['meta_learner'].get('convergence_status', 'unknown')

        # Apply synthesis strategy from config
        strategy = self.config['orchestrator']['synthesis_strategy']
        if strategy == 'weighted_consensus':
            synthesis = self._apply_weighted_consensus(synthesis, results)
        elif strategy == 'hierarchical':
            synthesis = self._apply_hierarchical_synthesis(synthesis, results)

        return synthesis

    def _apply_weighted_consensus(
        self,
        synthesis: Dict[str, Any],
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply weighted consensus to synthesize results.

        High-priority agents have more weight in final decisions.
        """
        # Weight threshold changes based on validation results
        if synthesis['validation'].get('false_positive_rate', 0) > 0.05:
            # High false positives - be more conservative with thresholds
            for param, change in synthesis['threshold_changes'].items():
                if change < 0:  # Lowering threshold
                    synthesis['threshold_changes'][param] = change * 0.5  # Reduce change

        return synthesis

    def _apply_hierarchical_synthesis(
        self,
        synthesis: Dict[str, Any],
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply hierarchical synthesis where certain agents can veto others.
        """
        # Meta learner can override if convergence detected
        if synthesis['convergence_status'] == 'converged':
            synthesis['threshold_changes'] = {}  # No more threshold changes
            synthesis['meta_insights'].append("Convergence detected - stopping threshold adjustments")

        return synthesis

    async def _apply_learnings(self, synthesis: Dict[str, Any], iteration: int):
        """
        Apply the synthesized learnings to the system.

        Args:
            synthesis: Synthesized insights
            iteration: Current iteration number
        """
        # Store new patterns
        for pattern in synthesis['patterns']:
            pattern.discovered_iteration = iteration
            pattern.last_seen_iteration = iteration
            await self.context_manager.add_pattern(pattern)

        # Update strategy performance
        for strategy_name, updates in synthesis['strategy_updates'].items():
            if 'performance' in updates:
                await self.context_manager.update_strategy_performance(
                    strategy_name=strategy_name,
                    success=updates['performance'].get('success', False),
                    confidence=updates['performance'].get('confidence', 0),
                    execution_time=updates['performance'].get('execution_time', 0),
                    iteration=iteration
                )

        # Store validated matches
        if 'validated_matches' in synthesis['validation']:
            for match in synthesis['validation']['validated_matches']:
                await self.context_manager.add_validated_match(
                    match_data=match,
                    validation_status=match.get('validation_status', 'uncertain'),
                    validator_confidence=match.get('validator_confidence', 0),
                    iteration=iteration
                )

    async def _create_context_snapshot(
        self,
        iteration_num: int,
        matched_records: pd.DataFrame,
        all_records: pd.DataFrame,
        synthesis: Dict[str, Any]
    ) -> LearningContext:
        """
        Create a complete context snapshot for checkpointing.

        Args:
            iteration_num: Current iteration
            matched_records: Matched records
            all_records: All records
            synthesis: Synthesized results

        Returns:
            Complete learning context
        """
        # Get current patterns from database
        patterns = await self.context_manager.query_patterns(min_confidence=0.5)

        # Get strategy performances
        strategy_performances = {}
        for agent_name in self.agents:
            # This would query from the strategies database
            pass

        # Create context
        context = LearningContext(
            current_iteration=iteration_num,
            total_matches=len(matched_records),
            total_records=len(all_records),
            patterns=patterns[:100],  # Top 100
            strategy_performances=strategy_performances,
            threshold_history=synthesis.get('threshold_history', []),
            convergence_metrics={
                'match_rate': len(matched_records) / len(all_records),
                'convergence_status': synthesis['convergence_status']
            },
            meta_insights=synthesis['meta_insights'],
            validated_matches=synthesis['validation'].get('validated_matches', [])[:100],
            failed_attempts=[],  # Would be populated from validation
            timestamp=datetime.now().isoformat()
        )

        return context

    def _get_current_thresholds(self) -> Dict[str, float]:
        """Get current threshold values from configuration."""
        thresholds = {}
        for param in self.config['agents']['threshold_tuning']['parameters']:
            thresholds[param['name']] = param['current']
        return thresholds

    async def evaluate_convergence(self) -> Dict[str, Any]:
        """
        Evaluate whether optimization has converged.

        Returns:
            Convergence analysis with recommendations
        """
        if len(self.iteration_history) < 3:
            return {
                'converged': False,
                'reason': 'insufficient_iterations',
                'recommendation': 'continue'
            }

        recent_metrics = [
            {'match_rate': r.match_rate}
            for r in self.iteration_history[-3:]
        ]

        convergence_analysis = await self.context_manager.analyze_convergence(
            recent_metrics,
            window_size=3
        )

        return convergence_analysis

    async def generate_report(self, iteration_num: int) -> str:
        """
        Generate a comprehensive report for the iteration.

        Args:
            iteration_num: Iteration to report on

        Returns:
            Formatted report as string
        """
        if iteration_num > len(self.iteration_history):
            return "No data for requested iteration"

        result = self.iteration_history[iteration_num - 1]

        report = f"""
# AI Optimization Report - Iteration {iteration_num}

## Summary
- **Match Rate**: {result.match_rate:.2%}
- **Total Matches**: {result.total_matches}
- **Convergence Status**: {result.convergence_status}
- **Execution Time**: {result.execution_time:.2f}s

## New Patterns Discovered
{self._format_patterns(result.new_patterns)}

## Strategy Updates
{self._format_dict(result.strategy_updates)}

## Threshold Changes
{self._format_dict(result.threshold_changes)}

## Validation Results
{self._format_dict(result.validation_results)}

## Meta Insights
{self._format_list(result.meta_insights)}

## Recommendations
Based on the analysis, the system recommends:
{self._generate_recommendations(result)}

---
Generated: {result.timestamp}
        """

        return report

    def _format_patterns(self, patterns: List[Pattern]) -> str:
        """Format patterns for report."""
        if not patterns:
            return "No new patterns discovered"

        lines = []
        for p in patterns[:10]:  # Top 10
            lines.append(f"- **{p.pattern_type}**: {p.pattern_value} (confidence: {p.confidence:.2f})")
        return "\n".join(lines)

    def _format_dict(self, d: Dict) -> str:
        """Format dictionary for report."""
        if not d:
            return "No changes"

        lines = []
        for k, v in d.items():
            lines.append(f"- **{k}**: {v}")
        return "\n".join(lines)

    def _format_list(self, lst: List) -> str:
        """Format list for report."""
        if not lst:
            return "No insights"

        return "\n".join(f"- {item}" for item in lst)

    def _generate_recommendations(self, result: IterationResult) -> str:
        """Generate recommendations based on iteration results."""
        recommendations = []

        if result.convergence_status == 'converged':
            recommendations.append("- Optimization has converged. Consider stopping.")
        elif result.match_rate < 0.10:
            recommendations.append("- Match rate is low. Consider more aggressive thresholds.")
        elif result.validation_results.get('false_positive_rate', 0) > 0.05:
            recommendations.append("- False positive rate is high. Tighten thresholds.")

        if not recommendations:
            recommendations.append("- Continue optimization with current settings.")

        return "\n".join(recommendations)
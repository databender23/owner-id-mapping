"""
Context Manager for Persistent State Management

This module manages the persistent context store that maintains learning
across iterations, enabling the AI system to build upon previous insights.
"""

import json
import sqlite3
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from contextlib import asynccontextmanager
import logging
from dataclasses import dataclass, asdict
import aiosqlite
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Pattern:
    """Represents a discovered matching pattern."""
    id: Optional[int] = None
    pattern_type: str = ""
    pattern_value: str = ""
    frequency: int = 0
    confidence: float = 0.0
    discovered_iteration: int = 0
    last_seen_iteration: int = 0
    examples: List[Dict] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.examples is None:
            self.examples = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class StrategyPerformance:
    """Tracks performance metrics for matching strategies."""
    strategy_name: str
    total_attempts: int = 0
    successful_matches: int = 0
    false_positives: int = 0
    average_confidence: float = 0.0
    average_execution_time: float = 0.0
    record_types_effective_for: List[str] = None
    iteration_history: List[Dict] = None

    def __post_init__(self):
        if self.record_types_effective_for is None:
            self.record_types_effective_for = []
        if self.iteration_history is None:
            self.iteration_history = []

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_attempts == 0:
            return 0.0
        return self.successful_matches / self.total_attempts


@dataclass
class LearningContext:
    """Complete learning context for the system."""
    current_iteration: int
    total_matches: int
    total_records: int
    patterns: List[Pattern]
    strategy_performances: Dict[str, StrategyPerformance]
    threshold_history: List[Dict]
    convergence_metrics: Dict[str, float]
    meta_insights: List[str]
    validated_matches: List[Dict]
    failed_attempts: List[Dict]
    timestamp: str


class ContextManager:
    """
    Manages persistent context across optimization iterations.

    Maintains:
    - Pattern library
    - Strategy performance history
    - Validated matches
    - Learning checkpoints
    - Threshold evolution
    """

    def __init__(self, context_store_path: str):
        """
        Initialize context manager with database paths.

        Args:
            context_store_path: Base directory for context databases
        """
        self.context_path = Path(context_store_path)
        self.context_path.mkdir(parents=True, exist_ok=True)

        self.patterns_db = self.context_path / "patterns.db"
        self.strategies_db = self.context_path / "strategies.db"
        self.validated_db = self.context_path / "validated_matches.db"
        self.checkpoint_file = self.context_path / "learning_checkpoint.json"

        # Initialize databases
        asyncio.create_task(self._initialize_databases())

        self._pattern_cache = {}
        self._strategy_cache = {}

    async def _initialize_databases(self):
        """Create database schemas if they don't exist."""
        # Patterns database
        async with aiosqlite.connect(self.patterns_db) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_type TEXT NOT NULL,
                    pattern_value TEXT NOT NULL,
                    frequency INTEGER DEFAULT 1,
                    confidence REAL DEFAULT 0.0,
                    discovered_iteration INTEGER,
                    last_seen_iteration INTEGER,
                    examples TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(pattern_type, pattern_value)
                )
            """)

            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_pattern_type ON patterns(pattern_type)
            """)

            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_confidence ON patterns(confidence DESC)
            """)

            await db.commit()

        # Strategies database
        async with aiosqlite.connect(self.strategies_db) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS strategy_performance (
                    strategy_name TEXT PRIMARY KEY,
                    total_attempts INTEGER DEFAULT 0,
                    successful_matches INTEGER DEFAULT 0,
                    false_positives INTEGER DEFAULT 0,
                    average_confidence REAL DEFAULT 0.0,
                    average_execution_time REAL DEFAULT 0.0,
                    record_types TEXT,
                    iteration_history TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            await db.execute("""
                CREATE TABLE IF NOT EXISTS strategy_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    iteration INTEGER,
                    record_id TEXT,
                    selected_strategy TEXT,
                    confidence REAL,
                    success BOOLEAN,
                    execution_time REAL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            await db.commit()

        # Validated matches database
        async with aiosqlite.connect(self.validated_db) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS validated_matches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    old_id TEXT,
                    old_name TEXT,
                    new_id TEXT,
                    new_name TEXT,
                    match_strategy TEXT,
                    confidence_score REAL,
                    name_score REAL,
                    address_score REAL,
                    validation_status TEXT,
                    validator_confidence REAL,
                    iteration INTEGER,
                    feedback TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_old_id ON validated_matches(old_id)
            """)

            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_validation_status
                ON validated_matches(validation_status)
            """)

            await db.commit()

        logger.info("Context databases initialized successfully")

    async def add_pattern(self, pattern: Pattern) -> int:
        """
        Add or update a pattern in the pattern library.

        Args:
            pattern: Pattern object to store

        Returns:
            Pattern ID
        """
        async with aiosqlite.connect(self.patterns_db) as db:
            # Check if pattern exists
            cursor = await db.execute(
                """SELECT id, frequency FROM patterns
                   WHERE pattern_type = ? AND pattern_value = ?""",
                (pattern.pattern_type, pattern.pattern_value)
            )
            existing = await cursor.fetchone()

            if existing:
                # Update existing pattern
                pattern_id, current_freq = existing
                await db.execute(
                    """UPDATE patterns
                       SET frequency = ?, confidence = ?, last_seen_iteration = ?,
                           examples = ?, metadata = ?
                       WHERE id = ?""",
                    (
                        current_freq + 1,
                        pattern.confidence,
                        pattern.last_seen_iteration,
                        json.dumps(pattern.examples[-10:]),  # Keep last 10 examples
                        json.dumps(pattern.metadata),
                        pattern_id
                    )
                )
            else:
                # Insert new pattern
                cursor = await db.execute(
                    """INSERT INTO patterns
                       (pattern_type, pattern_value, frequency, confidence,
                        discovered_iteration, last_seen_iteration, examples, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        pattern.pattern_type,
                        pattern.pattern_value,
                        pattern.frequency,
                        pattern.confidence,
                        pattern.discovered_iteration,
                        pattern.last_seen_iteration,
                        json.dumps(pattern.examples),
                        json.dumps(pattern.metadata)
                    )
                )
                pattern_id = cursor.lastrowid

            await db.commit()

            # Update cache
            self._pattern_cache[pattern_id] = pattern

            return pattern_id

    async def query_patterns(
        self,
        pattern_type: Optional[str] = None,
        min_confidence: float = 0.0,
        min_frequency: int = 1
    ) -> List[Pattern]:
        """
        Query patterns from the library.

        Args:
            pattern_type: Filter by pattern type
            min_confidence: Minimum confidence threshold
            min_frequency: Minimum frequency threshold

        Returns:
            List of matching patterns
        """
        query = """
            SELECT id, pattern_type, pattern_value, frequency, confidence,
                   discovered_iteration, last_seen_iteration, examples, metadata
            FROM patterns
            WHERE confidence >= ? AND frequency >= ?
        """
        params = [min_confidence, min_frequency]

        if pattern_type:
            query += " AND pattern_type = ?"
            params.append(pattern_type)

        query += " ORDER BY confidence DESC, frequency DESC"

        patterns = []
        async with aiosqlite.connect(self.patterns_db) as db:
            async with db.execute(query, params) as cursor:
                async for row in cursor:
                    pattern = Pattern(
                        id=row[0],
                        pattern_type=row[1],
                        pattern_value=row[2],
                        frequency=row[3],
                        confidence=row[4],
                        discovered_iteration=row[5],
                        last_seen_iteration=row[6],
                        examples=json.loads(row[7]) if row[7] else [],
                        metadata=json.loads(row[8]) if row[8] else {}
                    )
                    patterns.append(pattern)

        return patterns

    async def update_strategy_performance(
        self,
        strategy_name: str,
        success: bool,
        confidence: float,
        execution_time: float,
        record_type: Optional[str] = None,
        iteration: Optional[int] = None
    ):
        """
        Update performance metrics for a strategy.

        Args:
            strategy_name: Name of the strategy
            success: Whether the match was successful
            confidence: Confidence score of the match
            execution_time: Time taken to execute
            record_type: Type of record this worked for
            iteration: Current iteration number
        """
        async with aiosqlite.connect(self.strategies_db) as db:
            # Get current stats
            cursor = await db.execute(
                """SELECT total_attempts, successful_matches, average_confidence,
                          average_execution_time, record_types, iteration_history
                   FROM strategy_performance
                   WHERE strategy_name = ?""",
                (strategy_name,)
            )
            row = await cursor.fetchone()

            if row:
                total, successes, avg_conf, avg_time, types_json, history_json = row
                record_types = json.loads(types_json) if types_json else []
                history = json.loads(history_json) if history_json else []

                # Update stats
                total += 1
                if success:
                    successes += 1
                avg_conf = (avg_conf * (total - 1) + confidence) / total
                avg_time = (avg_time * (total - 1) + execution_time) / total

                if record_type and record_type not in record_types:
                    record_types.append(record_type)

                if iteration:
                    history.append({
                        'iteration': iteration,
                        'success_rate': successes / total,
                        'avg_confidence': avg_conf
                    })

                await db.execute(
                    """UPDATE strategy_performance
                       SET total_attempts = ?, successful_matches = ?,
                           average_confidence = ?, average_execution_time = ?,
                           record_types = ?, iteration_history = ?,
                           updated_at = CURRENT_TIMESTAMP
                       WHERE strategy_name = ?""",
                    (
                        total, successes, avg_conf, avg_time,
                        json.dumps(record_types), json.dumps(history[-10:]),
                        strategy_name
                    )
                )
            else:
                # Insert new strategy
                await db.execute(
                    """INSERT INTO strategy_performance
                       (strategy_name, total_attempts, successful_matches,
                        average_confidence, average_execution_time, record_types)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        strategy_name, 1, 1 if success else 0,
                        confidence, execution_time,
                        json.dumps([record_type] if record_type else [])
                    )
                )

            await db.commit()

    async def add_validated_match(
        self,
        match_data: Dict[str, Any],
        validation_status: str,
        validator_confidence: float,
        iteration: int,
        feedback: Optional[str] = None
    ):
        """
        Store a validated match for future learning.

        Args:
            match_data: Original match data
            validation_status: "confirmed", "false_positive", "uncertain"
            validator_confidence: Validator's confidence in assessment
            iteration: Current iteration number
            feedback: Optional human or AI feedback
        """
        async with aiosqlite.connect(self.validated_db) as db:
            await db.execute(
                """INSERT INTO validated_matches
                   (old_id, old_name, new_id, new_name, match_strategy,
                    confidence_score, name_score, address_score,
                    validation_status, validator_confidence, iteration, feedback)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    match_data.get('old_id'),
                    match_data.get('old_name'),
                    match_data.get('new_id'),
                    match_data.get('new_name'),
                    match_data.get('match_strategy'),
                    match_data.get('confidence_score'),
                    match_data.get('name_score'),
                    match_data.get('address_score'),
                    validation_status,
                    validator_confidence,
                    iteration,
                    feedback
                )
            )
            await db.commit()

    async def get_validated_matches(
        self,
        validation_status: Optional[str] = None,
        min_confidence: float = 0.0
    ) -> pd.DataFrame:
        """
        Retrieve validated matches for learning.

        Args:
            validation_status: Filter by validation status
            min_confidence: Minimum validator confidence

        Returns:
            DataFrame of validated matches
        """
        query = """
            SELECT * FROM validated_matches
            WHERE validator_confidence >= ?
        """
        params = [min_confidence]

        if validation_status:
            query += " AND validation_status = ?"
            params.append(validation_status)

        query += " ORDER BY created_at DESC"

        async with aiosqlite.connect(self.validated_db) as db:
            df = pd.read_sql_query(query, db, params=params)

        return df

    async def save_checkpoint(self, context: LearningContext):
        """
        Save complete learning context as checkpoint.

        Args:
            context: Complete learning context to save
        """
        checkpoint_data = {
            'current_iteration': context.current_iteration,
            'total_matches': context.total_matches,
            'total_records': context.total_records,
            'patterns': [asdict(p) for p in context.patterns[:100]],  # Top 100
            'strategy_performances': {
                k: asdict(v) for k, v in context.strategy_performances.items()
            },
            'threshold_history': context.threshold_history,
            'convergence_metrics': context.convergence_metrics,
            'meta_insights': context.meta_insights,
            'validated_matches': context.validated_matches[:100],  # Recent 100
            'failed_attempts': context.failed_attempts[:50],  # Recent 50
            'timestamp': datetime.now().isoformat()
        }

        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

        logger.info(f"Checkpoint saved for iteration {context.current_iteration}")

    async def load_checkpoint(self) -> Optional[LearningContext]:
        """
        Load the most recent learning checkpoint.

        Returns:
            Learning context or None if no checkpoint exists
        """
        if not self.checkpoint_file.exists():
            logger.info("No checkpoint found, starting fresh")
            return None

        try:
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)

            # Reconstruct objects from dictionary data
            patterns = [Pattern(**p) for p in data.get('patterns', [])]

            strategy_performances = {}
            for name, perf_data in data.get('strategy_performances', {}).items():
                strategy_performances[name] = StrategyPerformance(**perf_data)

            context = LearningContext(
                current_iteration=data['current_iteration'],
                total_matches=data['total_matches'],
                total_records=data['total_records'],
                patterns=patterns,
                strategy_performances=strategy_performances,
                threshold_history=data.get('threshold_history', []),
                convergence_metrics=data.get('convergence_metrics', {}),
                meta_insights=data.get('meta_insights', []),
                validated_matches=data.get('validated_matches', []),
                failed_attempts=data.get('failed_attempts', []),
                timestamp=data.get('timestamp', '')
            )

            logger.info(f"Checkpoint loaded from iteration {context.current_iteration}")
            return context

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    async def get_best_strategies_for_record(
        self,
        record_features: Dict[str, Any]
    ) -> List[Tuple[str, float]]:
        """
        Get recommended strategies for a specific record based on historical performance.

        Args:
            record_features: Features of the record to match

        Returns:
            List of (strategy_name, expected_success_rate) tuples
        """
        strategies = []

        async with aiosqlite.connect(self.strategies_db) as db:
            cursor = await db.execute(
                """SELECT strategy_name, total_attempts, successful_matches,
                          average_confidence, record_types
                   FROM strategy_performance
                   WHERE total_attempts > 10
                   ORDER BY (successful_matches * 1.0 / total_attempts) DESC"""
            )

            async for row in cursor:
                strategy_name, total, successes, avg_conf, types_json = row
                success_rate = successes / total if total > 0 else 0

                # Adjust success rate based on record features
                # This is a simplified heuristic - could be ML model
                adjustment = 1.0

                if record_features.get('has_attention_line') and strategy_name == 'address_first':
                    adjustment *= 1.2  # Boost for attention line handling

                if record_features.get('name_length', 0) < 5 and strategy_name == 'fuzzy_name':
                    adjustment *= 0.8  # Penalty for short names

                adjusted_rate = min(success_rate * adjustment, 1.0)
                strategies.append((strategy_name, adjusted_rate))

        return strategies[:5]  # Return top 5

    async def analyze_convergence(
        self,
        recent_metrics: List[Dict[str, float]],
        window_size: int = 3
    ) -> Dict[str, Any]:
        """
        Analyze convergence patterns to determine optimization status.

        Args:
            recent_metrics: Recent iteration metrics
            window_size: Number of iterations to consider

        Returns:
            Convergence analysis including recommendations
        """
        if len(recent_metrics) < window_size:
            return {
                'status': 'insufficient_data',
                'recommendation': 'continue',
                'confidence': 0.5
            }

        recent = recent_metrics[-window_size:]
        improvements = [
            recent[i]['match_rate'] - recent[i-1]['match_rate']
            for i in range(1, len(recent))
        ]

        avg_improvement = sum(improvements) / len(improvements)
        is_diminishing = all(imp < 0.01 for imp in improvements)

        if is_diminishing:
            return {
                'status': 'converged',
                'recommendation': 'stop_optimization',
                'confidence': 0.9,
                'average_improvement': avg_improvement,
                'iterations_since_significant': window_size
            }
        elif avg_improvement > 0.02:
            return {
                'status': 'improving',
                'recommendation': 'continue',
                'confidence': 0.8,
                'average_improvement': avg_improvement
            }
        else:
            return {
                'status': 'plateau',
                'recommendation': 'pivot_strategy',
                'confidence': 0.7,
                'average_improvement': avg_improvement
            }

    async def cleanup_old_data(self, keep_iterations: int = 10):
        """
        Clean up old data to prevent database bloat.

        Args:
            keep_iterations: Number of recent iterations to keep
        """
        # Implementation would clean up old iteration data
        # while preserving important patterns and insights
        pass
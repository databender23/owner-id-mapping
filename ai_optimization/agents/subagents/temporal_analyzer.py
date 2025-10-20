"""
Temporal Analyzer Subagent

Specializes in identifying and handling temporal changes in owner names,
particularly focusing on the critical pattern where the core/prefix name
remains stable while appended information changes over time.
"""

import asyncio
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from collections import defaultdict

from ..context_manager import Pattern

logger = logging.getLogger(__name__)


@dataclass
class TemporalChange:
    """Represents a temporal change pattern in names."""
    old_name: str
    core_name: str
    prefix: str
    suffix: str
    separator: str
    change_type: str  # "attention", "estate", "care_of", "trustee", etc.
    confidence: float
    match_score_without_suffix: float
    match_score_with_suffix: float
    potential_new_name: Optional[str] = None
    potential_new_id: Optional[str] = None


class TemporalAnalyzerAgent:
    """
    Analyzes temporal changes in owner names to improve matching.

    Key Insight: Owner names frequently change over time, particularly in the
    portion AFTER separators like "Attn:", "Estate of", "c/o", etc.

    Core Principle: The prefix/core name (before separators) is stable.
    The appended information (after separators) changes frequently due to:
    - Contact person changes
    - Estate transitions
    - Trustee changes
    - Administrative updates
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize temporal analyzer.

        Args:
            config: Agent configuration
        """
        self.config = config
        self.separators = config.get('separators', [
            ';', 'Attn:', 'ATTN:', 'c/o', 'C/O', '%',
            'Estate of', 'ESTATE OF', 'Estate Of'
        ])
        self.core_match_threshold = config.get('core_match_threshold', 85)
        self.temporal_indicators = config.get('temporal_indicators', [
            'estate', 'deceased', 'successor', 'trustee',
            'executor', 'administrator', 'heir'
        ])

        # Pattern storage
        self.temporal_patterns = []
        self.core_name_mappings = {}

    async def execute(self, task_type: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute temporal analysis task.

        Args:
            task_type: Type of task to execute
            input_data: Input data for the task

        Returns:
            Temporal patterns and recommendations
        """
        if task_type == 'analyze_temporal_changes':
            return await self.analyze_temporal_changes(
                input_data['unmatched'],
                input_data.get('new_owners_df'),
                input_data.get('iteration', 0)
            )
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    async def analyze_temporal_changes(
        self,
        unmatched_df: pd.DataFrame,
        new_owners_df: Optional[pd.DataFrame] = None,
        iteration: int = 0
    ) -> Dict[str, Any]:
        """
        Analyze unmatched records for temporal name changes.

        Args:
            unmatched_df: DataFrame of unmatched records
            new_owners_df: Optional DataFrame of new owners for matching
            iteration: Current iteration number

        Returns:
            Dictionary containing temporal patterns and recommendations
        """
        logger.info(f"Analyzing {len(unmatched_df)} records for temporal changes")

        # Extract core names from all unmatched records
        temporal_changes = []
        for _, row in unmatched_df.iterrows():
            change = self._extract_temporal_change(row)
            if change:
                temporal_changes.append(change)

        logger.info(f"Found {len(temporal_changes)} potential temporal changes")

        # If new owners provided, try to match core names
        if new_owners_df is not None and len(temporal_changes) > 0:
            matched_changes = await self._match_core_names(
                temporal_changes,
                new_owners_df
            )
        else:
            matched_changes = []

        # Analyze patterns in temporal changes
        patterns = self._analyze_temporal_patterns(temporal_changes)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            temporal_changes,
            matched_changes,
            patterns
        )

        # Convert patterns to Pattern objects
        pattern_objects = []
        for pattern_info in patterns:
            pattern = Pattern(
                pattern_type='temporal_change',
                pattern_value=pattern_info['pattern'],
                frequency=pattern_info['frequency'],
                confidence=pattern_info['confidence'],
                discovered_iteration=iteration,
                last_seen_iteration=iteration,
                examples=pattern_info['examples'][:5],
                metadata={
                    'separator': pattern_info.get('separator'),
                    'change_type': pattern_info.get('change_type'),
                    'avg_core_match_improvement': pattern_info.get('avg_improvement')
                }
            )
            pattern_objects.append(pattern)

        return {
            'temporal_patterns': pattern_objects,
            'temporal_change_count': len(temporal_changes),
            'matched_core_names': len(matched_changes),
            'patterns': patterns,
            'recommendations': recommendations,
            'high_confidence_matches': self._get_high_confidence_matches(matched_changes),
            'statistics': self._calculate_statistics(temporal_changes, matched_changes)
        }

    def _extract_temporal_change(self, row: pd.Series) -> Optional[TemporalChange]:
        """
        Extract temporal change information from a record.

        Args:
            row: Record row

        Returns:
            TemporalChange object or None if no temporal pattern found
        """
        # Determine name column - use Snowflake column names
        name = str(row.get('old_owner_name', row.get('EXCLUDE_OWNER_NAME', '')))
        clean_name = str(row.get('clean_name', ''))

        if not name or len(name) < 3:
            return None

        # Try each separator
        for separator in self.separators:
            if separator in name:
                # Split on separator
                parts = name.split(separator, 1)
                if len(parts) == 2:
                    prefix = parts[0].strip()
                    suffix = parts[1].strip()

                    # Clean the prefix for core name
                    core_name = self._clean_core_name(prefix)

                    # Determine change type
                    change_type = self._determine_change_type(suffix, separator)

                    return TemporalChange(
                        old_name=name,
                        core_name=core_name,
                        prefix=prefix,
                        suffix=suffix,
                        separator=separator,
                        change_type=change_type,
                        confidence=0.0,  # Will be calculated during matching
                        match_score_without_suffix=0.0,
                        match_score_with_suffix=0.0
                    )

        # Check for prefix patterns (e.g., "Estate of X")
        for indicator in ['Estate of', 'ESTATE OF', 'Estate Of']:
            if name.startswith(indicator):
                core_name = name[len(indicator):].strip()
                return TemporalChange(
                    old_name=name,
                    core_name=self._clean_core_name(core_name),
                    prefix=indicator,
                    suffix='',
                    separator='',
                    change_type='estate_prefix',
                    confidence=0.0,
                    match_score_without_suffix=0.0,
                    match_score_with_suffix=0.0
                )

        return None

    def _clean_core_name(self, name: str) -> str:
        """
        Clean and normalize core name.

        Args:
            name: Raw core name

        Returns:
            Cleaned core name
        """
        # Remove extra whitespace
        name = ' '.join(name.split())

        # Remove trailing punctuation
        name = name.rstrip('.,;:')

        # Lowercase for matching
        name = name.lower()

        # Remove common noise words at the end
        noise_words = ['the', 'a', 'an', 'and', 'or']
        words = name.split()
        while words and words[-1] in noise_words:
            words.pop()
        name = ' '.join(words)

        return name

    def _determine_change_type(self, suffix: str, separator: str) -> str:
        """
        Determine the type of temporal change.

        Args:
            suffix: Text after separator
            separator: The separator used

        Returns:
            Change type classification
        """
        suffix_lower = suffix.lower()
        separator_lower = separator.lower()

        if 'attn' in separator_lower or 'attention' in separator_lower:
            return 'attention_change'
        elif 'c/o' in separator_lower or 'care of' in separator_lower:
            return 'care_of_change'
        elif 'estate' in suffix_lower:
            return 'estate_transition'
        elif any(indicator in suffix_lower for indicator in self.temporal_indicators):
            return 'trustee_change'
        elif separator == ';':
            return 'administrative_update'
        else:
            return 'other_temporal_change'

    async def _match_core_names(
        self,
        temporal_changes: List[TemporalChange],
        new_owners_df: pd.DataFrame
    ) -> List[TemporalChange]:
        """
        Try to match core names against new owners.

        Args:
            temporal_changes: List of temporal changes
            new_owners_df: DataFrame of new owners

        Returns:
            List of temporal changes with potential matches
        """
        matched_changes = []

        # Build index of new owner names for faster matching
        new_owner_names = {}
        for _, row in new_owners_df.iterrows():
            clean_name = str(row.get('clean_name', ''))
            new_id = str(row.get('NEW_OWNER_ID', ''))
            full_name = str(row.get('EXCLUDE_OWNER_NAME', ''))

            if clean_name:
                new_owner_names[clean_name] = {
                    'id': new_id,
                    'full_name': full_name
                }

        # Try to match each core name
        for change in temporal_changes:
            core_name = change.core_name

            # Find best match for core name
            best_match = None
            best_score = 0

            for new_name, info in new_owner_names.items():
                # Check if new name contains core name
                if core_name in new_name:
                    score = 100
                else:
                    # Fuzzy match
                    score = fuzz.ratio(core_name, new_name)

                    # Also try matching against the beginning of the new name
                    if len(core_name) > 5:
                        prefix_score = fuzz.partial_ratio(core_name, new_name)
                        score = max(score, prefix_score)

                if score > best_score and score >= self.core_match_threshold:
                    best_score = score
                    best_match = info

            if best_match:
                # Calculate scores with and without suffix
                change.match_score_without_suffix = best_score
                change.match_score_with_suffix = fuzz.ratio(
                    change.old_name.lower(),
                    best_match['full_name'].lower()
                )
                change.confidence = best_score / 100.0
                change.potential_new_name = best_match['full_name']
                change.potential_new_id = best_match['id']

                matched_changes.append(change)

                logger.debug(
                    f"Core match: '{change.core_name}' -> '{best_match['full_name']}' "
                    f"(core score: {best_score}, full score: {change.match_score_with_suffix})"
                )

        return matched_changes

    def _analyze_temporal_patterns(
        self,
        temporal_changes: List[TemporalChange]
    ) -> List[Dict[str, Any]]:
        """
        Analyze patterns in temporal changes.

        Args:
            temporal_changes: List of temporal changes

        Returns:
            List of pattern dictionaries
        """
        patterns = []

        # Group by change type
        change_type_groups = defaultdict(list)
        for change in temporal_changes:
            change_type_groups[change.change_type].append(change)

        # Analyze each change type
        for change_type, changes in change_type_groups.items():
            if len(changes) >= 2:  # Minimum frequency
                # Calculate average score improvement
                improvements = []
                for change in changes:
                    if change.match_score_without_suffix > 0:
                        improvement = (
                            change.match_score_without_suffix -
                            change.match_score_with_suffix
                        )
                        improvements.append(improvement)

                avg_improvement = np.mean(improvements) if improvements else 0

                patterns.append({
                    'pattern': f"{change_type}_pattern",
                    'change_type': change_type,
                    'frequency': len(changes),
                    'confidence': min(len(changes) / len(temporal_changes), 1.0),
                    'avg_improvement': avg_improvement,
                    'examples': [
                        {
                            'old_name': c.old_name,
                            'core_name': c.core_name,
                            'separator': c.separator
                        }
                        for c in changes[:5]
                    ]
                })

        # Group by separator
        separator_groups = defaultdict(list)
        for change in temporal_changes:
            if change.separator:
                separator_groups[change.separator].append(change)

        # Analyze separator patterns
        for separator, changes in separator_groups.items():
            if len(changes) >= 3:
                patterns.append({
                    'pattern': f"separator_{separator}_pattern",
                    'separator': separator,
                    'change_type': 'separator_based',  # Add default change_type
                    'frequency': len(changes),
                    'confidence': 0.9,
                    'examples': [
                        {
                            'old_name': c.old_name,
                            'core_name': c.core_name
                        }
                        for c in changes[:3]
                    ]
                })

        # Sort by frequency
        patterns.sort(key=lambda x: x['frequency'], reverse=True)

        return patterns

    def _generate_recommendations(
        self,
        temporal_changes: List[TemporalChange],
        matched_changes: List[TemporalChange],
        patterns: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Generate actionable recommendations based on temporal analysis.

        Args:
            temporal_changes: All temporal changes found
            matched_changes: Successfully matched changes
            patterns: Discovered patterns

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Calculate potential improvement
        if temporal_changes:
            potential_matches = len([
                c for c in temporal_changes
                if c.match_score_without_suffix >= self.core_match_threshold
            ])

            if potential_matches > 10:
                recommendations.append(
                    f"Implement core name extraction: Could recover {potential_matches} "
                    f"matches ({potential_matches / len(temporal_changes) * 100:.1f}% "
                    "of temporal changes)"
                )

        # Check for dominant patterns
        if patterns and patterns[0]['frequency'] > 20:
            dominant = patterns[0]
            recommendations.append(
                f"Focus on {dominant['change_type']}: {dominant['frequency']} occurrences found"
            )

        # Check score improvements
        high_improvements = [
            c for c in matched_changes
            if c.match_score_without_suffix - c.match_score_with_suffix > 30
        ]
        if len(high_improvements) > 5:
            avg_improvement = np.mean([
                c.match_score_without_suffix - c.match_score_with_suffix
                for c in high_improvements
            ])
            recommendations.append(
                f"Core name matching shows {avg_improvement:.1f}% average improvement "
                "over full name matching for temporal changes"
            )

        # Separator-specific recommendations
        attention_changes = len([
            c for c in temporal_changes
            if c.change_type == 'attention_change'
        ])
        if attention_changes > 10:
            recommendations.append(
                f"Add special handling for attention lines: {attention_changes} found"
            )

        estate_changes = len([
            c for c in temporal_changes
            if c.change_type in ['estate_transition', 'estate_prefix']
        ])
        if estate_changes > 10:
            recommendations.append(
                f"Implement estate transition detection: {estate_changes} estate-related changes"
            )

        # Threshold recommendation
        if matched_changes:
            scores = [c.match_score_without_suffix for c in matched_changes]
            if scores:
                percentile_80 = np.percentile(scores, 20)  # Lower 20%
                if percentile_80 < self.core_match_threshold:
                    recommendations.append(
                        f"Consider lowering core match threshold from {self.core_match_threshold} "
                        f"to {int(percentile_80)} to capture more temporal matches"
                    )

        return recommendations

    def _get_high_confidence_matches(
        self,
        matched_changes: List[TemporalChange]
    ) -> List[Dict[str, Any]]:
        """
        Get high confidence temporal matches for review.

        Args:
            matched_changes: List of matched temporal changes

        Returns:
            List of high confidence match dictionaries
        """
        high_confidence = []

        for change in matched_changes:
            if change.confidence >= 0.9:
                high_confidence.append({
                    'old_name': change.old_name,
                    'core_name': change.core_name,
                    'new_name': change.potential_new_name,
                    'new_id': change.potential_new_id,
                    'change_type': change.change_type,
                    'confidence': change.confidence,
                    'core_score': change.match_score_without_suffix,
                    'full_score': change.match_score_with_suffix,
                    'improvement': change.match_score_without_suffix - change.match_score_with_suffix
                })

        # Sort by confidence
        high_confidence.sort(key=lambda x: x['confidence'], reverse=True)

        return high_confidence[:20]  # Top 20

    def _calculate_statistics(
        self,
        temporal_changes: List[TemporalChange],
        matched_changes: List[TemporalChange]
    ) -> Dict[str, Any]:
        """
        Calculate statistics about temporal changes.

        Args:
            temporal_changes: All temporal changes
            matched_changes: Successfully matched changes

        Returns:
            Statistics dictionary
        """
        stats = {
            'total_temporal_changes': len(temporal_changes),
            'matched_core_names': len(matched_changes),
            'match_rate': len(matched_changes) / len(temporal_changes) if temporal_changes else 0,
            'change_type_distribution': {},
            'separator_distribution': {},
            'score_improvements': {}
        }

        # Change type distribution
        change_types = defaultdict(int)
        for change in temporal_changes:
            change_types[change.change_type] += 1
        stats['change_type_distribution'] = dict(change_types)

        # Separator distribution
        separators = defaultdict(int)
        for change in temporal_changes:
            if change.separator:
                separators[change.separator] += 1
        stats['separator_distribution'] = dict(separators)

        # Score improvements for matched changes
        if matched_changes:
            improvements = [
                c.match_score_without_suffix - c.match_score_with_suffix
                for c in matched_changes
            ]
            stats['score_improvements'] = {
                'mean': float(np.mean(improvements)),
                'median': float(np.median(improvements)),
                'max': float(np.max(improvements)),
                'min': float(np.min(improvements))
            }

        return stats
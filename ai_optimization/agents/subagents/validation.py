"""
Validation Subagent

This agent validates matching results to identify false positives, ensure match quality,
and apply domain-specific business rules for oil & gas ownership records.
"""

import asyncio
import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd
import numpy as np
from rapidfuzz import fuzz
import anthropic

from ..context_manager import Pattern

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validating a single match."""
    old_owner_id: str
    new_owner_id: str
    original_confidence: float
    validation_status: str  # 'approved', 'rejected', 'needs_review'
    adjusted_confidence: float
    validation_reasons: List[str]
    risk_factors: List[str]
    suggested_action: str


@dataclass
class ValidationSummary:
    """Summary of validation results for a batch."""
    total_validated: int
    approved_count: int
    rejected_count: int
    needs_review_count: int
    false_positive_rate: float
    confidence_adjustments: Dict[str, int]
    common_issues: List[Tuple[str, int]]
    recommendations: List[str]


class ValidationAgent:
    """
    Validates matching results to ensure quality and prevent false positives.

    Key responsibilities:
    - Review matches for false positives
    - Apply domain-specific validation rules
    - Adjust confidence scores based on validation
    - Flag suspicious matches for human review
    - Learn from validation patterns to improve future matching
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize validation agent.

        Args:
            config: Agent configuration from main config
        """
        self.config = config
        self.batch_size = config.get('batch_size', 20)
        self.confidence_threshold = config.get('confidence_threshold', 70)
        self.strict_mode = config.get('strict_mode', False)

        # Validation thresholds
        self.name_similarity_threshold = config.get('name_similarity_threshold', 60)
        self.address_similarity_threshold = config.get('address_similarity_threshold', 70)
        self.high_confidence_threshold = config.get('high_confidence_threshold', 90)
        self.low_confidence_threshold = config.get('low_confidence_threshold', 60)

        # Initialize Claude client if AI validation is enabled
        self.use_ai = config.get('use_ai', False)
        if self.use_ai:
            try:
                self.claude_client = anthropic.Anthropic()
            except Exception as e:
                logger.warning(f"Could not initialize Claude client: {e}. Falling back to rule-based validation.")
                self.use_ai = False

        # Domain-specific patterns for oil & gas
        self.trust_keywords = self._load_trust_keywords()
        self.estate_patterns = self._load_estate_patterns()
        self.known_false_positive_patterns = self._load_false_positive_patterns()

        # Validation statistics
        self.validation_stats = {
            'total_validated': 0,
            'approved': 0,
            'rejected': 0,
            'needs_review': 0,
            'false_positives': 0
        }

    def _load_trust_keywords(self) -> List[str]:
        """Load trust-related keywords for validation."""
        return [
            'trust', 'trustee', 'estate', 'living trust', 'family trust',
            'revocable trust', 'irrevocable trust', 'testamentary trust',
            'grantor trust', 'beneficiary', 'successor trustee'
        ]

    def _load_estate_patterns(self) -> List[re.Pattern]:
        """Load estate-related patterns for validation."""
        return [
            re.compile(r'estate\s+of\s+(\w+)', re.IGNORECASE),
            re.compile(r'(\w+)\s+estate', re.IGNORECASE),
            re.compile(r'deceased', re.IGNORECASE),
            re.compile(r'heir[s]?\s+of', re.IGNORECASE),
            re.compile(r'successor[s]?\s+in\s+interest', re.IGNORECASE)
        ]

    def _load_false_positive_patterns(self) -> List[Dict[str, Any]]:
        """Load known false positive patterns."""
        return [
            {
                'pattern': 'generic_company_names',
                'keywords': ['llc', 'inc', 'corp', 'company', 'partnership'],
                'risk': 'high'
            },
            {
                'pattern': 'single_word_match',
                'description': 'Matches based on single common word',
                'risk': 'high'
            },
            {
                'pattern': 'state_only_address',
                'description': 'Address matches only on state level',
                'risk': 'medium'
            },
            {
                'pattern': 'number_only_match',
                'description': 'Matches based primarily on numbers',
                'risk': 'high'
            }
        ]

    async def execute(self, task_type: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute validation task.

        Args:
            task_type: Type of task to execute
            input_data: Input data for the task

        Returns:
            Validation results and recommendations
        """
        if task_type == 'validate_matches':
            return await self.validate_matches(
                input_data['matches_df'],
                input_data.get('old_owners_df'),
                input_data.get('new_owners_df'),
                input_data.get('iteration', 0)
            )
        elif task_type == 'validate_single':
            return await self.validate_single_match(
                input_data['match_record'],
                input_data.get('old_record'),
                input_data.get('new_record')
            )
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    async def validate_matches(
        self,
        matches_df: pd.DataFrame,
        old_owners_df: Optional[pd.DataFrame],
        new_owners_df: Optional[pd.DataFrame],
        iteration: int
    ) -> Dict[str, Any]:
        """
        Validate a batch of matches.

        Args:
            matches_df: DataFrame with match results
            old_owners_df: Original old owners data
            new_owners_df: Original new owners data
            iteration: Current iteration number

        Returns:
            Validation results and insights
        """
        logger.info(f"Starting validation of {len(matches_df)} matches")

        # Filter to matched records only
        matched_df = matches_df[matches_df['mapped_new_id'].notna()].copy()

        if len(matched_df) == 0:
            logger.warning("No matches to validate")
            return {
                'validation_results': [],
                'summary': ValidationSummary(
                    total_validated=0, approved_count=0, rejected_count=0,
                    needs_review_count=0, false_positive_rate=0.0,
                    confidence_adjustments={}, common_issues=[],
                    recommendations=["No matches found to validate"]
                ),
                'insights': {}
            }

        validation_results = []

        # Process matches in batches
        for i in range(0, len(matched_df), self.batch_size):
            batch = matched_df.iloc[i:i+self.batch_size]

            # Validate each match in the batch
            batch_results = await asyncio.gather(*[
                self._validate_single_match_row(row, old_owners_df, new_owners_df)
                for _, row in batch.iterrows()
            ])

            validation_results.extend(batch_results)

        # Generate summary
        summary = self._generate_validation_summary(validation_results)

        # Extract insights
        insights = await self._extract_validation_insights(
            validation_results,
            matches_df,
            iteration
        )

        # Update statistics
        self._update_statistics(validation_results)

        return {
            'validation_results': validation_results,
            'summary': summary,
            'insights': insights,
            'stats': self.validation_stats
        }

    async def _validate_single_match_row(
        self,
        match_row: pd.Series,
        old_owners_df: Optional[pd.DataFrame],
        new_owners_df: Optional[pd.DataFrame]
    ) -> ValidationResult:
        """
        Validate a single match from a DataFrame row.

        Args:
            match_row: Series containing match information
            old_owners_df: Original old owners data
            new_owners_df: Original new owners data

        Returns:
            Validation result
        """
        # Extract relevant fields
        old_owner_id = match_row.get('old_owner_id', '')
        new_owner_id = match_row.get('mapped_new_id', '')
        confidence_score = match_row.get('confidence_score', 0)
        match_step = match_row.get('match_step', '')
        name_score = match_row.get('name_score', 0)
        address_score = match_row.get('address_score', 0)

        # Get original records if available
        old_record = None
        new_record = None
        if old_owners_df is not None and old_owner_id:
            old_records = old_owners_df[old_owners_df['OLD_OWNER_ID'] == old_owner_id]
            if not old_records.empty:
                old_record = old_records.iloc[0]

        if new_owners_df is not None and new_owner_id:
            new_records = new_owners_df[new_owners_df['NEW_OWNER_ID'] == new_owner_id]
            if not new_records.empty:
                new_record = new_records.iloc[0]

        # Perform validation
        return await self.validate_single_match({
            'old_owner_id': old_owner_id,
            'new_owner_id': new_owner_id,
            'confidence_score': confidence_score,
            'match_step': match_step,
            'name_score': name_score,
            'address_score': address_score
        }, old_record, new_record)

    async def validate_single_match(
        self,
        match_record: Dict[str, Any],
        old_record: Optional[pd.Series],
        new_record: Optional[pd.Series]
    ) -> ValidationResult:
        """
        Validate a single match.

        Args:
            match_record: Match information
            old_record: Original old owner record
            new_record: Original new owner record

        Returns:
            Validation result
        """
        validation_reasons = []
        risk_factors = []

        # Extract match information
        old_owner_id = match_record.get('old_owner_id', '')
        new_owner_id = match_record.get('new_owner_id', '')
        original_confidence = match_record.get('confidence_score', 0)
        match_step = match_record.get('match_step', '')
        name_score = match_record.get('name_score', 0)
        address_score = match_record.get('address_score', 0)

        # Start with original confidence
        adjusted_confidence = original_confidence

        # Rule 1: Check if it's a direct ID match (highest confidence)
        if match_step == 'DIRECT_ID_MATCH':
            validation_reasons.append("Direct ID match - highest confidence")
            validation_status = 'approved'
            adjusted_confidence = 100

        # Rule 2: Check name similarity
        elif name_score < self.name_similarity_threshold:
            validation_reasons.append(f"Name similarity too low: {name_score:.1f}%")
            risk_factors.append("low_name_similarity")
            adjusted_confidence *= 0.7

        # Rule 3: Check address similarity for address-based matches
        if 'ADDRESS' in match_step and address_score < self.address_similarity_threshold:
            validation_reasons.append(f"Address similarity too low for address-based match: {address_score:.1f}%")
            risk_factors.append("low_address_similarity")
            adjusted_confidence *= 0.6

        # Rule 4: Check for estate transitions
        if old_record is not None and new_record is not None:
            old_name = str(old_record.get('OWNER_NAME', '')).lower()
            new_name = str(new_record.get('OWNER_NAME', '')).lower()

            # Check for estate patterns
            old_is_estate = any(pattern.search(old_name) for pattern in self.estate_patterns)
            new_is_estate = any(pattern.search(new_name) for pattern in self.estate_patterns)

            if old_is_estate != new_is_estate:
                validation_reasons.append("Estate transition detected")
                risk_factors.append("estate_transition")
                # Estate transitions are common and valid
                adjusted_confidence *= 1.1  # Slight boost

        # Rule 5: Check for suspicious patterns
        if self._check_false_positive_patterns(old_record, new_record, match_record):
            validation_reasons.append("Potential false positive pattern detected")
            risk_factors.append("false_positive_pattern")
            adjusted_confidence *= 0.5

        # Rule 6: Check for temporal consistency
        if old_record is not None and new_record is not None:
            if 'ATTN' in str(old_record.get('OWNER_NAME', '')) or 'ATTN' in str(new_record.get('OWNER_NAME', '')):
                validation_reasons.append("Attention line change - common pattern")
                # This is expected and valid
                adjusted_confidence *= 1.05

        # Rule 7: Apply strict mode if enabled
        if self.strict_mode and adjusted_confidence < self.high_confidence_threshold:
            validation_reasons.append(f"Strict mode: confidence below {self.high_confidence_threshold}%")
            risk_factors.append("strict_mode_rejection")

        # Determine validation status
        if adjusted_confidence >= self.high_confidence_threshold and len(risk_factors) == 0:
            validation_status = 'approved'
            suggested_action = "Accept match"
        elif adjusted_confidence < self.low_confidence_threshold or len(risk_factors) >= 2:
            validation_status = 'rejected'
            suggested_action = "Reject match - investigate alternatives"
        else:
            validation_status = 'needs_review'
            suggested_action = "Manual review recommended"

        # Cap adjusted confidence at 100
        adjusted_confidence = min(adjusted_confidence, 100)

        return ValidationResult(
            old_owner_id=old_owner_id,
            new_owner_id=new_owner_id,
            original_confidence=original_confidence,
            validation_status=validation_status,
            adjusted_confidence=adjusted_confidence,
            validation_reasons=validation_reasons,
            risk_factors=risk_factors,
            suggested_action=suggested_action
        )

    def _check_false_positive_patterns(
        self,
        old_record: Optional[pd.Series],
        new_record: Optional[pd.Series],
        match_record: Dict[str, Any]
    ) -> bool:
        """
        Check if the match exhibits known false positive patterns.

        Args:
            old_record: Original old owner record
            new_record: Original new owner record
            match_record: Match information

        Returns:
            True if false positive patterns detected
        """
        if old_record is None or new_record is None:
            return False

        old_name = str(old_record.get('OWNER_NAME', '')).lower()
        new_name = str(new_record.get('OWNER_NAME', '')).lower()

        # Check for single word matches
        old_words = set(old_name.split())
        new_words = set(new_name.split())
        common_words = old_words.intersection(new_words)

        if len(common_words) == 1 and len(old_words) > 2 and len(new_words) > 2:
            # Only one word in common and both names have multiple words
            return True

        # Check for generic company name matches
        generic_terms = ['llc', 'inc', 'corp', 'company', 'partnership', 'lp', 'ltd']
        if any(term in old_name for term in generic_terms) and any(term in new_name for term in generic_terms):
            # Both are companies, check if names are too generic
            name_without_generic = ' '.join([w for w in common_words if w not in generic_terms])
            if len(name_without_generic) < 5:  # Very short common part
                return True

        # Check for number-only matches
        if match_record.get('match_step') == 'PARTIAL_NAME_MATCH':
            # Remove numbers and check what's left
            old_no_numbers = re.sub(r'\d+', '', old_name).strip()
            new_no_numbers = re.sub(r'\d+', '', new_name).strip()
            if fuzz.ratio(old_no_numbers, new_no_numbers) < 50:
                return True

        return False

    def _generate_validation_summary(self, validation_results: List[ValidationResult]) -> ValidationSummary:
        """
        Generate a summary of validation results.

        Args:
            validation_results: List of validation results

        Returns:
            Validation summary
        """
        if not validation_results:
            return ValidationSummary(
                total_validated=0, approved_count=0, rejected_count=0,
                needs_review_count=0, false_positive_rate=0.0,
                confidence_adjustments={}, common_issues=[], recommendations=[]
            )

        # Count statuses
        approved_count = sum(1 for r in validation_results if r.validation_status == 'approved')
        rejected_count = sum(1 for r in validation_results if r.validation_status == 'rejected')
        needs_review_count = sum(1 for r in validation_results if r.validation_status == 'needs_review')

        # Calculate false positive rate
        false_positive_rate = rejected_count / len(validation_results) if validation_results else 0

        # Analyze confidence adjustments
        confidence_adjustments = {
            'increased': sum(1 for r in validation_results if r.adjusted_confidence > r.original_confidence),
            'decreased': sum(1 for r in validation_results if r.adjusted_confidence < r.original_confidence),
            'unchanged': sum(1 for r in validation_results if r.adjusted_confidence == r.original_confidence)
        }

        # Identify common issues
        all_risk_factors = []
        for result in validation_results:
            all_risk_factors.extend(result.risk_factors)

        from collections import Counter
        risk_factor_counts = Counter(all_risk_factors)
        common_issues = risk_factor_counts.most_common(5)

        # Generate recommendations
        recommendations = []
        if false_positive_rate > 0.1:
            recommendations.append("High false positive rate detected - consider adjusting matching thresholds")
        if rejected_count > approved_count:
            recommendations.append("More matches rejected than approved - review matching strategy")
        if 'low_name_similarity' in risk_factor_counts and risk_factor_counts['low_name_similarity'] > len(validation_results) * 0.3:
            recommendations.append("Many low name similarity scores - consider improving name normalization")
        if 'estate_transition' in risk_factor_counts and risk_factor_counts['estate_transition'] > 5:
            recommendations.append("Multiple estate transitions detected - implement specialized estate matching logic")

        return ValidationSummary(
            total_validated=len(validation_results),
            approved_count=approved_count,
            rejected_count=rejected_count,
            needs_review_count=needs_review_count,
            false_positive_rate=false_positive_rate,
            confidence_adjustments=confidence_adjustments,
            common_issues=common_issues,
            recommendations=recommendations
        )

    async def _extract_validation_insights(
        self,
        validation_results: List[ValidationResult],
        matches_df: pd.DataFrame,
        iteration: int
    ) -> Dict[str, Any]:
        """
        Extract insights from validation results.

        Args:
            validation_results: List of validation results
            matches_df: Original matches DataFrame
            iteration: Current iteration number

        Returns:
            Dictionary of insights
        """
        insights = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'key_findings': [],
            'threshold_recommendations': {},
            'pattern_observations': [],
            'quality_metrics': {}
        }

        if not validation_results:
            return insights

        # Calculate quality metrics
        approved_results = [r for r in validation_results if r.validation_status == 'approved']
        if approved_results:
            avg_approved_confidence = np.mean([r.adjusted_confidence for r in approved_results])
            insights['quality_metrics']['avg_approved_confidence'] = avg_approved_confidence

        rejected_results = [r for r in validation_results if r.validation_status == 'rejected']
        if rejected_results:
            avg_rejected_confidence = np.mean([r.original_confidence for r in rejected_results])
            insights['quality_metrics']['avg_rejected_confidence'] = avg_rejected_confidence

            # Threshold recommendation
            if avg_rejected_confidence > 60:
                insights['threshold_recommendations']['confidence_threshold'] = {
                    'current': self.confidence_threshold,
                    'suggested': int(avg_rejected_confidence + 10),
                    'reason': 'Many high-confidence matches being rejected'
                }

        # Pattern observations
        review_needed = [r for r in validation_results if r.validation_status == 'needs_review']
        if len(review_needed) > len(validation_results) * 0.2:
            insights['pattern_observations'].append(
                f"{len(review_needed)} matches ({len(review_needed)/len(validation_results)*100:.1f}%) need manual review"
            )

        # Key findings
        if len(approved_results) > len(validation_results) * 0.8:
            insights['key_findings'].append("High match quality - over 80% of matches approved")
        elif len(rejected_results) > len(validation_results) * 0.3:
            insights['key_findings'].append("Significant false positive rate - review matching strategy")

        # Check for systematic issues
        all_reasons = []
        for result in validation_results:
            all_reasons.extend(result.validation_reasons)

        if 'Estate transition detected' in all_reasons:
            estate_count = all_reasons.count('Estate transition detected')
            insights['pattern_observations'].append(f"Found {estate_count} estate transitions")

        return insights

    def _update_statistics(self, validation_results: List[ValidationResult]) -> None:
        """
        Update internal statistics.

        Args:
            validation_results: List of validation results
        """
        for result in validation_results:
            self.validation_stats['total_validated'] += 1

            if result.validation_status == 'approved':
                self.validation_stats['approved'] += 1
            elif result.validation_status == 'rejected':
                self.validation_stats['rejected'] += 1
                self.validation_stats['false_positives'] += 1
            else:
                self.validation_stats['needs_review'] += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get current validation statistics."""
        stats = self.validation_stats.copy()
        if stats['total_validated'] > 0:
            stats['approval_rate'] = stats['approved'] / stats['total_validated']
            stats['rejection_rate'] = stats['rejected'] / stats['total_validated']
            stats['review_rate'] = stats['needs_review'] / stats['total_validated']
        return stats
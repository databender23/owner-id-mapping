"""
Matching strategy implementations for owner ID mapping.

This module implements a cascading matching strategy using multiple
approaches to maximize match accuracy while minimizing false positives.
"""
from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Tuple

import pandas as pd
from rapidfuzz import fuzz, process

from .config import (
    NAME_THRESHOLD,
    ADDRESS_THRESHOLD,
    ADDRESS_MIN,
    ADDRESS_FIRST_NAME_THRESHOLD
)
from .text_utils import extract_po_box_or_zip

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """
    Result of a matching attempt.

    Attributes:
        new_id: Matched new owner ID (None if no match)
        match_step: Name of the matching strategy that succeeded
        name_score: Name similarity score (0-100)
        address_score: Address similarity score (0-100)
        status: Match status description
        matched_on_attn: Whether attention name was used for matching
    """
    new_id: Optional[str]
    match_step: str
    name_score: float
    address_score: float
    status: str
    matched_on_attn: bool = False


class BaseMatchStrategy(ABC):
    """
    Abstract base class for matching strategies.

    Each strategy implements a specific matching approach and can be
    composed in a cascade for fallback behavior.
    """

    @abstractmethod
    def match(
        self,
        old_row: pd.Series,
        new_df: pd.DataFrame,
        new_subset: pd.DataFrame
    ) -> Optional[MatchResult]:
        """
        Attempt to match an old owner record to a new owner.

        Args:
            old_row: Row from old owners dataset with cleaned fields
            new_df: Full new owners dataset
            new_subset: State-filtered subset of new owners dataset

        Returns:
            MatchResult if match found, None otherwise
        """
        pass


# DirectIDMatcher removed - IDs are reindexed annually so matching on OLD_OWNER_ID is invalid


class ExactNameMatcher(BaseMatchStrategy):
    """Match on exact cleaned name."""

    def match(
        self,
        old_row: pd.Series,
        new_df: pd.DataFrame,
        new_subset: pd.DataFrame
    ) -> Optional[MatchResult]:
        """Exact string equality on cleaned names with city/state validation."""
        old_name = old_row['clean_name']
        old_original = str(old_row.get('EXCLUDE_OWNER_NAME', old_row.get('Owner Name', ''))).lower()
        old_city = str(old_row.get('EXCLUDE_CITY', '') or '').lower().strip()
        old_state = str(old_row.get('EXCLUDE_STATE', '') or '').lower().strip()

        new_names = new_subset['clean_name'].tolist()

        if old_name in new_names:
            match_idx = new_names.index(old_name)
            matched_row = new_subset.iloc[match_idx]

            # CRITICAL VALIDATION: Verify the original names are actually similar
            # This prevents false matches due to data quality issues where different
            # companies share the same NEW_OWNER_ID
            new_original = str(matched_row.get('PROD_OWNER_NAME', '')).lower()

            # Check if the original names are actually similar
            original_similarity = fuzz.token_set_ratio(old_original, new_original)
            if original_similarity < 70:  # Original names must be somewhat similar
                # This is likely a false positive due to data quality issues
                return None

            # STRICT city/state validation for exact name matches
            new_city = str(matched_row.get('PROD_CITY', '') or '').lower().strip()
            new_state = str(matched_row.get('PROD_STATE', '') or '').lower().strip()

            # States must match exactly
            if old_state and new_state:
                if old_state != new_state:
                    return None  # REJECT - different states

            # Cities must match or be very similar
            if old_city and new_city:
                city_similarity = fuzz.ratio(old_city, new_city)
                if city_similarity < 85:  # Cities must be very similar
                    return None  # REJECT - different cities

            addr_score = fuzz.partial_ratio(
                old_row['clean_address'],
                matched_row['clean_address']
            )

            return MatchResult(
                new_id=matched_row['NEW_OWNER_ID'],
                match_step='EXACT_NAME',
                name_score=100.0,
                address_score=addr_score,
                status=matched_row['ID_STATUS']
            )

        return None


class AddressFirstMatcher(BaseMatchStrategy):
    """
    Match primarily on address, then validate name.

    This is the most complex strategy, handling duplicate addresses
    and attention line matching.
    """

    def match(
        self,
        old_row: pd.Series,
        new_df: pd.DataFrame,
        new_subset: pd.DataFrame
    ) -> Optional[MatchResult]:
        """Address-first matching with name validation."""
        if len(new_subset) == 0:
            return None

        old_addr = old_row['clean_address']
        old_name = old_row['clean_name']
        old_city = str(old_row.get('EXCLUDE_CITY', '') or '').lower().strip()
        old_state = str(old_row.get('EXCLUDE_STATE', '') or '').lower().strip()
        # We don't use attention names for matching

        new_addrs = new_subset['clean_address'].tolist()

        if not new_addrs:
            return None

        # Find best address match
        best_addr_result = process.extractOne(
            old_addr,
            new_addrs,
            scorer=fuzz.token_set_ratio
        )

        if best_addr_result is None:
            return None

        best_addr, addr_score, addr_idx = best_addr_result

        if addr_score < ADDRESS_THRESHOLD:
            return None

        # Validate city and state match
        candidate_row = new_subset.iloc[addr_idx]
        candidate_city = str(candidate_row.get('PROD_CITY', '') or '').lower().strip()
        candidate_state = str(candidate_row.get('PROD_STATE', '') or '').lower().strip()

        # STRICT city and state validation - MUST match or be empty
        # States must match exactly
        if old_state and candidate_state:
            if old_state != candidate_state:
                return None  # REJECT - states don't match

        # Cities must match or be very similar
        if old_city and candidate_city:
            city_similarity = fuzz.ratio(old_city, candidate_city)
            if city_similarity < 85:  # Cities must be very similar
                return None  # REJECT - cities don't match

        # Handle duplicate addresses
        candidate_new_id = new_subset.iloc[addr_idx]['NEW_OWNER_ID']
        duplicates = new_subset[new_subset['NEW_OWNER_ID'] == candidate_new_id]

        if len(duplicates) > 1:
            # Find best address match among duplicates
            best_dup_score = 0
            best_dup_idx = 0

            for i, (dup_idx, dup_row) in enumerate(duplicates.iterrows()):
                dup_addr = dup_row['clean_address']
                dup_score = fuzz.token_set_ratio(old_addr, dup_addr)
                if dup_score > best_dup_score:
                    best_dup_score = dup_score
                    best_dup_idx = i

            best_dup_row = duplicates.iloc[best_dup_idx]
            addr_score = best_dup_score
            candidate_name = best_dup_row['clean_name']
            candidate_attn = best_dup_row['attn_name']
        else:
            candidate_name = new_subset.iloc[addr_idx]['clean_name']
            candidate_attn = new_subset.iloc[addr_idx]['attn_name']

        # Calculate name scores - ONLY use main name, NOT attention names
        main_name_score = fuzz.ratio(old_name, candidate_name)

        # Check if name score meets threshold
        if main_name_score <= ADDRESS_FIRST_NAME_THRESHOLD:
            return None

        # We ONLY match on the core business name, never on attention names
        match_step = 'ADDRESS_NAME_MATCH'
        status = "ID_CHANGED (ADDRESS+NAME)"

        return MatchResult(
            new_id=candidate_new_id,
            match_step=match_step,
            name_score=main_name_score,
            address_score=addr_score,
            status=status,
            matched_on_attn=False  # Always False - we don't match on ATTN
        )


class FuzzyNameMatcher(BaseMatchStrategy):
    """
    Fuzzy name matching with address validation.

    Uses token-based fuzzy matching for names that may have variations,
    with PO Box validation to prevent false positives.
    """

    def match(
        self,
        old_row: pd.Series,
        new_df: pd.DataFrame,
        new_subset: pd.DataFrame
    ) -> Optional[MatchResult]:
        """Fuzzy name matching with address validation."""
        old_name = old_row['clean_name']
        old_addr = old_row['clean_address']
        old_city = str(old_row.get('EXCLUDE_CITY', '') or '').lower().strip()
        old_state = str(old_row.get('EXCLUDE_STATE', '') or '').lower().strip()

        new_names = new_subset['clean_name'].tolist()

        if not new_names:
            return None

        # Try character-based ratio first
        best_result_char = process.extractOne(
            old_name,
            new_names,
            scorer=fuzz.ratio
        )

        if best_result_char is None:
            return None

        _, char_score, char_idx = best_result_char

        if char_score > 70:
            name_score = char_score
            name_idx = char_idx
        else:
            # Fallback to token-based matching
            best_result_token = process.extractOne(
                old_name,
                new_names,
                scorer=fuzz.token_sort_ratio
            )
            if best_result_token is not None:
                _, name_score, name_idx = best_result_token
            else:
                name_score = 0

        if name_score <= NAME_THRESHOLD:
            return None

        # STRICT city/state validation for fuzzy name matches
        matched_row = new_subset.iloc[name_idx]
        new_city = str(matched_row.get('PROD_CITY', '') or '').lower().strip()
        new_state = str(matched_row.get('PROD_STATE', '') or '').lower().strip()

        # CRITICAL: Verify original names are similar to avoid data quality false positives
        old_original = str(old_row.get('EXCLUDE_OWNER_NAME', old_row.get('Owner Name', ''))).lower()
        new_original = str(matched_row.get('PROD_OWNER_NAME', '')).lower()
        original_similarity = fuzz.token_set_ratio(old_original, new_original)
        if original_similarity < 50:  # Original names must have some similarity for fuzzy matches
            return None  # REJECT - likely a data quality issue

        # States must match exactly for fuzzy name matches
        if old_state and new_state:
            if old_state != new_state:
                return None  # REJECT - different states

        # Cities must match or be very similar for fuzzy name matches
        if old_city and new_city:
            city_similarity = fuzz.ratio(old_city, new_city)
            if city_similarity < 80:  # Cities must be similar
                return None  # REJECT - different cities

        # Validate PO Box / ZIP consistency
        pob_old = extract_po_box_or_zip(old_addr)
        pob_new = extract_po_box_or_zip(new_subset.iloc[name_idx]['clean_address'])

        if pob_old and pob_new:
            if fuzz.ratio(pob_old, pob_new) < 80:
                return None

        # Check address score
        addr_score = fuzz.partial_ratio(
            old_addr,
            new_subset.iloc[name_idx]['clean_address']
        )

        # Reject if address doesn't validate
        if addr_score < ADDRESS_MIN:
            return None

        # Calculate combined score for confidence
        final_score = (name_score * 0.6) + (addr_score * 0.4)

        if final_score > 85:
            match_step = 'CONFIRMED_FUZZY'
            name_score = 100.0  # Boost for high confidence
        else:
            match_step = 'FUZZY_NAME'

        # matched_row already defined above for city/state validation

        return MatchResult(
            new_id=matched_row['NEW_OWNER_ID'],
            match_step=match_step,
            name_score=name_score,
            address_score=addr_score,
            status=matched_row['ID_STATUS']
        )


class CrossStateMatcher(BaseMatchStrategy):
    """
    Cross-state fallback matching.

    Searches across all states for high-confidence name matches
    with address validation.
    """

    def match(
        self,
        old_row: pd.Series,
        new_df: pd.DataFrame,
        new_subset: pd.DataFrame
    ) -> Optional[MatchResult]:
        """Cross-state name matching."""
        old_name = old_row['clean_name']
        old_addr = old_row['clean_address']

        if len(old_name) == 0:
            return None

        all_names = new_df['clean_name'].tolist()

        best_cross_state = process.extractOne(
            old_name,
            all_names,
            scorer=fuzz.ratio
        )

        if best_cross_state is None:
            return None

        _, cross_name_score, cross_idx = best_cross_state

        if cross_name_score <= 75:  # Reduced from 90 for more matches
            return None

        candidate_row = new_df.iloc[cross_idx]
        candidate_addr = candidate_row['clean_address']
        candidate_state = candidate_row['EXCLUDE_STATE']

        cross_addr_score = fuzz.partial_ratio(old_addr, candidate_addr)

        if cross_addr_score <= 40:  # Reduced from 60 for more matches
            return None

        return MatchResult(
            new_id=candidate_row['NEW_OWNER_ID'],
            match_step='CROSS_STATE_REVIEW',
            name_score=cross_name_score,
            address_score=cross_addr_score,
            status=f"{candidate_row['ID_STATUS']} (CROSS_STATE:{candidate_state})"
        )


class EstateTransitionMatcher(BaseMatchStrategy):
    """
    Matches records with estate and trust transitions.

    Specialized handler for estate transitions, trust changes, and ownership transfers.
    Handles patterns like "Estate of John Smith" -> "John Smith Trust", etc.
    """

    def match(
        self,
        old_row: pd.Series,
        new_df: pd.DataFrame,
        new_subset: pd.DataFrame
    ) -> Optional[MatchResult]:
        """Match estate and trust transitions."""
        old_name = str(old_row.get('EXCLUDE_OWNER_NAME', old_row.get('Owner Name', '')))
        old_clean = old_row['clean_name']
        old_addr = old_row['clean_address']

        if len(old_name) < 3:
            return None

        # Estate/Trust patterns
        estate_patterns = [
            (r'^estate\s+of\s+(.+)', 'estate_prefix'),
            (r'^(.+?)\s+estate$', 'estate_suffix'),
            (r'^heir[s]?\s+of\s+(.+)', 'heir_prefix'),
            (r'^(.+?)\s+heir[s]?$', 'heir_suffix'),
            (r'^(.+?)\s+(?:revocable\s+)?trust$', 'trust_suffix'),
            (r'^(.+?)\s+living\s+trust$', 'living_trust'),
            (r'^(.+?)\s+family\s+trust$', 'family_trust'),
            (r'^the\s+(.+?)\s+trust$', 'the_trust'),
        ]

        core_name = None
        pattern_type = None

        # Try to extract core name from estate/trust patterns
        for pattern, ptype in estate_patterns:
            match = re.search(pattern, old_name.lower())
            if match:
                core_name = match.group(1).strip()
                pattern_type = ptype
                break

        if not core_name:
            # Check if name contains trust keywords even if not matching exact pattern
            trust_keywords = ['trust', 'estate', 'heir', 'beneficiary', 'successor']
            if not any(keyword in old_name.lower() for keyword in trust_keywords):
                return None
            # Use the full name as core
            core_name = old_name

        # Clean the core name
        from .text_utils import clean_text
        clean_core = clean_text(core_name)

        if len(clean_core) < 3:
            return None

        # Look for matches with related trust/estate names
        for idx, new_row in new_df.iterrows():
            new_name = str(new_row.get('PROD_OWNER_NAME', ''))
            new_clean = new_row['clean_name']

            # Check for various estate/trust relationships
            if (clean_core in new_clean or new_clean in clean_core or
                fuzz.ratio(clean_core, new_clean) >= 70):

                # Extra validation for trust/estate transitions
                if any(keyword in new_name.lower() for keyword in ['trust', 'estate', 'heir']):
                    # Validate with address
                    addr_score = fuzz.ratio(old_addr, new_row['clean_address'])
                    if addr_score >= 40:  # Very lenient for estate transitions
                        return MatchResult(
                            new_id=new_row['NEW_OWNER_ID'],
                            match_step='ESTATE_TRANSITION',
                            name_score=80.0,  # High confidence for estate transitions
                            address_score=addr_score,
                            status=f'ESTATE_TRANSITION ({pattern_type})',
                            matched_on_attn=False
                        )

        return None


class TemporalPatternMatcher(BaseMatchStrategy):
    """
    Matches records with temporal name changes.

    Handles cases where owner names have changed over time, especially after
    separators like ";", "Attn:", "Estate of", "c/o", etc.
    Based on AI-discovered patterns from 2075+ temporal changes.
    """

    def match(
        self,
        old_row: pd.Series,
        new_df: pd.DataFrame,
        new_subset: pd.DataFrame
    ) -> Optional[MatchResult]:
        """Match based on core name before temporal separators."""
        old_name = str(old_row.get('EXCLUDE_OWNER_NAME', old_row.get('Owner Name', '')))
        old_clean = old_row['clean_name']
        old_addr = old_row['clean_address']

        if len(old_name) < 3:
            return None

        # Temporal separators discovered by AI - prioritized list (comma is most common)
        separators = [',', ';', ' Attn:', 'ATTN:', 'c/o', 'C/O', '%', 'Estate of', 'ESTATE OF',
                     ' - ', ' aka ', ' AKA ', ' f/k/a ', ' n/k/a ', ' DBA ', ' d/b/a ', ' & ']

        # Extract core name (before separator)
        core_name = old_name
        separator_found = None
        for separator in separators:
            if separator in old_name:
                parts = old_name.split(separator, 1)
                if len(parts) > 0 and len(parts[0].strip()) > 3:
                    core_name = parts[0].strip()
                    separator_found = separator
                    break

        # Also check for "Estate of" at beginning
        if old_name.lower().startswith('estate of'):
            core_name = old_name[9:].strip()
            separator_found = 'Estate of'

        # If no separator found but name is long, try matching the first part
        if not separator_found:
            # For long names, try matching the first significant portion
            if len(old_name) > 25:
                words = old_name.split()
                if len(words) >= 3:
                    # Use first 2-3 words as core name
                    core_name = ' '.join(words[:min(3, len(words)-1)])
                    separator_found = 'long_name'
                else:
                    return None
            else:
                return None

        # Clean the core name for matching
        from .text_utils import clean_text
        clean_core = clean_text(core_name)

        if len(clean_core) < 3:
            return None

        # Now try to match the core name
        all_names_list = new_df['clean_name'].tolist()

        # First try exact core match
        for idx, new_row in new_df.iterrows():
            new_clean = new_row['clean_name']

            # Check if new name starts with our core name
            if new_clean.startswith(clean_core):
                # Validate with address - lower threshold for temporal matches
                addr_score = fuzz.ratio(old_addr, new_row['clean_address'])
                if addr_score >= max(40, ADDRESS_MIN - 10):  # More lenient for temporal
                    return MatchResult(
                        new_id=new_row['NEW_OWNER_ID'],
                        match_step='TEMPORAL_CORE_MATCH',
                        name_score=85.0,  # High confidence for temporal matches
                        address_score=addr_score,
                        status=f'TEMPORAL_MATCH ({separator_found} change)',
                        matched_on_attn=False
                    )

        # Try fuzzy matching on core name
        core_matches = process.extract(clean_core, all_names_list, scorer=fuzz.token_sort_ratio, limit=10)

        for match_name, name_score, idx in core_matches:
            if name_score >= 65:  # Lower threshold for core names since temporal changes are common
                # Find the corresponding row
                matched_df = new_df[new_df['clean_name'] == match_name]
                if matched_df.empty:
                    continue

                new_row = matched_df.iloc[0]

                # Validate with address - lower threshold for temporal matches
                addr_score = fuzz.ratio(old_addr, new_row['clean_address'])
                if addr_score >= max(40, ADDRESS_MIN - 10):  # More lenient for temporal
                    return MatchResult(
                        new_id=new_row['NEW_OWNER_ID'],
                        match_step='TEMPORAL_FUZZY_MATCH',
                        name_score=name_score,
                        address_score=addr_score,
                        status=f'TEMPORAL_MATCH ({separator_found} change)',
                        matched_on_attn=False
                    )

        return None


class InitialMatcher(BaseMatchStrategy):
    """
    Matches based on first/last initials and address.

    Useful for abbreviated names or records with only initials.
    """

    def match(
        self,
        old_row: pd.Series,
        new_df: pd.DataFrame,
        new_subset: pd.DataFrame
    ) -> Optional[MatchResult]:
        """Match based on initials and address."""
        old_name = old_row['clean_name']
        old_addr = old_row['clean_address']
        old_city = str(old_row.get('EXCLUDE_CITY', '') or '').lower().strip()
        old_state = str(old_row.get('EXCLUDE_STATE', '') or '').lower().strip()

        if len(old_name) < 2:
            return None

        # Extract initials
        words = old_name.split()
        if not words:
            return None

        # Get first and last initials
        first_initial = words[0][0] if words[0] else ''
        last_initial = words[-1][0] if len(words) > 1 and words[-1] else ''

        if not first_initial:
            return None

        # Look for matches with same initials
        for idx, new_row in new_df.iterrows():
            new_clean = new_row['clean_name']
            new_addr = new_row['clean_address']
            new_city = str(new_row.get('PROD_CITY', '') or '').lower().strip()
            new_state = str(new_row.get('PROD_STATE', '') or '').lower().strip()

            # Check city and state for initial matching
            if old_state and new_state and old_state != new_state:
                continue  # States must match for initial-based matching

            if old_city and new_city:
                city_similarity = fuzz.ratio(old_city, new_city)
                if city_similarity < 70:  # Cities should be similar
                    continue

            new_words = new_clean.split()
            if not new_words:
                continue

            new_first_initial = new_words[0][0] if new_words[0] else ''
            new_last_initial = new_words[-1][0] if len(new_words) > 1 and new_words[-1] else ''

            # Check if initials match
            if first_initial == new_first_initial:
                if not last_initial or last_initial == new_last_initial:
                    # Check address similarity
                    addr_score = fuzz.ratio(old_addr, new_addr)
                    if addr_score >= 60:  # Need good address match for initial matching
                        # Calculate name similarity
                        name_score = fuzz.ratio(old_name, new_clean)
                        if name_score >= 30:  # Low threshold since we're matching on initials
                            return MatchResult(
                                new_id=new_row['NEW_OWNER_ID'],
                                match_step='INITIAL_MATCH',
                                name_score=name_score,
                                address_score=addr_score,
                                status=f'INITIAL_MATCH ({first_initial}{last_initial})',
                                matched_on_attn=False
                            )

        return None


class AddressOnlyMatcher(BaseMatchStrategy):
    """
    Matches based primarily on address with minimal name validation.

    For cases where ownership has transferred but the property address is identical.
    """

    def match(
        self,
        old_row: pd.Series,
        new_df: pd.DataFrame,
        new_subset: pd.DataFrame
    ) -> Optional[MatchResult]:
        """Match based on address with minimal name check."""
        old_name = old_row['clean_name']
        old_addr = old_row['clean_address']
        old_city = str(old_row.get('EXCLUDE_CITY', '') or '').lower().strip()
        old_state = str(old_row.get('EXCLUDE_STATE', '') or '').lower().strip()

        if len(old_addr) < 5:  # Need meaningful address
            return None

        # Look for exact or near-exact address matches WITH city/state validation
        for idx, new_row in new_df.iterrows():
            new_addr = new_row['clean_address']
            new_name = new_row['clean_name']
            new_city = str(new_row.get('PROD_CITY', '') or '').lower().strip()
            new_state = str(new_row.get('PROD_STATE', '') or '').lower().strip()

            # CRITICAL: Check city and state FIRST to avoid false positives
            # States must match exactly (or be empty)
            if old_state and new_state and old_state != new_state:
                continue  # Skip if states don't match

            # Cities must be very similar (or empty)
            if old_city and new_city:
                city_similarity = fuzz.ratio(old_city, new_city)
                if city_similarity < 80:  # Cities must be very similar
                    continue  # Skip if cities don't match

            # Calculate address similarity
            addr_score = fuzz.ratio(old_addr, new_addr)

            # High address threshold for address-only matching
            if addr_score >= 85:
                # Very minimal name check - just check for any commonality
                name_score = fuzz.partial_ratio(old_name, new_name)
                if name_score >= 25:  # Extremely low threshold
                    return MatchResult(
                        new_id=new_row['NEW_OWNER_ID'],
                        match_step='ADDRESS_ONLY',
                        name_score=name_score,
                        address_score=addr_score,
                        status='ADDRESS_ONLY_MATCH',
                        matched_on_attn=False
                    )

        return None


class LastResortMatcher(BaseMatchStrategy):
    """
    Ultra-aggressive last resort matching.

    Tries to match on very minimal criteria - any significant word match
    combined with any address similarity. Use with caution.
    """

    def match(
        self,
        old_row: pd.Series,
        new_df: pd.DataFrame,
        new_subset: pd.DataFrame
    ) -> Optional[MatchResult]:
        """Last resort matching on minimal criteria."""
        old_name = old_row['clean_name']
        old_addr = old_row['clean_address']
        old_city = str(old_row.get('EXCLUDE_CITY', '') or '').lower().strip()
        old_state = str(old_row.get('EXCLUDE_STATE', '') or '').lower().strip()

        if len(old_name) < 3:
            return None

        # Extract significant words
        words = old_name.split()
        significant_words = [w for w in words if len(w) > 3]

        if not significant_words:
            return None

        # Try to match on any significant word
        for word in significant_words[:3]:  # Check first 3 significant words
            if len(word) < 4:
                continue

            for idx, new_row in new_df.iterrows():
                new_clean = new_row['clean_name']
                new_addr = new_row['clean_address']
                new_city = str(new_row.get('PROD_CITY', '') or '').lower().strip()
                new_state = str(new_row.get('PROD_STATE', '') or '').lower().strip()

                # For last resort, at least state should match (if present)
                if old_state and new_state and old_state != new_state:
                    continue

                # Check if the word appears in the new name
                if word in new_clean:
                    # Very minimal address validation
                    addr_score = fuzz.partial_ratio(old_addr, new_addr)
                    if addr_score >= 20:  # Extremely low threshold
                        # Calculate name similarity
                        name_score = fuzz.ratio(old_name, new_clean)
                        if name_score >= 35:  # Extremely low threshold
                            return MatchResult(
                                new_id=new_row['NEW_OWNER_ID'],
                                match_step='LAST_RESORT',
                                name_score=name_score,
                                address_score=addr_score,
                                status=f'LAST_RESORT (word: {word})',
                                matched_on_attn=False
                            )

        return None


class PartialNameMatcher(BaseMatchStrategy):
    """
    Partial name matching for substring/expansion cases.

    Handles cases like "Steko" vs "Steko Investments of Texas".
    """

    def match(
        self,
        old_row: pd.Series,
        new_df: pd.DataFrame,
        new_subset: pd.DataFrame
    ) -> Optional[MatchResult]:
        """Partial name matching."""
        old_name = old_row['clean_name']
        old_addr = old_row['clean_address']

        if len(old_name) <= 15:
            return None

        # Extract significant words
        words = old_name.split()
        skip_words = {'the', 'a', 'an', 'of', 'and', 'or', 'for', 'to', 'in', 'on'}
        significant_words = [w for w in words if w not in skip_words and len(w) > 3]

        if not significant_words:
            return None

        first_word = significant_words[0]
        all_names_list = new_df['clean_name'].tolist()
        matching_starts = [name for name in all_names_list if name.startswith(first_word)]

        if not matching_starts:
            return None

        best_partial = process.extractOne(
            old_name,
            matching_starts,
            scorer=fuzz.token_set_ratio
        )

        if best_partial is None:
            return None

        _, partial_score, _ = best_partial

        if partial_score <= 85:
            return None

        partial_idx = all_names_list.index(best_partial[0])
        candidate_row = new_df.iloc[partial_idx]
        candidate_addr = candidate_row['clean_address']

        partial_addr_score = fuzz.partial_ratio(old_addr, candidate_addr)

        if partial_addr_score <= 50:
            return None

        return MatchResult(
            new_id=candidate_row['NEW_OWNER_ID'],
            match_step='PARTIAL_NAME_REVIEW',
            name_score=partial_score,
            address_score=partial_addr_score,
            status=f"{candidate_row['ID_STATUS']} (PARTIAL_MATCH)"
        )


class OwnerMapper:
    """
    Orchestrates the cascading match strategy.

    Runs each matching strategy in sequence until a match is found,
    or returns an unmatched result with diagnostic information.
    """

    def __init__(self):
        """Initialize with ordered list of matching strategies."""
        self.strategies: List[BaseMatchStrategy] = [
            # DirectIDMatcher removed - IDs are reindexed annually
            ExactNameMatcher(),
            AddressFirstMatcher(),
            EstateTransitionMatcher(),  # Handle estate/trust transitions early
            FuzzyNameMatcher(),
            TemporalPatternMatcher(),  # Handle 2075+ temporal changes
            CrossStateMatcher(),
            AddressOnlyMatcher(),  # Match on address with minimal name check
            InitialMatcher(),  # Match on initials + address
            PartialNameMatcher(),
            LastResortMatcher()  # Ultra-aggressive last resort
        ]

    def match_owner(
        self,
        old_row: pd.Series,
        new_df: pd.DataFrame,
        new_subset: pd.DataFrame
    ) -> MatchResult:
        """
        Attempt to match an owner using cascading strategies.

        Args:
            old_row: Row from old owners dataset
            new_df: Full new owners dataset
            new_subset: State-filtered subset

        Returns:
            MatchResult with match details or unmatched status
        """
        # Try each strategy in order
        for strategy in self.strategies:
            result = strategy.match(old_row, new_df, new_subset)
            if result is not None:
                return result

        # No match found - generate diagnostic information
        return self._create_unmatched_result(old_row, new_df)

    def _create_unmatched_result(
        self,
        old_row: pd.Series,
        new_df: pd.DataFrame
    ) -> MatchResult:
        """Create result for unmatched owner with diagnostic info."""
        old_name = old_row['clean_name']

        if len(old_name) < 5:
            return MatchResult(
                new_id=None,
                match_step='UNMATCHED',
                name_score=0.0,
                address_score=0.0,
                status='UNMATCHED (NAME_TOO_SHORT)'
            )

        # Find best possible match for diagnostics
        all_names = new_df['clean_name'].tolist()
        best_possible = process.extractOne(old_name, all_names, scorer=fuzz.ratio)

        if best_possible:
            best_score = best_possible[1]
            status = f'UNMATCHED (BEST_SCORE:{best_score:.0f}%)'
        else:
            status = 'UNMATCHED (NO_SIMILAR_NAMES)'

        return MatchResult(
            new_id=None,
            match_step='UNMATCHED',
            name_score=0.0,
            address_score=0.0,
            status=status
        )

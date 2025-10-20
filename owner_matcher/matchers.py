"""
Matching strategy implementations for owner ID mapping.

This module implements a cascading matching strategy using multiple
approaches to maximize match accuracy while minimizing false positives.
"""
from __future__ import annotations

import logging
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


class DirectIDMatcher(BaseMatchStrategy):
    """Match on exact OLD_OWNER_ID field."""

    def match(
        self,
        old_row: pd.Series,
        new_df: pd.DataFrame,
        new_subset: pd.DataFrame
    ) -> Optional[MatchResult]:
        """Direct ID lookup."""
        old_id = old_row['Previous Owner ID']

        if pd.isna(old_id):
            return None

        direct_match = new_df[new_df['OLD_OWNER_ID'] == old_id]

        if not direct_match.empty:
            matched_row = direct_match.iloc[0]
            addr_score = fuzz.partial_ratio(
                old_row['clean_address'],
                matched_row['clean_address']
            )

            return MatchResult(
                new_id=matched_row['NEW_OWNER_ID'],
                match_step='DIRECT_ID_MATCH',
                name_score=100.0,
                address_score=addr_score,
                status=matched_row['ID_STATUS']
            )

        return None


class ExactNameMatcher(BaseMatchStrategy):
    """Match on exact cleaned name."""

    def match(
        self,
        old_row: pd.Series,
        new_df: pd.DataFrame,
        new_subset: pd.DataFrame
    ) -> Optional[MatchResult]:
        """Exact string equality on cleaned names."""
        old_name = old_row['clean_name']
        new_names = new_subset['clean_name'].tolist()

        if old_name in new_names:
            match_idx = new_names.index(old_name)
            matched_row = new_subset.iloc[match_idx]
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
        old_attn = old_row['attn_name']

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

        # Calculate name scores
        main_name_score = fuzz.ratio(old_name, candidate_name)

        attn_name_score = 0
        if old_attn and len(old_attn) > 5:
            attn_to_main = fuzz.ratio(old_attn, candidate_name)
            attn_to_attn = fuzz.ratio(old_attn, candidate_attn) if candidate_attn else 0
            attn_name_score = max(attn_to_main, attn_to_attn)

        best_name_score = max(main_name_score, attn_name_score)

        # Check if name score meets threshold
        if best_name_score <= ADDRESS_FIRST_NAME_THRESHOLD:
            return None

        # Determine if matched on attention line
        matched_on_attn = (
            attn_name_score > main_name_score and
            attn_name_score > ADDRESS_FIRST_NAME_THRESHOLD
        )

        if matched_on_attn:
            match_step = 'ADDRESS_ATTN_MATCH'
            status = "ID_CHANGED (ADDRESS+ATTN)"
        else:
            match_step = 'ADDRESS_NAME_MATCH'
            status = "ID_CHANGED (ADDRESS+NAME)"

        return MatchResult(
            new_id=candidate_new_id,
            match_step=match_step,
            name_score=best_name_score,
            address_score=addr_score,
            status=status,
            matched_on_attn=matched_on_attn
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
        old_attn = old_row['attn_name']

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
            # Exception: if there's an attention line, require higher address match
            if old_attn and len(old_attn) > 5:
                return None
            else:
                return None

        # Calculate combined score for confidence
        final_score = (name_score * 0.6) + (addr_score * 0.4)

        if final_score > 85:
            match_step = 'CONFIRMED_FUZZY'
            name_score = 100.0  # Boost for high confidence
        else:
            match_step = 'FUZZY_NAME'

        matched_row = new_subset.iloc[name_idx]

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

        if cross_name_score <= 90:
            return None

        candidate_row = new_df.iloc[cross_idx]
        candidate_addr = candidate_row['clean_address']
        candidate_state = candidate_row['EXCLUDE_STATE']

        cross_addr_score = fuzz.partial_ratio(old_addr, candidate_addr)

        if cross_addr_score <= 60:
            return None

        return MatchResult(
            new_id=candidate_row['NEW_OWNER_ID'],
            match_step='CROSS_STATE_REVIEW',
            name_score=cross_name_score,
            address_score=cross_addr_score,
            status=f"{candidate_row['ID_STATUS']} (CROSS_STATE:{candidate_state})"
        )


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
            DirectIDMatcher(),
            ExactNameMatcher(),
            AddressFirstMatcher(),
            FuzzyNameMatcher(),
            CrossStateMatcher(),
            PartialNameMatcher()
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

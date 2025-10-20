"""
Text cleaning and normalization utilities for owner name processing.

This module provides functions to clean, normalize, and extract information
from owner names and addresses, preparing them for fuzzy matching.
"""
from __future__ import annotations

import re
from typing import Optional

import pandas as pd

from .config import CLEAN_PUNCT, TRUST_KEYWORDS


def extract_attention_name(text: Optional[str]) -> str:
    """
    Extract attention line name from owner text.

    Searches for patterns like "ATTN:", "Attention:", "c/o", or semicolon-delimited
    attention lines and extracts the name portion.

    Args:
        text: Raw owner name text that may contain attention lines

    Returns:
        Cleaned attention name if found, empty string otherwise

    Examples:
        >>> extract_attention_name("SBI West Texas; Attn: Extex Operating")
        'extex operating'
        >>> extract_attention_name("John Smith Oil LLC")
        ''
        >>> extract_attention_name("c/o Jane Doe Properties")
        'jane doe properties'
    """
    if pd.isna(text):
        return ''

    text = str(text).strip()

    # Patterns to match attention lines
    attn_patterns = [
        r'(?:attn:|attention:|c/o|care of)[\s:]*([^;,\n]+)',  # ATTN: name
        r';[\s]*([^;,\n]+?)(?=\s*$)',  # ; name at end
    ]

    for pattern in attn_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            attn_name = match.group(1).strip()
            # Only return if it looks like a real name (more than 5 chars)
            if len(attn_name) > 5:
                return clean_text(attn_name)

    return ''


def clean_text(text: Optional[str]) -> str:
    """
    Clean and normalize text for matching.

    Performs comprehensive text cleaning including:
    - Lowercase conversion
    - Removal of attention lines and special characters
    - Stripping trust-related keywords
    - Punctuation removal
    - Whitespace normalization

    Args:
        text: Raw text to clean

    Returns:
        Cleaned and normalized text

    Examples:
        >>> clean_text("Smith Family Living Trust, LLC")
        'smith'
        >>> clean_text("  John   Doe;  Attn: Manager  ")
        'john doe'
        >>> clean_text("ABC Oil & Gas, Jr.")
        'abc oil gas'
    """
    if pd.isna(text):
        return ''

    text = str(text).lower().strip()

    # Remove everything after semicolon (attention lines)
    text = re.sub(r';.*$', '', text)

    # Remove common prefixes
    text = re.sub(r'^(?:c/o|\s*attn:\s*|%|\-)', '', text)

    # Remove leading ordinal numbers (like "1st", "2nd")
    text = re.sub(r'^\s*\d+[a-z]{2,}', '', text)

    # Remove trust-related keywords
    for keyword in TRUST_KEYWORDS:
        # Use word boundary to avoid partial matches
        text = re.sub(
            r'\b' + re.escape(keyword.lower()) + r'\b(?!\w)',
            '',
            text,
            flags=re.IGNORECASE
        )

    # Apply entity suffix normalization (before removing punctuation)
    text = normalize_entity_suffixes(text)

    # Remove punctuation
    text = re.sub(CLEAN_PUNCT, ' ', text)

    # Normalize whitespace
    text = text.strip()
    return ' '.join(text.split())


def normalize_entity_suffixes(text: str) -> str:
    """
    Normalize entity suffixes to standard forms based on AI-discovered patterns.

    This function standardizes common business entity suffixes to improve matching.
    Based on patterns discovered by AI optimization in Iteration 1.

    Args:
        text: Text potentially containing entity suffixes

    Returns:
        Text with normalized entity suffixes

    Examples:
        >>> normalize_entity_suffixes("Smith Oil Company")
        'Smith Oil co'
        >>> normalize_entity_suffixes("ABC Limited Liability Company")
        'ABC llc'
    """
    if not text:
        return text

    # Entity suffix mappings discovered by AI
    suffix_mappings = {
        # LLC variations
        r'\blimited liability company\b': 'llc',
        r'\bl\.l\.c\.\b': 'llc',
        r'\bl l c\b': 'llc',

        # Inc variations
        r'\bincorporated\b': 'inc',
        r'\binc\.\b': 'inc',

        # Corp variations
        r'\bcorporation\b': 'corp',
        r'\bcorp\.\b': 'corp',

        # Ltd variations
        r'\blimited\b': 'ltd',
        r'\bltd\.\b': 'ltd',

        # LP variations
        r'\blimited partnership\b': 'lp',
        r'\bl\.p\.\b': 'lp',
        r'\bl p\b': 'lp',

        # Company variations
        r'\bcompany\b': 'co',
        r'\bco\.\b': 'co',
    }

    result = text.lower()
    for pattern, replacement in suffix_mappings.items():
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    return result


def extract_po_box_or_zip(address: str) -> Optional[str]:
    """
    Extract PO Box number or ZIP code from address string.

    Used for validating address matches when both addresses contain
    PO Boxes or ZIP codes.

    Args:
        address: Address string to search

    Returns:
        PO Box number or ZIP code if found, None otherwise

    Examples:
        >>> extract_po_box_or_zip("PO BOX 5190, San Antonio, TX")
        '5190'
        >>> extract_po_box_or_zip("123 Main St, Austin, TX 78701")
        '78701'
    """
    address_upper = address.upper()
    match = re.search(r'PO BOX\s*(\d+)|(\d{5})', address_upper)
    if match:
        return match.group(1) or match.group(2)
    return None

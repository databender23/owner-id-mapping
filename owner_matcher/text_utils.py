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

    # Remove punctuation
    text = re.sub(CLEAN_PUNCT, ' ', text)

    # Normalize whitespace
    text = text.strip()
    return ' '.join(text.split())


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

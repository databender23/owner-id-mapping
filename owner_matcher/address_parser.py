"""
Address parsing and normalization utilities.

This module handles parsing of address strings into structured components
using the usaddress library and provides fallback parsing for non-standard formats.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import usaddress

logger = logging.getLogger(__name__)


@dataclass
class ParsedAddress:
    """
    Structured representation of a parsed address.

    Attributes:
        street: Street address including number, name, and unit
        city: City name
        state: State abbreviation
        zip: ZIP code (5 or 9 digits)
    """
    street: str
    city: str
    state: str
    zip: str


def parse_address(address_str: Optional[str]) -> ParsedAddress:
    """
    Parse an address string into structured components.

    Uses the usaddress library for standardized parsing, with a regex-based
    fallback for addresses that don't parse cleanly.

    Args:
        address_str: Raw address string to parse

    Returns:
        ParsedAddress object with structured address components

    Examples:
        >>> addr = parse_address("123 Main St, Austin, TX 78701")
        >>> addr.street
        '123 Main St'
        >>> addr.city
        'Austin'
        >>> addr.state
        'TX'
        >>> addr.zip
        '78701'
    """
    if pd.isna(address_str):
        return ParsedAddress(street='', city='', state='', zip='')

    try:
        # Attempt to parse using usaddress library
        parsed, address_type = usaddress.tag(str(address_str))

        # Extract street components
        street_parts = []
        street_keys = [
            'AddressNumber',
            'StreetNamePreDirectional',
            'StreetName',
            'StreetNamePostType',
            'StreetNamePostDirectional',
            'OccupancyType',
            'OccupancyIdentifier'
        ]

        for key in street_keys:
            if key in parsed:
                street_parts.append(parsed[key])

        return ParsedAddress(
            street=' '.join(street_parts).strip(),
            city=parsed.get('PlaceName', ''),
            state=parsed.get('StateName', ''),
            zip=parsed.get('ZipCode', '')
        )

    except Exception as e:
        # Fallback to regex-based parsing
        logger.debug(f"usaddress failed for '{address_str}': {e}. Using regex fallback.")

        # Pattern: "Street Address, City, ST ZIP"
        pattern = r'^(.+?),\s*([^,]+),\s*([A-Z]{2})\s*(\d{5}(?:-\d{4})?)$'
        match = re.match(pattern, str(address_str))

        if match:
            return ParsedAddress(
                street=match.group(1).strip(),
                city=match.group(2).strip(),
                state=match.group(3).strip(),
                zip=match.group(4).strip()
            )

        # If regex also fails, return the full string as street
        logger.debug(f"Regex parsing also failed for '{address_str}'. Returning as street only.")
        return ParsedAddress(
            street=str(address_str),
            city='',
            state='',
            zip=''
        )

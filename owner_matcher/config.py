"""
Configuration settings for the Owner ID Mapping System.

This module centralizes all configuration values including file paths,
matching thresholds, and text cleaning patterns.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
OUTPUT_DIR = BASE_DIR / "outputs"

# Input/Output file paths (can be overridden via environment variables)
OLD_FILE = os.getenv(
    "OLD_OWNERS_FILE",
    str(RAW_DATA_DIR / "missing_owners_report.xlsx")
)
NEW_FILE = os.getenv(
    "NEW_OWNERS_FILE",
    str(RAW_DATA_DIR / "excluded_owners_comparison.xlsx")
)
OUTPUT_FILE = os.getenv(
    "OUTPUT_FILE",
    str(OUTPUT_DIR / "mapped_owners.csv")
)

# Matching thresholds (Ultra-aggressive optimization for 85% match rate goal)
NAME_THRESHOLD = 50  # Minimum fuzzy name match score (ultra-aggressive for 85% goal)
ADDRESS_THRESHOLD = 60  # Minimum address match score for address-first strategy
ADDRESS_MIN = 35  # Minimum address score for name-based matches (very lenient)
ADDRESS_FIRST_NAME_THRESHOLD = 40  # Minimum name score when address matches first

# Text cleaning patterns
CLEAN_PUNCT = r'[^\w\s]'  # Regex pattern to remove punctuation

# Trust keywords to remove during name cleaning
TRUST_KEYWORDS: List[str] = [
    "oil & gas trust",
    "mineral trust",
    "spousal trust",
    "family living trust",
    "family living",
    "family trust",
    "fam gst tr",
    "irrev tr",
    "life est",
    "test tr",
    "trust/dtd",
    "rev min tr",
    "gst trust",
    "marital trust",
    "rev liv tr",
    "revoc trust",
    "fam rev tr",
    "revocable living trust",
    "living trust",
    "estate living tr",
    "living tr",
    "mgmt tr",
    "fam tr",
    "sep prop tr",
    "min tr",
    "estate",
    "est",
    "rev tr",
    "trustee",
    "trust",
    "tst",
    "tr",
    "jr",
    "sr",
    "iii",
    "ii",
    "iv",
    "etal",
    "irrv"
]

# Column names for old owners dataset
OLD_COLUMNS = {
    'owner_id': 'Previous Owner ID',
    'owner_name': 'Owner Name',
    'address': 'Last Known Address',
    'state': 'State'
}

# Column names for new owners dataset
NEW_COLUMNS = {
    'old_id': 'OLD_OWNER_ID',
    'new_id': 'NEW_OWNER_ID',
    'owner_name': 'EXCLUDE_OWNER_NAME',
    'prod_owner_name': 'PROD_OWNER_NAME',  # Added: production name for comparison
    'prod_address': 'PROD_ADDRESS',
    'prod_city': 'PROD_CITY',
    'prod_state': 'PROD_STATE',
    'prod_zip': 'PROD_ZIP',
    'exclude_address': 'EXCLUDE_ADDRESS',
    'exclude_city': 'EXCLUDE_CITY',
    'exclude_state': 'EXCLUDE_STATE',
    'exclude_zip': 'EXCLUDE_ZIP',
    'id_status': 'ID_STATUS'
}

# Confidence score thresholds for match types
CONFIDENCE_SCORES = {
    # DIRECT_ID_MATCH removed - IDs are reindexed annually
    'EXACT_NAME': 100,
    'ADDRESS_NAME_MATCH': 90,
    'CONFIRMED_FUZZY': 85,
    'FUZZY_NAME': (75, 85),  # Range based on name_score
    'CROSS_STATE_REVIEW': (60, 80),  # Range based on name_score - 10
    'PARTIAL_NAME_REVIEW': (50, 75)  # Range based on name_score - 15
}

# Progress reporting interval
PROGRESS_INTERVAL = 100  # Report progress every N owners

# ============================================================================
# Snowflake Connection Configuration
# ============================================================================

# Snowflake connection parameters (can be overridden via environment variables)
SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT", "OIBLOOJ-DJ95069")
SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER", "STREAMLIT_APP_USER")
SNOWFLAKE_ROLE = os.getenv("SNOWFLAKE_ROLE", "STREAMLIT_APP_ROLE")
SNOWFLAKE_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")
SNOWFLAKE_DATABASE = os.getenv("SNOWFLAKE_DATABASE", "MINERALHOLDERS_DB")
SNOWFLAKE_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC")

# Private key authentication
SNOWFLAKE_PRIVATE_KEY_PATH = os.getenv(
    "SNOWFLAKE_PRIVATE_KEY_PATH",
    str(BASE_DIR / "show_goat_rsa_key.pem")
)

# SQL query file path
SNOWFLAKE_QUERY_FILE = BASE_DIR / "sql" / "generate_excluded_owners_comparison.sql"

"""
Main entry point for the Owner ID Mapping System.

This script orchestrates the complete matching pipeline including:
- Data loading and validation
- Preprocessing (text cleaning, address parsing)
- Cascading match execution
- Results output and summary statistics
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from . import __version__
from .address_parser import parse_address
from .config import (
    OLD_FILE,
    NEW_FILE,
    OUTPUT_FILE,
    OLD_COLUMNS,
    NEW_COLUMNS,
    CONFIDENCE_SCORES
)
from .matchers import OwnerMapper, MatchResult
from .text_utils import clean_text, extract_attention_name

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Optional Snowflake import (only if using --use-snowflake flag)
try:
    from .snowflake_client import fetch_new_owners_from_snowflake
    SNOWFLAKE_AVAILABLE = True
except ImportError:
    SNOWFLAKE_AVAILABLE = False
    logger.warning("Snowflake connector not available. Install with: pip install snowflake-connector-python cryptography")


def validate_file_exists(file_path: str, description: str) -> None:
    """
    Validate that a file exists.

    Args:
        file_path: Path to check
        description: Human-readable description for error messages

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Error: {description} not found at: {file_path}\n"
            f"Please ensure the file exists at this location."
        )


def load_old_owners(file_path: str) -> pd.DataFrame:
    """
    Load and validate old owners dataset.

    Args:
        file_path: Path to Excel file

    Returns:
        DataFrame with validated structure

    Raises:
        ValueError: If required columns are missing
    """
    logger.info(f"Loading old owners from: {file_path}")
    df = pd.read_excel(file_path)

    # Validate required columns
    required_cols = list(OLD_COLUMNS.values())
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"Missing required columns in old owners file: {missing_cols}\n"
            f"Expected columns: {required_cols}"
        )

    logger.info(f"Loaded {len(df)} old owner records")
    return df


def load_new_owners(file_path: str = None, use_snowflake: bool = False) -> pd.DataFrame:
    """
    Load and validate new owners dataset from Excel file or Snowflake.

    Args:
        file_path: Path to Excel file (ignored if use_snowflake=True)
        use_snowflake: If True, fetch data directly from Snowflake

    Returns:
        DataFrame with validated structure

    Raises:
        ValueError: If required columns are missing
        RuntimeError: If Snowflake is requested but not available
    """
    if use_snowflake:
        if not SNOWFLAKE_AVAILABLE:
            raise RuntimeError(
                "Snowflake connector is not available.\n"
                "Install with: pip install snowflake-connector-python cryptography"
            )

        logger.info("Fetching new owners data from Snowflake...")
        df = fetch_new_owners_from_snowflake()
        logger.info(f"Loaded {len(df)} new owner records from Snowflake")
    else:
        logger.info(f"Loading new owners from: {file_path}")
        df = pd.read_excel(file_path)
        logger.info(f"Loaded {len(df)} new owner records from Excel file")

    # Validate required columns (lowercase for case-insensitive check)
    df_cols_lower = [col.lower() for col in df.columns]
    required_cols = [
        NEW_COLUMNS['old_id'],
        NEW_COLUMNS['new_id'],
        NEW_COLUMNS['owner_name']
    ]

    # Check if columns exist (case-insensitive)
    missing_cols = []
    for col in required_cols:
        if col.lower() not in df_cols_lower:
            missing_cols.append(col)

    if missing_cols:
        raise ValueError(
            f"Missing required columns in new owners data: {missing_cols}\n"
            f"Expected columns: {required_cols}\n"
            f"Available columns: {list(df.columns)}"
        )

    return df


def preprocess_old_owners(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess old owners data: clean text and parse addresses.

    Args:
        df: Raw old owners DataFrame

    Returns:
        DataFrame with additional cleaned and parsed fields
    """
    logger.info("Preprocessing old owners data...")

    # Clean names
    df['clean_name'] = df[OLD_COLUMNS['owner_name']].apply(clean_text)
    df['attn_name'] = df[OLD_COLUMNS['owner_name']].apply(extract_attention_name)

    # Parse addresses
    logger.info("Parsing old owner addresses...")
    df['parsed'] = df[OLD_COLUMNS['address']].apply(parse_address)
    df['street'] = df['parsed'].apply(lambda x: x.street)
    df['city'] = df['parsed'].apply(lambda x: x.city)
    df['state_parsed'] = df['parsed'].apply(lambda x: x.state)
    df['zip'] = df['parsed'].apply(lambda x: x.zip)

    # Create full address string
    df['full_address'] = (
        df['street'].fillna('').astype(str) + ' ' +
        df['city'].fillna('').astype(str) + ' ' +
        df['state_parsed'].fillna('').astype(str) + ' ' +
        df['zip'].fillna('').astype(str)
    ).str.strip()

    df['clean_address'] = df['full_address'].apply(clean_text)

    logger.info("Old owners preprocessing complete")
    return df


def preprocess_new_owners(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess new owners data: clean text and combine addresses.

    Args:
        df: Raw new owners DataFrame

    Returns:
        DataFrame with additional cleaned fields
    """
    logger.info("Preprocessing new owners data...")

    # Clean names
    df['clean_name'] = df[NEW_COLUMNS['owner_name']].apply(clean_text)
    df['attn_name'] = df[NEW_COLUMNS['owner_name']].apply(extract_attention_name)

    # Combine production address
    df['full_prod_addr'] = (
        df[NEW_COLUMNS['prod_address']].fillna('').astype(str) + ' ' +
        df[NEW_COLUMNS['prod_city']].fillna('').astype(str) + ' ' +
        df[NEW_COLUMNS['prod_state']].fillna('').astype(str) + ' ' +
        df[NEW_COLUMNS['prod_zip']].fillna('').astype(str)
    ).str.strip()

    # Combine exclude address
    df['full_exclude_addr'] = (
        df[NEW_COLUMNS['exclude_address']].fillna('').astype(str) + ' ' +
        df[NEW_COLUMNS['exclude_city']].fillna('').astype(str) + ' ' +
        df[NEW_COLUMNS['exclude_state']].fillna('').astype(str) + ' ' +
        df[NEW_COLUMNS['exclude_zip']].fillna('').astype(str)
    ).str.strip()

    # Clean addresses
    df['clean_prod_addr'] = df['full_prod_addr'].apply(clean_text)
    df['clean_exclude_addr'] = df['full_exclude_addr'].apply(clean_text)

    # Use production address if available, otherwise exclude address
    df['clean_address'] = df['clean_prod_addr'].where(
        df['clean_prod_addr'] != '',
        df['clean_exclude_addr']
    )

    logger.info("New owners preprocessing complete")
    return df


def calculate_confidence(match_result: MatchResult) -> float:
    """
    Calculate confidence score based on match type and scores.

    Args:
        match_result: Result from matching process

    Returns:
        Confidence score (0-100)
    """
    match_step = match_result.match_step

    if match_step == 'EXACT_NAME':
        return CONFIDENCE_SCORES['EXACT_NAME']
    elif match_step == 'ADDRESS_NAME_MATCH':
        return CONFIDENCE_SCORES['ADDRESS_NAME_MATCH']
    elif match_step == 'CONFIRMED_FUZZY':
        return CONFIDENCE_SCORES['CONFIRMED_FUZZY']
    elif match_step == 'ESTATE_TRANSITION':
        # High confidence for estate/trust transitions
        return 85.0
    elif match_step == 'TEMPORAL_CORE_MATCH':
        # High confidence for temporal exact core matches
        return 88.0
    elif match_step == 'TEMPORAL_FUZZY_MATCH':
        # Good confidence for temporal fuzzy matches
        return max(75.0, min(85.0, match_result.name_score))
    elif match_step == 'FUZZY_NAME':
        min_conf, max_conf = CONFIDENCE_SCORES['FUZZY_NAME']
        return max(min_conf, min(max_conf, match_result.name_score))
    elif match_step == 'CROSS_STATE_REVIEW':
        min_conf, max_conf = CONFIDENCE_SCORES['CROSS_STATE_REVIEW']
        return max(min_conf, min(max_conf, match_result.name_score - 10))
    elif match_step == 'PARTIAL_NAME_REVIEW':
        min_conf, max_conf = CONFIDENCE_SCORES['PARTIAL_NAME_REVIEW']
        return max(min_conf, min(max_conf, match_result.name_score - 15))
    elif match_step == 'ADDRESS_ONLY':
        # Moderate-low confidence for address-only matches
        return max(55.0, min(70.0, match_result.address_score - 10))
    elif match_step == 'INITIAL_MATCH':
        # Moderate confidence for initial matches with good address
        return max(60.0, min(75.0, match_result.address_score))
    elif match_step == 'LAST_RESORT':
        # Low confidence for last resort matches
        return max(50.0, min(65.0, match_result.name_score))
    else:
        return 0.0


def determine_review_priority(match_result: MatchResult, old_name: str, new_df: pd.DataFrame) -> tuple[str, str]:
    """
    Determine review priority and suggested action for unmatched records.

    Args:
        match_result: Match result
        old_name: Cleaned old owner name
        new_df: New owners dataframe

    Returns:
        Tuple of (review_priority, suggested_action)
    """
    if match_result.new_id is not None:
        return 'MATCHED', 'Apply mapping'

    # Extract best score from status if available
    import re
    score_match = re.search(r'BEST_SCORE:(\d+)%', match_result.status)

    if score_match:
        best_score = int(score_match.group(1))
        if best_score >= 65:
            return 'HIGH', 'Manual search - close match exists'
        elif best_score >= 50:
            return 'MEDIUM', 'Review for typos/variations'
        else:
            return 'LOW', 'Likely inactive - consider removing'
    elif len(old_name) < 5:
        return 'LOW', 'Likely data quality issue'
    else:
        return 'LOW', 'Likely inactive - consider removing'


def map_owners_legacy(old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute the cascading match strategy for all owners (legacy workflow).
    Used when loading from two separate Excel files.

    Args:
        old_df: Preprocessed old owners DataFrame
        new_df: Preprocessed new owners DataFrame

    Returns:
        DataFrame with match results
    """
    logger.info("Starting cascading match process (legacy workflow)...")

    mapper = OwnerMapper()
    results: List[Dict] = []

    # Process each old owner with progress bar
    for idx, old_row in tqdm(
        old_df.iterrows(),
        total=len(old_df),
        desc="Matching owners",
        unit="owner"
    ):
        # Filter new owners by state if available
        state = old_row[OLD_COLUMNS['state']]
        if pd.notna(state):
            new_subset = new_df[new_df[NEW_COLUMNS['exclude_state']] == state].reset_index(drop=True)
        else:
            new_subset = new_df

        # Attempt match
        match_result = mapper.match_owner(old_row, new_df, new_subset)

        # Calculate confidence
        confidence = calculate_confidence(match_result)

        # Determine review priority
        review_priority, suggested_action = determine_review_priority(
            match_result,
            old_row['clean_name'],
            new_df
        )

        # Get new owner details if matched
        new_name = None
        new_address = None
        new_city = None
        new_state = None
        new_zip = None
        if match_result.new_id:
            matched_rec = new_df[new_df[NEW_COLUMNS['new_id']] == match_result.new_id]
            if not matched_rec.empty:
                new_name = matched_rec.iloc[0].get(NEW_COLUMNS['prod_owner_name'], '')
                new_address = matched_rec.iloc[0].get(NEW_COLUMNS['prod_address'], '')
                new_city = matched_rec.iloc[0].get(NEW_COLUMNS['prod_city'], '')
                new_state = matched_rec.iloc[0].get(NEW_COLUMNS['prod_state'], '')
                new_zip = matched_rec.iloc[0].get(NEW_COLUMNS['prod_zip'], '')

        # Build result record
        results.append({
            OLD_COLUMNS['owner_id']: old_row[OLD_COLUMNS['owner_id']],
            OLD_COLUMNS['owner_name']: old_row[OLD_COLUMNS['owner_name']],
            'OLD_ADDRESS': old_row.get(OLD_COLUMNS['address'], ''),
            'OLD_CITY': old_row.get('city', ''),
            'OLD_STATE': state,
            'OLD_ZIP': old_row.get('zip', ''),
            'mapped_new_id': match_result.new_id,
            'NEW_NAME': new_name,
            'NEW_ADDRESS': new_address,
            'NEW_CITY': new_city,
            'NEW_STATE': new_state,
            'NEW_ZIP': new_zip,
            'match_step': match_result.match_step,
            'matched_on_attention': 'Yes' if match_result.matched_on_attn else 'No',
            'confidence_score': round(confidence, 1),
            'name_score': round(match_result.name_score, 2),
            'address_score': round(match_result.address_score, 2),
            'status': match_result.status,
            'review_priority': review_priority,
            'suggested_action': suggested_action
        })

    logger.info("Matching complete")
    return pd.DataFrame(results)


def preprocess_snowflake_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Preprocess Snowflake data, separating matched and unmatched records.

    Args:
        df: Raw Snowflake DataFrame

    Returns:
        Tuple of (original_df, already_matched_df, needs_matching_df)
    """
    logger.info("Preprocessing Snowflake data...")

    # Separate already matched vs unmatched
    already_matched = df[df['NEW_OWNER_ID'].notna()].copy()
    needs_matching = df[df['NEW_OWNER_ID'].isna()].copy()

    logger.info(f"Already matched by SQL: {len(already_matched)} records")
    if len(already_matched) > 0 and 'ID_STATUS' in already_matched.columns:
        logger.info(f"  - ID_UNCHANGED: {(already_matched['ID_STATUS'] == 'ID_UNCHANGED').sum()}")
        logger.info(f"  - ID_CHANGED: {(already_matched['ID_STATUS'] == 'ID_CHANGED').sum()}")
    logger.info(f"Need fuzzy matching: {len(needs_matching)} records")

    # Preprocess unmatched records (these are the "old owners" to match)
    if len(needs_matching) > 0:
        needs_matching['clean_name'] = needs_matching['EXCLUDE_OWNER_NAME'].apply(clean_text)
        needs_matching['attn_name'] = needs_matching['EXCLUDE_OWNER_NAME'].apply(extract_attention_name)

        # Combine and clean address
        needs_matching['full_address'] = (
            needs_matching['EXCLUDE_ADDRESS'].fillna('').astype(str) + ' ' +
            needs_matching['EXCLUDE_CITY'].fillna('').astype(str) + ' ' +
            needs_matching['EXCLUDE_STATE'].fillna('').astype(str) + ' ' +
            needs_matching['EXCLUDE_ZIP'].fillna('').astype(str)
        ).str.strip()
        needs_matching['clean_address'] = needs_matching['full_address'].apply(clean_text)

        # Parse address
        needs_matching['parsed'] = needs_matching['full_address'].apply(parse_address)
        needs_matching['street'] = needs_matching['parsed'].apply(lambda x: x.street if x else '')
        needs_matching['city'] = needs_matching['parsed'].apply(lambda x: x.city if x else '')
        needs_matching['state_parsed'] = needs_matching['parsed'].apply(lambda x: x.state if x else '')
        needs_matching['zip'] = needs_matching['parsed'].apply(lambda x: x.zip if x else '')
        needs_matching['clean_state'] = needs_matching['EXCLUDE_STATE'].fillna(needs_matching['state_parsed'])

        # Add columns expected by matcher
        needs_matching['Previous Owner ID'] = needs_matching['OLD_OWNER_ID']
        needs_matching['Owner Name'] = needs_matching['EXCLUDE_OWNER_NAME']
        needs_matching['State'] = needs_matching['EXCLUDE_STATE']

    # Preprocess potential matches (records with production IDs)
    if len(already_matched) > 0:
        already_matched['clean_name'] = already_matched['PROD_OWNER_NAME'].apply(clean_text)
        already_matched['attn_name'] = already_matched['PROD_OWNER_NAME'].apply(extract_attention_name)

        # Combine and clean address
        already_matched['full_address'] = (
            already_matched['PROD_ADDRESS'].fillna('').astype(str) + ' ' +
            already_matched['PROD_CITY'].fillna('').astype(str) + ' ' +
            already_matched['PROD_STATE'].fillna('').astype(str) + ' ' +
            already_matched['PROD_ZIP'].fillna('').astype(str)
        ).str.strip()
        already_matched['clean_address'] = already_matched['full_address'].apply(clean_text)

        # Parse address
        already_matched['parsed'] = already_matched['full_address'].apply(parse_address)
        already_matched['street'] = already_matched['parsed'].apply(lambda x: x.street if x else '')
        already_matched['city'] = already_matched['parsed'].apply(lambda x: x.city if x else '')
        already_matched['state_parsed'] = already_matched['parsed'].apply(lambda x: x.state if x else '')
        already_matched['zip'] = already_matched['parsed'].apply(lambda x: x.zip if x else '')
        already_matched['clean_state'] = already_matched['PROD_STATE'].fillna(already_matched['state_parsed'])

    logger.info("Snowflake preprocessing complete")
    return df, already_matched, needs_matching


def map_owners_snowflake(snowflake_df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute matching workflow for Snowflake data.
    Handles SQL-matched and fuzzy matching in one workflow.

    Args:
        snowflake_df: Raw Snowflake DataFrame

    Returns:
        DataFrame with all match results
    """
    # Preprocess Snowflake data
    original_df, already_matched, needs_matching = preprocess_snowflake_data(snowflake_df)

    # Initialize results list
    results: List[Dict] = []

    # Add SQL-matched records first
    for idx, row in already_matched.iterrows():
        results.append({
            'OLD_OWNER_ID': row['OLD_OWNER_ID'],
            'NEW_OWNER_ID': row['NEW_OWNER_ID'],
            'OLD_NAME': row['EXCLUDE_OWNER_NAME'],
            'OLD_ADDRESS': row.get('EXCLUDE_ADDRESS', ''),
            'OLD_CITY': row.get('EXCLUDE_CITY', ''),
            'OLD_STATE': row.get('EXCLUDE_STATE', ''),
            'OLD_ZIP': row.get('EXCLUDE_ZIP', ''),
            'NEW_NAME': row['PROD_OWNER_NAME'],
            'NEW_ADDRESS': row.get('PROD_ADDRESS', ''),
            'NEW_CITY': row.get('PROD_CITY', ''),
            'NEW_STATE': row.get('PROD_STATE', ''),
            'NEW_ZIP': row.get('PROD_ZIP', ''),
            'MATCH_TYPE': 'SQL_DIRECT',
            'ID_STATUS': row.get('ID_STATUS', 'MATCHED'),
            'ADDRESS_MATCH': row.get('ADDRESS_MATCH_STATUS', 'UNKNOWN'),
            'match_step': 'SQL_MATCH',
            'confidence_score': 100.0,
            'name_score': 100.0,
            'address_score': 100.0,
            'status': f"ID_{row.get('ID_STATUS', 'MATCHED')}",
            'review_priority': 'MATCHED',
            'suggested_action': 'Apply mapping'
        })

    # Run fuzzy matching only on unmatched records
    if len(needs_matching) > 0 and len(already_matched) > 0:
        logger.info("Running fuzzy matching on unmatched records...")

        mapper = OwnerMapper()
        batch_size = 100
        total_matched = 0

        # Process in batches for progress tracking
        for i in tqdm(
            range(0, len(needs_matching), batch_size),
            desc="Fuzzy matching",
            unit="batch"
        ):
            batch = needs_matching.iloc[i:i+batch_size]

            for idx, old_row in batch.iterrows():
                # Get state for filtering
                old_state = old_row.get('clean_state', '')

                # Create state-filtered subset from already matched records
                if old_state and pd.notna(old_state):
                    state_matches = already_matched[already_matched['clean_state'] == old_state]
                else:
                    state_matches = already_matched

                # Run matcher
                match_result = mapper.match_owner(old_row, already_matched, state_matches)

                # Calculate confidence
                confidence = calculate_confidence(match_result)

                # Determine review priority
                review_priority, suggested_action = determine_review_priority(
                    match_result,
                    old_row['clean_name'],
                    already_matched
                )

                # Build result record
                result_dict = {
                    'OLD_OWNER_ID': old_row['OLD_OWNER_ID'],
                    'NEW_OWNER_ID': match_result.new_id,
                    'OLD_NAME': old_row['EXCLUDE_OWNER_NAME'],
                    'OLD_ADDRESS': old_row.get('EXCLUDE_ADDRESS', ''),
                    'OLD_CITY': old_row.get('EXCLUDE_CITY', ''),
                    'OLD_STATE': old_row.get('EXCLUDE_STATE', ''),
                    'OLD_ZIP': old_row.get('EXCLUDE_ZIP', ''),
                    'NEW_NAME': None,
                    'NEW_ADDRESS': None,
                    'NEW_CITY': None,
                    'NEW_STATE': None,
                    'NEW_ZIP': None,
                    'MATCH_TYPE': f'FUZZY_{match_result.match_step}' if match_result.new_id else 'UNMATCHED',
                    'ID_STATUS': 'FUZZY_MATCHED' if match_result.new_id else 'NOT_FOUND',
                    'ADDRESS_MATCH': 'FUZZY' if match_result.new_id else 'NO_MATCH',
                    'match_step': match_result.match_step,
                    'confidence_score': round(confidence, 1),
                    'name_score': round(match_result.name_score, 2),
                    'address_score': round(match_result.address_score, 2),
                    'status': match_result.status,
                    'review_priority': review_priority,
                    'suggested_action': suggested_action
                }

                # Add matched details if found
                if match_result.new_id:
                    matched_rec = already_matched[already_matched['NEW_OWNER_ID'] == match_result.new_id]
                    if not matched_rec.empty:
                        result_dict['NEW_NAME'] = matched_rec.iloc[0]['PROD_OWNER_NAME']
                        result_dict['NEW_ADDRESS'] = matched_rec.iloc[0].get('PROD_ADDRESS', '')
                        result_dict['NEW_CITY'] = matched_rec.iloc[0].get('PROD_CITY', '')
                        result_dict['NEW_STATE'] = matched_rec.iloc[0].get('PROD_STATE', '')
                        result_dict['NEW_ZIP'] = matched_rec.iloc[0].get('PROD_ZIP', '')
                    total_matched += 1

                results.append(result_dict)

        logger.info(f"Fuzzy matching complete: {total_matched} matches found")
    elif len(needs_matching) > 0:
        # Add unmatched records with no fuzzy matching possible
        for idx, row in needs_matching.iterrows():
            results.append({
                'OLD_OWNER_ID': row['OLD_OWNER_ID'],
                'NEW_OWNER_ID': None,
                'OLD_NAME': row['EXCLUDE_OWNER_NAME'],
                'OLD_ADDRESS': row.get('EXCLUDE_ADDRESS', ''),
                'OLD_CITY': row.get('EXCLUDE_CITY', ''),
                'OLD_STATE': row.get('EXCLUDE_STATE', ''),
                'OLD_ZIP': row.get('EXCLUDE_ZIP', ''),
                'NEW_NAME': None,
                'NEW_ADDRESS': None,
                'NEW_CITY': None,
                'NEW_STATE': None,
                'NEW_ZIP': None,
                'MATCH_TYPE': 'UNMATCHED',
                'ID_STATUS': 'NOT_FOUND',
                'ADDRESS_MATCH': 'NO_MATCH',
                'match_step': 'NO_MATCH',
                'confidence_score': 0.0,
                'name_score': 0.0,
                'address_score': 0.0,
                'status': 'NO_MATCH_FOUND',
                'review_priority': 'LOW',
                'suggested_action': 'Manual review required'
            })

    return pd.DataFrame(results)


def print_summary(mapped_df: pd.DataFrame, is_snowflake: bool = False) -> None:
    """
    Print summary statistics of matching results.

    Args:
        mapped_df: Results DataFrame
        is_snowflake: Whether data was loaded from Snowflake
    """
    print("\n" + "=" * 80)
    print("OWNER ID MATCHING SUMMARY")
    print("=" * 80)
    print(f"Total records processed: {len(mapped_df)}")

    # Check which ID column to use based on workflow
    id_col = 'NEW_OWNER_ID' if 'NEW_OWNER_ID' in mapped_df.columns else 'mapped_new_id'

    # Overall match statistics
    matched_count = mapped_df[id_col].notna().sum()
    unmatched_count = mapped_df[id_col].isna().sum()

    print(f"\nOverall Results:")
    print(f"  MATCHED:   {matched_count:6d} ({(matched_count/len(mapped_df))*100:5.1f}%)")
    print(f"  UNMATCHED: {unmatched_count:6d} ({(unmatched_count/len(mapped_df))*100:5.1f}%)")

    if is_snowflake and 'MATCH_TYPE' in mapped_df.columns:
        # Enhanced summary for Snowflake workflow
        print("\nMatches by Type:")

        # SQL matches
        sql_matches = mapped_df[mapped_df['MATCH_TYPE'] == 'SQL_DIRECT']
        if len(sql_matches) > 0:
            print(f"  SQL Direct Matches: {len(sql_matches)} ({len(sql_matches)/len(mapped_df)*100:.1f}%)")
            if 'ID_STATUS' in sql_matches.columns:
                id_unchanged = (sql_matches['ID_STATUS'] == 'ID_UNCHANGED').sum()
                id_changed = (sql_matches['ID_STATUS'] == 'ID_CHANGED').sum()
                print(f"    - ID_UNCHANGED: {id_unchanged}")
                print(f"    - ID_CHANGED: {id_changed}")

        # Fuzzy matches
        fuzzy_matches = mapped_df[mapped_df['MATCH_TYPE'].str.startswith('FUZZY_', na=False)]
        if len(fuzzy_matches) > 0:
            print(f"  Fuzzy Matches: {len(fuzzy_matches)} ({len(fuzzy_matches)/len(mapped_df)*100:.1f}%)")

            # Breakdown by step
            fuzzy_steps = fuzzy_matches['match_step'].value_counts()
            for step, count in fuzzy_steps.items():
                if step != 'NO_MATCH':
                    print(f"    - {step}: {count}")
    else:
        # Original summary for legacy workflow
        print("\nMatch Step Breakdown:")
        step_counts = mapped_df['match_step'].value_counts()
        for step, count in step_counts.items():
            pct = (count / len(mapped_df)) * 100
            print(f"  {step:30s}: {count:5d} ({pct:5.1f}%)")

    # Confidence distribution
    if 'confidence_score' in mapped_df.columns:
        matched_df = mapped_df[mapped_df[id_col].notna()]
        if len(matched_df) > 0:
            print("\nConfidence Score Distribution (matched records):")
            print(f"  High (90-100):   {(matched_df['confidence_score'] >= 90).sum()} matches")
            print(f"  Medium (75-89):  {((matched_df['confidence_score'] >= 75) & (matched_df['confidence_score'] < 90)).sum()} matches")
            print(f"  Low (50-74):     {((matched_df['confidence_score'] >= 50) & (matched_df['confidence_score'] < 75)).sum()} matches")
            print(f"  Very Low (<50):  {(matched_df['confidence_score'] < 50).sum()} matches")

    print("=" * 80)


def main(args: argparse.Namespace = None) -> int:
    """
    Main execution function.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    if args is None:
        parser = create_parser()
        args = parser.parse_args()

    # Set log level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    logger.info(f"Owner ID Mapping System v{__version__}")

    try:
        if args.use_snowflake:
            # New Snowflake workflow
            logger.info("Using Snowflake workflow")

            # Load data from Snowflake
            snowflake_df = fetch_new_owners_from_snowflake()

            # Execute the complete matching workflow
            mapped_df = map_owners_snowflake(snowflake_df)

            # Sort by confidence score
            mapped_df = mapped_df.sort_values('confidence_score', ascending=False).reset_index(drop=True)

            # Ensure output directory exists
            output_path = Path(args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save results
            mapped_df.to_csv(args.output_file, index=False)
            logger.info(f"Results saved to: {args.output_file}")

            # Generate additional report files if timestamp requested
            if getattr(args, 'timestamp', False):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

                # Save unmatched records separately
                unmatched = mapped_df[mapped_df['NEW_OWNER_ID'].isna()]
                if len(unmatched) > 0:
                    unmatched_file = output_path.parent / f'unmatched_records_{timestamp}.csv'
                    unmatched.to_csv(unmatched_file, index=False)
                    logger.info(f"Unmatched records saved to: {unmatched_file}")

                # Generate summary report
                report_file = output_path.parent / f'matching_report_{timestamp}.txt'
                with open(report_file, 'w') as f:
                    f.write("OWNER ID MATCHING REPORT\n")
                    f.write("="*60 + "\n")
                    f.write(f"Generated: {datetime.now()}\n\n")
                    f.write(f"Total Records Processed: {len(mapped_df)}\n")
                    f.write(f"Total Matched: {mapped_df['NEW_OWNER_ID'].notna().sum()}\n")
                    f.write(f"Total Unmatched: {mapped_df['NEW_OWNER_ID'].isna().sum()}\n\n")

                    # Add match type breakdown
                    if 'MATCH_TYPE' in mapped_df.columns:
                        f.write("Match Breakdown:\n")
                        match_types = mapped_df['MATCH_TYPE'].value_counts()
                        for match_type, count in match_types.items():
                            f.write(f"  {match_type}: {count}\n")

                logger.info(f"Summary report saved to: {report_file}")

            # Print summary
            print_summary(mapped_df, is_snowflake=True)

        else:
            # Legacy workflow with two Excel files
            logger.info("Using legacy Excel workflow")

            # Validate input files exist
            validate_file_exists(args.old_file, "Old owners file")
            validate_file_exists(args.new_file, "New owners file")

            # Load data
            old_df = load_old_owners(args.old_file)
            new_df = load_new_owners(args.new_file, use_snowflake=False)

            # Preprocess
            old_df = preprocess_old_owners(old_df)
            new_df = preprocess_new_owners(new_df)

            # Execute matching
            mapped_df = map_owners_legacy(old_df, new_df)

            # Sort by confidence score
            mapped_df = mapped_df.sort_values('confidence_score', ascending=False).reset_index(drop=True)

            # Ensure output directory exists
            output_path = Path(args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save results
            mapped_df.to_csv(args.output_file, index=False)
            logger.info(f"Results saved to: {args.output_file}")

            # Print summary
            print_summary(mapped_df, is_snowflake=False)

        return 0

    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        return 1


def create_parser() -> argparse.ArgumentParser:
    """
    Create command-line argument parser.

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description='Owner ID Mapping System - Fuzzy string matching for owner ID migration',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--old-file',
        default=OLD_FILE,
        help=f'Path to old owners Excel file (default: {OLD_FILE})'
    )

    parser.add_argument(
        '--new-file',
        default=NEW_FILE,
        help=f'Path to new owners Excel file (default: {NEW_FILE})'
    )

    parser.add_argument(
        '--output-file',
        '-o',
        default=OUTPUT_FILE,
        help=f'Path to output CSV file (default: {OUTPUT_FILE})'
    )

    parser.add_argument(
        '--use-snowflake',
        action='store_true',
        help='Fetch new owners data directly from Snowflake instead of Excel file'
    )

    parser.add_argument(
        '--timestamp',
        action='store_true',
        help='Generate timestamped report files (unmatched records and summary report)'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {__version__}'
    )

    return parser


if __name__ == '__main__':
    sys.exit(main())

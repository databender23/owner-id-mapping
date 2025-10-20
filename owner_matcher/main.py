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

# Optional Snowflake import (only if using --use-snowflake flag)
try:
    from .snowflake_client import fetch_new_owners_from_snowflake
    SNOWFLAKE_AVAILABLE = True
except ImportError:
    SNOWFLAKE_AVAILABLE = False
    logger.warning("Snowflake connector not available. Install with: pip install snowflake-connector-python cryptography")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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

    if match_step in ['DIRECT_ID_MATCH', 'EXACT_NAME']:
        return CONFIDENCE_SCORES['DIRECT_ID_MATCH']
    elif match_step == 'ADDRESS_ATTN_MATCH':
        return CONFIDENCE_SCORES['ADDRESS_ATTN_MATCH']
    elif match_step == 'ADDRESS_NAME_MATCH':
        return CONFIDENCE_SCORES['ADDRESS_NAME_MATCH']
    elif match_step == 'CONFIRMED_FUZZY':
        return CONFIDENCE_SCORES['CONFIRMED_FUZZY']
    elif match_step == 'FUZZY_NAME':
        min_conf, max_conf = CONFIDENCE_SCORES['FUZZY_NAME']
        return max(min_conf, min(max_conf, match_result.name_score))
    elif match_step == 'CROSS_STATE_REVIEW':
        min_conf, max_conf = CONFIDENCE_SCORES['CROSS_STATE_REVIEW']
        return max(min_conf, min(max_conf, match_result.name_score - 10))
    elif match_step == 'PARTIAL_NAME_REVIEW':
        min_conf, max_conf = CONFIDENCE_SCORES['PARTIAL_NAME_REVIEW']
        return max(min_conf, min(max_conf, match_result.name_score - 15))
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


def map_owners(old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute the cascading match strategy for all owners.

    Args:
        old_df: Preprocessed old owners DataFrame
        new_df: Preprocessed new owners DataFrame

    Returns:
        DataFrame with match results
    """
    logger.info("Starting cascading match process...")

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

        # Build result record
        results.append({
            OLD_COLUMNS['owner_id']: old_row[OLD_COLUMNS['owner_id']],
            OLD_COLUMNS['owner_name']: old_row[OLD_COLUMNS['owner_name']],
            'Cleaned Name': old_row['clean_name'],
            'Attention Name': old_row['attn_name'] if old_row['attn_name'] else '',
            OLD_COLUMNS['address']: old_row[OLD_COLUMNS['address']],
            OLD_COLUMNS['state']: state,
            'mapped_new_id': match_result.new_id,
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


def print_summary(mapped_df: pd.DataFrame) -> None:
    """
    Print summary statistics of matching results.

    Args:
        mapped_df: Results DataFrame
    """
    print("\n" + "=" * 60)
    print("MATCH SUMMARY:")
    print("=" * 60)

    # Match step breakdown
    step_counts = mapped_df['match_step'].value_counts()
    for step, count in step_counts.items():
        pct = (count / len(mapped_df)) * 100
        print(f"{step:30s}: {count:5d} ({pct:5.1f}%)")

    # Overall stats
    matched_count = mapped_df['mapped_new_id'].notna().sum()
    unmatched_count = mapped_df['mapped_new_id'].isna().sum()
    print("=" * 60)
    print(f"{'TOTAL MATCHED':30s}: {matched_count:5d} ({(matched_count/len(mapped_df))*100:5.1f}%)")
    print(f"{'TOTAL UNMATCHED':30s}: {unmatched_count:5d} ({(unmatched_count/len(mapped_df))*100:5.1f}%)")
    print("=" * 60)


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
        # Validate input files exist
        validate_file_exists(args.old_file, "Old owners file")

        if not args.use_snowflake:
            validate_file_exists(args.new_file, "New owners file")

        # Load data
        old_df = load_old_owners(args.old_file)
        new_df = load_new_owners(args.new_file, use_snowflake=args.use_snowflake)

        # Preprocess
        old_df = preprocess_old_owners(old_df)
        new_df = preprocess_new_owners(new_df)

        # Execute matching
        mapped_df = map_owners(old_df, new_df)

        # Sort by confidence score
        mapped_df = mapped_df.sort_values('confidence_score', ascending=False).reset_index(drop=True)

        # Ensure output directory exists
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save results
        mapped_df.to_csv(args.output_file, index=False)
        logger.info(f"Results saved to: {args.output_file}")

        # Print summary
        print_summary(mapped_df)

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

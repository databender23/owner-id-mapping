# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **Owner ID Mapping System** that performs fuzzy string matching between two datasets of oil & gas owners to map previous owner IDs to new owner IDs. The system uses a cascading match strategy with multiple fallback approaches to maximize matching accuracy.

The codebase has been refactored into a modular Python package with clear separation of concerns, type hints, logging, and a CLI interface.

## Repository Structure

```
owner_id_mapping/
├── owner_matcher/              # Main Python package
│   ├── __init__.py            # Package metadata
│   ├── config.py              # Centralized configuration (thresholds, paths, constants)
│   ├── text_utils.py          # Text cleaning and normalization
│   ├── address_parser.py      # Address parsing with usaddress library
│   ├── matchers.py            # Strategy pattern matching implementations
│   ├── snowflake_client.py    # Snowflake connection and query execution
│   └── main.py                # CLI entry point and orchestration
├── data/
│   └── raw/                   # Input Excel files
│       ├── missing_owners_report.xlsx
│       └── excluded_owners_comparison.xlsx
├── outputs/                   # Generated CSV results
│   └── mapped_owners.csv
├── sql/                       # Snowflake queries
│   ├── README.md             # SQL documentation
│   └── generate_excluded_owners_comparison.sql  # Query to generate input data
├── docs/                      # Documentation
│   └── archive/              # Original docx files
├── MATCHING_STRATEGY.md      # Comprehensive algorithm documentation
├── CLAUDE.md                 # This file
├── README.md                 # Quick start guide
└── requirements.txt          # Python dependencies
```

## Key Architecture

### Modular Design

The system is organized into focused modules:

1. **config.py** - All configuration in one place
   - File paths (with environment variable support)
   - Matching thresholds
   - Trust keywords
   - Column name mappings
   - Confidence score definitions

2. **text_utils.py** - Text processing utilities
   - `clean_text()`: Comprehensive text normalization
   - `extract_attention_name()`: Parse ATTN/c/o lines
   - `extract_po_box_or_zip()`: Extract numeric identifiers
   - All functions have type hints and docstrings

3. **address_parser.py** - Address parsing
   - `parse_address()`: Uses usaddress library with regex fallback
   - Returns structured `ParsedAddress` dataclass
   - Handles edge cases gracefully

4. **snowflake_client.py** - Snowflake database connection
   - `SnowflakeClient`: Connection manager with private key auth
   - `fetch_new_owners_from_snowflake()`: Convenience function to query data
   - Reads SQL from `sql/generate_excluded_owners_comparison.sql`
   - Context manager support for automatic connection cleanup

5. **matchers.py** - Strategy pattern implementation
   - `BaseMatchStrategy`: Abstract base class
   - Six concrete strategy classes:
     - `DirectIDMatcher`
     - `ExactNameMatcher`
     - `AddressFirstMatcher` (most complex)
     - `FuzzyNameMatcher`
     - `CrossStateMatcher`
     - `PartialNameMatcher`
   - `OwnerMapper`: Orchestrator that runs cascade
   - `MatchResult`: Dataclass for match results

6. **main.py** - CLI and orchestration
   - Argument parsing with argparse (includes `--use-snowflake` flag)
   - Data loading and validation (Excel or Snowflake)
   - Preprocessing pipeline
   - Match execution with progress bars
   - Results output and summary statistics

### Cascading Match Strategy (6 Steps)

The system processes each owner through a cascade of increasingly fuzzy matching strategies:

0. **Direct ID Match**: Exact match on `OLD_OWNER_ID` field
1. **Exact Name Match**: Character-for-character name matching after cleaning
2. **Address-First Matching**: The most complex and unique step
   - Matches primarily on address (75% threshold)
   - Then validates name similarity (60% threshold)
   - Handles duplicate address scenarios by finding best match
   - Can match on either main name OR attention name
   - This is the PRIMARY matching strategy that differs from typical name-first approaches
3. **Fuzzy Name Matching**: Token-based matching with address validation
4. **Cross-State Matching**: Remove state filter for high-confidence matches
5. **Partial Name Matching**: Substring/expansion matching for long names

See `MATCHING_STRATEGY.md` for detailed algorithm documentation.

### Data Flow

```
Snowflake Database (MINERALHOLDERS_DB)
    ↓ (sql/generate_excluded_owners_comparison.sql)
data/raw/excluded_owners_comparison.xlsx
data/raw/missing_owners_report.xlsx
    ↓ (Load & validate)
pd.DataFrame (old_owners, new_owners)
    ↓ (Preprocess: clean text, parse addresses)
pd.DataFrame (with clean_name, clean_address fields)
    ↓ (Match: cascade through strategies)
List[MatchResult]
    ↓ (Calculate confidence, determine review priority)
pd.DataFrame (results)
    ↓ (Sort by confidence, save)
outputs/mapped_owners.csv
```

**Data Generation:**
- Input Excel files are generated from Snowflake using queries in `sql/` directory
- `excluded_owners_comparison.xlsx`: Maps EXCLUDE_OWNERS to production IDs
- `missing_owners_report.xlsx`: Owners from old system not in production
- See `sql/README.md` for query documentation and customization

## Running the System

### Setup
```bash
# Install dependencies
pip install -r requirements.txt
```

### Basic Usage
```bash
# Run with Excel files (default)
python -m owner_matcher.main

# Query directly from Snowflake (recommended)
python -m owner_matcher.main --use-snowflake

# Custom input/output files
python -m owner_matcher.main \
  --old-file path/to/old_owners.xlsx \
  --new-file path/to/new_owners.xlsx \
  --output-file path/to/output.csv

# Enable debug logging
python -m owner_matcher.main --debug

# Show help
python -m owner_matcher.main --help
```

The script:
- Validates input files exist
- Loads and preprocesses data
- Runs cascading match strategy with progress bar
- Saves results to CSV
- Prints summary statistics

## Configuration

Edit `owner_matcher/config.py` to adjust:

**Thresholds:**
- `NAME_THRESHOLD = 70`: Minimum fuzzy name match score
- `ADDRESS_THRESHOLD = 75`: Minimum address match score for address-first strategy
- `ADDRESS_MIN = 60`: Minimum address score for name-based matches
- `ADDRESS_FIRST_NAME_THRESHOLD = 60`: Minimum name score when address matches first

**File Paths:**
- Can be overridden via environment variables:
  - `OLD_OWNERS_FILE`
  - `NEW_OWNERS_FILE`
  - `OUTPUT_FILE`

**Trust Keywords:**
- List of terms to remove during name cleaning
- Add/remove keywords based on your data

**Snowflake Connection:**
- Default configuration uses private key authentication
- Connection parameters:
  - Account: `OIBLOOJ-DJ95069`
  - User: `STREAMLIT_APP_USER`
  - Database: `MINERALHOLDERS_DB`
  - Private Key: `show_goat_rsa_key.pem`
- Override via environment variables:
  - `SNOWFLAKE_ACCOUNT`
  - `SNOWFLAKE_USER`
  - `SNOWFLAKE_PRIVATE_KEY_PATH`
  - etc. (see `config.py` for all options)

## Critical Implementation Details

### Address-First Strategy (matchers.py:AddressFirstMatcher)

This is the unique aspect of this system. Unlike typical name-first matching:

1. **Duplicate Address Handling:**
   - When multiple records share the same `NEW_OWNER_ID`, re-score each duplicate
   - Select the best address match among duplicates
   - Prevents false matches when companies have multiple locations

2. **Attention Name Matching:**
   - Extracts "ATTN:" or "C/O" names separately
   - Matches old_name → candidate_name OR old_attn → candidate_name/attn
   - Uses the BEST score of all combinations
   - Handles mail routing vs actual ownership scenarios

3. **Why Address-First?**
   - Properties don't move; ownership does
   - Address stability > name stability in oil & gas
   - Common names (e.g., "Smith Trust") require location validation

### Text Cleaning Pipeline (text_utils.py)

Comprehensive normalization:
- Lowercase conversion
- Attention line extraction (before removal)
- Trust keyword removal (40+ keywords)
- Punctuation removal
- Whitespace normalization

Trust keywords include estate, trust, living trust, family trust, jr, sr, etc.

### Address Parsing (address_parser.py)

Uses `usaddress` library:
- Extracts structured components (street, city, state, zip)
- Handles directionals (N, S, E, W)
- Standardizes street types (ST, AVE, BLVD)
- Regex fallback for non-standard formats

### Scoring System

Each match produces:
- `confidence_score`: Overall match quality (0-100)
- `name_score`: Name similarity (0-100)
- `address_score`: Address similarity (0-100)

Confidence stratification:
- Direct ID/Exact Name: 100
- Address+Attention: 95
- Address+Name: 90
- Confirmed Fuzzy: 85
- Fuzzy Name: 75-85
- Cross-state/Partial: 50-80

See `config.CONFIDENCE_SCORES` for exact mappings.

## Development Workflow

### Adding a New Matching Strategy

1. Create a new class in `matchers.py` inheriting from `BaseMatchStrategy`
2. Implement the `match()` method
3. Return `MatchResult` or `None`
4. Add the strategy to `OwnerMapper.strategies` list
5. Add confidence calculation to `main.calculate_confidence()`

### Modifying Thresholds

1. Edit values in `config.py`
2. Re-run the script
3. Compare output CSV with previous run
4. Check summary statistics for match rate changes

### Adding New Text Cleaning Rules

1. Edit `text_utils.clean_text()`
2. Add new regex patterns or keyword lists
3. Update `config.TRUST_KEYWORDS` if needed
4. Test with sample data

### Debugging Matches

```bash
# Enable debug logging to see match decisions
python -m owner_matcher.main --debug
```

Debug logs show:
- Which strategy attempted each match
- Score calculations
- Why matches were accepted/rejected
- Address parsing failures

## Testing & Development

No formal test suite currently exists. To test changes:

1. Backup current `outputs/mapped_owners.csv`
2. Run script with modifications
3. Compare new output with backup:
   ```python
   import pandas as pd
   old = pd.read_csv('outputs/mapped_owners.csv.backup')
   new = pd.read_csv('outputs/mapped_owners.csv')
   # Compare match_step, confidence_score distributions
   ```
4. Check summary statistics for regressions

## Output Interpretation

The CSV contains diagnostic fields:
- `match_step`: Which cascade step succeeded (e.g., "ADDRESS_NAME_MATCH")
- `matched_on_attention`: Whether attention name was used (Yes/No)
- `confidence_score`: Algorithmic confidence (0-100, higher = safer)
- `name_score`: Name similarity percentage
- `address_score`: Address similarity percentage
- `status`: Human-readable status (e.g., "ID_CHANGED (ADDRESS+NAME)")
- `review_priority`: HIGH/MEDIUM/LOW for unmatched records
- `suggested_action`: Recommended next step for manual review

Unmatched records include best possible score for investigation.

## Documentation

- **MATCHING_STRATEGY.md**: Comprehensive algorithm documentation
  - Detailed explanation of all 6 matching steps
  - Performance metrics and rationale
  - Review guidelines
  - Configuration tuning guide

- **README.md**: Quick start guide for new users

- **sql/README.md**: Snowflake query documentation
  - Data generation workflow
  - Query customization guide
  - Performance notes

- **docs/archive/**: Original Word documents with implementation notes

## Common Tasks

### Adjust matching to be more aggressive (more matches, higher false positive risk)
```python
# In config.py
NAME_THRESHOLD = 65  # was 70
ADDRESS_THRESHOLD = 70  # was 75
ADDRESS_MIN = 50  # was 60
```

### Adjust matching to be more conservative (fewer matches, lower false positive risk)
```python
# In config.py
NAME_THRESHOLD = 80  # was 70
ADDRESS_THRESHOLD = 85  # was 75
ADDRESS_MIN = 70  # was 60
```

### Process different input files
```bash
python -m owner_matcher.main \
  --old-file data/raw/custom_missing_owners.xlsx \
  --new-file data/raw/custom_excluded_owners.xlsx \
  --output-file outputs/custom_results.csv
```

### Export high-confidence matches only
```python
import pandas as pd
df = pd.read_csv('outputs/mapped_owners.csv')
high_conf = df[df['confidence_score'] >= 90]
high_conf.to_csv('outputs/high_confidence_only.csv', index=False)
```

### Query Snowflake directly from Python
```python
from owner_matcher.snowflake_client import fetch_new_owners_from_snowflake

# Fetch data
df = fetch_new_owners_from_snowflake()
print(f"Loaded {len(df)} owners from Snowflake")
```

### Use different Snowflake credentials
```bash
export SNOWFLAKE_ACCOUNT="different-account"
export SNOWFLAKE_USER="different-user"
export SNOWFLAKE_PRIVATE_KEY_PATH="/path/to/different-key.pem"

python -m owner_matcher.main --use-snowflake
```

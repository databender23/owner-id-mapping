# Owner ID Mapping System

A fuzzy string matching system for mapping old owner IDs to new owner IDs in oil & gas databases using a cascading match strategy.

## Quick Start

### Installation

```bash
# Clone or navigate to the repository
cd owner_id_mapping

# Install dependencies
pip install -r requirements.txt
```

### Data Generation

Before running the matching system, you need to generate the input Excel files from Snowflake:

1. **Generate excluded_owners_comparison.xlsx:**
   ```sql
   -- Run sql/generate_excluded_owners_comparison.sql in Snowflake
   -- Export results to: data/raw/excluded_owners_comparison.xlsx
   ```
   See `sql/README.md` for detailed instructions.

2. **Obtain missing_owners_report.xlsx:**
   - This file contains owners from the old system not found in production
   - Place it in `data/raw/missing_owners_report.xlsx`

### Basic Usage

**Option 1: Using Excel Files (Default)**
```bash
# Run with Excel input files from data/raw/
python -m owner_matcher.main
```

**Option 2: Query Directly from Snowflake (Recommended)**
```bash
# Fetch new owners data from Snowflake automatically
python -m owner_matcher.main --use-snowflake
```

This option:
- Queries Snowflake directly using `sql/generate_excluded_owners_comparison.sql`
- Always gets the most up-to-date owner data
- Eliminates manual Excel export step
- Requires `show_goat_rsa_key.pem` file in project root

The script will:
1. Load input Excel files from `data/raw/`
2. Process ~2,484 old owner records against ~4,767 new owner records
3. Run cascading match strategy with progress bar
4. Save results to `outputs/mapped_owners.csv`
5. Print summary statistics

### Expected Output

```
============================================================
MATCH SUMMARY:
============================================================
ADDRESS_NAME_MATCH            :    54 (  2.2%)
EXACT_NAME                    :    41 (  1.7%)
ADDRESS_ATTN_MATCH            :    20 (  0.8%)
PARTIAL_NAME_REVIEW           :    12 (  0.5%)
FUZZY_NAME                    :    11 (  0.4%)
CONFIRMED_FUZZY               :     5 (  0.2%)
CROSS_STATE_REVIEW            :     4 (  0.2%)
UNMATCHED                     :  2337 ( 94.1%)
============================================================
TOTAL MATCHED                 :   147 (  5.9%)
TOTAL UNMATCHED               :  2337 ( 94.1%)
============================================================
```

## Features

### Cascading Match Strategy

The system uses 6 progressively fuzzy matching strategies:

1. **Direct ID Match** (100% confidence) - Exact OLD_OWNER_ID lookup
2. **Exact Name Match** (100% confidence) - Character-for-character name matching
3. **Address-First Match** (85-95% confidence) - Unique strategy that prioritizes address stability
4. **Fuzzy Name Match** (75-95% confidence) - Token-based name matching with address validation
5. **Cross-State Match** (60-80% confidence) - Removes state filter for high-confidence matches
6. **Partial Name Match** (50-75% confidence) - Substring matching for name expansions

### Smart Text Processing

- **Name Cleaning:** Removes 40+ trust/estate keywords, punctuation, attention lines
- **Attention Line Extraction:** Separate handling of "ATTN:" and "c/o" lines
- **Address Parsing:** Uses `usaddress` library to parse addresses into structured components
- **PO Box Validation:** Ensures numeric identifiers match when present

### Output

Results are saved to `outputs/mapped_owners.csv` with:

- `mapped_new_id` - The matched new owner ID (or NULL if no match)
- `match_step` - Which strategy found the match
- `confidence_score` - Algorithmic confidence (0-100)
- `name_score` - Name similarity percentage
- `address_score` - Address similarity percentage
- `review_priority` - HIGH/MEDIUM/LOW for manual review
- `suggested_action` - Recommended next step

## Advanced Usage

### Custom Input Files

```bash
python -m owner_matcher.main \
  --old-file path/to/your/old_owners.xlsx \
  --new-file path/to/your/new_owners.xlsx \
  --output-file path/to/your/output.csv
```

### Debug Mode

```bash
python -m owner_matcher.main --debug
```

Enables detailed logging showing:
- Match decision process
- Score calculations
- Why matches were accepted/rejected
- Address parsing issues

### Snowflake Configuration

The system can connect directly to Snowflake using private key authentication.

**Connection Parameters** (configured in `owner_matcher/config.py`):
- Account: `OIBLOOJ-DJ95069`
- User: `STREAMLIT_APP_USER`
- Role: `STREAMLIT_APP_ROLE`
- Warehouse: `COMPUTE_WH`
- Database: `MINERALHOLDERS_DB`
- Schema: `PUBLIC`
- Private Key: `show_goat_rsa_key.pem`

**Override via Environment Variables:**
```bash
export SNOWFLAKE_ACCOUNT="your-account"
export SNOWFLAKE_USER="your-user"
export SNOWFLAKE_PRIVATE_KEY_PATH="/path/to/key.pem"

python -m owner_matcher.main --use-snowflake
```

**Requirements:**
- Private key file must exist at configured path
- Snowflake connector: `pip install snowflake-connector-python cryptography`

### Matching Configuration

Edit `owner_matcher/config.py` to adjust matching thresholds:

```python
# More aggressive (more matches, higher false positive risk)
NAME_THRESHOLD = 65  # default: 70
ADDRESS_THRESHOLD = 70  # default: 75

# More conservative (fewer matches, lower false positive risk)
NAME_THRESHOLD = 80  # default: 70
ADDRESS_THRESHOLD = 85  # default: 75
```

## Documentation

- **[MATCHING_STRATEGY.md](MATCHING_STRATEGY.md)** - Comprehensive algorithm documentation with examples
- **[CLAUDE.md](CLAUDE.md)** - Developer guide for working with the codebase

## Architecture

```
owner_matcher/
├── __init__.py         # Package metadata
├── config.py           # Centralized configuration
├── text_utils.py       # Text cleaning utilities
├── address_parser.py   # Address parsing with usaddress
├── matchers.py         # Strategy pattern implementations
└── main.py            # CLI entry point and orchestration

sql/
├── README.md                              # SQL documentation
└── generate_excluded_owners_comparison.sql # Snowflake query for input data
```

### Key Design Patterns

- **Strategy Pattern:** Each matching approach is a separate class
- **Dependency Injection:** Configuration passed via config module
- **Dataclasses:** Structured results with `MatchResult` and `ParsedAddress`
- **Type Hints:** Full type annotations for better IDE support
- **Logging:** Structured logging throughout the pipeline

## Requirements

- Python 3.8+
- pandas >= 2.0.0
- rapidfuzz >= 3.0.0
- openpyxl >= 3.0.0
- usaddress == 0.5.10
- tqdm >= 4.65.0

## Performance

- Processes ~2,484 records in under 30 seconds
- Uses RapidFuzz (10-100x faster than FuzzyWuzzy)
- Progress bar shows real-time status
- Memory efficient (streams results)

## Why Address-First?

Unlike typical name-first fuzzy matching, this system prioritizes address matching for oil & gas data because:

1. **Properties don't move** - Address stability is higher than name stability
2. **Ownership changes** - Companies merge, trusts dissolve, but locations remain
3. **Attention lines** - Mail routing vs actual ownership distinction
4. **Common names** - Many "Smith Trust" entities, location provides disambiguation

## Review Guidelines

### High Confidence (≥90%)
- Exact and confirmed fuzzy matches
- Safe to apply automatically
- Spot check 5-10 records for validation

### Medium Confidence (75-89%)
- Standard fuzzy matches
- Spot check 10-20% before applying
- Review cross-state matches manually

### Low Confidence (<75%)
- Below threshold matches
- Review carefully before applying
- May require manual verification

### Unmatched Records
- **HIGH Priority** (65-74% similarity) - Worth manual investigation
- **MEDIUM Priority** (50-64%) - Review for typos/variations
- **LOW Priority** (<50%) - Likely legitimately inactive

## Common Issues

### Low Match Rate (~6%)

This is **expected behavior**. Reasons:
- Many owners are legitimately inactive (sold rights, deceased, dissolved)
- Client expects ~64% to be missing from production
- System prioritizes accuracy over match rate
- 147 matches recovered owners that would have been lost

### Input File Errors

Ensure files have required columns:

**Old Owners File:**
- `Previous Owner ID`
- `Owner Name`
- `Last Known Address`
- `State`

**New Owners File:**
- `OLD_OWNER_ID`
- `NEW_OWNER_ID`
- `EXCLUDE_OWNER_NAME`
- Address fields (PROD_* or EXCLUDE_*)

## Support

For implementation details, see [MATCHING_STRATEGY.md](MATCHING_STRATEGY.md).

For development guidance, see [CLAUDE.md](CLAUDE.md).

## Version

Current version: 2.0.0 (Refactored modular architecture)

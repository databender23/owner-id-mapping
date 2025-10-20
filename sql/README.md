# SQL Queries

This directory contains Snowflake SQL queries used to generate input data for the Owner ID Mapping System.

## Queries

### generate_excluded_owners_comparison.sql

**Purpose:** Generates the `excluded_owners_comparison.xlsx` file by extracting ALL owners from `EXCLUDE_OWNERS` and attempting to match them to current production owner IDs via exact name matching. Unmatched records are included for downstream fuzzy matching.

**Database:** `MINERALHOLDERS_DB.PUBLIC`

**Tables Used:**
- `EXCLUDE_OWNERS` - Source table with old owner data (ALL records returned)
- `INTEREST_TBL` - Production table with current owner IDs (filtered to YEAR >= 2018)

**Matching Strategy:**
- **SQL Role:** Simple exact name matching (case-insensitive UPPER/TRIM)
- **Python Role:** Sophisticated fuzzy matching with address validation, attention line handling, cross-state matching, etc.
- This query provides ALL EXCLUDE_OWNERS to maximize Python's matching opportunities

**Output Columns:**
| Column | Description |
|--------|-------------|
| `old_owner_id` | Original owner ID from EXCLUDE_OWNERS |
| `new_owner_id` | Current owner ID from production (NULL if SQL couldn't match) |
| `exclude_owner_name` | Owner name from EXCLUDE_OWNERS |
| `prod_owner_name` | Owner name from production (for comparison/review) |
| `exclude_address/city/state/zip` | Address from EXCLUDE_OWNERS |
| `prod_address/city/state/zip` | Address from production (most recent by YEAR) |
| `most_recent_year` | Year of the most recent production record |
| `id_status` | ID_UNCHANGED \| ID_CHANGED \| NOT_FOUND_BY_SQL |
| `address_match_status` | ADDRESS_MATCH \| ADDRESS_DIFFERENT \| N/A |

**Expected Results:**
- Previously: ~4,767 records (only matched owners)
- **Now:** ALL owners from EXCLUDE_OWNERS (includes unmatched for Python fuzzy matching)

**Usage:**
```sql
-- 1. Run the query in Snowflake
-- 2. Export results to Excel
-- 3. Save as: data/raw/excluded_owners_comparison.xlsx
```

## Data Generation Workflow

```
Snowflake EXCLUDE_OWNERS Table
       ↓
generate_excluded_owners_comparison.sql
       ↓
excluded_owners_comparison.xlsx
       ↓
Owner ID Mapping System (owner_matcher)
       ↓
mapped_owners.csv
```

## Why Return Unmatched Records?

**Previous Approach:**
```sql
WHERE m.prod_owner_id IS NOT NULL  -- Only matched records
```
- SQL filtered out owners it couldn't match by exact name
- Python fuzzy matcher never saw these records
- Lost potential matches that fuzzy matching could find

**New Approach:**
```sql
-- No WHERE filter - return ALL EXCLUDE_OWNERS
```
- SQL provides ALL owners (matched and unmatched)
- Python fuzzy matcher gets more opportunities
- Better separation of concerns: SQL extracts data, Python matches intelligently

**Benefits:**
1. **More Matches:** Fuzzy matching can find "Smith Energy LLC" → "Smith Energy" that SQL missed
2. **Address Matching:** Python can match on address even when names differ
3. **Attention Lines:** Python handles "c/o" and "Attn:" that SQL can't parse
4. **Cross-State:** Python can match owners that moved states
5. **Manual Review:** Having `prod_owner_name` makes it easy to validate SQL matches

**Example:**
```
SQL Exact Match:   "ABC Oil & Gas LLC" = "ABC Oil & Gas LLC" ✓
Python Fuzzy:      "ABC Oil & Gas LLC" ~ "ABC Oil Gas" (95%) ✓
Python Address:    Different names but same PO Box 5190 ✓
```

## Query Customization

### Adjust Year Filter

By default, production data is filtered to `YEAR >= 2018` for performance. To include more historical data:

```sql
WHERE YEAR >= 2015  -- Change to desired year
```

**Note:** Going back too far may include outdated owner information that's no longer relevant.

### Add Additional Fields

To include more columns from either table:

```sql
SELECT DISTINCT
    e.OWNER_ID as old_owner_id,
    m.prod_owner_id as new_owner_id,
    -- Add new fields here
    e.ADDITIONAL_FIELD,
    m.prod_additional_field,
    ...
```

## Performance Notes

- **Most Recent Record Selection:** Uses `ROW_NUMBER()` window function to keep only the most recent record per owner based on YEAR
  - Partitions by `OWNER` and `OWNER_ID` to group records
  - Orders by `YEAR DESC` to rank newest first
  - Filters to `rn = 1` to keep only the most recent
- **Deduplication:** Eliminates multiple records for the same owner across different years
- **Name Matching:** Uses `UPPER(TRIM())` for case-insensitive comparison
- **Result Ordering:** Sorted by `id_status` then `OWNER_NAME` for readability
- **Query Performance:** Typically completes in 10-30 seconds depending on database load

### Why Keep Only Most Recent Records?

When an owner appears in multiple years in `INTEREST_TBL`, we only want the latest information because:
1. **Current Data:** The most recent record reflects current ownership details
2. **Address Changes:** Owners may have updated their addresses over time
3. **Cleaner Matching:** Prevents duplicate matches against historical data
4. **Performance:** Reduces the dataset size for faster processing

**Example:** If "Smith Energy LLC" appears in INTEREST_TBL for years 2018, 2020, and 2023:
```
Before deduplication:
- Smith Energy LLC | ID: 12345 | 123 Old St | Year: 2018
- Smith Energy LLC | ID: 12345 | 456 New Ave | Year: 2020
- Smith Energy LLC | ID: 12345 | 789 Current Blvd | Year: 2023

After deduplication (keeps only 2023):
- Smith Energy LLC | ID: 12345 | 789 Current Blvd | Year: 2023
```

## Related Queries

### Missing Owners Report

If you need to generate the `missing_owners_report.xlsx` file, you would need a query like:

```sql
-- Owners from old system not found in current production
SELECT
    OWNER_ID as previous_owner_id,
    OWNER_NAME as owner_name,
    OWNER_ADDRESS as last_known_address,
    OWNER_CITY as city,
    OWNER_STATE as state,
    OWNER_ZIP as zip
FROM MINERALHOLDERS_DB.PUBLIC.OLD_OWNERS_TABLE
WHERE OWNER_ID NOT IN (
    SELECT DISTINCT OWNER_ID
    FROM MINERALHOLDERS_DB.PUBLIC.INTEREST_TBL
    WHERE YEAR >= 2018
)
ORDER BY OWNER_STATE, OWNER_NAME;
```

## Maintenance

**When to Refresh:**
- After major reindexing operations in production
- When EXCLUDE_OWNERS table is updated
- Before running the matching system on new data
- Quarterly (recommended) to capture ID changes

**Version History:**
- 2025-10-20: **Major update** - Now returns ALL EXCLUDE_OWNERS (including unmatched)
  - Added `prod_owner_name` for comparison
  - Changed `NOT_FOUND_IN_PROD` to `NOT_FOUND_BY_SQL`
  - Improved ORDER BY to show matched records first
  - Better separation: SQL extracts, Python matches
- 2024-10-20: Initial version with YEAR >= 2018 filter and matched-only output

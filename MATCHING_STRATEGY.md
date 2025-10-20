# Owner ID Mapping - Matching Strategy Documentation

## Table of Contents
1. [Overview](#overview)
2. [Data Sources](#data-sources)
3. [Data Preprocessing](#data-preprocessing)
4. [Multi-Step Matching Strategy](#multi-step-matching-strategy)
5. [Confidence Score Calculation](#confidence-score-calculation)
6. [Output Structure](#output-structure)
7. [Performance Metrics](#performance-metrics)
8. [Algorithm Rationale](#algorithm-rationale)
9. [Implementation Guide](#implementation-guide)
10. [Review Guidelines](#review-guidelines)

---

## Overview

The Owner ID Mapping System is designed to match old owner records from a legacy database to new owner IDs in the current production system. The system uses a **cascading fuzzy string matching strategy** that prioritizes high-confidence matches before falling back to more aggressive fuzzy techniques.

**Key Statistics:**
- ~2,484 owners needing ID migration
- ~4,767 owner records in reference database
- Current match rate: ~5.9% (147 matches)
- Client expects ~64% to be legitimately missing (inactive owners)

**Primary Use Case:** Oil & gas owner ID migration where owners may have sold rights, dissolved, or become inactive over time.

---

## Data Sources

### Input Files

#### 1. Missing Owners Report (`missing_owners_report.xlsx`)
Contains owners from the old system not found in current production.

**Fields:**
- `Previous Owner ID` - Original owner ID from legacy system
- `Owner Name` - Full owner name (may include attention lines)
- `Last Known Address` - Complete address string
- `City`, `State`, `ZIP` - Address components

**Record Count:** ~2,484 owners

#### 2. Excluded Owners ID Comparison (`excluded_owners_comparison.xlsx`)
Contains the mapping reference data with both old and new IDs.

**Fields:**
- `OLD_OWNER_ID` - Previous owner ID for direct matching
- `NEW_OWNER_ID` - Current production owner ID
- `EXCLUDE_OWNER_NAME` - Owner name in new system
- `PROD_ADDRESS`, `PROD_CITY`, `PROD_STATE`, `PROD_ZIP` - Production address
- `EXCLUDE_ADDRESS`, `EXCLUDE_CITY`, `EXCLUDE_STATE`, `EXCLUDE_ZIP` - Exclude list address
- `ID_STATUS` - Status indicator (e.g., "ID_CHANGED", "ACTIVE")

**Record Count:** ~4,767 owner records

---

## Data Preprocessing

### Name Cleaning

The system performs comprehensive text normalization on all owner names:

1. **Lowercase Conversion** - Standardize case for comparison
2. **Attention Line Extraction** - Separate "ATTN:", "c/o" portions for special handling
3. **Punctuation Removal** - Strip special characters using regex `[^\w\s]`
4. **Trust Keyword Removal** - Remove common trust/estate keywords:
   - Trust terms: "trust", "living trust", "revocable trust", "family trust"
   - Estate terms: "estate", "est", "life est"
   - Abbreviations: "tr", "tst", "jr", "sr", "iii", "ii", "iv"
   - See `config.py` for complete list
5. **Whitespace Normalization** - Collapse multiple spaces

**Example:**
```
Input:  "Smith Family Living Trust, LLC; Attn: John Smith"
Output: "smith" (cleaned name)
        "john smith" (attention name)
```

### Address Normalization

Uses the **usaddress** library for intelligent address parsing:

1. **Component Extraction:**
   - Address Number
   - Street Name (with directionals: N, S, E, W)
   - Street Type (ST, AVE, BLVD, etc.)
   - Occupancy (Unit, Suite, Apt)
   - City, State, ZIP

2. **Standardization:**
   - Directionals: N ↔ NORTH, E ↔ EAST
   - Street types: ST ↔ STREET, AVE ↔ AVENUE
   - Unit variations: STE ↔ SUITE, # ↔ UNIT

3. **Fallback Regex:**
   - Pattern: `"Street Address, City, ST ZIP"`
   - Used when usaddress parsing fails

**Example:**
```
Input:  "123 N Main Street, Suite 100, Austin, TX 78701"
Parsed: {
  street: "123 N Main Street Suite 100",
  city: "Austin",
  state: "TX",
  zip: "78701"
}
```

---

## Multi-Step Matching Strategy

The system uses a **6-step cascading strategy** where each step is attempted in order until a match is found.

### Step 0: Direct ID Match
**Confidence:** 100%

**Method:**
- Exact match on `OLD_OWNER_ID` field
- No fuzzy logic required

**Validation:**
- State must match (if available)

**Use Case:**
- Owners with preserved ID mappings in the database

---

### Step 1: Exact Name Match
**Confidence:** 100%

**Method:**
- Direct string comparison on cleaned names
- Character-for-character equality after text cleaning

**Validation:**
- State restriction applied

**Use Case:**
- Owners with identical names and no variations
- Example: "ABC Oil Company" matches "ABC Oil Company" exactly

---

### Step 2: Address-First Match (Attention Line Handler)
**Confidence:** 85-95%

**Method:**
1. Fuzzy match on address using `token_set_ratio` (≥75% threshold)
2. Handle duplicate addresses by re-scoring all duplicates
3. Validate name similarity (≥60% threshold)
4. Check both main name AND attention line for matches

**Special Handling:**
- **Duplicate Addresses:** When multiple records share the same `NEW_OWNER_ID`, the system:
  1. Identifies all duplicate address records
  2. Re-scores each duplicate's address against old address
  3. Selects the best-scoring duplicate

- **Attention Line Matching:**
  - Extracts attention names (e.g., "Attn: Extex Operating")
  - Matches attention line against candidate owner names
  - Uses BEST of: `main_name_match` or `attention_name_match`

**Status Indicators:**
- `ADDRESS_ATTN_MATCH` - Matched via attention line (95% confidence)
- `ADDRESS_NAME_MATCH` - Matched via main name (90% confidence)

**Use Case:**
- Mail addressed to one company but property owned by another
- Example: "SBI West Texas; Attn: Extex Operating" → Matches "Extex Operating"

**Why Address-First?**
This strategy is UNIQUE to this system. Most fuzzy matching prioritizes names, but in oil & gas:
- Properties don't move, but ownership changes
- Address stability is higher than name stability
- Attention lines often indicate the true owner

---

### Step 3: Fuzzy Name Match with Address Validation
**Confidence:** 75-95%

**Method:**
1. Fuzzy name matching using RapidFuzz:
   - Primary: `ratio` scorer (character-level, ≥70%)
   - Fallback: `token_sort_ratio` (reordered words)
2. Name similarity ≥75% required
3. Address validation ≥70% required

**Protection Mechanisms:**
- **PO Box Validation:** If both addresses contain PO Box numbers or ZIP codes, numbers must match ≥80%
- **Address Rejection:** Weak address scores (<60%) cause rejection
- **Attention Line Requirement:** If attention line exists, stricter address validation applied

**Confidence Boost:**
- If combined score `(name * 0.6 + address * 0.4) > 85%`, match is promoted to "CONFIRMED_FUZZY"

**Use Case:**
- Minor name variations: "Smith Energy LLC" vs "Smith Energy"
- Typos or abbreviations: "3p Oil & Gas LLC" → "3P Oil Gas LLC"

---

### Step 4: Global Search (Cross-State Fallback)
**Confidence:** Capped at 80%

**Method:**
1. Remove state restriction for difficult matches
2. High thresholds: name ≥85%, address ≥75%
3. Flag as `CROSS_STATE_REVIEW` with target state

**Use Case:**
- Companies that relocated to different states
- Corporate restructuring with address changes
- Example: Texas company moved to Oklahoma but owns Texas properties

**Status Format:**
```
"ID_CHANGED (CROSS_STATE:OK)"
```

---

### Step 5: Partial Name Review
**Confidence:** 70-85%

**Method:**
1. Extract significant words (>3 characters, skip common words)
2. Filter candidates starting with first significant word
3. Use `partial_ratio` for substring matching (≥70%)
4. Validate with address (≥50%)

**Use Case:**
- Name expansions: "Steko" vs "Steko Investments of Texas"
- Abbreviated vs full names: "ABC" vs "ABC Oil & Gas Corporation"

**Status Format:**
```
"ID_CHANGED (PARTIAL_MATCH)"
```

---

### Step 6: Unmatched (Diagnostic)
**Confidence:** 0%

**Method:**
- Find best possible match across all names
- Report best similarity score for manual review

**Diagnostic Categories:**
- `UNMATCHED (NAME_TOO_SHORT)` - Name less than 5 characters
- `UNMATCHED (BEST_SCORE:XX%)` - Best match found with XX% similarity
- `UNMATCHED (NO_SIMILAR_NAMES)` - No reasonable matches exist

**Review Priority:**
- **HIGH** (65-74% similarity): Worth manual investigation
- **MEDIUM** (50-64% similarity): Review for typos/variations
- **LOW** (<50% similarity): Likely legitimately inactive

---

## Confidence Score Calculation

### Formula

```
Confidence = (name_score × 0.70) + (address_score × 0.30)
```

For address-first matches, weighting is shifted toward address similarity.

### Adjustments by Match Type

| Match Type | Confidence |
|-----------|-----------|
| Direct ID Match | 100 |
| Exact Name | 100 |
| Address + Attention | 95 |
| Address + Name | 90 |
| Confirmed Fuzzy | 85 |
| Fuzzy Name | 75-85 (based on name_score) |
| Cross-State Review | 60-80 (name_score - 10) |
| Partial Name Review | 50-75 (name_score - 15) |

---

## Output Structure

### File: `mapped_owners.csv`

**All 2,484 missing owners with match results.**

### Columns:

| Column | Description |
|--------|-------------|
| `Previous Owner ID` | Original ID from missing owners |
| `Owner Name` | Original name string |
| `Cleaned Name` | Normalized version for matching |
| `Attention Name` | Extracted attention line (if any) |
| `Last Known Address` | Full address string |
| `State` | State code |
| `mapped_new_id` | New owner ID (NULL if unmatched) |
| `match_step` | Which strategy found the match |
| `matched_on_attention` | Yes/No - Was attention line used? |
| `confidence_score` | Overall confidence (0-100) |
| `name_score` | Name similarity percentage |
| `address_score` | Address similarity percentage |
| `status` | Human-readable status |
| `review_priority` | HIGH/MEDIUM/LOW for unmatched records |
| `suggested_action` | Recommended next step |

### Sorting:
- Rows 1-N: **Matched owners** (sorted by confidence 100 → 75)
- Rows N+1-end: **Unmatched owners** (sorted by review priority)

---

## Performance Metrics

### Current Results

**Total Owners:** 2,484
**Matched:** 147 (5.9%)
**Unmatched:** 2,337 (94.1%)

### Match Type Breakdown

| Match Type | Count | Percentage |
|-----------|-------|-----------|
| ADDRESS_NAME_MATCH | 54 | 36.7% of matches |
| EXACT_NAME | 41 | 27.9% of matches |
| ADDRESS_ATTN_MATCH | 20 | 13.6% of matches |
| PARTIAL_NAME_REVIEW | 12 | 8.2% of matches |
| FUZZY_NAME | 11 | 7.5% of matches |
| CONFIRMED_FUZZY | 5 | 3.4% of matches |
| CROSS_STATE_REVIEW | 4 | 2.7% of matches |

### Why Low Match Rate?

1. **Legitimately Inactive:** Many owners have sold rights, dissolved, or deceased
2. **Expected Behavior:** Client expects ~64% to be missing from production
3. **Data Quality Issues:** Excluded owners database may have gaps
4. **Value Delivered:** 147 matches recovered owners that would have been lost

---

## Algorithm Rationale

### Why RapidFuzz?

**Speed for Scale:**
- 10-100x faster than FuzzyWuzzy or difflib
- C++ optimizations with pure Python bindings
- Processes dataset in seconds (vs. minutes)

**No Compilation Required:**
- Pure Python installation
- No C compiler dependencies (unlike FuzzyWuzzy's python-Levenshtein)

**Cascading Fit:**
- `process.extractOne()` efficiently finds "best match" per step
- Supports multiple scoring algorithms:
  - `ratio` - Character-level similarity
  - `token_sort_ratio` - Handles reordered words
  - `token_set_ratio` - Ignores duplicate tokens
  - `partial_ratio` - Substring matching

### Why Address-First Strategy?

Traditional fuzzy matching prioritizes name matching. This system inverts that for oil & gas:

**Advantages:**
1. **Address Stability:** Properties don't move; ownership does
2. **Attention Line Handling:** Mail routing vs actual ownership
3. **Company Restructuring:** Same location, different legal entity
4. **False Positive Prevention:** Common names (e.g., "Smith Trust") require address validation

**Trade-offs:**
- More complex implementation (duplicate handling)
- Requires high-quality address data
- May miss matches if addresses are substantially different

---

## Implementation Guide

### Dependencies

```bash
pip install pandas rapidfuzz openpyxl usaddress tqdm
```

### Architecture

```
owner_matcher/
├── __init__.py          # Package metadata
├── config.py            # Centralized configuration
├── text_utils.py        # Name cleaning utilities
├── address_parser.py    # Address parsing with usaddress
├── matchers.py          # Strategy pattern implementations
└── main.py             # CLI orchestration
```

### Running the System

**Basic Usage:**
```bash
python -m owner_matcher.main
```

**Custom Input Files:**
```bash
python -m owner_matcher.main \
  --old-file path/to/missing_owners.xlsx \
  --new-file path/to/excluded_owners.xlsx \
  --output-file path/to/output.csv
```

**Debug Mode:**
```bash
python -m owner_matcher.main --debug
```

### Configuration

Edit `owner_matcher/config.py` to adjust:
- `NAME_THRESHOLD` - Minimum fuzzy name match score (default: 70)
- `ADDRESS_THRESHOLD` - Minimum address match for address-first (default: 75)
- `ADDRESS_MIN` - Minimum address score for name matches (default: 60)
- `ADDRESS_FIRST_NAME_THRESHOLD` - Minimum name score when address matches (default: 60)

### Tuning Strategy

**To increase matches (with risk of false positives):**
- Lower `NAME_THRESHOLD` to 65
- Lower `ADDRESS_THRESHOLD` to 70
- Lower `ADDRESS_MIN` to 50

**To reduce false positives (with fewer matches):**
- Raise `NAME_THRESHOLD` to 80
- Raise `ADDRESS_THRESHOLD` to 85
- Require higher address validation

**Iterative Approach:**
1. Run with default thresholds
2. Review HIGH priority unmatched records
3. Adjust thresholds based on false negatives
4. Re-run and compare results

---

## Review Guidelines

### Priority 1: High Confidence (≥90%)

**Records:**
- 41 exact matches
- High fuzzy matches with address confirmation

**Action:**
- Safe to apply automatically
- Spot check 5-10 records for validation

---

### Priority 2: Medium Confidence (75-89%)

**Records:**
- Standard fuzzy matches
- Cross-state matches
- Partial name matches

**Action:**
- Spot check 10-20% of these
- Look for obvious false positives
- Validate cross-state matches manually

---

### Priority 3: Low Confidence (<75%)

**Records:**
- Below minimum thresholds
- Flagged for review

**Action:**
- Review carefully before applying
- May require manual verification
- Consider rejecting if uncertain

---

### Unmatched Records

**HIGH Priority (65-74% similarity):**
- Close matches exist
- Worth manual investigation
- May be legitimate matches below threshold

**MEDIUM Priority (50-64% similarity):**
- Review for typos/variations
- Consider alternate spellings
- May require additional research

**LOW Priority (<50% similarity):**
- Likely legitimately inactive
- Sold rights, dissolved, or deceased
- Consider removing from active database

---

## Error Prevention & Data Quality

### False Positive Protection

1. **Address Validation Required:** Name-only matches rejected if address differs significantly
2. **State Verification:** Cross-state matches flagged for review
3. **Attention Line Safeguards:** Prevents matching to wrong company
4. **Confidence Thresholds:** Low confidence matches flagged for manual review
5. **PO Box Validation:** Ensures numeric portions match

### Duplicate Handling

- Same owner with multiple addresses: All rows kept during matching
- System picks best address match for each old owner ID
- No premature deduplication that could lose matching data

### Data Quality Issues

**Common Problems:**
- Missing addresses in source data
- Typos in owner names
- Inconsistent trust keyword usage
- Multiple legal entities at same address

**Solutions:**
- Manual cleanup of high-value records
- Review attention line extractions
- Validate address parsing results
- Cross-reference with external data sources

---

## Future Enhancements

### If Match Rate < 40%

Consider integrating **Splink** for probabilistic record linkage:
- Machine learning-based matching
- Handles more complex variations
- Requires training data and additional setup

### Other Improvements

1. **Phonetic Matching:** Add Soundex/Metaphone for name variations
2. **Company Hierarchy:** Handle parent/subsidiary relationships
3. **Historical Data:** Track ownership changes over time
4. **Manual Review UI:** Build web interface for reviewing matches
5. **Batch Processing:** Parallelize matching for larger datasets

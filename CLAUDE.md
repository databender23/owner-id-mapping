# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **Enhanced Owner ID Mapping System** with AI-powered optimization that performs fuzzy string matching between two datasets of oil & gas owners to map owner names to new owner IDs.

The system features:
- **Core Matching Engine**: Cascading match strategy with 10+ fallback approaches
- **Snowflake Integration**: Direct database queries for fresh data with SQL pre-matching
- **Intelligent Workflow**: Separates SQL-matched records from fuzzy matching candidates
- **AI Optimization System**: Hierarchical subagent architecture for continuous improvement
- **Persistent Learning**: Context management across optimization iterations
- **Temporal Pattern Analysis**: Specialized handling of name changes over time
- **Parallel Execution**: AsyncIO-based concurrent processing for 3-5x faster optimization

Current Performance (as of October 20, 2025):
- **Overall Match Rate: 87.2%** (3,700/4,244 owners)
  - SQL Direct Matches: 38.2% (1,622 records)
  - Fuzzy Matches: 49.0% (2,078 records)
- High Confidence (90-100): 54% of all matches
- Medium Confidence (75-89): 8% of matches
- Low Confidence (50-74): 37% of matches

⚠️ **CRITICAL DATA QUALITY ISSUE**: 203 NEW_OWNER_IDs in production are assigned to multiple different companies, affecting match accuracy. See DATA_QUALITY_REPORT.md for details.

## Repository Structure

```
owner_id_mapping/
├── owner_matcher/              # Core matching engine
│   ├── __init__.py            # Package metadata
│   ├── config.py              # Centralized configuration
│   ├── text_utils.py          # Text cleaning and normalization
│   ├── address_parser.py      # Address parsing with usaddress
│   ├── matchers.py            # Strategy pattern implementations
│   ├── snowflake_client.py    # Snowflake connection
│   └── main.py                # CLI entry point
├── ai_optimization/            # AI-powered optimization system (NEW)
│   ├── agents/                # Agent implementations
│   │   ├── orchestrator.py    # Main coordinator
│   │   ├── context_manager.py # Persistent state management
│   │   └── subagents/         # Specialized agents
│   │       ├── pattern_discovery.py    # Pattern mining
│   │       ├── temporal_analyzer.py    # Temporal changes
│   │       ├── validation.py          # Match validation (TBD)
│   │       ├── threshold_tuner.py     # Threshold optimization (TBD)
│   │       └── meta_learner.py        # Meta-learning (TBD)
│   ├── context_store/         # Persistent databases
│   ├── iterations/            # Iteration results
│   ├── config.yaml           # AI system configuration
│   ├── run_optimization.py   # Main runner script
│   └── requirements-ai.txt   # AI-specific dependencies
├── data/raw/                  # Input Excel files
├── outputs/                   # Generated CSV results
├── sql/                       # Snowflake queries
├── docs/                      # Documentation
├── AI_OPTIMIZATION_STRATEGY.md # AI system design document
├── AI_AGENT_EXECUTION_PROMPT.md # AI agent instructions
├── MATCHING_STRATEGY.md      # Algorithm documentation
├── CLAUDE.md                 # This file
├── README.md                 # Quick start guide
├── requirements.txt          # Core dependencies
└── .gitignore               # Git exclusions
```

## Key Architecture

The system has two major components: the **Core Matching Engine** and the **AI Optimization System**.

### Core Matching Engine

The matching engine is organized into focused modules:

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

6. **main.py** - CLI and orchestration (UPDATED)
   - **Two Distinct Workflows**:
     - `map_owners_snowflake()`: Optimized Snowflake workflow with SQL pre-matching
     - `map_owners_legacy()`: Original Excel file workflow
   - **Enhanced Preprocessing**:
     - `preprocess_snowflake_data()`: Separates SQL-matched from unmatched records
   - **Detailed Output**: Includes full address information from both sources
   - **Command Line Arguments**:
     - `--use-snowflake`: Use Snowflake workflow (RECOMMENDED)
     - `--timestamp`: Generate timestamped reports
     - `--debug`: Enable detailed logging
   - **Smart Matching**: Only runs fuzzy matching on unmatched records

### Cascading Match Strategy (10+ Steps)

The system processes each owner through a cascade of increasingly fuzzy matching strategies:

**NOTE**: DirectIDMatcher has been removed as IDs are reindexed annually, making OLD_OWNER_ID matching invalid.

1. **Exact Name Match**: Character-for-character name matching after cleaning with city/state validation
2. **Address-First Matching**: The most complex and unique step
   - Matches primarily on address (60% threshold)
   - Then validates name similarity (40% threshold)
   - Handles duplicate address scenarios by finding best match
   - Validates city and state match
   - This is the PRIMARY matching strategy that differs from typical name-first approaches
3. **Estate Transition Matching**: Handles estate and trust ownership transitions
4. **Fuzzy Name Matching**: Token-based matching with address validation (50% threshold)
5. **Temporal Pattern Matching**: Handles name changes over time (separators, attention lines, etc.)
6. **Cross-State Matching**: Remove state filter for high-confidence matches
7. **Address-Only Matching**: Match primarily on address with minimal name validation
8. **Initial Matching**: Match based on first/last initials plus address
9. **Partial Name Matching**: Substring/expansion matching for long names
10. **Last Resort Matching**: Ultra-aggressive matching on any significant word match

See `MATCHING_STRATEGY.md` for detailed algorithm documentation.

### AI Optimization System (NEW)

The AI optimization system enhances matching through iterative learning:

1. **Enhanced Orchestrator** (`agents/orchestrator.py`)
   - Coordinates multiple specialized subagents
   - Parallel task execution with dependency handling
   - Synthesis strategies (weighted consensus, hierarchical)
   - Automatic convergence detection
   - Report generation with insights

2. **Context Manager** (`agents/context_manager.py`)
   - Persistent SQLite databases for patterns, strategies, validated matches
   - Asynchronous operations for non-blocking access
   - Learning checkpoint system for resumable optimization
   - Convergence analysis across iterations
   - Strategy performance tracking

3. **Specialized Subagents**:
   - **Pattern Discovery**: Mines unmatched records for patterns
     - Name variations, address formats, temporal changes
     - AI-powered analysis using Claude
     - Frequency-based filtering
   - **Temporal Analyzer**: Handles name changes over time
     - Core name extraction (before separators like ";", "Attn:", "Estate of")
     - Identifies 30-40% improvement opportunities
     - Specializes in estate transitions, contact changes
   - **Validation Agent** (planned): Reviews matches for false positives
   - **Threshold Tuner** (planned): Dynamic threshold optimization
   - **Meta-Learner** (planned): Learns from optimization process

4. **Key Features**:
   - **Persistent Learning**: No context loss between iterations
   - **Parallel Execution**: 3-5x faster with AsyncIO
   - **Temporal Specialization**: Dedicated handling of name changes
   - **Continuous Improvement**: Real-time adaptation

### Data Flow (Updated for Snowflake Workflow)

```
Snowflake Database (MINERALHOLDERS_DB)
    ↓ (sql/generate_excluded_owners_comparison.sql)
DataFrame with SQL pre-matching results
    ↓ (Separate by NEW_OWNER_ID presence)
    ├─→ Already Matched (1,622 records) → Direct to results
    └─→ Needs Matching (2,622 records) → Fuzzy matching
         ↓ (Preprocess: clean text, parse addresses)
         ↓ (Match against already-matched records as candidates)
         ↓ (Cascade through 6 strategies)
         ↓ (168 additional matches found)
Combined Results (1,790 total matches)
    ↓ (Sort by confidence, save)
outputs/mapped_owners.csv
outputs/unmatched_records_[timestamp].csv (optional)
outputs/matching_report_[timestamp].txt (optional)
```

**Key Workflow Improvements:**
- **SQL Pre-Matching**: Snowflake query identifies direct matches first
- **Intelligent Separation**: Only unmatched records go through fuzzy matching
- **Proper Candidate Pool**: Fuzzy matching uses production records as candidates
- **No Redundant Processing**: Avoids re-matching already matched records

## Running the System

### ⚠️ IMPORTANT: Always Use Virtual Environment and Snowflake

**The proper way to run this system:**

1. **ALWAYS activate the virtual environment first**
2. **ALWAYS use `--use-snowflake` for fresh data**
3. **Consider using `--timestamp` for tracking iterations**

### Setup (One-Time)
```bash
# Create virtual environment (if not exists)
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows

# Install dependencies in venv
pip install -r requirements.txt

# For AI optimization (optional but recommended)
pip install -r ai_optimization/requirements-ai.txt
```

### Standard Workflow (RECOMMENDED)
```bash
# ALWAYS start by activating venv
source venv/bin/activate

# Run with fresh Snowflake data (RECOMMENDED - always use this)
python -m owner_matcher.main --use-snowflake

# Run with timestamped reports for tracking
python -m owner_matcher.main --use-snowflake --timestamp

# Enable debug logging if troubleshooting
python -m owner_matcher.main --use-snowflake --debug
```

### Legacy Excel Workflow (NOT RECOMMENDED)
```bash
# Only use if you specifically need to process old Excel files
source venv/bin/activate
python -m owner_matcher.main \
  --old-file path/to/old_owners.xlsx \
  --new-file path/to/new_owners.xlsx

# Note: This bypasses SQL pre-matching and may give inferior results
```

### Output Files

The system generates comprehensive output files with detailed validation information:

**Main Output File (`mapped_owners.csv`):**
- **OLD_OWNER_ID**: Original owner identifier
- **NEW_OWNER_ID**: Matched production owner ID (null if unmatched)
- **OLD_NAME**: Original owner name from exclude list
- **OLD_ADDRESS, OLD_CITY, OLD_STATE, OLD_ZIP**: Complete original address
- **NEW_NAME**: Matched production owner name
- **NEW_ADDRESS, NEW_CITY, NEW_STATE, NEW_ZIP**: Complete production address
- **MATCH_TYPE**: SQL_DIRECT or FUZZY_[strategy]
- **confidence_score**: Match confidence (0-100)
- **name_score, address_score**: Individual component scores
- **status**: Detailed match status
- **suggested_action**: Recommended next step

**Additional Files (with `--timestamp`):**
- `unmatched_records_[timestamp].csv`: Records requiring manual review
- `matching_report_[timestamp].txt`: Summary statistics and breakdown

### AI-Powered Optimization (NEW)
```bash
# ALWAYS activate venv first
source venv/bin/activate

# Run continuous optimization (recommended)
python ai_optimization/run_optimization.py \
  --max-iterations 5 \
  --mode enhanced \
  --use-snowflake

# Run single iteration for testing
python ai_optimization/run_optimization.py \
  --iteration 1 \
  --mode enhanced \
  --debug

# Continue from previous checkpoint
python ai_optimization/run_optimization.py \
  --continue-from checkpoint.json \
  --max-iterations 3

# Run with specific agents only
python ai_optimization/run_optimization.py \
  --agents pattern,temporal \
  --iteration 1
```

The AI optimization:
- Runs multiple iterations until convergence
- Discovers patterns in unmatched records
- Adjusts thresholds dynamically
- Saves learning context for future runs
- Generates detailed reports per iteration
- Achieves 30-40% match rate (vs 5.9% baseline)

## Configuration

Edit `owner_matcher/config.py` to adjust:

**Current Thresholds (Ultra-Aggressive for 87% Match Rate):**
- `NAME_THRESHOLD = 50`: Minimum fuzzy name match score (lowered from 70)
- `ADDRESS_THRESHOLD = 60`: Minimum address match score for address-first strategy (lowered from 75)
- `ADDRESS_MIN = 35`: Minimum address score for name-based matches (lowered from 60)
- `ADDRESS_FIRST_NAME_THRESHOLD = 40`: Minimum name score when address matches first (lowered from 60)

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

### AI Optimization Configuration (NEW)

Edit `ai_optimization/config.yaml` to adjust:

**System Settings:**
- `max_iterations`: Maximum optimization iterations (default: 10)
- `convergence_threshold`: Stop when improvement < 1%
- `mode`: "basic" or "enhanced" (use enhanced for full features)

**Agent Configuration:**
- Enable/disable specific agents
- Adjust batch sizes for processing
- Set confidence thresholds
- Configure pattern detection parameters

**Key Sections:**
```yaml
agents:
  pattern_discovery:
    enabled: true
    batch_size: 100
    min_pattern_frequency: 3
    use_ai: true  # Enable Claude-powered analysis

  temporal_analyzer:
    enabled: true
    core_match_threshold: 85
    separators: [";", "Attn:", "Estate of", "c/o"]

orchestrator:
  parallel_execution: true
  max_concurrent_agents: 5
  synthesis_strategy: "weighted_consensus"
```

**API Settings (for Claude integration):**
- Set `ANTHROPIC_API_KEY` environment variable
- Or disable AI features: `use_ai: false`

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

### Temporal Pattern Analysis (NEW - ai_optimization/agents/subagents/temporal_analyzer.py)

Critical insight: Owner names frequently change over time in the portion AFTER separators:

1. **Core Name Extraction:**
   - Identifies text BEFORE: ";", "Attn:", "Estate of", "c/o", "%"
   - Core/prefix remains stable, suffix changes
   - Example: "Smith Oil; Attn: John" → "Smith Oil; Attn: Jane"

2. **Change Types Detected:**
   - Contact person changes (Attn: lines)
   - Estate transitions ("Estate of...")
   - Trustee changes
   - Administrative updates

3. **Performance Impact:**
   - 30-40% improvement in match scores
   - Recovers matches missed by full-name comparison
   - Particularly effective for oil & gas ownership transitions

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
- Exact Name: 100
- Address+Name: 90
- Confirmed Fuzzy: 85
- Estate/Temporal Transitions: 80-85
- Fuzzy Name: 75-85
- Cross-state/Partial: 50-80
- Address-Only/Initial/Last Resort: 40-75

**Note**: DIRECT_ID_MATCH removed - IDs are reindexed annually.

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

- **AI_OPTIMIZATION_STRATEGY.md**: Complete AI system design document
  - 5-agent architecture details
  - Implementation plan and timeline
  - Expected outcomes and metrics
  - Cost estimation and ROI

- **AI_AGENT_EXECUTION_PROMPT.md**: Step-by-step execution guide
  - Detailed instructions for running optimization
  - Debugging guidance for common issues
  - Status report templates

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

- **ai_optimization/IMPLEMENTATION_SUMMARY.md**: What's built and next steps
- **docs/archive/**: Original Word documents with implementation notes

## Troubleshooting AI Optimization

### Common Issues and Solutions

#### ImportError: No module named 'anthropic'
```bash
# Claude API is optional - disable if not needed
# In ai_optimization/config.yaml:
# agents -> pattern_discovery -> use_ai: false
```

#### SQLite database locked
```bash
# Remove journal files
rm -f ai_optimization/context_store/*.db-journal
```

#### Memory issues with large datasets
```yaml
# Reduce batch sizes in config.yaml:
agents:
  pattern_discovery:
    batch_size: 50  # Reduce from 100
```

#### Convergence not detected
- Check `convergence_threshold` in config.yaml (default: 0.01)
- May need more iterations for difficult datasets
- Review `ai_optimization/iterations/*/report.md` for insights

#### API rate limits (if using Claude)
- Implement exponential backoff (already in code)
- Reduce `batch_size` in agent configs
- Set `api -> retry_delay` higher in config.yaml

### Monitoring and Debugging

```bash
# Watch logs in real-time
tail -f ai_optimization/logs/optimization.log

# Check for errors
grep -i "error\|exception" ai_optimization/logs/*.log

# View iteration reports
cat ai_optimization/iterations/iteration_001/report.md

# Check database status
sqlite3 ai_optimization/context_store/patterns.db "SELECT COUNT(*) FROM patterns;"
```

## Best Practices

1. **Start Small**: Run 1-2 iterations first to validate setup
2. **Monitor Patterns**: Check discovered patterns make sense for your domain
3. **Validate Matches**: Review high-confidence matches before trusting fully
4. **Use Checkpoints**: System auto-saves progress, leverage for long runs
5. **Parallel Agents**: Keep `parallel_execution: true` for speed
6. **Threshold Tuning**: Let AI suggest changes, but review before applying

## Recent Changes (October 20, 2025)

### Critical Data Quality Issue Discovered
- **203 NEW_OWNER_IDs** are incorrectly assigned to multiple different companies in production
- Example: ID 1088737 is assigned to BP America, State Of Texas, Buckhorn Minerals, and Texco Partners
- This creates false positives where unrelated companies appear to match
- See `DATA_QUALITY_REPORT.md` and `outputs/duplicate_owner_ids_report.csv` for full details

### Matching System Updates
1. **Removed DirectIDMatcher**: IDs are reindexed annually, making OLD_OWNER_ID matching invalid
2. **Enhanced Validation**: Added original name similarity checks to prevent false positives from data quality issues
3. **City/State Validation**: Strict location validation for all matching strategies
4. **Expanded Strategies**: Added 5 new matching strategies (Estate, Temporal, Address-Only, Initial, Last Resort)
5. **Aggressive Thresholds**: Lowered thresholds significantly to achieve 87.2% match rate

### AI Optimization Fixes
The AI optimization system had several critical issues that have been resolved:

1. **Pattern Discovery Agent**: Fixed column name handling
   - Now adapts to actual column names in the data
   - Integrated `clean_text()` function for on-the-fly cleaning
   - All pattern discovery methods updated for flexibility

2. **Data Preprocessing**: Added `clean_name` column creation
   - Preprocessing step in `run_matching_iteration()`
   - Uses consistent text normalization

3. **Validation Agent**: Fixed parameter passing
   - Corrected parameter name from 'matches' to 'matches_df'
   - Proper None values for optional parameters

4. **Temporal Analyzer**: Updated column name handling
   - Checks multiple possible column names
   - Graceful fallback through variations

5. **Robustness Improvements**:
   - All agents handle multiple column name formats
   - Graceful fallbacks when expected columns missing
   - Works with both Snowflake and Excel workflows
   - No hard-coded assumptions about column names

### Key Functions Updated
- `map_owners_snowflake()`: Optimized workflow for Snowflake data
- `preprocess_snowflake_data()`: Intelligently separates and preprocesses matched/unmatched records
- Enhanced output with full address details for validation
- All matchers updated with city/state validation and original name verification

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

### AI Optimization Tasks (NEW)

#### Run optimization and monitor progress
```bash
# Start optimization
python ai_optimization/run_optimization.py --max-iterations 5 --mode enhanced

# Monitor in another terminal
tail -f ai_optimization/logs/optimization.log
```

#### Check optimization results
```python
import pandas as pd
import yaml

# Load latest iteration results
iteration_num = 1  # Or whichever iteration you want
results_path = f"ai_optimization/iterations/iteration_{iteration_num:03d}/"

# Load matches
matches = pd.read_csv(f"{results_path}/matches.csv")
print(f"Match rate: {matches['mapped_new_id'].notna().mean():.2%}")

# Load iteration summary
with open(f"{results_path}/results.yaml") as f:
    results = yaml.safe_load(f)
print(f"Convergence status: {results['convergence_status']}")
```

#### Query discovered patterns
```python
import sqlite3
conn = sqlite3.connect("ai_optimization/context_store/patterns.db")
patterns = pd.read_sql_query(
    "SELECT * FROM patterns ORDER BY confidence DESC LIMIT 20",
    conn
)
print(patterns[['pattern_type', 'pattern_value', 'frequency', 'confidence']])
```

#### Resume from checkpoint
```bash
# System automatically continues from last checkpoint
python ai_optimization/run_optimization.py --max-iterations 5

# Or explicitly specify checkpoint
python ai_optimization/run_optimization.py \
  --continue-from ai_optimization/context_store/learning_checkpoint.json
```

#### Debug specific agent
```bash
# Run only temporal analysis
python -c "
import asyncio
from ai_optimization.agents.subagents.temporal_analyzer import TemporalAnalyzerAgent
# ... agent testing code ...
"
```

## Performance Metrics

### Current System Performance (October 20, 2025)
- **Baseline (Manual)**: 5.9% match rate (147/2,484)
- **Current Performance**: 87.2% match rate (3,700/4,244 records)
  - SQL Direct: 1,622 (38.2%)
  - Fuzzy Matches: 2,078 (49.0%)
- **Confidence Distribution**:
  - High (90-100): 2,016 matches (54%)
  - Medium (75-89): 313 matches (8%)
  - Low (50-74): 1,371 matches (37%)
- **Processing Speed**: 3-5x faster with parallel agents
- **False Positive Risk**: HIGH due to aggressive thresholds and data quality issues

### Match Type Breakdown (Fuzzy Matches)
- ADDRESS_ONLY: 1,171 (56% of fuzzy matches) - LOW CONFIDENCE
- ADDRESS_NAME_MATCH: 366 (18%)
- TEMPORAL_FUZZY_MATCH: 264 (13%)
- LAST_RESORT: 178 (9%) - VERY LOW CONFIDENCE
- EXACT_NAME: 28 (1%)
- Other strategies: 71 (3%)

### Optimization Benchmarks
- Pattern Discovery: ~100-200 patterns per iteration
- Temporal Analysis: 30-40% improvement on name changes
- Threshold Tuning: 2-5% incremental improvements
- Overall Runtime: 5-10 minutes per iteration

### Data Quality Impact
- 203 NEW_OWNER_IDs with multiple company names
- Affects ~800-1000 records
- Creates ambiguity in matching results
- Requires manual validation for affected IDs

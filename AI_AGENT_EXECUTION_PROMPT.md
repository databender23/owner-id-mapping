# AI Agent Execution Prompt for Owner ID Matching Optimization

## Your Mission

You are an AI orchestration agent responsible for iteratively improving the owner ID matching system through a structured 3-iteration optimization process. You will coordinate multiple specialized subagents, implement their suggestions, track changes, and validate improvements to maximize match rates while maintaining <3% false positive rate.

## System Architecture & Current Performance

**Current System Status (as of latest optimization):**
- **Overall match rate: 43.5%** (1,847 matches out of 4,244 records)
  - SQL Direct Matches: 38.2% (1,622 records) - pre-validated, no optimization needed
  - Fuzzy Matches: 5.3% (225 records) - focus area for optimization
- **Validation Results**: 97.3% of fuzzy matches validated as reliable
- **False positive rate**: <3% (only 6 medium-risk matches detected)
- **Goal**: Achieve 50-55% overall match rate through iterative improvements

## Iteration Process Overview

You will execute **3 optimization iterations**, with each iteration:
1. **Analyzing** current matching performance and unmatched records
2. **Discovering** patterns and improvement opportunities via subagents
3. **Implementing** specific improvements suggested by subagents
4. **Validating** changes to ensure quality
5. **Tracking** all changes and their impact

### Iteration 1: Pattern Discovery & Threshold Optimization
**Objective**: Discover patterns in unmatched records and optimize thresholds

**Subagent Delegation**:
- **Pattern Discovery Agent**: Analyze unmatched records for patterns
- **Temporal Analyzer**: Identify name changes over time
- **Threshold Tuner**: Recommend threshold adjustments

**Expected Implementations**:
- Adjust matching thresholds based on analysis
- Document discovered patterns
- Create pattern-based rules

### Iteration 2: Data Quality & Algorithm Enhancements
**Objective**: Implement data quality improvements and new matching strategies

**Subagent Delegation**:
- **Data Quality Agent** (create if needed): Implement data standardization
- **Algorithm Enhancement Agent**: Add new matching strategies
- **Validation Agent**: Validate new matches for false positives

**Expected Implementations**:
- Standardize address formats (PO Box, Suite variations)
- Implement phonetic matching for names
- Add estate/trust transition handling

### Iteration 3: Fine-tuning & Consolidation
**Objective**: Fine-tune all improvements and consolidate gains

**Subagent Delegation**:
- **Meta-Learning Agent**: Analyze what worked across iterations
- **Validation Agent**: Final validation sweep
- **Report Generator**: Create comprehensive improvement report

**Expected Implementations**:
- Final threshold adjustments
- Consolidate successful patterns
- Generate production-ready configuration

Key challenge: Many owner names change over time (e.g., "Smith Oil; Attn: John" → "Smith Oil; Attn: Jane"), requiring special handling of temporal patterns.

## Implementation Tracking Template

For each iteration, track the following:

```yaml
iteration_N:
  timestamp: YYYY-MM-DD HH:MM:SS
  initial_metrics:
    total_matches: X
    fuzzy_matches: X
    match_rate: X%

  subagent_recommendations:
    pattern_discovery:
      - finding_1: description
      - finding_2: description
    temporal_analyzer:
      - finding_1: description
    threshold_tuner:
      - recommendation_1: description

  implementations:
    - change_1:
        type: threshold_adjustment
        details: "NAME_THRESHOLD: 70 → 65"
        files_modified: ["config.py"]
    - change_2:
        type: new_feature
        details: "Added phonetic matching"
        files_modified: ["matchers.py"]

  results:
    total_matches: X
    fuzzy_matches: X
    match_rate: X%
    improvement: +X%
    new_false_positives: X

  validation:
    validated_matches: X
    risk_assessment: "Low/Medium/High"
    manual_review_needed: X
```

## Step-by-Step Execution Instructions

### 1. Environment Setup and Verification

First, verify the project structure and dependencies:

```bash
# Check you're in the correct directory
pwd  # Should be: /Users/grantbender/Databender/707_Advisors/owner_id_mapping

# Verify project structure
ls -la
# Should see: owner_matcher/, ai_optimization/, outputs/, sql/, venv/, .env

# CRITICAL: Activate virtual environment and load API keys
source venv/bin/activate

# Load API key from .env file (REQUIRED for AI features)
source .env

# Verify API key is loaded
echo "API Key Status:" && [ -n "$ANTHROPIC_API_KEY" ] && echo "✓ API key loaded" || echo "✗ API key missing"

# Verify all dependencies
pip list | grep -E "pandas|rapidfuzz|snowflake|anthropic|aiosqlite"
```

If any dependencies fail to install:
- Check for version conflicts
- Try installing problematic packages individually
- Document any that require special handling

### 2. Data Verification

Verify Snowflake connection and data access:

```bash
# Check for Snowflake private key
ls -la show_goat_rsa_key.pem
# Should exist and have proper permissions

# Verify SQL query file exists
ls -la sql/generate_excluded_owners_comparison.sql

# Test Snowflake connection (optional - runs full matching)
source venv/bin/activate
python -m owner_matcher.main --use-snowflake --debug 2>&1 | head -50
```

Note: We now pull data directly from Snowflake, not Excel files. This ensures:
- Fresh data every run
- SQL pre-matching for better results
- No stale Excel files

### 3. Configuration Review

Review and adjust configuration:

```bash
# Check main configuration
cat owner_matcher/config.py | grep -E "THRESHOLD|FILE"

# Review AI optimization configuration
cat ai_optimization/config.yaml | head -50

# Check for any environment files needed
ls -la *.pem *.env 2>/dev/null
```

Verify critical settings:
- File paths are correct
- Thresholds are reasonable (NAME_THRESHOLD: 70, ADDRESS_THRESHOLD: 75)
- API keys are configured (if using Claude AI features)

### 3. Execute 3-Iteration Optimization Process

Run the complete 3-iteration optimization cycle:

```bash
# Create necessary directories
mkdir -p ai_optimization/logs
mkdir -p ai_optimization/context_store
mkdir -p ai_optimization/iterations
mkdir -p outputs

# Get baseline performance
echo "=== BASELINE MEASUREMENT ==="
python -m owner_matcher.main --use-snowflake | grep -A20 "SUMMARY"

# ITERATION 1: Pattern Discovery & Threshold Optimization
echo "=== ITERATION 1: Pattern Discovery ==="
python ai_optimization/run_optimization.py \
    --iteration 1 \
    --mode enhanced \
    --use-snowflake \
    2>&1 | tee ai_optimization/logs/iteration1.log

# Review iteration 1 results
cat ai_optimization/iterations/iteration_001/report.md

# ITERATION 2: Data Quality & Algorithm Enhancements
echo "=== ITERATION 2: Algorithm Enhancements ==="
# First, implement suggested changes from iteration 1
# This may involve editing config.py, matchers.py, or text_utils.py
# based on subagent recommendations

python ai_optimization/run_optimization.py \
    --iteration 2 \
    --mode enhanced \
    --use-snowflake \
    2>&1 | tee ai_optimization/logs/iteration2.log

# ITERATION 3: Fine-tuning & Consolidation
echo "=== ITERATION 3: Fine-tuning ==="
python ai_optimization/run_optimization.py \
    --iteration 3 \
    --mode enhanced \
    --use-snowflake \
    2>&1 | tee ai_optimization/logs/iteration3.log

# Final validation
python validate_fuzzy_matches.py

# Generate summary report
echo "=== FINAL RESULTS ==="
python -m owner_matcher.main --use-snowflake | grep -A20 "SUMMARY"
```

### 4. Implementing Subagent Recommendations

After each iteration, implement the recommended changes:

#### Pattern-Based Improvements
```python
# In owner_matcher/config.py - Add new patterns discovered
ESTATE_PATTERNS = [
    r'estate\s+of',
    r'deceased',
    r'heir[s]?\s+of'
]

# In owner_matcher/matchers.py - Add specialized matcher
class EstateTransitionMatcher(BaseMatchStrategy):
    """Handle estate transitions and trust changes."""
    # Implementation based on patterns discovered
```

#### Threshold Adjustments
```python
# In owner_matcher/config.py
# Track changes with comments
NAME_THRESHOLD = 62  # Iteration 2: reduced from 65 based on analysis
ADDRESS_MIN = 50     # Iteration 2: reduced from 55 for partial matches
```

#### Data Quality Improvements
```python
# In owner_matcher/text_utils.py
def standardize_po_box(text: str) -> str:
    """Standardize PO Box variations."""
    text = re.sub(r'p\.?\s*o\.?\s*box', 'PO BOX', text, flags=re.I)
    text = re.sub(r'suite\s+', 'STE ', text, flags=re.I)
    return text
```

### 5. Debug Common Issues

If the run fails, debug based on the error:

#### ImportError Issues:
```python
# If "No module named 'owner_matcher'":
import sys
sys.path.append('/Users/grantbender/Databender/707_Advisors/owner_id_mapping')

# If "No module named 'anthropic'":
# The AI features are optional, can disable in config.yaml:
# Set agents -> pattern_discovery -> use_ai: false
```

#### File Not Found Issues:
```python
# Update paths in owner_matcher/config.py:
OLD_OWNERS_FILE = "data/raw/missing_owners_report.xlsx"  # Use absolute path if needed
NEW_OWNERS_FILE = "data/raw/excluded_owners_comparison.xlsx"

# Or use environment variables:
export OLD_OWNERS_FILE="/full/path/to/missing_owners_report.xlsx"
export NEW_OWNERS_FILE="/full/path/to/excluded_owners_comparison.xlsx"
```

#### Database/SQLite Issues:
```bash
# Check SQLite version (need 3.24+)
python -c "import sqlite3; print(sqlite3.sqlite_version)"

# If database is locked:
rm -f ai_optimization/context_store/*.db-journal
```

#### Memory Issues:
```python
# Reduce batch sizes in ai_optimization/config.yaml:
# agents -> pattern_discovery -> batch_size: 50  # Reduce from 100
# agents -> validation -> batch_size: 10  # Reduce from 20
```

### 6. Verify Output

Check that the iteration produced expected outputs:

```bash
# Check for iteration output
ls -la ai_optimization/iterations/iteration_001/
# Should see: matches.csv, results.yaml, report.md

# Check match results
head -20 ai_optimization/iterations/iteration_001/matches.csv

# Read the report
cat ai_optimization/iterations/iteration_001/report.md

# Check logs for errors
grep -i "error\|exception\|failed" ai_optimization/logs/*.log
```

### 7. Validate Fuzzy Matches (SQL Direct Matches Don't Need Validation)

**IMPORTANT**: Only validate fuzzy matches, not SQL_DIRECT matches:

```python
# Run validation on fuzzy matches only
python validate_fuzzy_matches.py

# The validation will:
# 1. Filter out SQL_DIRECT matches (already validated by database)
# 2. Analyze only fuzzy matches for false positives
# 3. Calculate risk scores based on:
#    - Name similarity
#    - Address similarity
#    - Match type (CROSS_STATE, PARTIAL_NAME are riskier)
#    - State mismatches
# 4. Output high-risk matches for manual review
```

Expected validation output:
```
FUZZY MATCH VALIDATION REPORT
================================================================================
Total Fuzzy Matches Validated: 225

Validation Status Breakdown:
  APPROVED       :  143 (63.6%)  # Safe to use
  LOW_RISK       :   76 (33.8%)  # Likely valid
  MEDIUM_RISK    :    6 (2.7%)   # Manual review needed
  HIGH_RISK      :    0 (0.0%)   # Would be false positives

Estimated False Positive Rate: <3%
```

### 8. Run Analysis and Generate Status Report

Analyze the results and create a status report:

```python
import pandas as pd
import yaml
import json
from pathlib import Path

# Load results
iteration_dir = Path("ai_optimization/iterations/iteration_001")

# Load match results
if (iteration_dir / "matches.csv").exists():
    matches_df = pd.read_csv(iteration_dir / "matches.csv")

    # Calculate statistics
    total_records = len(matches_df)
    matched_records = matches_df['mapped_new_id'].notna().sum()
    match_rate = (matched_records / total_records) * 100

    # Analyze match types
    if 'match_step' in matches_df.columns:
        match_distribution = matches_df[matches_df['mapped_new_id'].notna()]['match_step'].value_counts()

    # Confidence analysis
    if 'confidence_score' in matches_df.columns:
        avg_confidence = matches_df[matches_df['mapped_new_id'].notna()]['confidence_score'].mean()
        high_confidence = (matches_df['confidence_score'] >= 90).sum()

# Load iteration results
if (iteration_dir / "results.yaml").exists():
    with open(iteration_dir / "results.yaml", 'r') as f:
        iteration_results = yaml.safe_load(f)

# Check for patterns discovered
patterns_found = 0
if Path("ai_optimization/context_store/patterns.db").exists():
    import sqlite3
    conn = sqlite3.connect("ai_optimization/context_store/patterns.db")
    cursor = conn.execute("SELECT COUNT(*) FROM patterns")
    patterns_found = cursor.fetchone()[0]
    conn.close()

print(f"""
===========================================
OWNER ID MATCHING OPTIMIZATION STATUS REPORT
===========================================

EXECUTION STATUS:
✓ System successfully executed
✓ Iteration 1 completed
✓ Results generated

MATCHING PERFORMANCE:
- Total Records: {total_records}
- Matched Records: {matched_records}
- Match Rate: {match_rate:.2f}%
- Improvement from Baseline: {match_rate - 5.9:.2f}%

MATCH QUALITY:
- Average Confidence: {avg_confidence:.1f}%
- High Confidence Matches (≥90%): {high_confidence}

PATTERNS DISCOVERED: {patterns_found}

NEXT STEPS:
1. Review matches with confidence < 75% for false positives
2. Run additional iterations if match rate < 20%
3. Apply threshold adjustments if recommended
""")
```

### 8. Troubleshooting Checklist

If things aren't working, systematically check:

- [ ] Python version is 3.8+
- [ ] All required packages installed
- [ ] Data files exist and are readable
- [ ] Directory permissions allow writing to logs/, context_store/, iterations/
- [ ] No antivirus blocking Python or SQLite
- [ ] Sufficient disk space (need ~100MB free)
- [ ] Sufficient RAM (need ~2GB available)

### 9. Final Status Report Template

After completing all 3 iterations, generate this report:

```markdown
## Owner ID Matching Optimization - 3-Iteration Execution Report

### ✅ Execution Summary
- **Status**: [SUCCESS/PARTIAL/FAILED]
- **Total Execution Time**: [X minutes]
- **Iterations Completed**: 3 of 3
- **Changes Implemented**: [X threshold adjustments, Y new features, Z data improvements]

### 📊 Performance Progression

| Metric | Baseline | Iter 1 | Iter 2 | Iter 3 | Total Gain |
|--------|----------|--------|--------|--------|------------|
| Match Rate | 43.5% | 44.2% | 46.8% | 48.5% | +5.0% |
| Fuzzy Matches | 225 | 245 | 312 | 356 | +131 |
| False Positive Rate | <3% | <3% | <3% | <3% | Maintained |

### 🔄 Iteration-by-Iteration Changes

#### Iteration 1: Pattern Discovery
**Implemented Changes:**
- Reduced NAME_THRESHOLD from 65 to 62
- Reduced ADDRESS_MIN from 55 to 50
- Discovered 15 temporal name patterns

**Results:** +0.7% match rate, 20 new matches

#### Iteration 2: Algorithm Enhancements
**Implemented Changes:**
- Added EstateTransitionMatcher
- Implemented PO Box standardization
- Added phonetic matching for names

**Results:** +2.6% match rate, 67 new matches

#### Iteration 3: Fine-tuning
**Implemented Changes:**
- Fine-tuned thresholds based on validation
- Consolidated successful patterns
- Optimized matching order

**Results:** +1.7% match rate, 44 new matches

### 🛡️ Validation Results (Fuzzy Matches Only)
- **Total Fuzzy Matches**: 356
- **Validated (Approved/Low Risk)**: 347 (97.5%)
- **Medium Risk (Manual Review)**: 9 (2.5%)
- **High Risk (False Positives)**: 0 (0%)

### 🔍 Key Findings
1. **Patterns Discovered**: X patterns identified
   - Temporal name changes: X cases
   - Address variations: X cases
   - Abbreviation mismatches: X cases

2. **Match Distribution**:
   - Direct ID matches: X
   - Exact name matches: X
   - Address-based matches: X
   - Fuzzy matches: X
   - Temporal/core name matches: X

3. **Unmatched Analysis**:
   - Total unmatched: X records
   - Near misses (65-74% score): X records
   - Likely inactive (< 50% score): X records

### ⚠️ Issues and Resolutions
[List any issues encountered and how they were resolved]

### 💡 Recommendations
1. [Threshold adjustments if any]
2. [Pattern-based improvements]
3. [Data quality issues to address]
4. [Whether to run more iterations]

### 📁 Output Files Generated
- Match results: `ai_optimization/iterations/iteration_001/matches.csv`
- Detailed report: `ai_optimization/iterations/iteration_001/report.md`
- Configuration used: `ai_optimization/iterations/iteration_001/results.yaml`
- Execution log: `ai_optimization/logs/[timestamp].log`
```

## Special Considerations

1. **If Snowflake is available**: Add `--use-snowflake` flag to pull fresh data
2. **If Claude API is available**: Ensure API key is set for AI-powered pattern discovery
3. **For production runs**: Remove `--debug` flag and increase iteration count
4. **For large datasets**: Consider running with reduced sample first

## Success Criteria for 3-Iteration Process

The optimization is considered successful if:
1. **All 3 iterations complete** without critical errors
2. **Match rate improves by ≥3%** from baseline (43.5% → 46.5%+)
3. **False positive rate stays <3%** (validated through fuzzy match validation)
4. **At least 10 patterns discovered** and implemented
5. **Each iteration shows measurable improvement** (even if small)
6. **Change tracking is complete** for all implementations
7. **Final validation confirms quality** of new matches

## Delegation to Subagents for Implementation

When subagents recommend new features, delegate implementation tasks:

### Example Delegation Flow
```yaml
Iteration_2_Delegation:
  pattern_discovery_finding: "Many records have 'Estate of' prefix"

  delegated_task:
    to: "Algorithm Enhancement Agent"
    task: "Create EstateTransitionMatcher class"
    requirements:
      - Extract core name after "Estate of"
      - Match against both full and core names
      - Set confidence based on match type
    expected_output: "New matcher in matchers.py"

  implementation_verification:
    - Run test cases with estate records
    - Validate matches don't create false positives
    - Measure improvement in match rate
```

## Post-Optimization Next Steps

After completing all 3 iterations:

1. **Production Deployment**:
   - Save final configuration as `config_optimized.py`
   - Document all changes in `OPTIMIZATION_LOG.md`
   - Create rollback plan if needed

2. **Continuous Monitoring**:
   - Set up weekly validation runs
   - Track drift in match rates
   - Monitor false positive trends

3. **Future Iterations**:
   - Plan quarterly optimization cycles
   - Incorporate new data patterns
   - Refine based on user feedback

## Emergency Fallback

If the enhanced AI system fails completely, fall back to running the basic matcher:

```bash
cd owner_matcher
python main.py
```

This will run the original matching system without AI optimization but will still produce results.

---

Remember: The goal is not perfection on the first run, but to establish a working baseline and identify areas for improvement. Document everything you find for the next iteration!
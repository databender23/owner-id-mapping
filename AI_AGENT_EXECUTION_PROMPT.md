# AI Agent Execution Prompt for Owner ID Matching Optimization

## Your Mission

You are an AI engineering assistant tasked with running the enhanced owner ID matching optimization system, debugging any issues that arise, and providing a comprehensive status report. This system uses multiple AI agents working in parallel to optimize fuzzy matching between old and new owner records in an oil & gas database.

## Background Context

This system attempts to match ~2,484 old owner records to ~4,767 new owner records. The current baseline match rate is 5.9% (147 matches). The goal is to increase this to 30-40% through iterative optimization while maintaining <5% false positive rate.

Key challenge: Many owner names change over time (e.g., "Smith Oil; Attn: John" → "Smith Oil; Attn: Jane"), requiring special handling of temporal patterns.

## Step-by-Step Execution Instructions

### 1. Environment Setup and Verification

First, verify the project structure and dependencies:

```bash
# Check you're in the correct directory
pwd  # Should be: /Users/grantbender/Databender/707_Advisors/owner_id_mapping

# Verify project structure
ls -la
# Should see: owner_matcher/, ai_optimization/, data/, sql/, etc.

# Check Python version (need 3.8+)
python --version

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate  # On Mac/Linux
# OR: venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
pip install -r ai_optimization/requirements-ai.txt
```

If any dependencies fail to install:
- Check for version conflicts
- Try installing problematic packages individually
- Document any that require special handling

### 2. Data Verification

Check that required data files exist:

```bash
# Check for input data files
ls -la data/raw/
# Should see: missing_owners_report.xlsx and excluded_owners_comparison.xlsx

# If files are missing, check for alternative locations
find . -name "*.xlsx" -type f 2>/dev/null

# Verify data file sizes (should not be empty)
du -h data/raw/*.xlsx
```

If data files are missing:
- Check if they need to be generated from Snowflake
- Look for backup locations
- Document what's missing

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

### 4. Run Single Optimization Iteration

Execute one iteration of the optimization:

```bash
# Create necessary directories
mkdir -p ai_optimization/logs
mkdir -p ai_optimization/context_store
mkdir -p ai_optimization/iterations

# Run with local Excel files (simpler for testing)
python ai_optimization/run_optimization.py \
    --iteration 1 \
    --mode enhanced \
    --debug \
    2>&1 | tee ai_optimization/logs/run_$(date +%Y%m%d_%H%M%S).log
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

### 7. Run Analysis and Generate Status Report

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

After completing execution, report:

```markdown
## Owner ID Matching Optimization - Execution Report

### ✅ Execution Summary
- **Status**: [SUCCESS/PARTIAL/FAILED]
- **Execution Time**: [X minutes]
- **Errors Encountered**: [None/List issues]
- **Iterations Completed**: [1 of 1]

### 📊 Performance Metrics
- **Baseline Match Rate**: 5.9% (147 matches)
- **Current Match Rate**: X.X% (XXX matches)
- **Improvement**: +X.X%
- **False Positive Risk**: [Low/Medium/High based on confidence scores]

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

## Success Criteria

The execution is considered successful if:
1. The system runs without critical errors
2. Match rate improves from baseline (5.9%)
3. Output files are generated correctly
4. At least some patterns are discovered
5. The system provides actionable recommendations

## Emergency Fallback

If the enhanced AI system fails completely, fall back to running the basic matcher:

```bash
cd owner_matcher
python main.py
```

This will run the original matching system without AI optimization but will still produce results.

---

Remember: The goal is not perfection on the first run, but to establish a working baseline and identify areas for improvement. Document everything you find for the next iteration!
#!/bin/bash

# Repository Cleanup Script
# Date: October 20, 2025
# Purpose: Remove temporary, test, and redundant files after repository reorganization

echo "==================================================================="
echo "Owner ID Mapping Repository Cleanup"
echo "==================================================================="
echo ""
echo "This script will remove the following files:"
echo ""

# Files to remove (with explanations)
echo "1. Documentation files (integrated into CLAUDE.md):"
echo "   - AI_OPTIMIZATION_FIXES_SUMMARY.md"
echo ""

echo "2. Test/debugging scripts:"
echo "   - generate_fuzzy_validation.py"
echo ""

echo "3. System files:"
echo "   - .DS_Store"
echo ""

echo "4. Old output files (keeping only the latest):"
echo "   Removing timestamped duplicates from outputs/"
echo "   - fuzzy_match_validation_20251020_*.csv (keeping latest)"
echo "   - unmatched_records_20251020_*.csv"
echo "   - matching_report_20251020_*.txt"
echo "   - complete_matches_20251020_*.csv"
echo ""

echo "5. Test/temporary context store (can be regenerated):"
echo "   - context_store/ directory"
echo ""

echo "6. High-risk matches file (was for testing):"
echo "   - outputs/high_risk_matches.csv"
echo ""

read -p "Do you want to proceed with cleanup? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo ""
    echo "Removing files..."

    # 1. Remove integrated documentation
    rm -f AI_OPTIMIZATION_FIXES_SUMMARY.md
    echo "✓ Removed AI_OPTIMIZATION_FIXES_SUMMARY.md"

    # 2. Remove test/debugging scripts
    rm -f generate_fuzzy_validation.py
    echo "✓ Removed generate_fuzzy_validation.py"

    # 3. Remove system files
    rm -f .DS_Store
    echo "✓ Removed .DS_Store"

    # 4. Clean up old output files (keep only the latest)
    # Remove all timestamped fuzzy validation files except the latest
    rm -f outputs/fuzzy_match_validation_20251020_140200.csv
    rm -f outputs/fuzzy_match_validation_20251020_140619.csv
    echo "✓ Removed old fuzzy validation files (kept latest)"

    # Remove all unmatched records files (they're regenerated each run)
    rm -f outputs/unmatched_records_*.csv
    echo "✓ Removed unmatched records files"

    # Remove all matching report files (they're regenerated each run)
    rm -f outputs/matching_report_*.txt
    echo "✓ Removed matching report files"

    # Remove complete matches file (old format)
    rm -f outputs/complete_matches_*.csv
    echo "✓ Removed complete matches file"

    # 5. Remove test context store (optional - ask first)
    read -p "Remove context_store/ directory? This will delete AI learning data. (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]
    then
        rm -rf context_store/
        echo "✓ Removed context_store/"
    else
        echo "⊗ Kept context_store/"
    fi

    # 6. Remove high-risk matches test file
    rm -f outputs/high_risk_matches.csv
    echo "✓ Removed high_risk_matches.csv"

    echo ""
    echo "==================================================================="
    echo "Cleanup complete!"
    echo ""
    echo "Files kept:"
    echo "  - outputs/mapped_owners.csv (main output)"
    echo "  - outputs/fuzzy_match_validation.csv (latest validation)"
    echo "  - outputs/fuzzy_match_validation_20251020_142339.csv (timestamped backup)"
    echo "  - outputs/duplicate_owner_ids_report.csv (data quality report)"
    echo "  - DATA_QUALITY_REPORT.md (critical issue documentation)"
    echo "  - All core system files and documentation"
    echo ""
    echo "Repository is now clean and organized."
    echo "==================================================================="
else
    echo "Cleanup cancelled."
fi
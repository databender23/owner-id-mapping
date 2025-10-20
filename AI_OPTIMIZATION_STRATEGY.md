# AI-Assisted Iterative Optimization System for Owner ID Matching

**Version:** 1.0
**Created:** 2025-10-20
**Status:** Design Phase

---

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Five-Agent System](#five-agent-system)
4. [Iteration Workflow](#iteration-workflow)
5. [Implementation Plan](#implementation-plan)
6. [Safety Mechanisms](#safety-mechanisms)
7. [Expected Outcomes](#expected-outcomes)
8. [Technology Stack](#technology-stack)
9. [Cost Estimation](#cost-estimation)
10. [Success Metrics](#success-metrics)

---

## Overview

### Problem Statement
The current owner ID matching system achieves a ~6% match rate (147/2484 owners). While this is expected due to many legitimately inactive owners, there's significant room for improvement through iterative optimization of matching logic and thresholds.

### Solution
Create an automated system where AI agents continuously improve the matching logic through:
1. **Execution** - Run matching process with current configuration
2. **Validation** - AI reviews matched records for correctness
3. **Analysis** - AI identifies patterns in unmatched records
4. **Improvement** - AI suggests specific code/config changes
5. **Iteration** - Apply changes and measure impact

### Goals
- Increase match rate from 6% to 20-30%
- Maintain false positive rate below 5%
- Automate optimization process
- Provide clear audit trail
- Enable continuous improvement

### Critical Name Pattern: Temporal Changes

**Key Insight:** Owner names frequently change over time, particularly in the portion AFTER separators like "Attn:", "Estate of", "c/o", etc.

**Pattern:**
```
Old Database:  "Smith Oil Company; Attn: John Smith"
New Database:  "Smith Oil Company; Attn: Jane Doe"
```

**Core Principle:** The **prefix/core name** (before separators) is stable. The **appended information** (after separators) changes frequently due to:
- Contact person changes
- Estate transitions ("Estate of...")
- Trustee changes
- Administrative updates

**Matching Strategy:**
1. **Extract Core Name:** Identify text BEFORE: "Attn:", ";", "Estate of", "c/o", "%"
2. **Match Core First:** Prioritize matching the stable prefix
3. **Secondary Validation:** Use appended info for additional validation, not primary matching

**Examples:**
```
GOOD MATCH:
  Old: "ABC Trust; Attn: Manager A"
  New: "ABC Trust; Attn: Manager B"
  → Core "ABC Trust" matches perfectly

GOOD MATCH:
  Old: "Smith Energy LLC"
  New: "Estate of Smith Energy LLC"
  → Core "Smith Energy LLC" matches

POTENTIAL MISS (current system):
  Old: "Jones Oil Company"
  New: "Jones Oil Company; c/o Estate Services"
  → Exact match fails, but core matches
```

**Agent Instructions:**
- **Validator:** Check if mismatches are due to appended info changes only
- **Analyzer:** Specifically identify this pattern in unmatched records
- **Suggester:** Propose core-name extraction improvements

---

## System Architecture

### Directory Structure
```
owner_id_mapping/
├── ai_optimization/
│   ├── README.md                          # System documentation
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── match_validator.py            # Agent 1: Validate matches
│   │   ├── unmatched_analyzer.py         # Agent 2: Analyze gaps
│   │   ├── improvement_suggester.py      # Agent 3: Suggest changes
│   │   ├── metrics_tracker.py            # Agent 4: Track metrics
│   │   ├── report_generator.py           # Agent 5: Generate reports
│   │   └── orchestrator.py               # Coordinate workflow
│   ├── prompts/
│   │   ├── match_validation.md           # Validation criteria
│   │   ├── unmatched_analysis.md         # Pattern analysis guide
│   │   ├── improvement_suggestion.md     # Change suggestion template
│   │   └── report_generation.md          # Report template
│   ├── iterations/
│   │   ├── iteration_001/
│   │   │   ├── input_config.json         # Config snapshot
│   │   │   ├── matches.csv               # Results
│   │   │   ├── validations.json          # AI validation output
│   │   │   ├── analysis.json             # Gap analysis
│   │   │   ├── suggestions.json          # Proposed improvements
│   │   │   └── metrics.json              # Performance metrics
│   │   ├── iteration_002/
│   │   └── ...
│   ├── reports/
│   │   ├── progress_report.md            # Human-readable summary
│   │   └── metrics_dashboard.html        # Visual metrics
│   ├── models/
│   │   ├── __init__.py
│   │   ├── validation.py                 # Pydantic models for validation
│   │   ├── analysis.py                   # Models for analysis
│   │   └── suggestion.py                 # Models for suggestions
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── claude_api.py                 # Claude API wrapper
│   │   ├── data_loader.py                # Load matching results
│   │   └── config_manager.py             # Config read/write
│   ├── config.yaml                        # Agent configuration
│   ├── run_optimization.py               # Main orchestration script
│   └── apply_changes.py                  # Apply approved changes
```

---

## Five-Agent System

### Agent 1: Match Validator

**Purpose:** Review matched records and identify potential false positives

**Input:**
- `matched_owners.csv` (confidence_score >= 75)
- Sample size: 100% of high-confidence matches, 50% of medium-confidence

**Process:**
1. Review name similarity coherence
   - Does "Smith Energy LLC" → "Smith Energy" make sense?
   - Are abbreviations standard? (Corp, Inc, Ltd)
2. Check address coherence
   - Same city/state/ZIP proximity?
   - Are addresses in same geographic region?
3. Validate business logic
   - Do oil & gas industry indicators align?
   - Are entity types compatible? (Trust → Trust, LLC → LLC)
4. Flag suspicious matches
   - Completely different states without justification
   - Wildly different names despite high score
   - Mismatched entity types

**Output:** `validations.json`
```json
{
  "iteration": 1,
  "timestamp": "2025-10-20T10:00:00",
  "total_reviewed": 147,
  "validated_matches": 135,
  "flagged_for_review": 12,
  "false_positives_detected": 3,
  "validation_details": {
    "by_confidence_tier": {
      "high (90-100)": {
        "count": 85,
        "validated": 83,
        "flagged": 2,
        "false_positive_rate": 0.024
      },
      "medium (75-89)": {
        "count": 50,
        "validated": 46,
        "flagged": 4,
        "false_positive_rate": 0.08
      },
      "low (60-74)": {
        "count": 12,
        "validated": 6,
        "flagged": 6,
        "false_positive_rate": 0.50
      }
    },
    "by_match_strategy": {
      "DIRECT_ID_MATCH": {"validated": 41, "flagged": 0},
      "EXACT_NAME": {"validated": 41, "flagged": 0},
      "ADDRESS_NAME_MATCH": {"validated": 50, "flagged": 4},
      "FUZZY_NAME": {"validated": 3, "flagged": 8}
    }
  },
  "flagged_records": [
    {
      "old_id": "12345",
      "old_name": "Smith Oil Company TX",
      "new_id": "78901",
      "new_name": "Smith Gas Corporation CA",
      "match_step": "FUZZY_NAME",
      "confidence_score": 72,
      "name_score": 85,
      "address_score": 45,
      "concern": "Different states (TX vs CA), different product focus (Oil vs Gas), low address score",
      "recommendation": "manual_review",
      "severity": "high"
    }
  ],
  "confidence_insights": {
    "threshold_effectiveness": "Current 75% threshold effectively filters out most false positives",
    "recommended_action": "Consider raising FUZZY_NAME threshold to 75 for better precision"
  }
}
```

---

### Agent 2: Unmatched Analyzer

**Purpose:** Identify patterns in unmatched records and near-misses

**Input:**
- `mapped_owners.csv` (all unmatched records + confidence_score < 75)
- Total: ~2337 records

**Process:**
1. **Near-Miss Detection**
   - Find records scoring 65-69 (just below NAME_THRESHOLD of 70)
   - Find records scoring 70-74 (just below ADDRESS_THRESHOLD of 75)
   - Calculate "gap" to next threshold

2. **Pattern Recognition**
   - Common name variations not handled
     - Abbreviations: "Corporation" vs "Corp"
     - Suffixes: "Jr", "Sr", "III"
     - Formatting: "Oil & Gas" vs "Oil and Gas"
   - Trust keyword variations
     - "Living Trust" vs "Liv Tr"
     - "Revocable Trust" vs "Rev Trust"
   - Address formatting issues
     - Suite variations: "Suite 100", "Ste 100", "#100"
     - PO Box: "P.O. Box", "PO Box", "POB"
   - **CRITICAL: Temporal Name Changes** (High Priority)
     - Identify names where core/prefix matches but appended info differs
     - Extract text BEFORE separators: ";", "Attn:", "Estate of", "c/o", "%"
     - Compare match scores: full name vs core name only
     - Flag cases where core name scores 85%+ but full name scores <70%
     - Examples to detect:
       - "Smith Oil; Attn: John" vs "Smith Oil; Attn: Jane"
       - "ABC LLC" vs "Estate of ABC LLC"
       - "Jones Trust" vs "Jones Trust; c/o Services Inc"

3. **State-Level Analysis**
   - Mismatches where owner moved states
   - Count by state pair (TX → CA, etc.)

4. **Threshold Sensitivity Analysis**
   - Model impact of threshold changes
   - Calculate potential gains at different thresholds

**Output:** `analysis.json`
```json
{
  "iteration": 1,
  "timestamp": "2025-10-20T10:15:00",
  "total_unmatched": 2337,
  "analysis_summary": {
    "near_misses": 156,
    "addressable_patterns": 87,
    "state_mismatches": 45,
    "data_quality_issues": 23
  },
  "near_misses": {
    "count": 156,
    "by_threshold": {
      "NAME_THRESHOLD (70)": {
        "range_65_69": 67,
        "range_60_64": 89
      },
      "ADDRESS_THRESHOLD (75)": {
        "range_70_74": 34
      }
    },
    "examples": [
      {
        "old_name": "ABC Oil & Gas Corporation",
        "best_match_name": "ABC Oil Gas Corp",
        "best_match_score": 68,
        "current_threshold": 70,
        "gap": 2,
        "pattern_identified": "corporation vs corp abbreviation",
        "recommendation": "add_keyword",
        "confidence": 0.95
      },
      {
        "old_name": "Smith Family Living Trust",
        "best_match_name": "Smith Fam Liv Tr",
        "best_match_score": 66,
        "current_threshold": 70,
        "gap": 4,
        "pattern_identified": "trust abbreviation",
        "recommendation": "improve_trust_normalization",
        "confidence": 0.90
      }
    ]
  },
  "common_patterns": {
    "missing_keywords": {
      "count": 87,
      "keywords": [
        {"word": "corporation", "occurrences": 23, "priority": "high"},
        {"word": "incorporated", "occurrences": 15, "priority": "high"},
        {"word": "ltd", "occurrences": 12, "priority": "medium"},
        {"word": "limited", "occurrences": 18, "priority": "medium"},
        {"word": "company", "occurrences": 19, "priority": "medium"}
      ]
    },
    "abbreviation_variations": {
      "count": 34,
      "patterns": [
        {"pattern": "& vs and", "count": 12},
        {"pattern": "suite variations", "count": 8},
        {"pattern": "po box formats", "count": 14}
      ]
    },
    "trust_variations": {
      "count": 45,
      "variations": [
        {"from": "living trust", "to": "liv tr", "count": 15},
        {"from": "revocable trust", "to": "rev trust", "count": 12},
        {"from": "family trust", "to": "fam tr", "count": 18}
      ]
    },
    "temporal_name_changes": {
      "count": 89,
      "description": "Names where core/prefix matches but appended info differs",
      "patterns": [
        {
          "pattern": "attention_line_change",
          "count": 34,
          "example": {
            "old": "Smith Oil Co; Attn: John Smith",
            "new": "Smith Oil Co; Attn: Jane Doe",
            "core_match": "Smith Oil Co",
            "match_score_without_suffix": 95,
            "match_score_with_suffix": 62
          }
        },
        {
          "pattern": "estate_prefix_added",
          "count": 28,
          "example": {
            "old": "ABC Energy LLC",
            "new": "Estate of ABC Energy LLC",
            "core_match": "ABC Energy LLC",
            "match_score_without_prefix": 100,
            "match_score_with_prefix": 73
          }
        },
        {
          "pattern": "care_of_appended",
          "count": 27,
          "example": {
            "old": "Jones Trust",
            "new": "Jones Trust; c/o Trust Services Inc",
            "core_match": "Jones Trust",
            "match_score_without_suffix": 100,
            "match_score_with_suffix": 58
          }
        }
      ],
      "recommendation": "Implement core-name extraction: match prefix before separators (';', 'Attn:', 'Estate of', 'c/o')"
    }
  },
  "state_mismatch_analysis": {
    "total": 45,
    "top_state_pairs": [
      {"from": "TX", "to": "OK", "count": 12},
      {"from": "CA", "to": "TX", "count": 8},
      {"from": "LA", "to": "TX", "count": 7}
    ],
    "recommendation": "Consider relaxing state restriction for high name scores"
  },
  "threshold_sensitivity": {
    "NAME_THRESHOLD": {
      "current": 70,
      "potential_gains": {
        "at_68": {"additional_matches": 23, "risk_level": "low"},
        "at_65": {"additional_matches": 67, "risk_level": "medium"},
        "at_60": {"additional_matches": 156, "risk_level": "high"}
      },
      "recommendation": "Reduce to 68 (low risk, 23 additional matches)"
    },
    "ADDRESS_THRESHOLD": {
      "current": 75,
      "potential_gains": {
        "at_73": {"additional_matches": 12, "risk_level": "low"},
        "at_70": {"additional_matches": 34, "risk_level": "medium"}
      },
      "recommendation": "Reduce to 73 (low risk, 12 additional matches)"
    },
    "ADDRESS_FIRST_NAME_THRESHOLD": {
      "current": 60,
      "potential_gains": {
        "at_55": {"additional_matches": 15, "risk_level": "medium"}
      },
      "recommendation": "Keep at 60 (insufficient data for safe reduction)"
    }
  },
  "data_quality_issues": {
    "count": 23,
    "issues": [
      {"type": "missing_address", "count": 8},
      {"type": "malformed_name", "count": 7},
      {"type": "duplicate_records", "count": 8}
    ]
  }
}
```

---

### Agent 3: Improvement Suggester

**Purpose:** Propose specific, actionable code changes based on validation and analysis

**Input:**
- `validations.json` from Agent 1
- `analysis.json` from Agent 2
- Current configuration snapshot

**Process:**
1. **Prioritize Suggestions**
   - High impact + low risk = Priority 1
   - High impact + medium risk = Priority 2
   - Low impact or high risk = Priority 3

2. **Generate Specific Changes**
   - File path
   - Line number
   - Current value
   - Proposed value
   - Justification with data

3. **Risk Assessment**
   - Estimate false positive potential
   - Consider validation agent feedback
   - Suggest mitigation strategies

4. **Impact Estimation**
   - Expected additional matches
   - Confidence level in estimate

**Output:** `suggestions.json`
```json
{
  "iteration": 1,
  "timestamp": "2025-10-20T10:30:00",
  "total_suggestions": 5,
  "suggestions": [
    {
      "id": "S001",
      "priority": "high",
      "type": "threshold_adjustment",
      "title": "Reduce NAME_THRESHOLD from 70 to 68",
      "rationale": "Analysis found 23 near-misses in 68-69 range with identifiable valid patterns. Validation shows current 70 threshold is effective at filtering false positives.",
      "supporting_data": {
        "near_misses_count": 23,
        "example_patterns": ["corporation vs corp", "incorporated vs inc"],
        "false_positive_risk": 0.04
      },
      "expected_impact": {
        "additional_matches": 23,
        "improvement_percentage": 1.56,
        "confidence": 0.85
      },
      "implementation": {
        "file": "owner_matcher/config.py",
        "line": 34,
        "current": "NAME_THRESHOLD = 70",
        "proposed": "NAME_THRESHOLD = 68",
        "diff": "- NAME_THRESHOLD = 70\n+ NAME_THRESHOLD = 68"
      },
      "risks": {
        "severity": "low",
        "description": "May introduce 1-2 false positives based on validation feedback",
        "mitigation": "Sample validation of new matches before full deployment"
      },
      "validation_checks": [
        "Run on test sample of 100 records",
        "Manual review of 20 new matches",
        "Compare false positive rate vs iteration 0"
      ],
      "approval_status": "pending"
    },
    {
      "id": "S002",
      "priority": "high",
      "type": "keyword_addition",
      "title": "Add missing corporate entity keywords",
      "rationale": "Found 87 unmatched records containing corporate entity terms not in current TRUST_KEYWORDS list. These are standard business abbreviations with negligible false positive risk.",
      "supporting_data": {
        "keyword_occurrences": {
          "corporation": 23,
          "incorporated": 15,
          "ltd": 12,
          "limited": 18,
          "company": 19
        },
        "total_affected": 87
      },
      "expected_impact": {
        "additional_matches": 87,
        "improvement_percentage": 5.88,
        "confidence": 0.95
      },
      "implementation": {
        "file": "owner_matcher/config.py",
        "line": 43,
        "current": "TRUST_KEYWORDS: List[str] = [...]",
        "proposed_additions": [
          "corporation",
          "corp",
          "incorporated",
          "inc",
          "ltd",
          "limited",
          "company",
          "co"
        ],
        "diff": "Add to end of TRUST_KEYWORDS list"
      },
      "risks": {
        "severity": "very_low",
        "description": "Standard abbreviations universally recognized",
        "mitigation": "None needed - standard practice"
      },
      "validation_checks": [
        "Verify no critical business names are affected"
      ],
      "approval_status": "pending"
    },
    {
      "id": "S003",
      "priority": "medium",
      "type": "threshold_adjustment",
      "title": "Reduce ADDRESS_THRESHOLD from 75 to 73",
      "rationale": "Found 12 near-misses at 73-74 range. Lower risk than NAME_THRESHOLD due to address stability.",
      "supporting_data": {
        "near_misses_count": 12,
        "validation_feedback": "Address-based matches show low false positive rate"
      },
      "expected_impact": {
        "additional_matches": 12,
        "improvement_percentage": 0.81,
        "confidence": 0.80
      },
      "implementation": {
        "file": "owner_matcher/config.py",
        "line": 35,
        "current": "ADDRESS_THRESHOLD = 75",
        "proposed": "ADDRESS_THRESHOLD = 73"
      },
      "risks": {
        "severity": "low",
        "description": "Slightly increased chance of different companies at similar addresses"
      },
      "approval_status": "pending"
    },
    {
      "id": "S004",
      "priority": "medium",
      "type": "code_enhancement",
      "title": "Improve trust abbreviation normalization",
      "rationale": "Found 45 trust variations that score below threshold due to abbreviations like 'liv tr' vs 'living trust'.",
      "expected_impact": {
        "additional_matches": 45,
        "improvement_percentage": 3.04,
        "confidence": 0.75
      },
      "implementation": {
        "file": "owner_matcher/text_utils.py",
        "type": "function_addition",
        "description": "Add trust-specific abbreviation expansion before cleaning",
        "pseudo_code": "Expand 'liv tr' to 'living trust', 'rev trust' to 'revocable trust', etc."
      },
      "risks": {
        "severity": "low",
        "description": "May over-normalize some legitimate differences"
      },
      "approval_status": "pending"
    },
    {
      "id": "S005",
      "priority": "high",
      "type": "code_enhancement",
      "title": "Implement core-name extraction for temporal changes",
      "rationale": "Found 89 unmatched records where core name (before separators) matches perfectly, but appended information (after 'Attn:', ';', 'Estate of', etc.) differs. These represent temporal changes where contact person or estate status changed over time.",
      "supporting_data": {
        "temporal_changes_count": 89,
        "patterns": {
          "attention_line_change": 34,
          "estate_prefix_added": 28,
          "care_of_appended": 27
        },
        "avg_core_score": 94,
        "avg_full_score": 61
      },
      "expected_impact": {
        "additional_matches": 89,
        "improvement_percentage": 6.01,
        "confidence": 0.90
      },
      "implementation": {
        "file": "owner_matcher/text_utils.py",
        "type": "function_addition",
        "description": "Add extract_core_name() function to extract primary name before separators",
        "pseudo_code": "def extract_core_name(text):\n    # Split on separators: ';', 'Attn:', 'Estate of', 'c/o', '%'\n    # Return prefix (core name)\n    # Use in matching pipeline before fuzzy comparison"
      },
      "implementation_steps": [
        "1. Add extract_core_name() to text_utils.py",
        "2. Update clean_text() to preserve core name",
        "3. Add core_name field to preprocessing",
        "4. Create CoreNameMatcher strategy in matchers.py",
        "5. Add to cascade before FUZZY_NAME"
      ],
      "risks": {
        "severity": "low",
        "description": "May match different entities with same core name but different modifiers",
        "mitigation": "Require address validation (60%+ score) when matching on core name"
      },
      "validation_checks": [
        "Test on 20 sample temporal changes",
        "Verify address scores are adequate",
        "Ensure no false matches on common names"
      ],
      "approval_status": "pending"
    },
    {
      "id": "S006",
      "priority": "low",
      "type": "threshold_adjustment",
      "title": "Reduce ADDRESS_FIRST_NAME_THRESHOLD from 60 to 55",
      "rationale": "Found 15 potential matches in 55-59 range, but higher risk due to name flexibility.",
      "expected_impact": {
        "additional_matches": 15,
        "improvement_percentage": 1.01,
        "confidence": 0.60
      },
      "implementation": {
        "file": "owner_matcher/config.py",
        "line": 37,
        "current": "ADDRESS_FIRST_NAME_THRESHOLD = 60",
        "proposed": "ADDRESS_FIRST_NAME_THRESHOLD = 55"
      },
      "risks": {
        "severity": "medium",
        "description": "Address-first matching with very low name similarity increases false positive risk"
      },
      "approval_status": "pending"
    }
  ],
  "summary": {
    "total_expected_matches": 182,
    "current_match_count": 147,
    "projected_match_count": 329,
    "projected_match_rate": "13.2%",
    "improvement": "+7.3%",
    "recommendation": "Approve S001, S002, S003 (low risk, high impact). Review S004, S005 with caution."
  }
}
```

---

### Agent 4: Metrics Tracker

**Purpose:** Track performance across iterations and detect convergence

**Output:** `metrics.json`
```json
{
  "iteration": 2,
  "timestamp": "2025-10-20T11:00:00",
  "baseline": {
    "iteration": 0,
    "match_count": 147,
    "match_rate": 0.059,
    "false_positive_rate": 0.02
  },
  "current": {
    "match_count": 170,
    "match_rate": 0.068,
    "false_positive_rate": 0.03,
    "improvement_from_baseline": {
      "absolute": 23,
      "percentage": 0.009,
      "relative_improvement": "15.6%"
    }
  },
  "coverage": {
    "total_records": 2484,
    "matched": 170,
    "unmatched": 2314,
    "match_rate": 0.068
  },
  "confidence_distribution": {
    "high (90-100)": {
      "count": 95,
      "percentage": 55.9,
      "change_from_previous": "+10"
    },
    "medium (75-89)": {
      "count": 55,
      "percentage": 32.4,
      "change_from_previous": "+5"
    },
    "low (60-74)": {
      "count": 20,
      "percentage": 11.8,
      "change_from_previous": "+8"
    }
  },
  "strategy_effectiveness": {
    "DIRECT_ID_MATCH": {
      "count": 41,
      "percentage": 24.1,
      "change": 0
    },
    "EXACT_NAME": {
      "count": 45,
      "percentage": 26.5,
      "change": "+4"
    },
    "ADDRESS_NAME_MATCH": {
      "count": 58,
      "percentage": 34.1,
      "change": "+4"
    },
    "ADDRESS_ATTN_MATCH": {
      "count": 11,
      "percentage": 6.5,
      "change": 0
    },
    "FUZZY_NAME": {
      "count": 15,
      "percentage": 8.8,
      "change": "+15"
    }
  },
  "changes_applied": [
    {
      "change_id": "S001",
      "description": "NAME_THRESHOLD: 70 → 68",
      "expected_impact": 23,
      "actual_impact": 15,
      "effectiveness": 0.65
    },
    {
      "change_id": "S002",
      "description": "Added keywords: corporation, corp, inc, ltd, limited, company, co",
      "expected_impact": 87,
      "actual_impact": 8,
      "effectiveness": 0.09,
      "note": "Lower than expected - many still failed address validation"
    }
  ],
  "iteration_comparison": {
    "iteration_0": {"matches": 147, "rate": 0.059},
    "iteration_1": {"matches": 158, "rate": 0.064, "gain": 11},
    "iteration_2": {"matches": 170, "rate": 0.068, "gain": 12}
  },
  "convergence_analysis": {
    "status": "improving",
    "improvement_rate": 0.004,
    "trend": "positive",
    "iterations_since_significant_improvement": 0,
    "estimated_iterations_to_convergence": "5-7",
    "recommendation": "Continue optimization"
  },
  "quality_metrics": {
    "false_positive_rate": 0.03,
    "false_positive_threshold": 0.05,
    "status": "within_acceptable_range",
    "high_confidence_percentage": 55.9,
    "target": 60.0
  }
}
```

---

### Agent 5: Report Generator

**Purpose:** Create human-readable summaries for stakeholders

**Output:** `reports/progress_report.md`

```markdown
# Owner ID Matching Optimization - Progress Report
**Iteration 2**
**Date:** 2025-10-20
**Status:** ✅ Improving

## Executive Summary
- **Match Rate:** 6.8% (↑ from 5.9% baseline)
- **Total Matches:** 170 (↑ 23 from baseline)
- **False Positive Rate:** 3.0% (within acceptable range)
- **Recommendation:** Continue optimization

## Iteration 2 Results

### Changes Applied
1. ✅ **NAME_THRESHOLD: 70 → 68**
   - Expected: +23 matches
   - Actual: +15 matches
   - Effectiveness: 65%

2. ✅ **Added Corporate Keywords**
   - Expected: +87 matches
   - Actual: +8 matches
   - Effectiveness: 9%
   - Note: Many still failed address validation

### Performance Metrics
| Metric | Iteration 0 | Iteration 1 | Iteration 2 | Change |
|--------|------------|-------------|-------------|---------|
| Matches | 147 | 158 | 170 | +23 |
| Match Rate | 5.9% | 6.4% | 6.8% | +0.9% |
| High Confidence | 85 | 90 | 95 | +10 |
| False Positives | 2.0% | 2.5% | 3.0% | +1.0% |

### Strategy Effectiveness
- **FUZZY_NAME:** +15 matches (most improved)
- **EXACT_NAME:** +4 matches
- **ADDRESS_NAME_MATCH:** +4 matches

## Iteration 3 Recommendations

### High Priority (Approve)
1. **Improve Trust Abbreviation Handling**
   - Impact: +45 matches
   - Risk: Low
   - Effort: Medium

2. **Reduce ADDRESS_THRESHOLD: 75 → 73**
   - Impact: +12 matches
   - Risk: Low
   - Effort: Minimal

### Medium Priority (Review)
3. **Cross-State Matching Relaxation**
   - Impact: +20-30 matches
   - Risk: Medium
   - Requires sample validation

## Unmatched Analysis
- **Total Unmatched:** 2314 owners
- **Near Misses:** 134 (potential quick wins)
- **Addressable Patterns:** 67

## Next Steps
1. Review and approve Iteration 3 suggestions
2. Apply approved changes
3. Run Iteration 3
4. Target: 200+ matches (8% rate)
```

---

## Iteration Workflow

### Step 1: Pre-Iteration Setup (Automated)
```bash
python ai_optimization/run_optimization.py --iteration 1
```

**Actions:**
1. Create iteration directory: `iterations/iteration_001/`
2. Snapshot current configuration → `input_config.json`
3. Run matching process: `python -m owner_matcher.main --use-snowflake`
4. Copy results → `matches.csv`
5. Initialize orchestrator

**Duration:** 2-3 minutes

---

### Step 2: AI Validation Phase (5-10 minutes)

**Match Validator Agent** reviews matched records:

```
FOR EACH matched record WITH confidence >= 75%:
    ANALYZE:
        - Name coherence (abbreviations make sense?)
        - Address proximity (same region?)
        - Business logic (entity types align?)

    IF suspicious:
        FLAG for manual review
        RECORD concern + severity

    TRACK:
        - False positive rate by confidence tier
        - False positive rate by match strategy
```

**Output:** `validations.json`

**Duration:** 5-10 minutes (API calls for ~150 records)

---

### Step 3: AI Analysis Phase (10-15 minutes)

**Unmatched Analyzer Agent** examines gaps:

```
ANALYZE unmatched records:
    1. Near-Miss Detection:
        - Find scores within 5 points of thresholds
        - Categorize by gap size

    2. Pattern Recognition:
        - Extract common name variations
        - Identify missing keywords
        - Detect trust abbreviations

    3. Threshold Sensitivity:
        - Model impact of -2, -5, -10 point reductions
        - Estimate additional matches at each level

    4. State Mismatch Analysis:
        - Count cross-state near-misses
        - Assess validity of relaxing state filter
```

**Output:** `analysis.json`

**Duration:** 10-15 minutes (processing ~2300 records)

---

### Step 4: AI Suggestion Phase (5 minutes)

**Improvement Suggester Agent** generates recommendations:

```
INPUT: validations.json + analysis.json

PROCESS:
    1. Prioritize by (impact × confidence) / risk
    2. Generate specific code changes:
        - File path
        - Line number
        - Current → Proposed
    3. Estimate impact + confidence
    4. Assess risks
    5. Create validation checklist

RANK suggestions:
    High: impact >= 20 matches, risk <= low
    Medium: impact >= 10 matches, risk <= medium
    Low: impact < 10 or risk = high
```

**Output:** `suggestions.json`

**Duration:** 5 minutes

---

### Step 5: Human Review & Approval (30 minutes)

**Human reviews** `suggestions.json`:

```bash
python ai_optimization/review_suggestions.py --iteration 1
```

**Interactive CLI:**
```
═══════════════════════════════════════════════════════════
ITERATION 1 - SUGGESTION REVIEW
═══════════════════════════════════════════════════════════

[S001] HIGH PRIORITY
Title: Reduce NAME_THRESHOLD from 70 to 68
Impact: +23 matches | Risk: Low | Confidence: 85%

Rationale:
Analysis found 23 near-misses in 68-69 range with identifiable
valid patterns. Validation shows current 70 threshold is effective.

Implementation:
  File: owner_matcher/config.py
  Line: 34
  Change: NAME_THRESHOLD = 70 → 68

Approve this suggestion? (y/n/skip): y
✓ S001 APPROVED

───────────────────────────────────────────────────────────

[S002] HIGH PRIORITY
Title: Add missing corporate entity keywords
Impact: +87 matches | Risk: Very Low | Confidence: 95%

Rationale:
Found 87 unmatched records containing standard corporate terms
not in current TRUST_KEYWORDS list.

Approve this suggestion? (y/n/skip): y
✓ S002 APPROVED

───────────────────────────────────────────────────────────
```

**Review Criteria:**
- Does the rationale make sense?
- Is the risk acceptable?
- Is the expected impact worth it?
- Are validation checks adequate?

**Decisions:**
- **Approve:** Mark for implementation
- **Reject:** Skip this change
- **Defer:** Review in next iteration

**Output:** `suggestions_approved.json`

**Duration:** 15-30 minutes

---

### Step 6: Implementation & Re-run (Automated)

```bash
python ai_optimization/apply_changes.py \
  --iteration 1 \
  --approved suggestions_approved.json
```

**Actions:**
1. Parse approved suggestions
2. Apply code changes:
   ```python
   # Example: Update config.py
   with open('owner_matcher/config.py', 'r') as f:
       content = f.read()

   content = content.replace(
       'NAME_THRESHOLD = 70',
       'NAME_THRESHOLD = 68'
   )

   with open('owner_matcher/config.py', 'w') as f:
       f.write(content)
   ```
3. Git commit with detailed message:
   ```
   Iteration 1: Apply optimization changes

   - Reduce NAME_THRESHOLD: 70 → 68
   - Add keywords: corporation, corp, inc, ltd

   Expected impact: +110 matches
   Risk level: Low

   See: iterations/iteration_001/suggestions_approved.json
   ```
4. Trigger new iteration:
   ```bash
   python ai_optimization/run_optimization.py --iteration 2
   ```

**Duration:** 5 minutes + matching runtime

---

### Step 7: Metrics Review (Automated)

**Metrics Tracker Agent** compares results:

```
COMPARE iteration N vs N-1:
    - Match count delta
    - Match rate improvement
    - Confidence distribution shift
    - Strategy effectiveness changes

ASSESS change effectiveness:
    - Expected vs actual impact
    - Effectiveness ratio

CHECK convergence criteria:
    - Improvement rate trend
    - False positive rate
    - Iterations since significant gain

GENERATE dashboard:
    - Line charts: match rate over time
    - Bar charts: matches by strategy
    - Heatmap: confidence distribution
```

**Outputs:**
- `metrics.json` (structured data)
- `reports/metrics_dashboard.html` (visual)
- `reports/progress_report.md` (narrative)

**Duration:** 2-3 minutes

---

## Implementation Plan

### Phase 1: Infrastructure Setup (Week 1)

**Goal:** Build foundation for agent system

**Tasks:**

1. **Directory Structure** (Day 1)
   - Create `ai_optimization/` hierarchy
   - Set up `agents/`, `prompts/`, `iterations/`, `reports/`
   - Initialize Python package structure

2. **Data Models** (Day 2)
   ```python
   # models/validation.py
   class ValidationResult(BaseModel):
       total_reviewed: int
       validated_matches: int
       flagged_for_review: int
       false_positive_rate: float
       flagged_records: List[FlaggedRecord]

   # models/analysis.py
   class AnalysisResult(BaseModel):
       total_unmatched: int
       near_misses: NearMissAnalysis
       common_patterns: PatternAnalysis
       threshold_sensitivity: ThresholdAnalysis

   # models/suggestion.py
   class Suggestion(BaseModel):
       id: str
       priority: Priority
       type: SuggestionType
       implementation: Implementation
       expected_impact: Impact
       risks: Risk
   ```

3. **Orchestrator Framework** (Day 3-4)
   ```python
   # orchestrator.py
   class OptimizationOrchestrator:
       def __init__(self, iteration: int):
           self.iteration = iteration
           self.agents = self._initialize_agents()

       def run_iteration(self):
           # Step 1: Setup
           self.setup_iteration()
           # Step 2-4: Agent execution
           validations = self.agents.validator.run()
           analysis = self.agents.analyzer.run()
           suggestions = self.agents.suggester.run(validations, analysis)
           # Step 5-7: Human review + metrics
           return suggestions
   ```

4. **Metrics System** (Day 5)
   - SQLite database for metrics history
   - Comparison functions
   - Dashboard generation (Plotly)

**Deliverables:**
- ✅ Working directory structure
- ✅ Pydantic models for all data structures
- ✅ Basic orchestration framework
- ✅ Metrics tracking system
- ✅ Git repository initialized

---

### Phase 2: Agent Development (Week 2-3)

**Goal:** Build three core agents

#### Week 2: Agents 1 & 2

**Agent 1: Match Validator** (Days 1-2)
```python
# agents/match_validator.py
class MatchValidator:
    def __init__(self, claude_client: ClaudeClient):
        self.client = claude_client
        self.prompt = load_prompt('match_validation.md')

    def run(self, matches_df: pd.DataFrame) -> ValidationResult:
        # Sample high/medium confidence matches
        sample = self._stratified_sample(matches_df)

        # Batch validate via Claude
        validations = []
        for batch in self._batch(sample, size=10):
            prompt = self._build_prompt(batch)
            response = self.client.complete(prompt)
            validations.extend(self._parse_response(response))

        # Aggregate results
        return ValidationResult(
            total_reviewed=len(sample),
            validated_matches=sum(v.is_valid for v in validations),
            flagged_records=[v for v in validations if v.flagged]
        )
```

**Prompt Template:** `prompts/match_validation.md`
```markdown
# Match Validation Task

You are validating owner ID matches for an oil & gas database.

## Records to Validate

{{#each batch}}
Record {{@index}}:
- Old Name: {{old_name}}
- New Name: {{new_name}}
- Old Address: {{old_address}}
- New Address: {{new_address}}
- Match Strategy: {{match_step}}
- Name Score: {{name_score}}
- Address Score: {{address_score}}
{{/each}}

## Validation Criteria

For each record, assess:
1. Name Coherence: Do the names represent the same entity?
2. Address Proximity: Are addresses in same/nearby locations?
3. Business Logic: Do entity types align?
4. Risk Level: High/Medium/Low

## Output Format (JSON)

Return array of validations:
[
  {
    "record_id": 0,
    "is_valid": true,
    "confidence": 0.95,
    "concerns": [],
    "recommendation": "approve"
  },
  ...
]
```

**Agent 2: Unmatched Analyzer** (Days 3-5)
```python
# agents/unmatched_analyzer.py
class UnmatchedAnalyzer:
    def run(self, matches_df: pd.DataFrame) -> AnalysisResult:
        # 1. Near-miss detection
        near_misses = self._find_near_misses(matches_df)

        # 2. Pattern recognition
        patterns = self._identify_patterns(matches_df)

        # 3. Threshold sensitivity
        sensitivity = self._analyze_thresholds(matches_df)

        # 4. Use Claude for complex pattern analysis
        advanced_patterns = self._claude_pattern_analysis(
            sample=near_misses[:50]
        )

        return AnalysisResult(
            total_unmatched=len(matches_df[matches_df.mapped_new_id.isna()]),
            near_misses=near_misses,
            common_patterns=patterns,
            threshold_sensitivity=sensitivity
        )

    def _find_near_misses(self, df):
        # Find records just below thresholds
        near_name = df[
            (df.name_score >= 65) &
            (df.name_score < 70) &
            (df.mapped_new_id.isna())
        ]
        return near_name
```

#### Week 3: Agent 3 & Integration

**Agent 3: Improvement Suggester** (Days 1-3)
```python
# agents/improvement_suggester.py
class ImprovementSuggester:
    def run(
        self,
        validations: ValidationResult,
        analysis: AnalysisResult
    ) -> List[Suggestion]:

        suggestions = []

        # 1. Threshold adjustments
        suggestions.extend(
            self._suggest_threshold_changes(analysis)
        )

        # 2. Keyword additions
        suggestions.extend(
            self._suggest_keyword_additions(analysis)
        )

        # 3. Code enhancements
        suggestions.extend(
            self._claude_suggest_enhancements(validations, analysis)
        )

        # 4. Prioritize and rank
        ranked = self._prioritize_suggestions(suggestions, validations)

        return ranked

    def _suggest_threshold_changes(self, analysis):
        # Data-driven threshold suggestions
        suggestions = []

        for threshold_name, sensitivity in analysis.threshold_sensitivity.items():
            if sensitivity.near_misses_at_minus_2 > 10:
                suggestions.append(Suggestion(
                    type="threshold_adjustment",
                    current_value=sensitivity.current,
                    proposed_value=sensitivity.current - 2,
                    expected_impact=sensitivity.near_misses_at_minus_2,
                    ...
                ))

        return suggestions
```

**Integration Testing** (Days 4-5)
- End-to-end test with sample data
- Verify JSON schema compliance
- Test error handling
- Validate Claude API integration

**Deliverables:**
- ✅ Working Match Validator agent
- ✅ Working Unmatched Analyzer agent
- ✅ Working Improvement Suggester agent
- ✅ Prompt templates for each agent
- ✅ Unit tests for core functions
- ✅ Integration test suite

---

### Phase 3: Workflow Automation (Week 4)

**Goal:** Automate end-to-end workflow

**Tasks:**

1. **Run Optimization Script** (Days 1-2)
   ```python
   # run_optimization.py
   @click.command()
   @click.option('--iteration', type=int, required=True)
   @click.option('--use-snowflake', is_flag=True)
   def main(iteration: int, use_snowflake: bool):
       """Run optimization iteration."""

       # Setup
       iteration_dir = create_iteration_dir(iteration)
       snapshot_config(iteration_dir)

       # Run matching
       run_matching_process(use_snowflake)

       # Initialize orchestrator
       orchestrator = OptimizationOrchestrator(iteration)

       # Run agents
       results = orchestrator.run_iteration()

       # Save outputs
       save_results(iteration_dir, results)

       # Generate reports
       generate_reports(iteration_dir, results)

       print(f"✓ Iteration {iteration} complete")
       print(f"  See: {iteration_dir}")
   ```

2. **Change Application Script** (Days 3-4)
   ```python
   # apply_changes.py
   def apply_changes(iteration: int, approved_file: str):
       """Apply approved changes from suggestions."""

       # Load approved suggestions
       approved = load_approved_suggestions(approved_file)

       # Apply each change
       for suggestion in approved:
           if suggestion.type == "threshold_adjustment":
               apply_threshold_change(suggestion)
           elif suggestion.type == "keyword_addition":
               apply_keyword_addition(suggestion)
           elif suggestion.type == "code_enhancement":
               print(f"⚠️  Manual implementation required: {suggestion.title}")

       # Git commit
       commit_changes(iteration, approved)

       print("✓ Changes applied successfully")
   ```

3. **Review Interface** (Day 5)
   ```python
   # review_suggestions.py
   def interactive_review(iteration: int):
       """Interactive CLI for reviewing suggestions."""

       suggestions = load_suggestions(iteration)
       approved = []

       for s in suggestions:
           display_suggestion(s)
           decision = prompt_user()  # y/n/skip

           if decision == 'y':
               approved.append(s)

       save_approved_suggestions(iteration, approved)
   ```

**Deliverables:**
- ✅ Automated iteration runner
- ✅ Change application system
- ✅ Interactive review CLI
- ✅ Git integration
- ✅ Error handling and rollback

---

### Phase 4: Reporting & Documentation (Week 5)

**Goal:** Create reporting system and documentation

**Tasks:**

1. **Metrics Dashboard** (Days 1-2)
   ```python
   # Generate HTML dashboard with Plotly
   def generate_dashboard(metrics_history):
       # Line chart: match rate over time
       fig1 = px.line(metrics_history, x='iteration', y='match_rate')

       # Bar chart: matches by strategy
       fig2 = px.bar(latest_metrics, x='strategy', y='count')

       # Heatmap: confidence distribution
       fig3 = px.imshow(confidence_matrix)

       # Combine into dashboard
       dashboard = create_html_dashboard([fig1, fig2, fig3])
       save(dashboard, 'reports/metrics_dashboard.html')
   ```

2. **Progress Reports** (Day 3)
   - Markdown report generator
   - Executive summary
   - Detailed metrics
   - Recommendations

3. **User Documentation** (Days 4-5)
   - How to run optimization
   - How to review suggestions
   - How to interpret results
   - Troubleshooting guide

**Deliverables:**
- ✅ Interactive metrics dashboard
- ✅ Automated report generation
- ✅ Complete user documentation
- ✅ API documentation
- ✅ Example workflows

---

## Safety Mechanisms

### Guardrails

**1. Threshold Limits**
```python
THRESHOLD_CONSTRAINTS = {
    'NAME_THRESHOLD': {'min': 50, 'max': 90},
    'ADDRESS_THRESHOLD': {'min': 60, 'max': 90},
    'ADDRESS_MIN': {'min': 40, 'max': 80},
    'ADDRESS_FIRST_NAME_THRESHOLD': {'min': 45, 'max': 75}
}

MAX_REDUCTION_PER_ITERATION = 5  # Never reduce more than 5 points at once
```

**Enforcement:**
```python
def validate_threshold_change(current, proposed):
    if proposed < THRESHOLD_CONSTRAINTS[name]['min']:
        raise ValueError(f"Threshold {name} cannot go below {min}")

    if abs(current - proposed) > MAX_REDUCTION_PER_ITERATION:
        raise ValueError(f"Maximum reduction is {MAX_REDUCTION_PER_ITERATION} points")
```

**2. Change Limits**
```python
MAX_CHANGES_PER_ITERATION = 5
REQUIRE_HUMAN_APPROVAL = True  # All changes need approval
TEST_BEFORE_APPLY = True  # Run on sample first
```

**3. Convergence Criteria**

**Stop Conditions:**
```python
def check_convergence(metrics_history):
    # 1. Maximum iterations
    if len(metrics_history) >= 10:
        return "max_iterations_reached"

    # 2. Diminishing returns (< 1% improvement for 3 iterations)
    recent = metrics_history[-3:]
    improvements = [m.improvement_rate for m in recent]
    if all(imp < 0.01 for imp in improvements):
        return "diminishing_returns"

    # 3. False positive rate exceeded
    if metrics_history[-1].false_positive_rate > 0.10:
        return "high_false_positive_rate"

    return None  # Continue
```

### Audit Trail

**Git Integration:**
```python
def commit_changes(iteration, approved_suggestions):
    """Create detailed git commit."""

    message = f"""Iteration {iteration}: Apply optimization changes

Changes Applied:
"""
    for s in approved_suggestions:
        message += f"- {s.title}\n"
        message += f"  Expected: {s.expected_impact.additional_matches} matches\n"
        message += f"  Risk: {s.risks.severity}\n"

    message += f"\nSee: iterations/iteration_{iteration:03d}/suggestions_approved.json"

    subprocess.run(['git', 'add', '.'])
    subprocess.run(['git', 'commit', '-m', message])
    subprocess.run(['git', 'tag', f'iteration-{iteration}'])
```

**Iteration Archives:**
```
iterations/
├── iteration_001/
│   ├── input_config.json       # Snapshot for rollback
│   ├── matches.csv             # Results
│   ├── validations.json        # AI validation
│   ├── analysis.json           # Gap analysis
│   ├── suggestions.json        # All suggestions
│   ├── suggestions_approved.json  # Approved changes
│   └── metrics.json            # Performance
```

**Rollback Capability:**
```python
def rollback_to_iteration(target_iteration: int):
    """Rollback to previous iteration."""

    # Checkout git tag
    subprocess.run(['git', 'checkout', f'iteration-{target_iteration}'])

    # Restore config
    restore_config(f'iterations/iteration_{target_iteration:03d}/input_config.json')

    print(f"✓ Rolled back to iteration {target_iteration}")
```

### Human Oversight

**Required Approvals:**
```python
APPROVAL_REQUIRED = {
    'threshold_adjustment': True,   # Always require approval
    'keyword_addition': True,       # Low risk but still approve
    'code_enhancement': True,       # Always manual review
}
```

**Sample Validation:**
```python
def validate_changes_on_sample(approved_suggestions):
    """Test changes on sample before full deployment."""

    # Apply changes temporarily
    with temporary_changes(approved_suggestions):
        # Run on sample (100 records)
        sample_results = run_matching_on_sample(n=100)

        # Manual review
        print("Sample Results:")
        display_sample_results(sample_results)

        proceed = input("Proceed with full deployment? (y/n): ")
        return proceed.lower() == 'y'
```

**Quality Gates:**
```python
def check_quality_gates(iteration_metrics):
    """Check if results meet quality standards."""

    gates = {
        'false_positive_rate': 0.10,    # Must be < 10%
        'high_confidence_min': 0.50,    # At least 50% high confidence
        'improvement_min': 0.01         # At least 1% improvement
    }

    failures = []

    if iteration_metrics.false_positive_rate > gates['false_positive_rate']:
        failures.append("False positive rate too high")

    if iteration_metrics.high_confidence_pct < gates['high_confidence_min']:
        failures.append("Too few high-confidence matches")

    if failures:
        print("❌ Quality gates failed:")
        for f in failures:
            print(f"  - {f}")
        return False

    return True
```

---

## Expected Outcomes

### Iteration Progression

**Iteration 0 (Baseline)**
```
Match Rate: 5.9% (147/2484)
Distribution:
  - High Confidence (90-100): 85 (57.8%)
  - Medium Confidence (75-89): 50 (34.0%)
  - Low Confidence (60-74): 12 (8.2%)

Strategy Distribution:
  - DIRECT_ID_MATCH: 41
  - EXACT_NAME: 41
  - ADDRESS_NAME_MATCH: 54
  - ADDRESS_ATTN_MATCH: 20
  - FUZZY_NAME: 11

Configuration:
  - NAME_THRESHOLD: 70
  - ADDRESS_THRESHOLD: 75
  - ADDRESS_MIN: 60
  - ADDRESS_FIRST_NAME_THRESHOLD: 60
```

**Iteration 3 (Projected)**
```
Match Rate: 12-15% (~300-375 matches)
Distribution:
  - High Confidence: 200+ (60%+)
  - Medium Confidence: 100+
  - Low Confidence: <50

Improvements Applied:
  - Optimized thresholds (NAME: 65, ADDRESS: 72)
  - Additional trust keywords (15+)
  - Better address normalization
  - Improved abbreviation handling

False Positive Rate: <5%
```

**Iteration 10 (Target/Convergence)**
```
Match Rate: 25-30% (~620-745 matches)
Distribution:
  - High Confidence: 400+ (60%+)
  - Medium Confidence: 200+
  - Low Confidence: <100

Indicators:
  - Improvement < 1% for 3 consecutive iterations
  - Remaining unmatched likely legitimate (inactive)
  - Diminishing returns on threshold reductions
  - Most addressable patterns handled

False Positive Rate: <5%
Recommendation: Conclude optimization, maintain configuration
```

### Projected Impact by Change Type

| Change Type | Expected Matches | Confidence | Risk |
|-------------|-----------------|------------|------|
| Threshold optimization | 100-150 | High | Low-Med |
| Keyword additions | 150-200 | High | Low |
| Trust abbreviation handling | 50-75 | Medium | Low |
| Address normalization | 75-100 | Medium | Low |
| Cross-state relaxation | 50-75 | Medium | Med |
| **Total** | **425-600** | | |

### Success Criteria by Phase

**Phase 1 (Iterations 1-3):**
- ✅ Match rate: 10%+
- ✅ System working end-to-end
- ✅ Agent suggestions are actionable
- ✅ Human review process is efficient

**Phase 2 (Iterations 4-7):**
- ✅ Match rate: 15-20%
- ✅ False positive rate < 5%
- ✅ 60%+ matches at high confidence
- ✅ Convergence indicators present

**Phase 3 (Iterations 8-10):**
- ✅ Match rate: 20-30%
- ✅ Diminishing returns detected
- ✅ System automatically suggests stopping
- ✅ Documentation complete for handoff

---

## Technology Stack

### AI/LLM
- **Claude 3.5 Sonnet** via Anthropic API
  - Structured output parsing
  - Context window: ~200K tokens
  - JSON mode for reliable parsing
  - Batch processing for efficiency

### Orchestration
- **Python 3.10+**
  - asyncio for concurrent agent execution
  - Type hints throughout
  - Pydantic for data validation

### CLI & UX
- **Click** - Command-line interface
- **Rich** - Beautiful terminal output
- **Questionary** - Interactive prompts

### Data Management
- **Pandas** - Data manipulation
- **Pydantic** - Schema validation
- **JSON** - Structured outputs
- **SQLite** - Metrics history (optional)

### Visualization
- **Plotly** - Interactive charts
- **Markdown** - Report generation
- **Jinja2** - Template rendering

### Version Control
- **Git** - Change tracking
- **GitPython** - Programmatic git operations

### Dependencies
```python
# requirements-ai-optimization.txt
anthropic>=0.25.0
pandas>=2.0.0
pydantic>=2.0.0
click>=8.1.0
rich>=13.0.0
questionary>=2.0.0
plotly>=5.18.0
jinja2>=3.1.0
gitpython>=3.1.0
```

---

## Cost Estimation

### Per Iteration Costs

**Agent 1: Match Validator**
- Records reviewed: ~150 (75% of matches)
- Tokens per record: ~500 (input: 300, output: 200)
- Total tokens: 75,000
- Cost: ~$0.50

**Agent 2: Unmatched Analyzer**
- Records analyzed: Sample of 200 from 2300
- Tokens per batch: ~1000 (complex analysis)
- Total tokens: 200,000
- Cost: ~$2.00

**Agent 3: Improvement Suggester**
- Input: Summary data (~5000 tokens)
- Output: Structured suggestions (~5000 tokens)
- Total tokens: 10,000
- Cost: ~$0.30

**Agent 5: Report Generator**
- Input: Metrics data (~2000 tokens)
- Output: Markdown report (~3000 tokens)
- Total tokens: 5,000
- Cost: ~$0.15

**Total per iteration:** ~$3-5

### Full Optimization Cost

**10 Iterations:** $30-50 total

**ROI Calculation:**
- Cost: $50
- Current manual matching time: 40 hours @ $50/hr = $2000
- If AI reduces manual work by 50%: Save $1000
- If we recover 400 additional owners: Potential value in thousands

**ROI:** 20x-40x return on investment

---

## Success Metrics

### Primary Metrics

**1. Match Coverage**
- **Baseline:** 5.9% (147/2484)
- **Target:** 20-30% (500-745 matches)
- **Measurement:** `matched_count / total_records`

**2. Confidence Quality**
- **Baseline:** 57.8% high confidence
- **Target:** 60%+ high confidence
- **Measurement:** `high_confidence_matches / total_matches`

**3. False Positive Rate**
- **Baseline:** ~2%
- **Target:** <5%
- **Measurement:** Manual sample validation

**4. Iteration Efficiency**
- **Target:** Meaningful improvement within 5-7 iterations
- **Measurement:** Improvement rate per iteration

**5. Human Time Saved**
- **Target:** Reduce manual matching by 50%
- **Measurement:** Hours before vs after

### Secondary Metrics

**6. Strategy Effectiveness**
- Track which matching strategies improve most
- Identify underperforming strategies

**7. Threshold Optimization**
- Find optimal balance between precision and recall
- Document final recommended thresholds

**8. Pattern Coverage**
- Percentage of identifiable patterns addressed
- Remaining unaddressable patterns

**9. Convergence Speed**
- Iterations to reach 90% of potential improvement
- Diminishing returns detection

**10. Agent Performance**
- Suggestion acceptance rate
- Expected vs actual impact accuracy
- False positive prediction accuracy

### Tracking Dashboard

```python
# Example metrics tracking
{
    "iteration": 5,
    "primary_metrics": {
        "match_coverage": {
            "current": 0.18,
            "target": 0.25,
            "progress": 0.72
        },
        "confidence_quality": {
            "current": 0.62,
            "target": 0.60,
            "progress": 1.03
        },
        "false_positive_rate": {
            "current": 0.04,
            "target": 0.05,
            "status": "within_target"
        }
    },
    "improvement_trajectory": {
        "last_3_iterations": [0.015, 0.012, 0.008],
        "trend": "diminishing",
        "recommendation": "2-3 more iterations then conclude"
    }
}
```

---

## Next Steps

### Immediate Actions (Week 1)
1. ✅ Review and approve this strategy document
2. 📋 Set up project directory structure
3. 📋 Initialize git repository for ai_optimization/
4. 📋 Set up Anthropic API access
5. 📋 Create initial data models (Pydantic)

### Week 2-5: Implementation
Follow the 4-phase implementation plan outlined above.

### Week 6+: Production Operation
1. Run first production iteration
2. Establish weekly iteration cadence
3. Monitor convergence
4. Document lessons learned
5. Prepare for handoff/maintenance

---

## Appendix

### Example Prompts

See `prompts/` directory for detailed prompt templates:
- `match_validation.md` - Validation criteria
- `unmatched_analysis.md` - Pattern analysis guide
- `improvement_suggestion.md` - Suggestion template
- `report_generation.md` - Report format

### Configuration Schema

See `models/` directory for Pydantic models:
- `validation.py` - ValidationResult schema
- `analysis.py` - AnalysisResult schema
- `suggestion.py` - Suggestion schema
- `metrics.py` - MetricsResult schema

### Troubleshooting

**Common Issues:**
1. Claude API rate limits → Add retry logic with exponential backoff
2. Large context windows → Implement batch processing
3. Parsing errors → Use structured output mode
4. Git conflicts → Ensure clean working directory before iterations

---

**Document Version:** 1.0
**Last Updated:** 2025-10-20
**Status:** Ready for Implementation
**Approver:** [Pending]

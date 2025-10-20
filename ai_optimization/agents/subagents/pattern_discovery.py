"""
Pattern Discovery Subagent

This agent continuously mines unmatched records to discover new matching patterns
and improve the system's understanding of data variations.
"""

import asyncio
import json
import logging
import re
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from rapidfuzz import fuzz
import anthropic

from ..context_manager import Pattern

logger = logging.getLogger(__name__)


@dataclass
class PatternCandidate:
    """Candidate pattern discovered from data."""
    pattern_type: str
    pattern_value: str
    examples: List[Dict[str, Any]]
    frequency: int
    confidence: float
    description: str


class PatternDiscoveryAgent:
    """
    Discovers patterns in unmatched records to improve matching strategies.

    Pattern types discovered:
    - Name variations (abbreviations, suffixes)
    - Address format variations
    - Temporal changes (estate transitions, contact changes)
    - Trust and entity patterns
    - Industry-specific conventions
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize pattern discovery agent.

        Args:
            config: Agent configuration from main config
        """
        self.config = config
        self.batch_size = config.get('batch_size', 100)
        self.min_pattern_frequency = config.get('min_pattern_frequency', 3)
        self.confidence_threshold = config.get('confidence_threshold', 0.8)
        self.pattern_types = config.get('pattern_types', [])

        # Initialize Claude client if API analysis is enabled
        self.use_ai = config.get('use_ai', True)
        self.claude_client = None
        if self.use_ai:
            try:
                # API key can be set via:
                # 1. Environment variable: ANTHROPIC_API_KEY
                # 2. .env file in project root
                # 3. Passed directly (not recommended for production)
                import os
                from pathlib import Path

                api_key = os.environ.get('ANTHROPIC_API_KEY')

                # Try loading from .env file if not in environment
                if not api_key:
                    env_file = Path(__file__).parent.parent.parent.parent / '.env'
                    if env_file.exists():
                        try:
                            with open(env_file, 'r') as f:
                                for line in f:
                                    if line.strip().startswith('ANTHROPIC_API_KEY='):
                                        api_key = line.split('=', 1)[1].strip().strip('"\'')
                                        if api_key:
                                            logger.info("Loaded API key from .env file")
                                            break
                        except Exception as e:
                            logger.debug(f"Could not read .env file: {e}")

                if api_key:
                    self.claude_client = anthropic.Anthropic(api_key=api_key)
                    logger.info("Claude AI client initialized successfully")
                else:
                    # Try default initialization (will use ANTHROPIC_API_KEY env var)
                    self.claude_client = anthropic.Anthropic()
                    logger.info("Claude AI client initialized with default settings")
            except Exception as e:
                logger.warning(f"Claude AI initialization failed: {e}")
                logger.warning("AI features disabled. Set ANTHROPIC_API_KEY environment variable or add to .env file to enable.")
                self.use_ai = False
                self.claude_client = None

        # Pattern storage
        self.discovered_patterns = []
        self.pattern_cache = defaultdict(list)

    async def execute(self, task_type: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute pattern discovery task.

        Args:
            task_type: Type of task to execute
            input_data: Input data for the task

        Returns:
            Discovered patterns and insights
        """
        if task_type == 'analyze_unmatched':
            return await self.analyze_unmatched_records(
                input_data['records'],
                input_data.get('previous_patterns', []),
                input_data.get('iteration', 0)
            )
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    async def analyze_unmatched_records(
        self,
        unmatched_df: pd.DataFrame,
        previous_patterns: List[Pattern],
        iteration: int
    ) -> Dict[str, Any]:
        """
        Analyze unmatched records to discover patterns.

        Args:
            unmatched_df: DataFrame of unmatched records
            previous_patterns: Previously discovered patterns
            iteration: Current iteration number

        Returns:
            Dictionary containing discovered patterns and analysis
        """
        logger.info(f"Analyzing {len(unmatched_df)} unmatched records for patterns")

        # Run pattern discovery methods in parallel
        tasks = []

        if 'name_variations' in self.pattern_types:
            tasks.append(self._discover_name_patterns(unmatched_df))

        if 'address_formats' in self.pattern_types:
            tasks.append(self._discover_address_patterns(unmatched_df))

        if 'temporal_changes' in self.pattern_types:
            tasks.append(self._discover_temporal_patterns(unmatched_df))

        if 'abbreviations' in self.pattern_types:
            tasks.append(self._discover_abbreviation_patterns(unmatched_df))

        if 'trust_patterns' in self.pattern_types:
            tasks.append(self._discover_trust_patterns(unmatched_df))

        # Execute pattern discovery tasks
        pattern_results = await asyncio.gather(*tasks)

        # Combine all discovered patterns
        all_patterns = []
        for result in pattern_results:
            all_patterns.extend(result)

        # Filter and rank patterns
        filtered_patterns = self._filter_patterns(all_patterns, previous_patterns)

        # Use AI to analyze complex patterns if enabled
        if self.use_ai and len(unmatched_df) > 0:
            ai_patterns = await self._ai_pattern_analysis(
                unmatched_df.sample(min(self.batch_size, len(unmatched_df)))
            )
            filtered_patterns.extend(ai_patterns)

        # Convert to Pattern objects
        final_patterns = []
        for candidate in filtered_patterns:
            pattern = Pattern(
                pattern_type=candidate.pattern_type,
                pattern_value=candidate.pattern_value,
                frequency=candidate.frequency,
                confidence=candidate.confidence,
                discovered_iteration=iteration,
                last_seen_iteration=iteration,
                examples=candidate.examples[:10],  # Keep top 10 examples
                metadata={'description': candidate.description}
            )
            final_patterns.append(pattern)

        # Generate insights
        insights = self._generate_insights(final_patterns, unmatched_df)

        return {
            'patterns': final_patterns,
            'pattern_count': len(final_patterns),
            'insights': insights,
            'unmatched_characteristics': self._analyze_unmatched_characteristics(unmatched_df)
        }

    async def _discover_name_patterns(self, df: pd.DataFrame) -> List[PatternCandidate]:
        """
        Discover patterns in owner names.

        Patterns include:
        - Common prefixes/suffixes
        - Word frequency patterns
        - Name structure variations
        """
        patterns = []

        # Use clean_name if available (from preprocessing), otherwise use EXCLUDE_OWNER_NAME from Snowflake
        if 'clean_name' in df.columns:
            name_col = 'clean_name'
        elif 'EXCLUDE_OWNER_NAME' in df.columns:
            name_col = 'EXCLUDE_OWNER_NAME'
        else:
            logger.warning("No name column found in dataframe. Available columns: %s", df.columns.tolist())
            return patterns

        # Analyze name components
        name_components = defaultdict(int)
        suffix_patterns = defaultdict(list)
        prefix_patterns = defaultdict(list)

        for idx, row in df.iterrows():
            # Get name - if using EXCLUDE_OWNER_NAME, clean it
            name = str(row.get(name_col, ''))

            # Clean the name if it's not already clean
            if name_col == 'EXCLUDE_OWNER_NAME':
                # Import clean_text from owner_matcher
                try:
                    import sys
                    from pathlib import Path
                    sys.path.append(str(Path(__file__).parent.parent.parent.parent))
                    from owner_matcher.text_utils import clean_text
                    name = clean_text(name)
                except Exception as e:
                    logger.warning(f"Could not import clean_text: {e}. Using raw name.")
                    name = name.lower()

            if len(name) < 3:
                continue

            # Extract words
            words = name.split()

            # Check for common suffixes
            if len(words) > 1:
                last_word = words[-1]
                if last_word in ['llc', 'inc', 'corp', 'ltd', 'lp', 'trust']:
                    suffix_patterns[last_word].append({
                        'name': name,
                        'id': row.get('old_owner_id', row.get('OLD_OWNER_ID', ''))
                    })

            # Check for common prefixes
            if len(words) > 1:
                first_word = words[0]
                if len(first_word) > 2:
                    prefix_patterns[first_word].append({
                        'name': name,
                        'id': row.get('old_owner_id', row.get('OLD_OWNER_ID', ''))
                    })

            # Count word frequencies
            for word in words:
                if len(word) > 3:  # Skip short words
                    name_components[word] += 1

        # Create patterns from frequent components
        for component, count in name_components.items():
            if count >= self.min_pattern_frequency:
                examples = []
                for idx, row in df.iterrows():
                    name = str(row.get(name_col, ''))
                    if name_col != 'clean_name':
                        try:
                            from owner_matcher.text_utils import clean_text
                            name = clean_text(name)
                        except:
                            name = name.lower()

                    if component in name:
                        examples.append({
                            'name': row.get(name_col, ''),
                            'clean_name': name
                        })
                        if len(examples) >= 5:
                            break

                patterns.append(PatternCandidate(
                    pattern_type='name_component',
                    pattern_value=component,
                    examples=examples,
                    frequency=count,
                    confidence=min(count / len(df), 1.0),
                    description=f"Common name component '{component}' appears {count} times"
                ))

        # Create suffix patterns
        for suffix, examples in suffix_patterns.items():
            if len(examples) >= self.min_pattern_frequency:
                patterns.append(PatternCandidate(
                    pattern_type='name_suffix',
                    pattern_value=suffix,
                    examples=examples[:5],
                    frequency=len(examples),
                    confidence=0.9,
                    description=f"Entity suffix '{suffix}' is common in unmatched records"
                ))

        return patterns

    async def _discover_address_patterns(self, df: pd.DataFrame) -> List[PatternCandidate]:
        """
        Discover patterns in address formats.

        Patterns include:
        - PO Box variations
        - Suite/Unit formats
        - Street type abbreviations
        """
        patterns = []

        if 'Last Known Address' not in df.columns:
            return patterns

        # Regex patterns for address components
        po_box_pattern = r'(p\.?\s*o\.?\s*box|pob|post\s*office\s*box)\s*(\d+)'
        suite_pattern = r'(suite|ste|unit|apt|#)\s*(\w+)'
        street_types = defaultdict(list)

        for idx, row in df.iterrows():
            address = str(row.get('Last Known Address', '')).lower()

            # Check for PO Box variations
            po_match = re.search(po_box_pattern, address, re.IGNORECASE)
            if po_match:
                variation = po_match.group(1)
                patterns.append(PatternCandidate(
                    pattern_type='address_po_box_variation',
                    pattern_value=variation,
                    examples=[{'address': address, 'id': row.get('OLD_OWNER_ID', '')}],
                    frequency=1,
                    confidence=0.8,
                    description=f"PO Box variation: '{variation}'"
                ))

            # Check for suite/unit variations
            suite_match = re.search(suite_pattern, address, re.IGNORECASE)
            if suite_match:
                variation = suite_match.group(1)
                street_types[variation].append(address)

        # Aggregate street type patterns
        for street_type, addresses in street_types.items():
            if len(addresses) >= self.min_pattern_frequency:
                patterns.append(PatternCandidate(
                    pattern_type='address_unit_variation',
                    pattern_value=street_type,
                    examples=[{'address': addr} for addr in addresses[:5]],
                    frequency=len(addresses),
                    confidence=0.85,
                    description=f"Unit/Suite variation: '{street_type}'"
                ))

        return patterns

    async def _discover_temporal_patterns(self, df: pd.DataFrame) -> List[PatternCandidate]:
        """
        Discover temporal change patterns in names.

        Focus on:
        - Estate indicators
        - Attention line changes
        - Trustee transitions
        """
        patterns = []
        temporal_indicators = [
            'estate of', 'deceased', 'successor', 'trustee',
            'attn:', 'c/o', 'care of', 'attention'
        ]

        temporal_matches = defaultdict(list)

        # Determine name column
        name_col = None
        # Use clean_name or EXCLUDE_OWNER_NAME from Snowflake
        if 'clean_name' in df.columns:
            name_col = 'clean_name'
        elif 'EXCLUDE_OWNER_NAME' in df.columns:
            name_col = 'EXCLUDE_OWNER_NAME'

        if name_col is None:
            logger.warning("No name column found for temporal pattern discovery")
            return patterns

        for idx, row in df.iterrows():
            name = str(row.get(name_col, '')).lower()

            for indicator in temporal_indicators:
                if indicator in name:
                    temporal_matches[indicator].append({
                        'name': row.get(name_col, ''),
                        'clean_name': row.get('clean_name', name),
                        'id': row.get('old_owner_id', row.get('OLD_OWNER_ID', ''))
                    })

        # Create patterns for frequent temporal indicators
        for indicator, examples in temporal_matches.items():
            if len(examples) >= self.min_pattern_frequency:
                patterns.append(PatternCandidate(
                    pattern_type='temporal_indicator',
                    pattern_value=indicator,
                    examples=examples[:5],
                    frequency=len(examples),
                    confidence=0.95,
                    description=f"Temporal indicator '{indicator}' suggests name changes over time"
                ))

        return patterns

    async def _discover_abbreviation_patterns(self, df: pd.DataFrame) -> List[PatternCandidate]:
        """
        Discover abbreviation patterns in unmatched records.

        Focus on:
        - Common abbreviations not being matched
        - Inconsistent abbreviation usage
        """
        patterns = []
        abbreviations = {
            'corporation': ['corp', 'corporation'],
            'incorporated': ['inc', 'incorporated'],
            'limited': ['ltd', 'limited'],
            'company': ['co', 'company'],
            'associates': ['assoc', 'associates'],
            'international': ['intl', 'international'],
            'management': ['mgmt', 'management'],
            'properties': ['props', 'properties']
        }

        abbrev_mismatches = defaultdict(list)

        # Determine name columns
        name_col = 'clean_name' if 'clean_name' in df.columns else 'EXCLUDE_OWNER_NAME'
        clean_col = 'clean_name' if 'clean_name' in df.columns else name_col

        for idx, row in df.iterrows():
            name = str(row.get(clean_col, '')).lower()

            for full_form, abbrevs in abbreviations.items():
                for abbrev in abbrevs:
                    if abbrev in name:
                        abbrev_mismatches[f"{full_form}_variations"].append({
                            'name': row.get(name_col, ''),
                            'variation': abbrev,
                            'id': row.get('old_owner_id', row.get('OLD_OWNER_ID', ''))
                        })

        # Create patterns for abbreviation mismatches
        for pattern_name, examples in abbrev_mismatches.items():
            if len(examples) >= self.min_pattern_frequency:
                patterns.append(PatternCandidate(
                    pattern_type='abbreviation_mismatch',
                    pattern_value=pattern_name,
                    examples=examples[:5],
                    frequency=len(examples),
                    confidence=0.85,
                    description=f"Abbreviation variations for {pattern_name.replace('_variations', '')}"
                ))

        return patterns

    async def _discover_trust_patterns(self, df: pd.DataFrame) -> List[PatternCandidate]:
        """
        Discover patterns specific to trust entities.

        Focus on:
        - Trust naming conventions
        - Trust type variations
        - Family trust patterns
        """
        patterns = []
        trust_types = defaultdict(list)
        trust_keywords = [
            'trust', 'living trust', 'revocable trust', 'family trust',
            'irrevocable trust', 'testamentary trust', 'charitable trust'
        ]

        # Determine name column
        name_col = None
        # Use clean_name or EXCLUDE_OWNER_NAME from Snowflake
        if 'clean_name' in df.columns:
            name_col = 'clean_name'
        elif 'EXCLUDE_OWNER_NAME' in df.columns:
            name_col = 'EXCLUDE_OWNER_NAME'

        if name_col is None:
            return patterns

        for idx, row in df.iterrows():
            name = str(row.get(name_col, '')).lower()

            for keyword in trust_keywords:
                if keyword in name:
                    trust_types[keyword].append({
                        'name': row.get(name_col, ''),
                        'clean_name': row.get('clean_name', name),
                        'id': row.get('old_owner_id', row.get('OLD_OWNER_ID', ''))
                    })

        # Create patterns for trust variations
        for trust_type, examples in trust_types.items():
            if len(examples) >= self.min_pattern_frequency:
                patterns.append(PatternCandidate(
                    pattern_type='trust_variation',
                    pattern_value=trust_type,
                    examples=examples[:5],
                    frequency=len(examples),
                    confidence=0.9,
                    description=f"Trust type '{trust_type}' common in unmatched records"
                ))

        return patterns

    async def _ai_pattern_analysis(self, sample_df: pd.DataFrame) -> List[PatternCandidate]:
        """
        Use Claude to analyze complex patterns that rule-based methods might miss.

        Args:
            sample_df: Sample of unmatched records

        Returns:
            AI-discovered patterns
        """
        if not self.use_ai or len(sample_df) == 0:
            return []

        # Determine column names
        name_col = 'clean_name' if 'clean_name' in sample_df.columns else 'EXCLUDE_OWNER_NAME'
        clean_col = 'clean_name' if 'clean_name' in sample_df.columns else name_col

        # Prepare data for AI analysis
        sample_data = []
        for _, row in sample_df.head(20).iterrows():
            sample_data.append({
                'name': row.get(name_col, ''),
                'clean_name': row.get(clean_col, ''),
                'address': row.get('Last Known Address', row.get('LAST_KNOWN_ADDRESS', '')),
                'state': row.get('State', row.get('OWNER_STATE', ''))
            })

        prompt = f"""
Analyze these unmatched owner records and identify patterns that might prevent matching:

Records:
{json.dumps(sample_data, indent=2)}

Please identify:
1. Common naming patterns or variations that might not match
2. Address format inconsistencies
3. Potential temporal changes (estate transitions, contact changes)
4. Any other patterns that could explain why these records don't match

Return your analysis as a JSON array of patterns, each with:
- pattern_type: Category of pattern
- pattern_value: Specific pattern identified
- description: Explanation of the pattern
- confidence: Your confidence in this pattern (0-1)
- examples: 2-3 example record indices that show this pattern
"""

        try:
            response = await asyncio.to_thread(
                self.claude_client.messages.create,
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse AI response
            ai_patterns = self._parse_ai_response(response.content[0].text)

            # Convert to PatternCandidate objects
            patterns = []
            for p in ai_patterns:
                patterns.append(PatternCandidate(
                    pattern_type=f"ai_{p.get('pattern_type', 'unknown')}",
                    pattern_value=p.get('pattern_value', ''),
                    examples=p.get('examples', []),
                    frequency=len(p.get('examples', [])),
                    confidence=p.get('confidence', 0.5),
                    description=f"AI: {p.get('description', '')}"
                ))

            return patterns

        except Exception as e:
            logger.error(f"AI pattern analysis failed: {e}")
            return []

    def _parse_ai_response(self, response_text: str) -> List[Dict]:
        """
        Parse AI response to extract patterns.

        Args:
            response_text: Raw response from Claude

        Returns:
            List of pattern dictionaries
        """
        try:
            # Try to extract JSON from response
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                return json.loads(json_str)
        except:
            pass

        # Fallback: extract patterns manually
        patterns = []
        lines = response_text.split('\n')
        current_pattern = {}

        for line in lines:
            line = line.strip()
            if 'pattern_type' in line.lower():
                if current_pattern:
                    patterns.append(current_pattern)
                current_pattern = {'pattern_type': line.split(':', 1)[1].strip() if ':' in line else 'unknown'}
            elif 'pattern_value' in line.lower() and current_pattern:
                current_pattern['pattern_value'] = line.split(':', 1)[1].strip() if ':' in line else ''
            elif 'description' in line.lower() and current_pattern:
                current_pattern['description'] = line.split(':', 1)[1].strip() if ':' in line else ''
            elif 'confidence' in line.lower() and current_pattern:
                try:
                    conf_str = line.split(':', 1)[1].strip() if ':' in line else '0.5'
                    current_pattern['confidence'] = float(conf_str)
                except:
                    current_pattern['confidence'] = 0.5

        if current_pattern:
            patterns.append(current_pattern)

        return patterns

    def _filter_patterns(
        self,
        candidates: List[PatternCandidate],
        previous_patterns: List[Pattern]
    ) -> List[PatternCandidate]:
        """
        Filter and deduplicate patterns.

        Args:
            candidates: New pattern candidates
            previous_patterns: Previously discovered patterns

        Returns:
            Filtered list of unique, high-quality patterns
        """
        # Create set of previous patterns for deduplication
        previous_set = set(
            (p.pattern_type, p.pattern_value)
            for p in previous_patterns
        )

        filtered = []
        seen = set()

        for candidate in candidates:
            # Check if pattern is new
            pattern_key = (candidate.pattern_type, candidate.pattern_value)

            if pattern_key in seen or pattern_key in previous_set:
                continue

            # Check if pattern meets quality thresholds
            if candidate.frequency >= self.min_pattern_frequency and \
               candidate.confidence >= self.confidence_threshold:
                filtered.append(candidate)
                seen.add(pattern_key)

        # Sort by confidence and frequency
        filtered.sort(key=lambda x: (x.confidence, x.frequency), reverse=True)

        return filtered[:50]  # Return top 50 patterns

    def _generate_insights(
        self,
        patterns: List[Pattern],
        unmatched_df: pd.DataFrame
    ) -> List[str]:
        """
        Generate actionable insights from discovered patterns.

        Args:
            patterns: Discovered patterns
            unmatched_df: Unmatched records

        Returns:
            List of insight strings
        """
        insights = []

        # Count pattern types
        pattern_type_counts = Counter(p.pattern_type for p in patterns)

        # Generate insights based on pattern types
        if pattern_type_counts.get('temporal_indicator', 0) > 5:
            insights.append(
                f"Found {pattern_type_counts['temporal_indicator']} temporal indicators - "
                "consider implementing core name extraction before temporal markers"
            )

        if pattern_type_counts.get('abbreviation_mismatch', 0) > 3:
            insights.append(
                "Multiple abbreviation mismatches detected - "
                "expand abbreviation normalization in text cleaning"
            )

        if pattern_type_counts.get('trust_variation', 0) > 3:
            insights.append(
                "Various trust naming conventions found - "
                "consider adding more trust keywords to normalization"
            )

        # Analyze unmatched characteristics
        if len(unmatched_df) > 100:
            short_names = len(unmatched_df[unmatched_df['clean_name'].str.len() < 5])
            if short_names > len(unmatched_df) * 0.1:
                insights.append(
                    f"{short_names} records have very short names (<5 chars) - "
                    "consider special handling for abbreviated names"
                )

        # Check for systematic issues
        if len(patterns) > 30:
            insights.append(
                f"High pattern diversity ({len(patterns)} patterns) suggests "
                "systematic matching issues - consider threshold adjustments"
            )

        return insights

    def _analyze_unmatched_characteristics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze characteristics of unmatched records.

        Args:
            df: Unmatched records

        Returns:
            Statistical characteristics
        """
        characteristics = {
            'total_count': len(df),
            'name_stats': {},
            'address_stats': {},
            'common_states': []
        }

        # Determine column names
        name_col = None
        if 'clean_name' in df.columns:
            name_col = 'clean_name'
        elif 'EXCLUDE_OWNER_NAME' in df.columns:
            name_col = 'EXCLUDE_OWNER_NAME'

        if name_col and name_col in df.columns:
            name_lengths = df[name_col].astype(str).str.len()
            if not name_lengths.empty:
                characteristics['name_stats'] = {
                    'avg_length': float(name_lengths.mean()),
                    'min_length': int(name_lengths.min()),
                    'max_length': int(name_lengths.max()),
                    'short_names': int((name_lengths < 5).sum()),
                    'single_word': int((df[name_col].astype(str).str.split().str.len() == 1).sum())
                }

        # State column
        state_col = 'State' if 'State' in df.columns else 'OWNER_STATE'
        if state_col in df.columns:
            state_counts = df[state_col].value_counts()
            characteristics['common_states'] = [
                {'state': state, 'count': int(count)}
                for state, count in state_counts.head(5).items()
            ]

        # Address column
        addr_col = None
        if 'Last Known Address' in df.columns:
            addr_col = 'Last Known Address'
        elif 'LAST_KNOWN_ADDRESS' in df.columns:
            addr_col = 'LAST_KNOWN_ADDRESS'

        if addr_col and addr_col in df.columns:
            has_address = df[addr_col].notna()
            characteristics['address_stats'] = {
                'has_address': int(has_address.sum()),
                'missing_address': int((~has_address).sum()),
                'has_po_box': int(df[addr_col].astype(str).str.contains(
                    r'p\.?\s*o\.?\s*box', case=False, na=False
                ).sum())
            }

        return characteristics
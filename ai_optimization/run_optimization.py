#!/usr/bin/env python
"""
Main runner for AI-powered owner ID matching optimization.

This script orchestrates the enhanced optimization system with hierarchical
subagent architecture and persistent context management.
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
import yaml

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from owner_matcher.main import (
    load_old_owners, load_new_owners,
    preprocess_old_owners, preprocess_new_owners,
    map_owners_snowflake, preprocess_snowflake_data
)
from owner_matcher.matchers import OwnerMapper
from owner_matcher.config import OUTPUT_FILE
from owner_matcher.snowflake_client import fetch_new_owners_from_snowflake

from agents.orchestrator import EnhancedOrchestrator
from agents.context_manager import ContextManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_optimization/logs/optimization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


async def run_matching_iteration(
    old_df: pd.DataFrame,
    new_df: pd.DataFrame,
    config_updates: Dict[str, Any] = None
) -> pd.DataFrame:
    """
    Run the owner matching process with optional config updates.

    Args:
        old_df: Old owners dataframe
        new_df: New owners dataframe
        config_updates: Optional configuration updates to apply

    Returns:
        DataFrame with matching results
    """
    # Apply any configuration updates
    if config_updates:
        logger.info(f"Applying configuration updates: {config_updates}")
        # This would update the owner_matcher config
        # For now, we'll simulate by modifying thresholds directly
        if 'thresholds' in config_updates:
            from owner_matcher import config
            for param, value in config_updates['thresholds'].items():
                setattr(config, param, value)

    # Run matching
    mapper = OwnerMapper()
    results = []

    for idx, old_row in old_df.iterrows():
        # Create state-filtered subset for matching
        old_state = old_row.get('EXCLUDE_STATE', old_row.get('clean_state', ''))
        if old_state and not pd.isna(old_state):
            new_subset = new_df[new_df['PROD_STATE'] == old_state]
        else:
            new_subset = new_df  # If no state, use full dataset

        match_result = mapper.match_owner(old_row, new_df, new_subset)

        # Convert to dict and add additional fields
        result_dict = vars(match_result)
        result_dict['old_owner_id'] = old_row.get('OLD_OWNER_ID', '')
        result_dict['old_owner_name'] = old_row.get('EXCLUDE_OWNER_NAME', '')

        results.append(result_dict)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Add clean_name column for AI agents
    if 'old_owner_name' in results_df.columns:
        from owner_matcher.text_utils import clean_text
        results_df['clean_name'] = results_df['old_owner_name'].apply(
            lambda x: clean_text(x) if pd.notna(x) else ''
        )

    return results_df


async def run_optimization_iteration(
    iteration_num: int,
    orchestrator: EnhancedOrchestrator,
    use_snowflake: bool = False,
    continue_from_checkpoint: bool = True
) -> Dict[str, Any]:
    """
    Run a single optimization iteration.

    Args:
        iteration_num: Current iteration number
        orchestrator: Enhanced orchestrator instance
        use_snowflake: Whether to use Snowflake for data loading
        continue_from_checkpoint: Whether to continue from previous checkpoint

    Returns:
        Iteration results
    """
    logger.info(f"=" * 60)
    logger.info(f"Starting Optimization Iteration {iteration_num}")
    logger.info(f"=" * 60)

    # Load data
    logger.info("Loading data...")
    if use_snowflake:
        # Load data directly from Snowflake
        logger.info("Fetching data from Snowflake...")
        snowflake_df = await asyncio.to_thread(fetch_new_owners_from_snowflake)

        # Separate and preprocess data
        original_df, already_matched, needs_matching = await asyncio.to_thread(
            preprocess_snowflake_data, snowflake_df
        )

        # For optimization, we work with the unmatched records
        # Set up data frames for matching
        if len(needs_matching) > 0:
            old_df = needs_matching  # Unmatched records become "old owners" to match
            new_df = already_matched  # Already matched records are candidates
            logger.info(f"Processing {len(old_df)} unmatched records against {len(new_df)} potential matches")
        else:
            logger.warning("No unmatched records to process")
            old_df = pd.DataFrame()
            new_df = pd.DataFrame()
    else:
        # Legacy Excel file loading
        from owner_matcher.config import OLD_FILE, NEW_FILE
        old_df = await asyncio.to_thread(load_old_owners, OLD_FILE)
        new_df = await asyncio.to_thread(load_new_owners, NEW_FILE, use_snowflake=False)

        # Preprocess Excel data
        old_df = await asyncio.to_thread(preprocess_old_owners, old_df)
        new_df = await asyncio.to_thread(preprocess_new_owners, new_df)

    # Skip preprocessing if already done for Snowflake
    if not use_snowflake:
        # Only preprocess for Excel workflow (already done for Snowflake)
        logger.info("Preprocessing data...")
        # Already preprocessed in the else block above
        pass

    # Get current configuration from orchestrator
    current_config = orchestrator._get_current_thresholds()

    # Run matching with current configuration
    logger.info("Running owner matching...")
    match_results = await run_matching_iteration(
        old_df,
        new_df,
        config_updates={'thresholds': current_config}
    )

    # Run AI optimization
    logger.info("Running AI optimization analysis...")
    iteration_result = await orchestrator.run_iteration(
        iteration_num=iteration_num,
        match_results_df=match_results,
        continue_from_checkpoint=continue_from_checkpoint
    )

    # Generate and display report
    report = await orchestrator.generate_report(iteration_num)
    logger.info("Iteration Report:\n" + report)

    # Check convergence
    convergence = await orchestrator.evaluate_convergence()
    logger.info(f"Convergence status: {convergence}")

    # Save results
    output_dir = Path(f"ai_optimization/iterations/iteration_{iteration_num:03d}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save match results
    match_results.to_csv(output_dir / "matches.csv", index=False)

    # Save iteration results
    with open(output_dir / "results.yaml", 'w') as f:
        yaml.dump({
            'iteration': iteration_num,
            'match_rate': iteration_result.match_rate,
            'total_matches': iteration_result.total_matches,
            'convergence_status': iteration_result.convergence_status,
            'execution_time': iteration_result.execution_time,
            'threshold_changes': iteration_result.threshold_changes
        }, f)

    # Save report
    with open(output_dir / "report.md", 'w') as f:
        f.write(report)

    return {
        'iteration_result': iteration_result,
        'convergence': convergence,
        'should_continue': convergence.get('recommendation') == 'continue'
    }


async def run_continuous_optimization(
    max_iterations: int = 10,
    use_snowflake: bool = False,
    mode: str = 'enhanced'
) -> None:
    """
    Run continuous optimization until convergence or max iterations.

    Args:
        max_iterations: Maximum number of iterations
        use_snowflake: Whether to use Snowflake for data
        mode: Optimization mode ('basic' or 'enhanced')
    """
    logger.info(f"Starting continuous optimization in {mode} mode")
    logger.info(f"Maximum iterations: {max_iterations}")

    # Initialize orchestrator
    orchestrator = EnhancedOrchestrator()

    # Check for existing checkpoint
    context_manager = ContextManager("ai_optimization/context_store")
    existing_context = await context_manager.load_checkpoint()

    start_iteration = 1
    if existing_context:
        start_iteration = existing_context.current_iteration + 1
        logger.info(f"Continuing from iteration {start_iteration}")

    # Run iterations
    for iteration in range(start_iteration, max_iterations + 1):
        try:
            result = await run_optimization_iteration(
                iteration_num=iteration,
                orchestrator=orchestrator,
                use_snowflake=use_snowflake,
                continue_from_checkpoint=(iteration > 1)
            )

            # Check if we should continue
            if not result['should_continue']:
                logger.info(f"Optimization converged at iteration {iteration}")
                break

            # Apply recommended changes if approved
            if result['iteration_result'].threshold_changes:
                logger.info("Threshold changes recommended:")
                for param, value in result['iteration_result'].threshold_changes.items():
                    logger.info(f"  {param}: {value}")

                # In production, this would require human approval
                # For now, we'll auto-apply if confidence is high
                if result['iteration_result'].convergence_status != 'converged':
                    logger.info("Applying threshold changes for next iteration")
                    # Changes will be applied in next iteration

        except Exception as e:
            logger.error(f"Error in iteration {iteration}: {e}", exc_info=True)
            break

    logger.info("Optimization complete")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run AI-powered owner ID matching optimization"
    )

    parser.add_argument(
        '--iteration',
        type=int,
        help='Run specific iteration number'
    )

    parser.add_argument(
        '--max-iterations',
        type=int,
        default=10,
        help='Maximum number of iterations (default: 10)'
    )

    parser.add_argument(
        '--mode',
        choices=['basic', 'enhanced'],
        default='enhanced',
        help='Optimization mode (default: enhanced)'
    )

    parser.add_argument(
        '--use-snowflake',
        action='store_true',
        help='Use Snowflake for data loading'
    )

    parser.add_argument(
        '--continue-from',
        type=str,
        help='Continue from checkpoint file'
    )

    parser.add_argument(
        '--agents',
        type=str,
        help='Comma-separated list of specific agents to run'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    args = parser.parse_args()

    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create necessary directories
    Path("ai_optimization/logs").mkdir(parents=True, exist_ok=True)
    Path("ai_optimization/iterations").mkdir(parents=True, exist_ok=True)
    Path("ai_optimization/context_store").mkdir(parents=True, exist_ok=True)

    try:
        if args.iteration:
            # Run specific iteration
            orchestrator = EnhancedOrchestrator()
            result = await run_optimization_iteration(
                iteration_num=args.iteration,
                orchestrator=orchestrator,
                use_snowflake=args.use_snowflake,
                continue_from_checkpoint=args.continue_from is not None
            )
            logger.info(f"Iteration {args.iteration} complete")
        else:
            # Run continuous optimization
            await run_continuous_optimization(
                max_iterations=args.max_iterations,
                use_snowflake=args.use_snowflake,
                mode=args.mode
            )

    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")
    except Exception as e:
        logger.error(f"Optimization failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
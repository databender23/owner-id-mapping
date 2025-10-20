-- ============================================================================
-- EXCLUDE_OWNERS ID MAPPING ANALYSIS
-- Maps old owner IDs from EXCLUDE_OWNERS table to new owner IDs in production
-- Shows which excluded owners have new IDs after reindexing
-- ============================================================================
--
-- PURPOSE:
--   This query generates the excluded_owners_comparison.xlsx file used as input
--   for the Owner ID Mapping System. It identifies owners from the EXCLUDE_OWNERS
--   table that exist in current production with potentially different IDs.
--
-- OUTPUT COLUMNS:
--   - old_owner_id: Original owner ID from EXCLUDE_OWNERS table
--   - new_owner_id: Current owner ID from production INTEREST_TBL (NULL if not matched)
--   - exclude_owner_name: Owner name from EXCLUDE_OWNERS
--   - prod_owner_name: Owner name from production INTEREST_TBL (for comparison)
--   - exclude_address/city/state/zip: Address from EXCLUDE_OWNERS
--   - prod_address/city/state/zip: Address from production INTEREST_TBL (most recent)
--   - most_recent_year: Year of the most recent production record for this owner
--   - id_status: ID_UNCHANGED | ID_CHANGED | NOT_FOUND_BY_SQL
--   - address_match_status: ADDRESS_MATCH | ADDRESS_DIFFERENT | N/A
--
-- USAGE:
--   1. Run this query in Snowflake
--   2. Export results to Excel
--   3. Save as: data/raw/excluded_owners_comparison.xlsx
--   4. Run the owner matching script
--
-- FILTERS:
--   - Production data filtered to YEAR >= 2018 for performance
--   - Returns ALL owners from EXCLUDE_OWNERS (matched and unmatched)
--   - SQL performs simple exact name matching; Python fuzzy matching handles the rest
--
-- LAST UPDATED: 2025-10-20
-- CHANGE LOG:
--   2025-10-20:
--     - Added ROW_NUMBER logic to keep only most recent record per owner by YEAR
--     - Removed WHERE filter to return ALL EXCLUDE_OWNERS (including unmatched)
--     - Added prod_owner_name column for easier manual review
--     - Changed id_status 'NOT_FOUND_IN_PROD' to 'NOT_FOUND_BY_SQL'
--     - Updated ORDER BY to show matched records first
-- ============================================================================

-- Main query to find ID mappings for excluded owners
WITH owner_mapping AS (
    -- Get most recent owner record for each owner (by YEAR)
    -- Uses ROW_NUMBER to rank records by year descending, keeping only the latest
    SELECT
        prod_owner_name,
        prod_owner_id,
        prod_address,
        prod_city,
        prod_state,
        prod_zip,
        most_recent_year
    FROM (
        SELECT
            OWNER as prod_owner_name,
            OWNER_ID as prod_owner_id,
            OWNER_ADDRESS as prod_address,
            OWNER_CITY as prod_city,
            OWNER_STATE as prod_state,
            OWNER_ZIP as prod_zip,
            YEAR as most_recent_year,
            ROW_NUMBER() OVER (
                PARTITION BY OWNER, OWNER_ID
                ORDER BY YEAR DESC
            ) as rn
        FROM MINERALHOLDERS_DB.PUBLIC.INTEREST_TBL
        WHERE YEAR >= 2018
            AND OWNER IS NOT NULL
    ) ranked
    WHERE rn = 1  -- Keep only the most recent record per owner
)
SELECT DISTINCT
    e.OWNER_ID as old_owner_id,
    m.prod_owner_id as new_owner_id,
    e.OWNER_NAME as exclude_owner_name,
    m.prod_owner_name,  -- Include production name for comparison/review
    e.OWNER_ADDRESS as exclude_address,
    e.OWNER_CITY as exclude_city,
    e.OWNER_STATE as exclude_state,
    e.OWNER_ZIP as exclude_zip,
    m.prod_address,
    m.prod_city,
    m.prod_state,
    m.prod_zip,
    m.most_recent_year,  -- Year of the most recent production record
    CASE
        WHEN m.prod_owner_id IS NULL THEN 'NOT_FOUND_BY_SQL'
        WHEN e.OWNER_ID = m.prod_owner_id THEN 'ID_UNCHANGED'
        ELSE 'ID_CHANGED'
    END as id_status,
    -- Check if address details match (simplified to just address field)
    CASE
        WHEN m.prod_owner_id IS NULL THEN 'N/A'
        WHEN m.prod_address ilike e.OWNER_ADDRESS THEN 'ADDRESS_MATCH'
        ELSE 'ADDRESS_DIFFERENT'
    END as address_match_status
FROM MINERALHOLDERS_DB.PUBLIC.EXCLUDE_OWNERS e
LEFT JOIN owner_mapping m
    ON UPPER(TRIM(e.OWNER_NAME)) = UPPER(TRIM(m.prod_owner_name))
-- REMOVED: WHERE m.prod_owner_id IS NOT NULL
-- We now return ALL EXCLUDE_OWNERS records, even if SQL couldn't match them.
-- The Python fuzzy matching system will handle unmatched records with sophisticated
-- matching strategies (address-first, fuzzy name, cross-state, etc.)
ORDER BY
    CASE
        WHEN m.prod_owner_id IS NOT NULL THEN 0  -- Matched records first
        ELSE 1  -- Unmatched records last
    END,
    e.OWNER_NAME;

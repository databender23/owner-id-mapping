"""
Snowflake connection and query execution utilities.

This module handles connecting to Snowflake using private key authentication
and executing queries to fetch owner data directly from the database.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import snowflake.connector
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization

from .config import (
    SNOWFLAKE_ACCOUNT,
    SNOWFLAKE_USER,
    SNOWFLAKE_ROLE,
    SNOWFLAKE_WAREHOUSE,
    SNOWFLAKE_DATABASE,
    SNOWFLAKE_SCHEMA,
    SNOWFLAKE_PRIVATE_KEY_PATH,
    SNOWFLAKE_QUERY_FILE
)

logger = logging.getLogger(__name__)


class SnowflakeClient:
    """
    Client for connecting to Snowflake and executing queries.

    Uses private key authentication for secure connection without passwords.
    """

    def __init__(
        self,
        account: str = SNOWFLAKE_ACCOUNT,
        user: str = SNOWFLAKE_USER,
        role: str = SNOWFLAKE_ROLE,
        warehouse: str = SNOWFLAKE_WAREHOUSE,
        database: str = SNOWFLAKE_DATABASE,
        schema: str = SNOWFLAKE_SCHEMA,
        private_key_path: str = SNOWFLAKE_PRIVATE_KEY_PATH
    ):
        """
        Initialize Snowflake client with connection parameters.

        Args:
            account: Snowflake account identifier
            user: Snowflake username
            role: Snowflake role to use
            warehouse: Snowflake warehouse for compute
            database: Target database
            schema: Target schema
            private_key_path: Path to RSA private key file (.pem)
        """
        self.account = account
        self.user = user
        self.role = role
        self.warehouse = warehouse
        self.database = database
        self.schema = schema
        self.private_key_path = private_key_path
        self._connection: Optional[snowflake.connector.SnowflakeConnection] = None

    def _load_private_key(self) -> bytes:
        """
        Load and decode the RSA private key from file.

        Returns:
            Encoded private key bytes

        Raises:
            FileNotFoundError: If private key file doesn't exist
            ValueError: If private key format is invalid
        """
        key_path = Path(self.private_key_path)

        if not key_path.exists():
            raise FileNotFoundError(
                f"Private key file not found: {self.private_key_path}\n"
                f"Please ensure the private key file exists at this location."
            )

        logger.info(f"Loading private key from: {self.private_key_path}")

        try:
            with open(key_path, "rb") as key_file:
                private_key = serialization.load_pem_private_key(
                    key_file.read(),
                    password=None,  # Assuming unencrypted key
                    backend=default_backend()
                )

            # Encode the key for Snowflake
            pkb = private_key.private_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )

            logger.debug("Private key loaded and encoded successfully")
            return pkb

        except Exception as e:
            raise ValueError(f"Failed to load private key: {e}")

    def connect(self) -> snowflake.connector.SnowflakeConnection:
        """
        Establish connection to Snowflake using private key authentication.

        Returns:
            Active Snowflake connection

        Raises:
            Exception: If connection fails
        """
        if self._connection is not None:
            logger.debug("Reusing existing Snowflake connection")
            return self._connection

        logger.info(f"Connecting to Snowflake account: {self.account}")

        try:
            # Load private key
            private_key_bytes = self._load_private_key()

            # Establish connection
            self._connection = snowflake.connector.connect(
                account=self.account,
                user=self.user,
                private_key=private_key_bytes,
                role=self.role,
                warehouse=self.warehouse,
                database=self.database,
                schema=self.schema
            )

            logger.info("Successfully connected to Snowflake")
            return self._connection

        except Exception as e:
            logger.error(f"Failed to connect to Snowflake: {e}")
            raise

    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute a SQL query and return results as a DataFrame.

        Args:
            query: SQL query string to execute

        Returns:
            DataFrame with query results

        Raises:
            Exception: If query execution fails
        """
        logger.info("Executing Snowflake query...")
        logger.debug(f"Query: {query[:200]}...")  # Log first 200 chars

        try:
            conn = self.connect()

            # Execute query and fetch results
            cursor = conn.cursor()
            cursor.execute(query)

            # Fetch all results
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]

            # Convert to DataFrame
            df = pd.DataFrame(results, columns=columns)

            logger.info(f"Query returned {len(df)} rows")
            cursor.close()

            return df

        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise

    def fetch_excluded_owners_comparison(self) -> pd.DataFrame:
        """
        Fetch excluded owners comparison data using the standard query.

        Reads the SQL query from sql/generate_excluded_owners_comparison.sql
        and executes it against Snowflake.

        Returns:
            DataFrame with excluded owners comparison data

        Raises:
            FileNotFoundError: If query file doesn't exist
            Exception: If query execution fails
        """
        query_file = Path(SNOWFLAKE_QUERY_FILE)

        if not query_file.exists():
            raise FileNotFoundError(
                f"Query file not found: {SNOWFLAKE_QUERY_FILE}\n"
                f"Please ensure the SQL query file exists at this location."
            )

        logger.info(f"Reading query from: {query_file}")

        # Read query from file
        with open(query_file, 'r') as f:
            query = f.read()

        # Remove comments and clean up query
        # Keep the actual SQL, remove comment blocks
        query_lines = []
        in_comment_block = False

        for line in query.split('\n'):
            stripped = line.strip()

            # Toggle comment block
            if '-- ========' in stripped:
                in_comment_block = not in_comment_block
                continue

            # Skip comment lines
            if stripped.startswith('--'):
                continue

            # Skip empty lines in comment blocks
            if in_comment_block:
                continue

            query_lines.append(line)

        clean_query = '\n'.join(query_lines).strip()

        # Execute query
        logger.info("Fetching excluded owners comparison data from Snowflake...")
        df = self.execute_query(clean_query)

        return df

    def close(self) -> None:
        """Close the Snowflake connection if open."""
        if self._connection is not None:
            logger.info("Closing Snowflake connection")
            self._connection.close()
            self._connection = None

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def fetch_new_owners_from_snowflake() -> pd.DataFrame:
    """
    Convenience function to fetch new owners data from Snowflake.

    Returns:
        DataFrame with excluded owners comparison data

    Example:
        >>> df = fetch_new_owners_from_snowflake()
        >>> print(f"Loaded {len(df)} owner records from Snowflake")
    """
    with SnowflakeClient() as client:
        return client.fetch_excluded_owners_comparison()

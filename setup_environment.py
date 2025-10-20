#!/usr/bin/env python
"""
Setup script to configure environment for the Owner ID Mapping System.

This script helps users:
1. Create a .env file from .env.example
2. Set up their Claude API key
3. Configure Snowflake credentials (optional)
4. Verify the setup
"""

import os
import sys
from pathlib import Path
import shutil


def setup_env_file():
    """Create .env file from .env.example if it doesn't exist."""
    env_file = Path(".env")
    env_example = Path(".env.example")

    if env_file.exists():
        print("✓ .env file already exists")
        return True

    if env_example.exists():
        response = input("\n.env file not found. Create from .env.example? (y/n): ")
        if response.lower() == 'y':
            shutil.copy(env_example, env_file)
            print("✓ Created .env file from .env.example")
            print("\n⚠️  Please edit .env and add your API keys:")
            print("   - ANTHROPIC_API_KEY for Claude AI features")
            print("   - Snowflake credentials (if using Snowflake)")
            return True
    else:
        print("❌ .env.example not found")
        return False


def check_claude_api():
    """Check if Claude API key is configured."""
    api_key = os.environ.get('ANTHROPIC_API_KEY')

    if api_key:
        if api_key == 'your-api-key-here':
            print("⚠️  ANTHROPIC_API_KEY is still set to placeholder value")
            print("   Please update it with your actual API key")
            return False
        else:
            # Mask the API key for security
            masked_key = api_key[:7] + "..." + api_key[-4:] if len(api_key) > 11 else "***"
            print(f"✓ ANTHROPIC_API_KEY is set ({masked_key})")
            return True
    else:
        print("⚠️  ANTHROPIC_API_KEY not found in environment")
        print("\nTo add your Claude API key, you can:")
        print("1. Add to .env file: ANTHROPIC_API_KEY=your-actual-key")
        print("2. Export in terminal: export ANTHROPIC_API_KEY='your-actual-key'")
        print("3. Add to ~/.bashrc or ~/.zshrc for permanent setup")
        return False


def check_snowflake():
    """Check if Snowflake credentials are configured."""
    account = os.environ.get('SNOWFLAKE_ACCOUNT')
    user = os.environ.get('SNOWFLAKE_USER')
    key_path = os.environ.get('SNOWFLAKE_PRIVATE_KEY_PATH')

    if account or user or key_path:
        print("\n✓ Snowflake configuration detected:")
        if account:
            print(f"  - Account: {account}")
        if user:
            print(f"  - User: {user}")
        if key_path:
            if Path(key_path).exists():
                print(f"  - Private key: {key_path} (exists)")
            else:
                print(f"  - Private key: {key_path} (NOT FOUND)")
        return True
    else:
        print("\n⚠️  Snowflake not configured (optional)")
        print("   Add credentials to .env if you want to use --use-snowflake")
        return False


def test_claude_import():
    """Test if Claude client can be imported and initialized."""
    try:
        import anthropic
        print("\n✓ Anthropic package is installed")

        if os.environ.get('ANTHROPIC_API_KEY') and os.environ.get('ANTHROPIC_API_KEY') != 'your-api-key-here':
            try:
                client = anthropic.Anthropic()
                print("✓ Claude client can be initialized")
                return True
            except Exception as e:
                print(f"❌ Claude client initialization failed: {e}")
                return False
        else:
            print("⚠️  Skipping Claude client test (no valid API key)")
            return False
    except ImportError:
        print("\n❌ Anthropic package not installed")
        print("   Run: pip install anthropic")
        return False


def main():
    """Main setup function."""
    print("=" * 60)
    print("Owner ID Mapping System - Environment Setup")
    print("=" * 60)

    # Check Python version
    if sys.version_info < (3, 8):
        print(f"❌ Python {sys.version_info.major}.{sys.version_info.minor} detected")
        print("   Python 3.8+ is required")
        sys.exit(1)
    else:
        print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")

    # Setup .env file
    print("\n" + "-" * 40)
    print("Checking environment configuration...")
    print("-" * 40)

    env_exists = setup_env_file()

    if env_exists:
        # Load .env file if it exists
        try:
            from dotenv import load_dotenv
            load_dotenv()
            print("✓ Loaded .env file")
        except ImportError:
            print("⚠️  python-dotenv not installed")
            print("   Run: pip install python-dotenv")
            print("   Or manually export environment variables")

    # Check configurations
    print("\n" + "-" * 40)
    print("Checking API configurations...")
    print("-" * 40)

    claude_ok = check_claude_api()
    snowflake_ok = check_snowflake()

    # Test imports
    print("\n" + "-" * 40)
    print("Testing imports...")
    print("-" * 40)

    claude_import_ok = test_claude_import()

    # Summary
    print("\n" + "=" * 60)
    print("Setup Summary")
    print("=" * 60)

    if claude_ok and claude_import_ok:
        print("✅ Claude AI features are ready to use")
    else:
        print("⚠️  Claude AI features are disabled")
        print("   The system will work but without AI-powered pattern discovery")

    if snowflake_ok:
        print("✅ Snowflake integration is configured")
    else:
        print("⚠️  Snowflake not configured (will use Excel files)")

    print("\n" + "-" * 40)
    print("Next steps:")
    print("-" * 40)

    if not claude_ok:
        print("1. Add your Claude API key to .env file")
        print("   Get a key at: https://console.anthropic.com/")

    print("\nTo run the system:")
    print("  Basic matching:    python -m owner_matcher.main")
    print("  AI optimization:   python ai_optimization/run_optimization.py")

    print("\n✨ Setup complete!")


if __name__ == "__main__":
    main()
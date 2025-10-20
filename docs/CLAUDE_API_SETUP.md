# Claude API Setup Guide

This guide explains how to configure your Claude API key for the AI optimization features.

## Quick Setup

### Option 1: Using .env File (Recommended)

1. **Copy the example environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit .env and add your API key:**
   ```bash
   # Open .env in your editor
   nano .env  # or vim, code, etc.

   # Update this line with your actual key:
   ANTHROPIC_API_KEY=sk-ant-api03-xxxxxxxxxxxxx
   ```

3. **Install python-dotenv to load the .env file:**
   ```bash
   pip install python-dotenv
   ```

4. **Run the setup script to verify:**
   ```bash
   python setup_environment.py
   ```

### Option 2: Environment Variable

Set the API key in your terminal session:

```bash
# For current session only
export ANTHROPIC_API_KEY='sk-ant-api03-xxxxxxxxxxxxx'

# Or add to ~/.bashrc or ~/.zshrc for permanent setup
echo "export ANTHROPIC_API_KEY='sk-ant-api03-xxxxxxxxxxxxx'" >> ~/.bashrc
source ~/.bashrc
```

### Option 3: System Environment Variable (Windows)

```powershell
# PowerShell
$env:ANTHROPIC_API_KEY = "sk-ant-api03-xxxxxxxxxxxxx"

# Or set permanently via System Properties
setx ANTHROPIC_API_KEY "sk-ant-api03-xxxxxxxxxxxxx"
```

## Getting Your Claude API Key

1. **Sign up or log in** at https://console.anthropic.com/
2. Navigate to **API Keys** section
3. Click **Create Key**
4. Copy the key (starts with `sk-ant-api03-`)
5. **Important:** Save the key securely - you won't be able to see it again!

## Verifying Your Setup

Run the setup script to check everything is configured correctly:

```bash
python setup_environment.py
```

Expected output when properly configured:
```
✓ Python 3.x detected
✓ Loaded .env file
✓ ANTHROPIC_API_KEY is set (sk-ant-...xxxx)
✓ Anthropic package is installed
✓ Claude client can be initialized
✅ Claude AI features are ready to use
```

## Testing Claude Integration

Run a test to ensure Claude is working:

```python
python -c "
import os
import anthropic

client = anthropic.Anthropic()
response = client.messages.create(
    model='claude-3-sonnet-20241022',
    max_tokens=100,
    messages=[{'role': 'user', 'content': 'Say hello'}]
)
print('✅ Claude is working:', response.content[0].text)
"
```

## Running Without Claude (Optional)

The system works without Claude API, but AI-powered pattern discovery will be disabled.

To disable Claude features, edit `ai_optimization/config.yaml`:

```yaml
agents:
  pattern_discovery:
    use_ai: false  # Disable Claude integration
```

## Troubleshooting

### "ANTHROPIC_API_KEY not found"
- Make sure you've set the environment variable
- If using .env file, ensure python-dotenv is installed
- Check the .env file is in the project root directory

### "Invalid API key"
- Verify the key starts with `sk-ant-api03-`
- Check for extra spaces or quotes in the key
- Ensure the key is still active in your Anthropic console

### "Rate limit exceeded"
- Claude has rate limits based on your plan
- Reduce `batch_size` in `ai_optimization/config.yaml`
- Add delays between requests with `retry_delay`

### "Module 'anthropic' not found"
```bash
pip install anthropic
# or
pip install -r ai_optimization/requirements-ai.txt
```

## API Usage and Costs

- The AI optimization uses Claude for pattern analysis
- Each iteration may make 5-20 API calls depending on data size
- Estimated cost: $0.50-2.00 per optimization iteration
- Monitor usage at: https://console.anthropic.com/usage

## Security Best Practices

1. **Never commit API keys** to version control
   - .env is in .gitignore by default
   - Don't hardcode keys in Python files

2. **Use environment variables** in production
   - Set via secure secrets management
   - Rotate keys periodically

3. **Limit API key permissions** if possible
   - Use project-specific keys
   - Monitor usage regularly

## Support

If you encounter issues:
1. Check the setup with `python setup_environment.py`
2. Review this guide and troubleshooting section
3. Check your Anthropic console for API key status
4. Ensure you're using a compatible Claude model

The system will work without Claude, but you'll miss out on:
- AI-powered pattern discovery
- Complex pattern analysis
- Intelligent suggestions for unmatched records
# Security Guidelines

## API Key Management

⚠️ **IMPORTANT**: Never commit API keys or secrets to the repository.

### Environment Variables Setup

This project requires several API keys for different services. Set them up as environment variables:

#### Required Environment Variables

1. **OpenAI API Key** (for synthetic dataset generation)
   ```bash
   export OPENAI_API_KEY="your-openai-api-key-here"
   ```

2. **Gemini API Key** (for vision analysis)
   ```bash
   export GEMINI_API_KEY="your-gemini-api-key-here"
   ```

3. **Anthropic API Key** (for Claude integration)
   ```bash
   export ANTHROPIC_AUTH_TOKEN="your-anthropic-api-key-here"
   ```

#### Windows Setup

Create a `.env` file in the project root (already in .gitignore):

```env
OPENAI_API_KEY=your-openai-api-key-here
GEMINI_API_KEY=your-gemini-api-key-here
ANTHROPIC_AUTH_TOKEN=your-anthropic-api-key-here
```

Or set via Command Prompt:
```cmd
set OPENAI_API_KEY=your-openai-api-key-here
set GEMINI_API_KEY=your-gemini-api-key-here
set ANTHROPIC_AUTH_TOKEN=your-anthropic-api-key-here
```

#### Linux/macOS Setup

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
export OPENAI_API_KEY="your-openai-api-key-here"
export GEMINI_API_KEY="your-gemini-api-key-here"
export ANTHROPIC_AUTH_TOKEN="your-anthropic-api-key-here"
```

Then reload your shell or run `source ~/.bashrc`

### Obtaining API Keys

1. **OpenAI API Key**:
   - Visit https://platform.openai.com/api-keys
   - Create a new secret key
   - Required for synthetic dataset generation

2. **Google Gemini API Key**:
   - Visit https://makersuite.google.com/app/apikey
   - Generate a new API key
   - Required for vision analysis features

3. **Anthropic Claude API Key**:
   - Visit https://console.anthropic.com/
   - Generate a new API key
   - Required for claude-DeepSeek.cmd script

### Security Best Practices

1. **Never commit secrets**: API keys should never be in your code
2. **Use environment variables**: Load keys from environment at runtime
3. **Use .env files locally**: Keep a local .env file (already in .gitignore)
4. **Rotate keys regularly**: Change API keys periodically
5. **Monitor usage**: Track API usage for unexpected activity

### Files That Use API Keys

- `claude-DeepSeek.cmd` - Uses ANTHROPIC_AUTH_TOKEN
- `Versions/V2 Nanobanana Sythetic cardboard bundle dataset/Nano_banana/nano_banana.py` - Uses GEMINI_API_KEY
- `Yolo DS To Qwen DS/Sythetic cardboard bundle dataset/synthetic dataset/generate_pallet_dataset.py` - Uses OPENAI_API_KEY
- `Yolo DS To Qwen DS/Sythetic cardboard bundle dataset/synthetic dataset/synthetic_cardboard_generator.py` - Uses OPENAI_API_KEY

### Emergency Response

If API keys are accidentally committed:

1. **Immediately rotate/disable** the exposed keys in their respective services
2. **Generate new keys** from the API providers
3. **Clean git history** using BFG Repo-Cleaner or git filter-branch
4. **Force push** the cleaned repository
5. **Update environment variables** with new keys

### .gitignore Protection

The following patterns are included in .gitignore to prevent accidental commits:

```gitignore
# API Keys and secrets (SECURITY)
*.env
.env*
config.json
secrets.json
*api_key*
*API_KEY*
*secret*
*SECRET*
*token*
*TOKEN*

# API Key patterns
sk-*
AIza*
gho_*
ghp_*
```

---

**Remember**: When in doubt, don't commit it. It's better to ask for review than to expose sensitive information.
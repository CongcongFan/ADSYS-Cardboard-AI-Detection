@echo off
set ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
set ANTHROPIC_AUTH_TOKEN=YOUR_ANTHROPIC_API_KEY_HERE
set ANTHROPIC_MODEL=deepseek-chat
set ANTHROPIC_SMALL_FAST_MODEL=deepseek-chat
claude --dangerously-skip-permissions%*

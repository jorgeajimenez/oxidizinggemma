#!/usr/bin/env bash
# ==============================================================================
# Gemma 4 Live Inference Streaming Client
# ==============================================================================
# Usage:
#   ./stream.sh "Your prompt here" [max_tokens] [temperature]
#   ./stream.sh --raw "Your prompt here"
# ==============================================================================

set -e

HOST="${GEMMA_HOST:-195.242.13.222:8080}"
RAW_MODE=false

if [ "$1" == "--raw" ]; then
  RAW_MODE=true
  shift
fi

PROMPT="${1:-Explain why Go and Rust make an unstoppable pair for high-performance AI inference.}"
MAX_TOKENS="${2:-100}"
TEMPERATURE="${3:-0.7}"

# Safely encode JSON with python or jq to prevent newline escaping bugs
JSON_PAYLOAD=$(python3 -c '
import json, sys
prompt, max_tokens, temp = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
print(json.dumps({"prompt": prompt, "max_tokens": max_tokens, "temperature": temp}))
' "$PROMPT" "$MAX_TOKENS" "$TEMPERATURE")

echo -e "\033[1;34m=== Gemma 4 Cloud Inference Client ===\033[0m"
echo -e "\033[0;33mEndpoint:\033[0m http://$HOST/v1/chat/completions"
echo -e "\033[0;33mPrompt:\033[0m   $PROMPT"
echo -e "\033[1;32m--- Streaming Tokens ---\033[0m"

if [ "$RAW_MODE" = true ]; then
  curl -N -s -X POST "http://$HOST/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$JSON_PAYLOAD"
else
  # Pretty print the streamed tokens in real-time
  curl -N -s -X POST "http://$HOST/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$JSON_PAYLOAD" | while IFS= read -r line; do
      if [[ "$line" =~ ^data:\ (\{.*\}) ]]; then
        json_data="${BASH_REMATCH[1]}"
        token=$(python3 -c "import json, sys; print(json.loads(sys.argv[1]).get('token', ''), end='')" "$json_data" 2>/dev/null || true)
        printf "%s" "$token"
      elif [[ "$line" == "data: [DONE]" ]]; then
        break
      fi
    done
fi

echo -e "\n\033[1;32m------------------------\033[0m"

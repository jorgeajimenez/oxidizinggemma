#!/usr/bin/env bash
# ==============================================================================
# Gemma 4 Multi-Cloud Streaming Client with Auto-Discovery
# ==============================================================================
# Usage:
#   ./stream.sh "Your prompt here"
#   ./stream.sh nebius "Your prompt here"
#   ./stream.sh gcp "Your prompt here"
#   ./stream.sh <IP:PORT> "Your prompt here"
#   ./stream.sh --raw [target] "Your prompt here"
# ==============================================================================

set -euo pipefail

RAW_MODE=false
if [[ "${1:-}" == "--raw" ]]; then
  RAW_MODE=true
  shift
fi

TARGET="${1:-}"
PROMPT=""
MAX_TOKENS=100
TEMPERATURE=0.7

# Detect if first argument is a cluster alias or IP
if [[ "$TARGET" == "nebius" ]]; then
  echo -e "\033[0;36mAuto-discovering SkyPilot cluster: gemma4-nebius...\033[0m"
  IP=$(sky status --ip gemma4-nebius 2>/dev/null || true)
  if [[ -z "$IP" ]]; then
    echo "Error: Cluster 'gemma4-nebius' not found in sky status." >&2
    exit 1
  fi
  HOST="${IP}:8080"
  shift
elif [[ "$TARGET" == "gcp" ]]; then
  echo -e "\033[0;36mAuto-discovering SkyPilot cluster: gemma4-gcp...\033[0m"
  IP=$(sky status --ip gemma4-gcp 2>/dev/null || true)
  if [[ -z "$IP" ]]; then
    echo "Error: Cluster 'gemma4-gcp' not found in sky status." >&2
    exit 1
  fi
  HOST="${IP}:8080"
  shift
elif [[ "$TARGET" == "local" ]]; then
  HOST="127.0.0.1:8080"
  shift
elif [[ "$TARGET" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+(:[0-9]+)?$ ]]; then
  if [[ "$TARGET" == *":"* ]]; then
    HOST="$TARGET"
  else
    HOST="${TARGET}:8080"
  fi
  shift
else
  # Default fallback resolution
  if [[ -n "${GEMMA_HOST:-}" ]]; then
    HOST="$GEMMA_HOST"
  else
    # Try discovering active Nebius cluster first, then GCP, then local
    NEBIUS_IP=$(sky status --ip gemma4-nebius 2>/dev/null || true)
    if [[ -n "$NEBIUS_IP" ]]; then
      HOST="${NEBIUS_IP}:8080"
    else
      GCP_IP=$(sky status --ip gemma4-gcp 2>/dev/null || true)
      if [[ -n "$GCP_IP" ]]; then
        HOST="${GCP_IP}:8080"
      else
        HOST="127.0.0.1:8080"
      fi
    fi
  fi
fi

PROMPT="${1:-Explain why Rust and Go make an unstoppable pair for high-performance AI inference.}"
MAX_TOKENS="${2:-100}"
TEMPERATURE="${3:-0.7}"

# Encode JSON safely
JSON_PAYLOAD=$(python3 -c '
import json, sys
prompt, max_tokens, temp = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
print(json.dumps({"prompt": prompt, "max_tokens": max_tokens, "temperature": temp}))
' "$PROMPT" "$MAX_TOKENS" "$TEMPERATURE")

echo -e "\033[1;34m=== Gemma 4 Cloud Inference Client ===\033[0m"
echo -e "\033[0;33mEndpoint:\033[0m http://$HOST/v1/chat/completions"
echo -e "\033[0;33mPrompt:\033[0m   $PROMPT"
echo -e "\033[1;32m--- Streaming Response ---\033[0m"

if [ "$RAW_MODE" = true ]; then
  curl -N -s -X POST "http://$HOST/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$JSON_PAYLOAD"
else
  # Stream tokens live in typewriter style
  curl -N -s -X POST "http://$HOST/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$JSON_PAYLOAD" | python3 -u -c '
import sys, json

for line in sys.stdin:
    line = line.strip()
    if line.startswith("data:"):
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            data = json.loads(payload)
            token = data.get("token", "")
            sys.stdout.write(token)
            sys.stdout.flush()
        except Exception:
            pass
'
fi

echo -e "\n\033[1;32m--------------------------\033[0m"

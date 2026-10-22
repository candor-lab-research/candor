#!/bin/bash
# Uploads the release GGUF and its licence to the private repo as candor-team, then drops the token.
set -euo pipefail
cd /root/candor
trap 'rm -f /root/.hf-token' EXIT
export HF_TOKEN=$(cat /root/.hf-token)
.venv/bin/hf auth whoami | grep -E "user(=|: )candor-team"
echo "== upload $(date -u +%T)"
.venv/bin/hf upload candor-lab/Qwen3.8-27B-Candor-Uncensored-GGUF /root/release . \
    --commit-message "Add the Q4_K_M GGUF and its licence"
echo "== done $(date -u +%T)"

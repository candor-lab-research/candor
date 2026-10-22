#!/bin/bash
# The release copy of trial 532's measured export: general.name set, general.finetune dropped, every
# tensor checked byte-identical, hashed, then loaded once in llama-server b11524.
set -euo pipefail
cd /root/candor
E=runs/qwen27b-medium/exports/trial-532/qwen27b-medium-trial-532-Q4_K_M.gguf
D=/root/release
F=$D/Qwen3.8-27B-Candor-Uncensored-Q4_K_M.gguf
mkdir -p $D
echo "== export $(date -u +%T)"
sha256sum $E
echo "== rename $(date -u +%T)"
.venv/bin/python -m gguf.scripts.gguf_new_metadata $E $F --general-name "Qwen3.8-27B-Candor-Uncensored" \
    --remove-metadata general.finetune --force 2>&1 | tail -2
echo "== compare $(date -u +%T)"
.venv/bin/python /root/same.py $E $F
sha256sum $F
ls -l $F
echo "== load $(date -u +%T)"
/root/.cache/candor/llama.cpp-b11524/linux-cuda/bin/llama-server -m $F --port 8181 -c 4096 -np 1 \
    >/root/logs/release-load.log 2>&1 &
server=$!
for _ in $(seq 1 150); do curl -sf localhost:8181/health >/dev/null && break; sleep 2; done
curl -sf localhost:8181/health && echo
kill $server
wait $server || true
echo "== done $(date -u +%T)"

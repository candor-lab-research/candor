#!/bin/bash
# Phase 5 step 6 on one A40: build; export trials 251 and 532, hashed; their held-out harmful prompts
# again with answers up to 1024 tokens, in a hard-linked copy of the run; decide.py's winner, then its
# export measured in full again, to compare with step 5's. Numbers only.
set -euo pipefail
R=runs/qwen27b-medium
L=runs/qwen27b-long
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 60 >>/root/logs/gpumem.log 2>&1 &
echo "== build $(date -u +%T)"
candor run $R
for t in 251 532; do
    echo "== export $t $(date -u +%T)"
    candor export $R --trial $t
done
echo "== hash $(date -u +%T)"
sha256sum $R/exports/*/*.gguf | tee /root/logs/exports.sha256
rm -rf $L
cp -al $R $L
rm -rf $L/.lock $L/logs $L/candor.toml
mkdir $L/logs
# An export is named for its run's folder.
for f in $L/exports/*/qwen27b-medium-*; do
    mv "$f" "${f/qwen27b-medium-/qwen27b-long-}"
done
sed 's/^answer_tokens = 256$/answer_tokens = 1024/' $R/candor.toml >$L/candor.toml
[ "$(diff $R/candor.toml $L/candor.toml | grep -c '^>')" = 1 ]
for t in 251 532; do
    echo "== long $t $(date -u +%T)"
    python -u /root/step6/long.py $L $t
done
echo "== decide $(date -u +%T)"
python /root/step6/decide.py $L | tee /root/logs/decide.txt
w=$(tail -1 /root/logs/decide.txt)
echo "== measure $w again $(date -u +%T)"
candor measure $R --trial $w --exported --name shipped-$w
echo "== done $(date -u +%T)"

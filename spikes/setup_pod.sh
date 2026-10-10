#!/usr/bin/env bash
# Prepares a Runpod PyTorch pod (runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404) for spikes/phase0.py.
# Expects this repo at /root/candor. Run it in tmux; the last line it prints is DONE.
set -euo pipefail

TAG=b11524
RELEASE=https://github.com/ggml-org/llama.cpp/releases/download/$TAG
LLAMA_SHA=5a68281f130a3ea793042f4534939b352669138ff5229ef4127030a01161f171
CUDART_SHA=3286cf2eefefc4ebda4c21a215fca38e65732dd14d6a7ba6a8ab8d6aae6dcb99

python -m venv --system-site-packages /root/venv
/root/venv/bin/pip install -q -e /root/candor "huggingface_hub[hf_xet]"

mkdir -p /root/llama
curl -sL -o /root/llama.tar.gz "$RELEASE/llama-$TAG-bin-ubuntu-cuda-12.8-x64.tar.gz"
echo "$LLAMA_SHA  /root/llama.tar.gz" | sha256sum -c -
tar -xzf /root/llama.tar.gz -C /root/llama
SERVER=$(find /root/llama -name llama-server -type f | head -1)
if ldd "$SERVER" | grep -q "not found"; then
    curl -sL -o /root/cudart.tar.gz "$RELEASE/cudart-llama-$TAG-bin-ubuntu-cuda-12.8-x64.tar.gz"
    echo "$CUDART_SHA  /root/cudart.tar.gz" | sha256sum -c -
    tar -xzf /root/cudart.tar.gz -C "$(dirname "$SERVER")"
fi
if LD_LIBRARY_PATH="$(dirname "$SERVER")" ldd "$SERVER" | grep "not found"; then
    echo "llama-server is missing libraries" >&2
    exit 1
fi
echo "llama-server: $SERVER"

git clone -q --depth 1 --branch "$TAG" https://github.com/ggml-org/llama.cpp /root/llama.cpp
python -m venv /root/convenv
/root/convenv/bin/pip install -q -r /root/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt

echo DONE

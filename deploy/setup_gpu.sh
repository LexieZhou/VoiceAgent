#!/usr/bin/env bash
# One-time setup on the GPU machine (e.g. a RunPod PyTorch pod).
set -euo pipefail
cd "$(dirname "$0")"

# espeak-ng is Kokoro's fallback phonemizer for out-of-vocabulary words.
if command -v apt-get >/dev/null; then
  apt-get update -qq && apt-get install -y -qq espeak-ng
fi

pip install -U pip
pip install -r requirements-gpu.txt

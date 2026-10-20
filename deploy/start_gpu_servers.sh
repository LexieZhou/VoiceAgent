#!/usr/bin/env bash
# Start STT (vLLM Whisper), LLM (vLLM) and TTS (Kokoro) on one GPU.
#
#   VLLM_API_KEY=... ./start_gpu_servers.sh
#
# Every setting below can be overridden via environment variables. Memory
# fractions are of total GPU memory and are sized for a 48 GB card.
set -euo pipefail
cd "$(dirname "$0")"

: "${VLLM_API_KEY:?Set VLLM_API_KEY (same value as in the client .env)}"

STT_MODEL=${STT_MODEL:-openai/whisper-large-v3-turbo}
STT_PORT=${STT_PORT:-8001}
STT_GPU_MEM=${STT_GPU_MEM:-0.15}

LLM_MODEL=${LLM_MODEL:-Qwen/Qwen3-8B}
LLM_PORT=${LLM_PORT:-8000}
LLM_GPU_MEM=${LLM_GPU_MEM:-0.60}
LLM_MAX_LEN=${LLM_MAX_LEN:-8192}

TTS_PORT=${TTS_PORT:-8880}

LOG_DIR=${LOG_DIR:-./logs}
mkdir -p "$LOG_DIR"

wait_for() {
  local name=$1 url=$2 pid=$3
  echo "Waiting for $name at $url ..."
  until curl -sf "$url" >/dev/null; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "$name exited during startup; see $LOG_DIR/$name.log" >&2
      exit 1
    fi
    sleep 5
  done
  echo "$name is ready"
}

# Start the vLLM servers one at a time: each one profiles free GPU memory at
# startup, so concurrent startups can miscount and fail.
nohup vllm serve "$STT_MODEL" \
  --port "$STT_PORT" \
  --gpu-memory-utilization "$STT_GPU_MEM" \
  --api-key "$VLLM_API_KEY" \
  > "$LOG_DIR/stt.log" 2>&1 &
wait_for stt "http://localhost:$STT_PORT/health" $!

nohup vllm serve "$LLM_MODEL" \
  --port "$LLM_PORT" \
  --gpu-memory-utilization "$LLM_GPU_MEM" \
  --max-model-len "$LLM_MAX_LEN" \
  --enable-prefix-caching \
  --api-key "$VLLM_API_KEY" \
  > "$LOG_DIR/llm.log" 2>&1 &
wait_for llm "http://localhost:$LLM_PORT/health" $!

nohup python kokoro_server.py \
  --port "$TTS_PORT" \
  --api-key "$VLLM_API_KEY" \
  > "$LOG_DIR/tts.log" 2>&1 &
wait_for tts "http://localhost:$TTS_PORT/health" $!

echo "All servers up: LLM :$LLM_PORT  STT :$STT_PORT  TTS :$TTS_PORT (logs in $LOG_DIR)"

# Voice Agent

A real-time English voice agent built on [Pipecat](https://github.com/pipecat-ai/pipecat), with barge-in support.

```
Browser mic ──WebRTC──▶ Silero VAD ─▶ Smart Turn v3 ─▶ STT ─▶ LLM ─▶ TTS ──WebRTC──▶ Browser speaker
                        └──────── Mac (local, CPU) ────────┘  └──── GPU server (HTTP) ────┘
```

| Stage | Default | Runs on |
|---|---|---|
| VAD | Silero | local CPU |
| Turn detection | Pipecat Smart Turn v3 (ONNX) | local CPU |
| STT | `openai/whisper-large-v3-turbo` on vLLM | GPU |
| LLM | `Qwen/Qwen3-8B` on vLLM (thinking disabled) | GPU |
| TTS | Kokoro-82M (`deploy/kokoro_server.py`) | GPU |

Every model and parameter lives in [config.yaml](config.yaml). STT, LLM and TTS are reached through
OpenAI-compatible APIs, so switching engines (for example, SGLang for the LLM) only means changing `base_url`.

## 1. GPU server

On a Linux machine with a 48 GB GPU (for example, a RunPod PyTorch pod with HTTP ports 8000, 8001 and 8880 exposed):

```bash
git clone <this repo> && cd VoiceAgent/deploy
./setup_gpu.sh                                  # espeak-ng + vllm[audio] + kokoro
VLLM_API_KEY=<secret> ./start_gpu_servers.sh   # STT :8001, LLM :8000, TTS :8880
```

Override models and memory split with env vars such as `LLM_MODEL`, `LLM_GPU_MEM`, `STT_MODEL` and `STT_GPU_MEM`
(see the top of `start_gpu_servers.sh`). Logs are written to `deploy/logs/`.

## 2. Local agent (Mac)

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
cp .env.example .env        # fill in the GPU URLs and VLLM_API_KEY
.venv/bin/python scripts/healthcheck.py      # TTS -> STT round trip + LLM time to first token
.venv/bin/python -m voice_agent.bot          # optional: --config other.yaml
```

Open http://localhost:7860/client, allow microphone access, and start talking.

## Configuration notes

- **Barge-in**: set `interruptions.enabled`. When the VAD hears you while the bot is speaking, the bot's audio
  stops, the in-flight LLM/TTS work is cancelled, and only what was actually spoken is kept in the context.
  Setting `interruptions.min_words: N` only interrupts after N transcribed words. With segmented Whisper,
  transcripts arrive after you pause, so interruptions then feel slower.
- **Turn detection**: `turn.provider: smart_turn_v3` uses an audio model to decide whether you're done talking,
  so mid-sentence pauses ("I want to... um...") don't trigger a reply. Use `none` for plain silence timeout (`turn.speech_timeout`).
- **Testing without a GPU**: set a section's `provider: openai` and `OPENAI_API_KEY` to use the OpenAI cloud API
  (also set `model`/`voice` to OpenAI names, such as `gpt-4.1-mini`, `whisper-1`, `gpt-4o-mini-tts` / `alloy`,
  and remove the vLLM-only `llm.extra_body`).
- **Latency**: with `metrics.enabled`, Pipecat logs TTFB for each service.

## Tests

```bash
.venv/bin/python -m pytest
```

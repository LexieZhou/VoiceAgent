"""Minimal OpenAI-compatible TTS server for Kokoro-82M.

POST /v1/audio/speech  {"model": "kokoro", "input": "...", "voice": "af_heart",
                        "response_format": "pcm", "speed": 1.0}
Streams raw 16-bit mono PCM at 24 kHz, one chunk per sentence-sized segment,
so the client can start playback before the whole reply is synthesized.

Run: python kokoro_server.py --port 8880 [--api-key KEY]
"""

import argparse
import io
import threading
import wave

import numpy as np
import torch
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from kokoro import KPipeline
from pydantic import BaseModel

SAMPLE_RATE = 24000

app = FastAPI()
_pipelines: dict[str, KPipeline] = {}
_lock = threading.Lock()  # KPipeline is not thread-safe
_api_key: str | None = None


class SpeechRequest(BaseModel):
    input: str
    model: str = "kokoro"
    voice: str = "af_heart"
    response_format: str = "pcm"
    speed: float = 1.0


def _pipeline_for(voice: str) -> KPipeline:
    # Kokoro voice names start with the language code: a=US English, b=UK English, ...
    lang = voice[0]
    if lang not in _pipelines:
        _pipelines[lang] = KPipeline(lang_code=lang, device="cuda" if torch.cuda.is_available() else "cpu")
    return _pipelines[lang]


def _check_auth(authorization: str | None = Header(default=None)):
    if _api_key and authorization != f"Bearer {_api_key}":
        raise HTTPException(status_code=401, detail="Invalid API key")


def _pcm_chunks(req: SpeechRequest):
    with _lock:
        pipeline = _pipeline_for(req.voice)
        for _, _, audio in pipeline(req.input, voice=req.voice, speed=req.speed):
            if audio is None:
                continue
            samples = audio.detach().cpu().numpy() if torch.is_tensor(audio) else np.asarray(audio)
            yield (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


@app.post("/v1/audio/speech", dependencies=[Depends(_check_auth)])
def speech(req: SpeechRequest):
    if req.response_format == "pcm":
        return StreamingResponse(_pcm_chunks(req), media_type="audio/pcm")
    if req.response_format == "wav":
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            for chunk in _pcm_chunks(req):
                w.writeframes(chunk)
        return StreamingResponse(iter([buf.getvalue()]), media_type="audio/wav")
    raise HTTPException(status_code=400, detail="response_format must be 'pcm' or 'wav'")


@app.get("/v1/models", dependencies=[Depends(_check_auth)])
def models():
    return {"object": "list", "data": [{"id": "kokoro", "object": "model"}]}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8880)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--warmup-voice", default="af_heart")
    args = parser.parse_args()
    _api_key = args.api_key

    # Load weights and run one synthesis up front so the first real request is fast.
    list(_pcm_chunks(SpeechRequest(input="Warming up.", voice=args.warmup_voice)))
    uvicorn.run(app, host=args.host, port=args.port)

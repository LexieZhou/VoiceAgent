"""Check that the STT, LLM and TTS servers in config.yaml are reachable, and time them.

TTS synthesizes a test sentence, STT transcribes that audio back (a round trip,
so no sample file is needed), and the LLM is timed to first token.

Run: python scripts/healthcheck.py [--config config.yaml]
"""

import argparse
import io
import sys
import time
import wave

from dotenv import load_dotenv
from openai import OpenAI

from voice_agent.config import load_config

TEST_SENTENCE = "Hello there, this is a quick test of the voice pipeline."


def client(provider: str, base_url: str | None, api_key: str | None) -> OpenAI:
    if provider == "openai_compatible":
        return OpenAI(base_url=base_url, api_key=api_key or "EMPTY")
    return OpenAI(api_key=api_key or None)


def check_tts(cfg) -> bytes:
    c = client(cfg.provider, cfg.base_url, cfg.api_key)
    start = time.perf_counter()
    first_chunk = None
    pcm = bytearray()
    with c.audio.speech.with_streaming_response.create(
        model=cfg.model, voice=cfg.voice, input=TEST_SENTENCE, response_format="pcm"
    ) as r:
        for chunk in r.iter_bytes(4096):
            if first_chunk is None:
                first_chunk = time.perf_counter() - start
            pcm.extend(chunk)
    total = time.perf_counter() - start
    secs = len(pcm) / 2 / cfg.sample_rate
    print(f"TTS  ok  first audio {first_chunk * 1000:.0f} ms, total {total * 1000:.0f} ms, {secs:.1f}s of audio")

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(cfg.sample_rate)
        w.writeframes(bytes(pcm))
    return buf.getvalue()


def check_stt(cfg, wav: bytes):
    c = client(cfg.provider, cfg.base_url, cfg.api_key)
    start = time.perf_counter()
    result = c.audio.transcriptions.create(
        model=cfg.model, file=("test.wav", wav, "audio/wav"), language=cfg.language
    )
    elapsed = time.perf_counter() - start
    print(f"STT  ok  {elapsed * 1000:.0f} ms  -> {result.text.strip()!r}")


def check_llm(cfg):
    c = client(cfg.provider, cfg.base_url, cfg.api_key)
    start = time.perf_counter()
    ttft = None
    text = []
    stream = c.chat.completions.create(
        model=cfg.model,
        messages=[
            {"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": "Say hi in five words."},
        ],
        max_tokens=50,
        stream=True,
        extra_body=cfg.extra_body or None,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            if ttft is None:
                ttft = time.perf_counter() - start
            text.append(delta)
    total = time.perf_counter() - start
    print(f"LLM  ok  first token {ttft * 1000:.0f} ms, total {total * 1000:.0f} ms -> {''.join(text).strip()!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()

    load_dotenv(override=True)
    cfg = load_config(args.config)

    failed = False
    wav = None
    for name, check in [
        ("TTS", lambda: check_tts(cfg.tts)),
        ("STT", lambda: check_stt(cfg.stt, wav)),
        ("LLM", lambda: check_llm(cfg.llm)),
    ]:
        if name == "STT" and wav is None:
            print("STT  skipped (needs TTS audio)")
            failed = True
            continue
        try:
            result = check()
            if name == "TTS":
                wav = result
        except Exception as e:
            print(f"{name}  FAILED: {e}")
            failed = True
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

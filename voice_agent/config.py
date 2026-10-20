"""Typed configuration for the voice agent, loaded from a YAML file.

String values may reference environment variables as ``${VAR}`` or
``${VAR:-default}``; they are expanded before validation.
"""

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ServerConfig(_Strict):
    host: str = "localhost"
    port: int = 7860


class VADConfig(_Strict):
    provider: Literal["silero"] = "silero"
    confidence: float = 0.7
    start_secs: float = 0.2
    stop_secs: float = 0.2
    min_volume: float = 0.6


class TurnConfig(_Strict):
    # smart_turn_v3: ML end-of-turn model on top of VAD.
    # none: end the turn after `speech_timeout` seconds of VAD silence.
    provider: Literal["smart_turn_v3", "none"] = "smart_turn_v3"
    stop_secs: float = 3.0
    speech_timeout: float = 0.6
    cpu_count: int = 1


class STTConfig(_Strict):
    provider: Literal["openai_compatible", "openai"] = "openai_compatible"
    base_url: str | None = None
    api_key: str | None = None
    model: str = "openai/whisper-large-v3-turbo"
    language: str = "en"
    prompt: str | None = None
    # P99 latency from end of speech to transcript; used by turn detection as a safety net.
    ttfs_p99_latency: float = 1.0


class LLMConfig(_Strict):
    provider: Literal["openai_compatible", "openai"] = "openai_compatible"
    base_url: str | None = None
    api_key: str | None = None
    model: str = "Qwen/Qwen3-8B"
    temperature: float | None = 0.7
    max_tokens: int | None = 300
    extra_body: dict[str, Any] = Field(default_factory=dict)
    system_prompt: str = "You are a helpful voice assistant."
    greeting: str | None = None


class TTSConfig(_Strict):
    provider: Literal["openai_compatible", "openai"] = "openai_compatible"
    base_url: str | None = None
    api_key: str | None = None
    model: str = "kokoro"
    voice: str = "af_heart"
    speed: float | None = None
    sample_rate: int = 24000


class InterruptionConfig(_Strict):
    enabled: bool = True
    # >0: require this many transcribed words before the user can interrupt the bot.
    min_words: int = 0


class MetricsConfig(_Strict):
    enabled: bool = True


class AppConfig(_Strict):
    server: ServerConfig = Field(default_factory=ServerConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    turn: TurnConfig = Field(default_factory=TurnConfig)
    stt: STTConfig
    llm: LLMConfig
    tts: TTSConfig
    interruptions: InterruptionConfig = Field(default_factory=InterruptionConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)


def expand_env(value: Any) -> Any:
    """Recursively expand ``${VAR}`` / ``${VAR:-default}`` in strings."""
    if isinstance(value, str):

        def replace(match: re.Match) -> str:
            name, default = match.group(1), match.group(2)
            resolved = os.environ.get(name)
            if resolved:
                return resolved
            if default is not None:
                return default
            raise ValueError(f"Environment variable {name} is not set (referenced in config)")

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    return value


def load_config(path: str | Path | None = None) -> AppConfig:
    path = Path(path or os.environ.get("VOICE_AGENT_CONFIG") or DEFAULT_CONFIG_PATH)
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return AppConfig.model_validate(expand_env(raw))

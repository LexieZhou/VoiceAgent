"""Build Pipecat components from an :class:`AppConfig`.

Each ``build_*`` function dispatches on the section's ``provider`` field, so
swapping a model or inference engine only requires a config change.
"""

from collections.abc import AsyncGenerator

from loguru import logger
from openai import BadRequestError

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.transcriptions.language import Language
from pipecat.turns.user_start import (
    MinWordsUserTurnStartStrategy,
    TranscriptionUserTurnStartStrategy,
    VADUserTurnStartStrategy,
)
from pipecat.turns.user_stop import (
    SpeechTimeoutUserTurnStopStrategy,
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.tracing.service_decorators import traced_tts

from voice_agent.config import AppConfig, LLMConfig, STTConfig, TTSConfig, VADConfig

# Self-hosted OpenAI-compatible servers (vLLM, Kokoro) ignore the key unless
# started with one, but the OpenAI client refuses to run without a value.
_PLACEHOLDER_KEY = "EMPTY"


def _api_key(provider: str, api_key: str | None) -> str | None:
    if api_key:
        return api_key
    # provider "openai" falls back to OPENAI_API_KEY via the OpenAI client.
    return _PLACEHOLDER_KEY if provider == "openai_compatible" else None


def _base_url(provider: str, base_url: str | None, section: str) -> str | None:
    if provider == "openai_compatible" and not base_url:
        raise ValueError(f"{section}.base_url is required for provider 'openai_compatible'")
    return base_url if provider == "openai_compatible" else None


def build_vad(cfg: VADConfig) -> SileroVADAnalyzer:
    return SileroVADAnalyzer(
        params=VADParams(
            confidence=cfg.confidence,
            start_secs=cfg.start_secs,
            stop_secs=cfg.stop_secs,
            min_volume=cfg.min_volume,
        )
    )


def build_turn_strategies(cfg: AppConfig) -> UserTurnStrategies:
    """Turn start strategies control barge-in; stop strategies control end-of-turn."""
    interrupt = cfg.interruptions.enabled
    if cfg.interruptions.min_words > 0:
        # Only transcripts can start a turn, so the bot keeps talking until the
        # user has said `min_words` words (filters out "uh-huh", coughs, etc.).
        start = [
            MinWordsUserTurnStartStrategy(
                min_words=cfg.interruptions.min_words, enable_interruptions=interrupt
            )
        ]
    else:
        start = [
            VADUserTurnStartStrategy(enable_interruptions=interrupt),
            TranscriptionUserTurnStartStrategy(enable_interruptions=interrupt),
        ]

    if cfg.turn.provider == "smart_turn_v3":
        stop = [
            TurnAnalyzerUserTurnStopStrategy(
                turn_analyzer=LocalSmartTurnAnalyzerV3(
                    cpu_count=cfg.turn.cpu_count,
                    params=SmartTurnParams(stop_secs=cfg.turn.stop_secs),
                )
            )
        ]
    else:
        stop = [SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=cfg.turn.speech_timeout)]

    return UserTurnStrategies(start=start, stop=stop)


def build_stt(cfg: STTConfig) -> OpenAISTTService:
    return OpenAISTTService(
        api_key=_api_key(cfg.provider, cfg.api_key),
        base_url=_base_url(cfg.provider, cfg.base_url, "stt"),
        ttfs_p99_latency=cfg.ttfs_p99_latency,
        settings=OpenAISTTService.Settings(
            model=cfg.model,
            language=Language(cfg.language),
            prompt=cfg.prompt,
        ),
    )


def build_llm(cfg: LLMConfig) -> OpenAILLMService:
    extra = {"extra_body": cfg.extra_body} if cfg.extra_body else {}
    return OpenAILLMService(
        api_key=_api_key(cfg.provider, cfg.api_key),
        base_url=_base_url(cfg.provider, cfg.base_url, "llm"),
        settings=OpenAILLMService.Settings(
            model=cfg.model,
            system_instruction=cfg.system_prompt,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            extra=extra,
        ),
    )


class OpenAICompatibleTTSService(OpenAITTSService):
    """OpenAI TTS client for self-hosted ``/v1/audio/speech`` servers.

    The stock service rejects voices outside OpenAI's fixed list; servers like
    Kokoro have their own voice names, so this passes the voice through as-is.
    """

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        create_params = {
            "input": text,
            "model": self._settings.model,
            "voice": self._settings.voice,
            "response_format": "pcm",
        }
        if self._settings.speed:
            create_params["speed"] = self._settings.speed

        try:
            async with self._client.audio.speech.with_streaming_response.create(
                **create_params
            ) as r:
                if r.status_code != 200:
                    error = await r.text()
                    logger.error(f"{self} error getting audio ({r.status_code}): {error}")
                    yield ErrorFrame(error=f"TTS error ({r.status_code}): {error}")
                    return

                await self.start_tts_usage_metrics(text)
                async for chunk in r.iter_bytes(self.chunk_size):
                    if chunk:
                        await self.stop_ttfb_metrics()
                        yield TTSAudioRawFrame(chunk, self.sample_rate, 1, context_id=context_id)
        except BadRequestError as e:
            yield ErrorFrame(error=f"TTS request failed: {e}")


def build_tts(cfg: TTSConfig) -> OpenAITTSService:
    service_cls = OpenAICompatibleTTSService if cfg.provider == "openai_compatible" else OpenAITTSService
    return service_cls(
        api_key=_api_key(cfg.provider, cfg.api_key),
        base_url=_base_url(cfg.provider, cfg.base_url, "tts"),
        sample_rate=cfg.sample_rate,
        settings=OpenAITTSService.Settings(
            model=cfg.model,
            voice=cfg.voice,
            speed=cfg.speed,
        ),
    )

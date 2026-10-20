from pathlib import Path

import pytest
from pydantic import ValidationError

from voice_agent.config import DEFAULT_CONFIG_PATH, expand_env, load_config

ENV = {
    "LLM_BASE_URL": "http://gpu:8000/v1",
    "STT_BASE_URL": "http://gpu:8001/v1",
    "TTS_BASE_URL": "http://gpu:8880/v1",
    "VLLM_API_KEY": "secret",
}


@pytest.fixture
def env(monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)


def test_default_config_loads(env):
    cfg = load_config(DEFAULT_CONFIG_PATH)
    assert cfg.llm.base_url == "http://gpu:8000/v1"
    assert cfg.stt.api_key == "secret"
    assert cfg.llm.extra_body["chat_template_kwargs"]["enable_thinking"] is False
    assert cfg.turn.provider == "smart_turn_v3"
    assert cfg.interruptions.enabled


def test_env_default_used_when_unset(monkeypatch):
    monkeypatch.delenv("SOME_UNSET_VAR", raising=False)
    assert expand_env("${SOME_UNSET_VAR:-fallback}") == "fallback"
    assert expand_env("${SOME_UNSET_VAR:-}") == ""


def test_missing_env_raises(monkeypatch):
    monkeypatch.delenv("SOME_UNSET_VAR", raising=False)
    with pytest.raises(ValueError, match="SOME_UNSET_VAR"):
        expand_env({"a": ["${SOME_UNSET_VAR}"]})


def test_unknown_field_rejected(tmp_path: Path, env):
    bad = DEFAULT_CONFIG_PATH.read_text().replace("model: kokoro", "model: kokoro\n  typo_field: 1")
    path = tmp_path / "config.yaml"
    path.write_text(bad)
    with pytest.raises(ValidationError, match="typo_field"):
        load_config(path)


def test_factory_builds_pipeline_components(env):
    from voice_agent import factory

    cfg = load_config(DEFAULT_CONFIG_PATH)
    factory.build_vad(cfg.vad)
    strategies = factory.build_turn_strategies(cfg)
    assert len(strategies.stop) == 1
    assert isinstance(factory.build_tts(cfg.tts), factory.OpenAICompatibleTTSService)
    factory.build_stt(cfg.stt)
    factory.build_llm(cfg.llm)


def test_min_words_uses_transcript_start_strategy(env):
    from pipecat.turns.user_start import MinWordsUserTurnStartStrategy

    from voice_agent import factory

    cfg = load_config(DEFAULT_CONFIG_PATH)
    cfg.interruptions.min_words = 3
    strategies = factory.build_turn_strategies(cfg)
    assert [type(s) for s in strategies.start] == [MinWordsUserTurnStartStrategy]

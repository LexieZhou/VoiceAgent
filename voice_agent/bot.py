"""Voice agent entry point.

Pipeline: browser mic (WebRTC) -> Silero VAD + Smart Turn -> STT -> LLM -> TTS -> browser.

Run:  python -m voice_agent.bot [--config config.yaml]
Then open http://localhost:7860/client
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import LLMRunFrame
from pipecat.observers.loggers.transcription_log_observer import TranscriptionLogObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.workers.runner import WorkerRunner

from voice_agent.config import load_config
from voice_agent.factory import (
    build_llm,
    build_stt,
    build_tts,
    build_turn_strategies,
    build_vad,
)

load_dotenv(override=True)

transport_params = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    cfg = load_config()
    logger.info(f"Starting bot: stt={cfg.stt.model} llm={cfg.llm.model} tts={cfg.tts.model}")

    stt = build_stt(cfg.stt)
    llm = build_llm(cfg.llm)
    tts = build_tts(cfg.tts)

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=build_vad(cfg.vad),
            user_turn_strategies=build_turn_strategies(cfg),
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=cfg.metrics.enabled,
            enable_usage_metrics=cfg.metrics.enabled,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        observers=[TranscriptionLogObserver()],
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        if cfg.llm.greeting:
            # A user-role cue keeps chat templates that only allow a leading
            # system message (e.g. Qwen) happy.
            context.add_message({"role": "user", "content": cfg.llm.greeting})
            await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Entry point discovered by the Pipecat development runner."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


def _prepare_argv():
    """Handle --config and default --host/--port/-t from the config file."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    args, rest = pre.parse_known_args()
    if args.config:
        os.environ["VOICE_AGENT_CONFIG"] = os.path.abspath(args.config)

    cfg = load_config()  # fail fast on a bad config, before the server starts
    if "--host" not in rest:
        rest += ["--host", cfg.server.host]
    if "--port" not in rest:
        rest += ["--port", str(cfg.server.port)]
    if "-t" not in rest and "--transport" not in rest:
        rest += ["-t", "webrtc"]
    sys.argv = [sys.argv[0], *rest]


if __name__ == "__main__":
    from pipecat.runner.run import main

    _prepare_argv()
    main()

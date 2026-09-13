from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from minpod.devices import detect_device, device_label
from minpod.metrics import PipelineMetrics, ResourceMonitor
from minpod.progress import estimate_audio_s
from minpod.script import Beat

SAMPLE_RATE = 24000
DEFAULT_VOICE = "af_heart"
REPO_ID = "hexgrad/Kokoro-82M"
TEST_SENTENCE = (
    "Kokoro is running locally. This is a short test of the MinPod text to speech pipeline."
)


@dataclass
class SynthesisResult:
    samples: np.ndarray
    sample_rate: int
    metrics: PipelineMetrics


class TTSEngine(Protocol):
    def synthesize(
        self,
        text: str,
        on_progress: Callable[[float, float], None] | None = None,
    ) -> SynthesisResult: ...

    def synthesize_beats(
        self,
        beats: list[Beat],
        on_progress: Callable[[float, float], None] | None = None,
    ) -> SynthesisResult: ...


def _to_numpy(audio: object) -> np.ndarray:
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def _silence(seconds: float) -> np.ndarray:
    n = max(0, int(SAMPLE_RATE * seconds))
    return np.zeros(n, dtype=np.float32)


def _hf_snapshot(repo_id: str) -> Path | None:
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    snapshots = hf_home / "hub" / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    if not snapshots.is_dir():
        return None
    for snap in snapshots.iterdir():
        if snap.is_dir():
            return snap
    return None


def _prefer_local_hub(repo_id: str, voice: str) -> None:
    """Use the HF disk cache only. Inference is local; the Hub is just the weight CDN."""
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        return
    snap = _hf_snapshot(repo_id)
    if snap is None:
        return
    model_name = "kokoro-v1_0.pth"
    needed = (
        snap / "config.json",
        snap / model_name,
        snap / "voices" / f"{voice}.pt",
    )
    if all(path.exists() for path in needed):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


class KokoroEngine:
    def __init__(self, voice: str = DEFAULT_VOICE, lang_code: str = "a") -> None:
        _prefer_local_hub(REPO_ID, voice)
        from kokoro import KPipeline

        self.voice = voice
        started = time.perf_counter()
        self.device = detect_device()
        self.pipeline = KPipeline(
            lang_code=lang_code,
            repo_id=REPO_ID,
            device=self.device,
        )
        self.load_s = time.perf_counter() - started

    def synthesize(
        self,
        text: str,
        on_progress: Callable[[float, float], None] | None = None,
    ) -> SynthesisResult:
        return self.synthesize_beats(
            [Beat(kind="speech", text=text)],
            on_progress=on_progress,
        )

    def synthesize_beats(
        self,
        beats: list[Beat],
        on_progress: Callable[[float, float], None] | None = None,
    ) -> SynthesisResult:
        monitor = ResourceMonitor()
        monitor.start()
        started = time.perf_counter()
        ttft_s = 0.0
        chunks: list[np.ndarray] = []
        heard_speech = False
        done_s = 0.0
        speech_s = 0.0
        words_done = 0
        total_words = sum(len(beat.text.split()) for beat in beats if beat.kind != "pause")
        total_pause = sum(beat.seconds for beat in beats if beat.kind == "pause")
        estimated_s = estimate_audio_s(beats)
        if on_progress is not None:
            on_progress(0.0, estimated_s)

        for beat in beats:
            if beat.kind == "pause":
                silence = _silence(beat.seconds)
                chunks.append(silence)
                done_s += beat.seconds
                if on_progress is not None:
                    on_progress(done_s, max(estimated_s, done_s))
                continue
            if not beat.text.strip():
                continue
            for graphemes, _ps, audio in self.pipeline(beat.text, voice=self.voice):
                if not heard_speech:
                    ttft_s = time.perf_counter() - started
                    heard_speech = True
                samples = _to_numpy(audio)
                chunks.append(samples)
                chunk_s = len(samples) / SAMPLE_RATE
                speech_s += chunk_s
                done_s += chunk_s
                words_done += len(graphemes.split())
                if words_done > 0 and total_words > 0:
                    estimated_s = (speech_s / words_done) * total_words + total_pause
                if on_progress is not None:
                    on_progress(done_s, max(estimated_s, done_s))

        gen_s = time.perf_counter() - started
        cpu_pct, ram_gb, vram_gb = monitor.stop()
        samples = (
            np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        )
        audio_s = float(len(samples) / SAMPLE_RATE)

        return SynthesisResult(
            samples=samples,
            sample_rate=SAMPLE_RATE,
            metrics=PipelineMetrics(
                device=device_label(self.device),
                load_s=self.load_s,
                ttft_s=ttft_s,
                gen_s=gen_s,
                audio_s=audio_s,
                cpu_pct=cpu_pct,
                ram_gb=ram_gb,
                vram_gb=vram_gb,
            ),
        )


def get_engine(name: str = "kokoro") -> TTSEngine:
    key = name.lower().strip()
    if key == "kokoro":
        return KokoroEngine()
    if key == "chatterbox":
        raise NotImplementedError("Chatterbox is not wired up yet.")
    raise ValueError(f"Unknown engine: {name}")

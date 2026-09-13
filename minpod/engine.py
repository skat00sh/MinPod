from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from minpod.devices import detect_device, device_label
from minpod.metrics import PipelineMetrics, ResourceMonitor
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
    def synthesize(self, text: str) -> SynthesisResult: ...

    def synthesize_beats(self, beats: list[Beat]) -> SynthesisResult: ...


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

    def synthesize(self, text: str) -> SynthesisResult:
        return self.synthesize_beats([Beat(kind="speech", text=text)])

    def synthesize_beats(self, beats: list[Beat]) -> SynthesisResult:
        monitor = ResourceMonitor()
        monitor.start()
        started = time.perf_counter()
        ttft_s = 0.0
        chunks: list[np.ndarray] = []
        heard_speech = False

        for beat in beats:
            if beat.kind == "pause":
                chunks.append(_silence(beat.seconds))
                continue
            if not beat.text.strip():
                continue
            for _gs, _ps, audio in self.pipeline(beat.text, voice=self.voice):
                if not heard_speech:
                    ttft_s = time.perf_counter() - started
                    heard_speech = True
                chunks.append(_to_numpy(audio))

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

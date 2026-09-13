from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 24000
AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}


def _ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found. Install with: brew install ffmpeg")
    return ffmpeg


def _run_ffmpeg(args: list[str]) -> None:
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"ffmpeg failed: {detail or 'unknown error'}")


def to_mp3(samples: np.ndarray, sample_rate: int, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        sf.write(tmp_path, samples, sample_rate)
        _run_ffmpeg(
            [
                _ffmpeg(),
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(tmp_path),
                "-b:a",
                "128k",
                str(output),
            ]
        )
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    return output


def collect_audio_paths(
    inputs: list[str | Path],
    skip: str | Path | None = None,
) -> list[Path]:
    skip_path = Path(skip).expanduser().resolve() if skip is not None else None
    collected: list[Path] = []
    seen: set[Path] = set()

    for item in inputs:
        path = Path(item).expanduser().resolve()
        if path.is_dir():
            found = sorted(
                (
                    child
                    for child in path.iterdir()
                    if child.is_file() and child.suffix.lower() in AUDIO_SUFFIXES
                ),
                key=lambda child: child.name.lower(),
            )
            if not found:
                raise FileNotFoundError(f"no audio files in {path}")
            candidates = found
        elif path.is_file():
            if path.suffix.lower() not in AUDIO_SUFFIXES:
                raise ValueError(f"not an audio file: {path}")
            candidates = [path]
        else:
            raise FileNotFoundError(f"missing audio path: {path}")

        for candidate in candidates:
            resolved = candidate.resolve()
            if skip_path is not None and resolved == skip_path:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            collected.append(resolved)

    if not collected:
        raise ValueError("need at least one input file")
    return collected


def merge_mp3(
    inputs: list[str | Path],
    output_path: str | Path,
    gap_s: float = 0.5,
) -> Path:
    output = Path(output_path)
    paths = collect_audio_paths(inputs, skip=output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = _ffmpeg()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        concat = tmp_dir / "concat.txt"
        lines: list[str] = []
        silence: Path | None = None
        if gap_s > 0 and len(paths) > 1:
            silence = tmp_dir / "gap.mp3"
            _run_ffmpeg(
                [
                    ffmpeg,
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    f"anullsrc=r={SAMPLE_RATE}:cl=mono",
                    "-t",
                    f"{gap_s:.3f}",
                    "-c:a",
                    "libmp3lame",
                    "-b:a",
                    "128k",
                    str(silence),
                ]
            )
        for i, path in enumerate(paths):
            lines.append(_concat_entry(path))
            if silence is not None and i < len(paths) - 1:
                lines.append(_concat_entry(silence))
        concat.write_text("\n".join(lines) + "\n", encoding="utf-8")
        _run_ffmpeg(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat),
                "-c",
                "copy",
                str(output.resolve()),
            ]
        )

    return output


def audio_duration_s(path: str | Path) -> float | None:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(Path(path)),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def _concat_entry(path: Path) -> str:
    escaped = str(path).replace("'", r"'\''")
    return f"file '{escaped}'"

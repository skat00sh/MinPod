from __future__ import annotations

import sys
from collections.abc import Callable

from minpod.script import Beat

WORDS_PER_MINUTE = 150.0


def estimate_audio_s(beats: list[Beat], wpm: float = WORDS_PER_MINUTE) -> float:
    words = 0
    pause_s = 0.0
    for beat in beats:
        if beat.kind == "pause":
            pause_s += beat.seconds
        else:
            words += len(beat.text.split())
    speech_s = (words / wpm) * 60.0 if wpm > 0 else 0.0
    return speech_s + pause_s


def format_clock(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


class ProgressPrinter:
    def __init__(self, stream=None) -> None:
        self.stream = sys.stderr if stream is None else stream
        self.enabled = hasattr(self.stream, "isatty") and self.stream.isatty()
        self._last = ""

    def set_text(self, text: str) -> None:
        if not self.enabled:
            self.stream.write(text + "\n")
            self.stream.flush()
            return
        pad = max(0, len(self._last) - len(text))
        self.stream.write("\r" + text + (" " * pad))
        self.stream.flush()
        self._last = text

    def update(self, done_s: float, total_s: float) -> None:
        if not self.enabled:
            return
        total_s = max(total_s, done_s)
        self.set_text(f"audio {format_clock(done_s)} / ~{format_clock(total_s)}")

    def close(self) -> None:
        if self.enabled and self._last:
            self.stream.write("\n")
            self.stream.flush()
            self._last = ""


ProgressCallback = Callable[[float, float], None]

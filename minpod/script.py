from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

PAUSE_SECONDS = 0.5

_NARRATION_HEADING = re.compile(r"^##[^\n]*narration[^\n]*$", re.IGNORECASE | re.MULTILINE)
_SECTION_BREAK = re.compile(r"^\s*---\s*$", re.MULTILINE)
_BRACKET_LABEL = re.compile(r"^\[.+\]$")
_HEADING = re.compile(r"^#{1,6}\s+")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")
_CODE = re.compile(r"`([^`]+)`")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")


@dataclass(frozen=True)
class Beat:
    kind: str
    text: str = ""
    seconds: float = PAUSE_SECONDS


def load_script(path: str | Path) -> list[Beat]:
    file = Path(path)
    raw = file.read_text(encoding="utf-8")
    if file.suffix.lower() in {".md", ".markdown"}:
        body = _narration_body(raw)
        return _beats_from_markdown(body)
    return _beats_from_plain(raw)


def _narration_body(markdown: str) -> str:
    match = _NARRATION_HEADING.search(markdown)
    if match:
        return markdown[match.end() :]
    if markdown.startswith("---"):
        end = markdown.find("\n---", 3)
        if end != -1:
            return markdown[end + 4 :]
    return markdown


def _beats_from_markdown(body: str) -> list[Beat]:
    beats: list[Beat] = []
    blocks = _SECTION_BREAK.split(body)
    for i, block in enumerate(blocks):
        text = _clean_markdown(block)
        if text:
            beats.append(Beat(kind="speech", text=text))
        if i < len(blocks) - 1:
            beats.append(Beat(kind="pause", seconds=PAUSE_SECONDS))
    return _trim_beats(beats)


def _beats_from_plain(body: str) -> list[Beat]:
    beats: list[Beat] = []
    blocks = _SECTION_BREAK.split(body)
    for i, block in enumerate(blocks):
        text = block.strip()
        if text:
            beats.append(Beat(kind="speech", text=text))
        if i < len(blocks) - 1:
            beats.append(Beat(kind="pause", seconds=PAUSE_SECONDS))
    return _trim_beats(beats)


def _clean_markdown(block: str) -> str:
    lines: list[str] = []
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line:
            lines.append("")
            continue
        if _BRACKET_LABEL.fullmatch(line) or _HEADING.match(line):
            continue
        line = _LINK.sub(r"\1", line)
        line = _BOLD.sub(r"\1", line)
        line = _ITALIC.sub(r"\1", line)
        line = _CODE.sub(r"\1", line)
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _trim_beats(beats: list[Beat]) -> list[Beat]:
    while beats and beats[0].kind == "pause":
        beats = beats[1:]
    while beats and beats[-1].kind == "pause":
        beats = beats[:-1]
    return beats

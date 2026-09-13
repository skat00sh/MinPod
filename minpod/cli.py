from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from minpod.ui import print_banner, print_error

DEFAULT_OUTPUT = Path("output/sample.mp3")
DEFAULT_MERGE_OUTPUT = Path("output/merged.mp3")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        print_banner()
        if argv and argv[0] == "merge":
            return _merge_main(argv[1:])
        return _speak_main(argv)
    except KeyboardInterrupt:
        print_error("interrupted")
        return 130
    except (FileNotFoundError, IsADirectoryError, ValueError, RuntimeError) as exc:
        print_error(_friendly_error(exc))
        return 1


def _friendly_error(exc: BaseException) -> str:
    text = str(exc).strip()
    if text.startswith("[Errno"):
        quoted = text.rsplit(": ", 1)[-1].strip().strip("'\"")
        if isinstance(exc, FileNotFoundError):
            return f"file not found: {quoted}"
        if isinstance(exc, IsADirectoryError):
            return f"not a file: {quoted}"
    return text


def _speak_main(argv: list[str]) -> int:
    from minpod.engine import TEST_SENTENCE, get_engine
    from minpod.export import to_mp3
    from minpod.progress import ProgressPrinter
    from minpod.script import load_script

    parser = argparse.ArgumentParser(
        description="Local Kokoro-82M text-to-speech to MP3",
        epilog="Merge existing files: minpod merge a.mp3 b.mp3 -o out.mp3",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", help="Text to speak")
    source.add_argument("--script", help="Markdown or text file to speak")
    parser.add_argument(
        "--output",
        help="MP3 output path (default: output/sample.mp3, or output/<script-stem>.mp3)",
    )
    args = parser.parse_args(argv)

    beats = None
    script_path = None
    if args.script:
        script_path = Path(args.script)
        if not script_path.exists():
            raise FileNotFoundError(f"file not found: {script_path}")
        if not script_path.is_file():
            raise IsADirectoryError(f"not a file: {script_path}")
        beats = load_script(script_path)
        if not beats:
            raise ValueError(f"no spoken text found in {script_path}")

    progress = ProgressPrinter()
    progress.set_text("loading model...")
    try:
        engine = get_engine("kokoro")
        if beats is not None and script_path is not None:
            result = engine.synthesize_beats(beats, on_progress=progress.update)
            output = Path(args.output) if args.output else Path("output") / f"{script_path.stem}.mp3"
        else:
            result = engine.synthesize(
                args.text or TEST_SENTENCE,
                on_progress=progress.update,
            )
            output = Path(args.output) if args.output else DEFAULT_OUTPUT
    finally:
        progress.close()

    written = to_mp3(result.samples, result.sample_rate, output)
    print(result.metrics.format(str(written)))
    return 0


def _merge_main(argv: list[str]) -> int:
    from minpod.export import audio_duration_s, collect_audio_paths, merge_mp3

    parser = argparse.ArgumentParser(
        prog="minpod merge",
        description="Merge audio files or folders into one",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Audio files or folders, in order. Folders are expanded alphabetically.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=str(DEFAULT_MERGE_OUTPUT),
        help="Merged MP3 path",
    )
    parser.add_argument(
        "--gap",
        type=float,
        default=0.5,
        help="Silence between files in seconds (default: 0.5)",
    )
    args = parser.parse_args(argv)

    paths = collect_audio_paths(args.inputs, skip=args.output)
    written = merge_mp3(paths, args.output, gap_s=args.gap)
    duration = audio_duration_s(written)
    print(f"output  {written}")
    print(f"files   {len(paths)}")
    print(f"gap     {args.gap:.1f}s")
    if duration is not None:
        print(f"audio   {duration:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

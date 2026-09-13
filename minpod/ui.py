from __future__ import annotations

import shutil
import subprocess
import sys


def print_banner(stream=None) -> None:
    stream = sys.stderr if stream is None else stream
    if not hasattr(stream, "isatty") or not stream.isatty():
        return
    figlet = shutil.which("figlet")
    if figlet is None:
        stream.write("MinPod\n\n")
        stream.flush()
        return
    result = subprocess.run(
        [figlet, "-f", "slant", "MinPod"],
        capture_output=True,
        text=True,
        check=False,
    )
    art = result.stdout if result.returncode == 0 and result.stdout.strip() else "MinPod\n"
    if not art.endswith("\n"):
        art += "\n"
    stream.write(art)
    if not art.endswith("\n\n"):
        stream.write("\n")
    stream.flush()


def print_error(message: str) -> None:
    sys.stderr.write(f"error: {message}\n")
    sys.stderr.flush()

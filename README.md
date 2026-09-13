# MinPod

Local Kokoro-82M text-to-speech. Speaks a test sentence to MP3.

## Setup

```bash
brew install ffmpeg espeak-ng
uv sync
```

`ffmpeg` is required for MP3 output. `espeak-ng` is the English fallback for out-of-dictionary words.

The first run downloads Kokoro weights from Hugging Face into the local cache (`~/.cache/huggingface`). After that, inference stays on-device and does not call the Hub.

## Run

```bash
uv run minpod
```

Optional:

```bash
uv run minpod --text "Hello from MinPod." --output output/hello.mp3
uv run minpod --script resources/phase1_hld_script.md
```

`--script` reads a markdown or text file. If the markdown has a `## Narration` heading, only that section is spoken. Bracket labels like `[Cold open]` are skipped, and `---` becomes a short pause. Output defaults to `output/<filename>.mp3`.

Merge existing MP3s (does not load the TTS model):

```bash
uv run minpod merge output/phase1_hld_script.mp3 output/phase2_components_script.mp3 -o output/series.mp3
uv run minpod merge output/ -o output/merged.mp3
```

A folder argument takes every audio file in that directory, sorted by name. The output file is skipped if it is in the same folder. `--gap` is silence between files (default 0.5s).

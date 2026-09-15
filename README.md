# auto_rus_subtitles

Automatic Russian subtitles for films that have none.

Give it a video (or audio) file and it will:

1. extract the soundtrack with ffmpeg,
2. transcribe it with Whisper (`faster-whisper`, on the GPU if you have one),
3. group the words into readable subtitle cues and write a `.srt`,
4. produce a copy of the video with the subtitles attached.

Made for language learners who can read Russian but want help following spoken dialogue. It works for other languages too: pass `--language en`, `--language de`, and so on.

## Install

You need three things:

1. **ffmpeg** on your PATH.
   - Debian/Ubuntu: `sudo apt install ffmpeg`
   - Fedora: `sudo dnf install ffmpeg`
   - macOS: `brew install ffmpeg`
   - Windows: `winget install ffmpeg`
   - No admin rights? Download a static build (for example from https://johnvansickle.com/ffmpeg/ on Linux) and put `ffmpeg` and `ffprobe` in a `bin/` folder inside this project. The script looks there if nothing is on PATH.
2. **uv** (Python project manager): https://docs.astral.sh/uv/getting-started/installation/
3. This repository:

```bash
git clone https://github.com/mild-rgb/auto_rus_subtitles.git
cd auto_rus_subtitles
```

The first `uv run` creates a Python 3.12 environment and installs everything else, including the CUDA libraries needed for NVIDIA GPUs. No system CUDA install is required. The Whisper model (about 1.6 GB) downloads on first use into `~/.cache/huggingface`.

**GPU:** any NVIDIA card with 4 GB or more of memory works. Without an NVIDIA GPU it runs on the CPU, which is many times slower but produces the same result.

## Usage

```bash
# Default: writes film.ru.srt and film.subbed.mkv (subtitle track, no re-encode, fast)
uv run subtitle.py /path/to/film.mkv

# Several files at once
uv run subtitle.py ~/Videos/*.avi

# Only the .srt file (VLC and mpv pick it up automatically if it sits next to the video)
uv run subtitle.py film.mp4 --srt-only

# Burn the text into the picture (re-encodes video, slow, for players that can't show soft subs)
uv run subtitle.py film.mp4 --burn

# Put results somewhere else
uv run subtitle.py film.mp4 --output-dir ~/Videos/subbed
```

Try it on the included sample, a 90 second public-domain reading of Chekhov's play "Предложение" (LibriVox):

```bash
uv run subtitle.py samples/chekhov_predlozhenie_90s.mp4
```

Outputs go next to the input file unless `--output-dir` is given:

| file | what it is |
|---|---|
| `film.ru.srt` | plain subtitle file, editable in any text editor |
| `film.subbed.mkv` | original video and audio copied as-is, plus a Russian subtitle track set as default |
| `film.subbed.mp4` | only with `--burn`: subtitles drawn into the picture |

Speed: about 10x real time on a modest GPU (GTX 1050 Ti), so a 2 hour film takes 12 to 15 minutes. Newer GPUs are much faster.

## Options

| flag | default | meaning |
|---|---|---|
| `--model` | `large-v3-turbo` | Whisper model. `medium` or `small` are faster but less accurate. `large-v3` is a bit more accurate but about 5x slower. |
| `--language` | `ru` | spoken language code |
| `--device` | `auto` | `cuda` or `cpu` |
| `--burn` | off | hard-code subtitles instead of adding a track |
| `--crf` | 20 | video quality for `--burn`, lower is better and bigger |
| `--srt-only` | off | skip the video step |
| `--quiet` | off | hide the progress line |

Subtitle layout limits (line length, cue duration, pause splitting) are constants at the top of `subtitle.py`.

## Tips for accuracy

- Whisper occasionally invents text during long music or silence. Voice activity detection is on to limit this. If you see a nonsense cue, just delete it from the `.srt`.
- Very noisy or overlapping dialogue gets worse results. Try `--model large-v3` for a difficult film.
- The `.srt` is the source of truth. Fix it in a text editor and re-attach with:

```bash
ffmpeg -i film.mkv -i film.ru.srt -map 0 -map 1 -c copy -c:s srt -metadata:s:s:0 language=rus film.subbed.mkv
```

## How it works

- `subtitle.py` is the whole pipeline, about 300 lines, no framework.
- Transcription uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (Whisper on CTranslate2) with word timestamps, voice activity detection, and `condition_on_previous_text=False` to avoid repeated-text loops.
- Cues are rebuilt from word timestamps: at most 2 lines of 42 characters, at most 6 seconds on screen, split at sentence ends, commas, and pauses longer than 0.8 s. Two-line cues are balanced rather than greedily wrapped.
- On Linux the script puts the pip-installed CUDA libraries on `LD_LIBRARY_PATH` and re-launches itself once, so the GPU works without a system CUDA install.

## License

MIT. The sample recording is from LibriVox and is in the public domain.

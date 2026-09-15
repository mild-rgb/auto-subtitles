#!/usr/bin/env python3
"""
Automatic Russian subtitles for a media file.

Pipeline:
  1. ffmpeg extracts the soundtrack as 16 kHz mono WAV.
  2. faster-whisper (Whisper via CTranslate2) transcribes it with word timestamps.
  3. Words are regrouped into readable subtitle cues and written as .srt.
  4. ffmpeg attaches the .srt to the video as a subtitle track (no re-encoding),
     or burns it into the picture if --burn is given.

Usage:
  uv run subtitle.py film.mkv
  uv run subtitle.py film.mp4 --burn
  uv run subtitle.py *.avi --srt-only
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent


def ensure_cuda_libs_on_path() -> None:
    """Put the pip-installed NVIDIA libraries (cuBLAS, cuDNN) on LD_LIBRARY_PATH.

    The dynamic loader only reads LD_LIBRARY_PATH at start-up, so if we had to
    add anything we re-launch this same process once with the new value.
    """
    if os.environ.get("_SUBTITLE_REEXEC"):
        return
    try:
        import nvidia  # the namespace package that holds nvidia/cublas, nvidia/cudnn, ...
    except ImportError:
        return
    lib_dirs = [str(d) for root in nvidia.__path__ for d in Path(root).glob("*/lib") if d.is_dir()]
    if not lib_dirs:
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if all(d in current.split(":") for d in lib_dirs):
        return
    os.environ["LD_LIBRARY_PATH"] = ":".join([*lib_dirs, current]).rstrip(":")
    os.environ["_SUBTITLE_REEXEC"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])

# Subtitle layout limits (roughly what TV broadcasters use).
MAX_LINE_CHARS = 42       # characters per line
MAX_LINES = 2             # lines per cue
MAX_CUE_SECONDS = 6.0     # longest a single cue stays on screen
MIN_CUE_SECONDS = 1.0     # shortest a cue may be shown (stretched if possible)
PAUSE_SPLIT_SECONDS = 0.8 # a silence this long between words starts a new cue
CUE_GAP_SECONDS = 0.08    # small gap so consecutive cues do not touch


# --------------------------------------------------------------------------- #
# ffmpeg helpers
# --------------------------------------------------------------------------- #

def find_tool(name: str) -> str:
    """Find ffmpeg/ffprobe on PATH, or fall back to ./bin next to this script."""
    on_path = shutil.which(name)
    if on_path:
        return on_path
    local = HERE / "bin" / name
    if local.exists():
        return str(local)
    sys.exit(f"error: {name} not found on PATH or in {HERE / 'bin'}")


def run(cmd: list[str], quiet: bool = True) -> None:
    if quiet:
        cmd = [cmd[0], "-hide_banner", "-loglevel", "error", *cmd[1:]]
    subprocess.run(cmd, check=True)


def extract_audio(ffmpeg: str, media: Path, wav: Path) -> None:
    """Decode the first audio stream to 16 kHz mono PCM, which is what Whisper wants."""
    run([ffmpeg, "-y", "-i", str(media), "-vn", "-sn", "-dn",
         "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)])


def has_video_stream(ffprobe: str, media: Path) -> bool:
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v", "-show_entries",
         "stream=codec_type", "-of", "csv=p=0", str(media)],
        capture_output=True, text=True).stdout
    return "video" in out


def attach_subtitles(ffmpeg: str, media: Path, srt: Path, out: Path) -> None:
    """Copy all streams as they are and add the .srt as a selectable subtitle track."""
    run([ffmpeg, "-y", "-i", str(media), "-i", str(srt),
         "-map", "0", "-map", "1:0",
         "-c", "copy", "-c:s", "srt",
         "-metadata:s:s:0", "language=rus",
         "-metadata:s:s:0", "title=Русские (авто)",
         "-disposition:s:0", "default",
         str(out)])


def burn_subtitles(ffmpeg: str, media: Path, srt: Path, out: Path, crf: int) -> None:
    """Re-encode the video with the subtitles drawn into the picture."""
    # The subtitles filter has its own quoting rules, so instead of escaping the
    # real path we hand it a plain-named copy and run ffmpeg from that folder.
    with tempfile.TemporaryDirectory(prefix="burn_") as tmp:
        shutil.copy(srt, Path(tmp) / "subs.srt")
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
               "-i", str(media.resolve()),
               "-vf", "subtitles=subs.srt:force_style='FontSize=20,Outline=1'",
               "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
               "-c:a", "copy", "-map", "0:v:0", "-map", "0:a?",
               str(out.resolve())]
        subprocess.run(cmd, check=True, cwd=tmp)


# --------------------------------------------------------------------------- #
# transcription
# --------------------------------------------------------------------------- #

@dataclass
class Word:
    start: float
    end: float
    text: str


def pick_device(requested: str) -> tuple[str, str]:
    """Return (device, compute_type). int8 runs on every GPU and on CPU."""
    import ctranslate2
    if requested in ("auto", "cuda") and ctranslate2.get_cuda_device_count() > 0:
        types = ctranslate2.get_supported_compute_types("cuda")
        return "cuda", ("int8_float16" if "int8_float16" in types else "int8")
    if requested == "cuda":
        print("warning: CUDA requested but not available, using CPU", file=sys.stderr)
    return "cpu", "int8"


def transcribe(wav: Path, model_name: str, language: str, device: str,
               progress: bool) -> list[Word]:
    from faster_whisper import WhisperModel

    dev, ctype = pick_device(device)
    print(f"[2/4] loading model {model_name} on {dev} ({ctype})")
    model = WhisperModel(model_name, device=dev, compute_type=ctype,
                         cpu_threads=os.cpu_count() or 4)

    segments, info = model.transcribe(
        str(wav),
        language=language,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,                       # skip silence and music
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False,      # stops repeated-text hallucination loops
        no_speech_threshold=0.6,
    )
    total = info.duration
    print(f"[3/4] transcribing {total/60:.1f} min of audio "
          f"(language={info.language}, p={info.language_probability:.2f})")

    words: list[Word] = []
    t0 = time.time()
    for seg in segments:
        # Drop segments Whisper itself is unsure contain speech.
        if seg.no_speech_prob > 0.8 and seg.avg_logprob < -1.0:
            continue
        for w in seg.words or []:
            text = w.word.strip()
            if text:
                words.append(Word(w.start, w.end, text))
        if progress and total:
            done = min(seg.end / total, 1.0)
            elapsed = time.time() - t0
            eta = elapsed / done - elapsed if done > 0.01 else 0
            print(f"\r      {done*100:5.1f}%  {seg.end/60:5.1f}/{total/60:.1f} min"
                  f"  eta {eta/60:4.1f} min", end="", flush=True)
    if progress:
        print()
    return words


# --------------------------------------------------------------------------- #
# cue building
# --------------------------------------------------------------------------- #

@dataclass
class Cue:
    start: float
    end: float
    text: str


STRONG_END = re.compile(r"[.!?…]+[»\"')]*$")
WEAK_END = re.compile(r"[,;:—-]+$")


def build_cues(words: list[Word]) -> list[Cue]:
    """Group words into cues that fit on screen and break at natural points."""
    max_chars = MAX_LINE_CHARS * MAX_LINES
    cues: list[Cue] = []
    cur: list[Word] = []

    def text_len(ws: list[Word]) -> int:
        return sum(len(x.text) for x in ws) + max(len(ws) - 1, 0)

    def flush(upto: int | None = None) -> None:
        """Emit cur[:upto] as a cue and keep the rest for the next one."""
        part = cur[:upto] if upto is not None else cur[:]
        rest = cur[upto:] if upto is not None else []
        if part:
            cues.append(Cue(part[0].start, part[-1].end, " ".join(w.text for w in part)))
        cur[:] = rest

    def last_good_break() -> int | None:
        """Index after the last punctuation-ended word, if it leaves a decent cue."""
        for i in range(len(cur) - 1, 0, -1):
            if (STRONG_END.search(cur[i - 1].text) or WEAK_END.search(cur[i - 1].text)) \
                    and text_len(cur[:i]) >= 12:
                return i
        return None

    for w in words:
        if cur:
            prev = cur[-1].text
            cur_len = text_len(cur)
            pause = w.start - cur[-1].end
            if pause > PAUSE_SPLIT_SECONDS:
                flush()                                  # natural break: silence
            elif STRONG_END.search(prev) and cur_len >= 15:
                flush()                                  # natural break: end of sentence
            elif WEAK_END.search(prev) and cur_len >= max_chars * 0.7:
                flush()                                  # comma late in the cue
            elif cur_len + 1 + len(w.text) > max_chars or w.end - cur[0].start > MAX_CUE_SECONDS:
                flush(last_good_break())                 # forced: back up to punctuation
        cur.append(w)
    flush()

    # Timing clean-up: enforce a minimum display time, never overlap the next cue.
    for i, c in enumerate(cues):
        nxt = cues[i + 1].start if i + 1 < len(cues) else None
        if c.end - c.start < MIN_CUE_SECONDS:
            c.end = c.start + MIN_CUE_SECONDS
        if nxt is not None and c.end > nxt - CUE_GAP_SECONDS:
            c.end = max(nxt - CUE_GAP_SECONDS, c.start + 0.3)
    return cues


def wrap_lines(text: str) -> str:
    """Split a cue into at most two lines of similar length, preferring a
    break after punctuation, so the second line is never a lonely word."""
    if len(text) <= MAX_LINE_CHARS:
        return text
    words = text.split()
    best, best_score = None, None
    for i in range(1, len(words)):
        a, b = " ".join(words[:i]), " ".join(words[i:])
        if len(a) > MAX_LINE_CHARS or len(b) > MAX_LINE_CHARS:
            continue
        score = abs(len(a) - len(b))
        if STRONG_END.search(a) or WEAK_END.search(a):
            score -= 12          # a break after punctuation reads better
        if best_score is None or score < best_score:
            best, best_score = (a, b), score
    if best is None:             # too many long words: fall back to plain wrapping
        return "\n".join(textwrap.wrap(text, width=len(text) // 2 + 1,
                                       break_long_words=False, break_on_hyphens=False))
    return "\n".join(best)


def srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(cues: list[Cue], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i, c in enumerate(cues, 1):
            f.write(f"{i}\n{srt_time(c.start)} --> {srt_time(c.end)}\n{wrap_lines(c.text)}\n\n")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def process(media: Path, args: argparse.Namespace, ffmpeg: str, ffprobe: str) -> None:
    out_dir = Path(args.output_dir) if args.output_dir else media.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    srt = out_dir / f"{media.stem}.{args.language}.srt"

    print(f"=== {media.name}")
    with tempfile.TemporaryDirectory(prefix="subs_") as tmp:
        wav = Path(tmp) / "audio.wav"
        print("[1/4] extracting audio")
        extract_audio(ffmpeg, media, wav)
        words = transcribe(wav, args.model, args.language, args.device,
                           progress=not args.quiet)

    cues = build_cues(words)
    write_srt(cues, srt)
    print(f"[4/4] wrote {len(cues)} cues -> {srt}")

    if args.srt_only or not has_video_stream(ffprobe, media):
        return
    if args.burn:
        out = out_dir / f"{media.stem}.subbed.mp4"
        print(f"      burning subtitles into {out.name} (re-encoding, this is slow)")
        burn_subtitles(ffmpeg, media, srt, out, args.crf)
    else:
        out = out_dir / f"{media.stem}.subbed.mkv"
        print(f"      attaching subtitle track -> {out.name}")
        attach_subtitles(ffmpeg, media, srt, out)
    print(f"      done: {out}")


def main() -> None:
    ensure_cuda_libs_on_path()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("media", nargs="+", type=Path, help="video or audio file(s)")
    p.add_argument("--model", default="large-v3-turbo",
                   help="Whisper model: tiny, base, small, medium, large-v3, "
                        "large-v3-turbo (default)")
    p.add_argument("--language", default="ru", help="spoken language code (default ru)")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--output-dir", help="where to write outputs (default: next to input)")
    p.add_argument("--srt-only", action="store_true", help="only write the .srt file")
    p.add_argument("--burn", action="store_true",
                   help="burn subtitles into the picture (re-encodes video) "
                        "instead of adding a subtitle track")
    p.add_argument("--crf", type=int, default=20, help="x264 quality for --burn (lower=better)")
    p.add_argument("--quiet", action="store_true", help="no progress line")
    args = p.parse_args()

    ffmpeg, ffprobe = find_tool("ffmpeg"), find_tool("ffprobe")
    failures = 0
    for media in args.media:
        if not media.exists():
            print(f"skip: {media} does not exist", file=sys.stderr)
            failures += 1
            continue
        try:
            process(media, args, ffmpeg, ffprobe)
        except subprocess.CalledProcessError as e:
            print(f"error: command failed ({e.returncode}): {' '.join(map(str, e.cmd))}",
                  file=sys.stderr)
            failures += 1
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

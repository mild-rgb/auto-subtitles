#!/usr/bin/env python3
"""
Automatic same-language subtitles for a media file, in any language Whisper knows.

Pipeline:
  1. ffmpeg extracts the soundtrack as 16 kHz mono WAV.
  2. faster-whisper (Whisper via CTranslate2) transcribes it with word timestamps.
  3. Words are regrouped into readable subtitle cues and written as .srt.
  4. ffmpeg attaches the .srt to the video as a subtitle track (no re-encoding),
     or burns it into the picture if --burn is given.

The spoken language is detected automatically (or given with --language).
Line length and word joining adapt to the script: Latin/Cyrillic get 42
characters per line, Chinese/Japanese/Korean 18, Thai-like scripts 32.

Inputs can also be URLs (YouTube and the other sites yt-dlp supports): the
video is downloaded first, then handled like a local file.

Usage:
  uv run subtitle.py film.mkv
  uv run subtitle.py film.mp4 --burn
  uv run subtitle.py *.avi --srt-only --language ru
  uv run subtitle.py https://www.youtube.com/watch?v=XXXX
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

# Subtitle layout limits (roughly what TV broadcasters use).
MAX_LINES = 2             # lines per cue
MAX_CUE_SECONDS = 6.0     # longest a single cue stays on screen
MIN_CUE_SECONDS = 1.0     # shortest a cue may be shown (stretched if possible)
PAUSE_SPLIT_SECONDS = 0.8 # a silence this long between words starts a new cue
CUE_GAP_SECONDS = 0.08    # small gap so consecutive cues do not touch
MIN_BREAK_CHARS = 12      # never back up to a punctuation break that leaves a shorter cue


@dataclass(frozen=True)
class Layout:
    """How to lay out text for a given writing system."""
    max_line_chars: int   # characters per line
    sep: str              # what goes between transcribed words: " " or ""


# Wide characters that carry a whole syllable or word each: fewer fit on a line,
# and the language writes no spaces between words.
CJK_LAYOUT = Layout(max_line_chars=18, sep="")
# Korean uses wide characters but does put spaces between words.
KOREAN_LAYOUT = Layout(max_line_chars=20, sep=" ")
# Thai, Lao, Khmer, Burmese: narrow letters, but no spaces between words.
NO_SPACE_LAYOUT = Layout(max_line_chars=32, sep="")
DEFAULT_LAYOUT = Layout(max_line_chars=42, sep=" ")

LAYOUT_BY_LANGUAGE = {
    "zh": CJK_LAYOUT, "yue": CJK_LAYOUT, "ja": CJK_LAYOUT,
    "ko": KOREAN_LAYOUT,
    "th": NO_SPACE_LAYOUT, "lo": NO_SPACE_LAYOUT, "km": NO_SPACE_LAYOUT, "my": NO_SPACE_LAYOUT,
}


def layout_for(language: str) -> Layout:
    return LAYOUT_BY_LANGUAGE.get(language, DEFAULT_LAYOUT)


# ISO 639-1 (what Whisper reports) -> ISO 639-2 (what video containers expect).
ISO_639_2 = {
    "ar": "ara", "bg": "bul", "ca": "cat", "cs": "ces", "da": "dan", "de": "deu",
    "el": "ell", "en": "eng", "es": "spa", "et": "est", "fa": "fas", "fi": "fin",
    "fr": "fra", "he": "heb", "hi": "hin", "hr": "hrv", "hu": "hun", "id": "ind",
    "it": "ita", "ja": "jpn", "ko": "kor", "lt": "lit", "lv": "lav", "ms": "msa",
    "nl": "nld", "no": "nor", "pl": "pol", "pt": "por", "ro": "ron", "ru": "rus",
    "sk": "slk", "sl": "slv", "sr": "srp", "sv": "swe", "th": "tha", "tr": "tur",
    "uk": "ukr", "ur": "urd", "vi": "vie", "zh": "zho", "yue": "yue", "be": "bel",
    "kk": "kaz", "ka": "kat", "hy": "hye", "az": "aze", "uz": "uzb", "tg": "tgk",
    "ky": "kir", "mn": "mon", "lo": "lao", "km": "khm", "my": "mya", "ta": "tam",
    "bn": "ben", "sw": "swa", "tl": "tgl", "cy": "cym", "eu": "eus", "gl": "glg",
    "is": "isl", "mk": "mkd", "sq": "sqi", "bs": "bos", "af": "afr", "ne": "nep",
}


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


def run_ffmpeg(ffmpeg: str, args: list[str], cwd: str | None = None) -> None:
    """Run ffmpeg with the given arguments, printing only errors."""
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *args],
                   check=True, cwd=cwd)


def extract_audio(ffmpeg: str, media: Path, wav: Path) -> None:
    """Decode the first audio stream to 16 kHz mono PCM, which is what Whisper wants."""
    run_ffmpeg(ffmpeg, ["-i", str(media), "-vn", "-sn", "-dn", "-map", "0:a:0",
                        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)])


def has_video_stream(ffprobe: str, media: Path) -> bool:
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v", "-show_entries",
         "stream=codec_type", "-of", "csv=p=0", str(media)],
        capture_output=True, text=True).stdout
    return "video" in out


def attach_subtitles(ffmpeg: str, media: Path, srt: Path, out: Path, language: str) -> None:
    """Copy all streams as they are and add the .srt as a selectable subtitle track."""
    # -dn drops data streams (e.g. the timecode track some MP4s carry), which
    # Matroska cannot hold and which would make ffmpeg refuse to write the file.
    run_ffmpeg(ffmpeg, ["-i", str(media), "-i", str(srt),
                        "-map", "0", "-map", "1:0", "-dn",
                        "-c", "copy", "-c:s", "srt",
                        "-metadata:s:s:0", f"language={ISO_639_2.get(language, 'und')}",
                        "-metadata:s:s:0", f"title={language} (auto)",
                        "-disposition:s:0", "default",
                        str(out)])


def burn_subtitles(ffmpeg: str, media: Path, srt: Path, out: Path, crf: int) -> None:
    """Re-encode the video with the subtitles drawn into the picture."""
    # The subtitles filter has its own quoting rules, so instead of escaping the
    # real path we hand it a plain-named copy and run ffmpeg from that folder.
    with tempfile.TemporaryDirectory(prefix="burn_") as tmp:
        shutil.copy(srt, Path(tmp) / "subs.srt")
        run_ffmpeg(ffmpeg, ["-i", str(media.resolve()),
                            "-vf", "subtitles=subs.srt:force_style='FontSize=20,Outline=1'",
                            "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
                            "-c:a", "copy", "-map", "0:v:0", "-map", "0:a?",
                            str(out.resolve())],
                   cwd=tmp)


# --------------------------------------------------------------------------- #
# download
# --------------------------------------------------------------------------- #

def is_url(s: str) -> bool:
    return s.startswith(("http://", "https://"))


def download(url: str, out_dir: Path, ffmpeg: str, max_height: int, quiet: bool) -> list[Path]:
    """Download a video (or every video of a playlist) with yt-dlp. Returns the file paths."""
    import yt_dlp

    opts = {
        "format": f"bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]/best",
        "merge_output_format": "mkv",
        "outtmpl": str(out_dir / "%(title).150B [%(id)s].%(ext)s"),
        "ffmpeg_location": str(Path(ffmpeg).parent),
        "quiet": quiet,
        "no_warnings": quiet,
        "noprogress": quiet,
        "ignoreerrors": "only_download",   # a broken playlist entry doesn't stop the rest
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    entries = info.get("entries") or [info]
    files = []
    for entry in entries:
        if not entry:
            continue
        downloads = entry.get("requested_downloads") or []
        path = downloads[0].get("filepath") if downloads else None
        if path and Path(path).exists():
            files.append(Path(path))
    if not files:
        raise RuntimeError(f"nothing downloaded from {url}")
    return files


# --------------------------------------------------------------------------- #
# transcription
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
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
               progress: bool) -> tuple[list[Word], str]:
    """Return the transcribed words and the language code that was used."""
    from faster_whisper import WhisperModel

    dev, ctype = pick_device(device)
    print(f"[2/4] loading model {model_name} on {dev} ({ctype})")
    model = WhisperModel(model_name, device=dev, compute_type=ctype,
                         cpu_threads=os.cpu_count() or 4)

    segments, info = model.transcribe(
        str(wav),
        language=None if language == "auto" else language,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,                       # skip silence and music
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False,      # stops repeated-text hallucination loops
        no_speech_threshold=0.6,
    )
    total = info.duration
    detected = "detected" if language == "auto" else "given"
    print(f"[3/4] transcribing {total/60:.1f} min of audio, language {info.language} "
          f"({detected}, confidence {info.language_probability:.0%})")

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
    return words, info.language


# --------------------------------------------------------------------------- #
# cue building
# --------------------------------------------------------------------------- #

@dataclass
class Cue:
    start: float
    end: float
    text: str


STRONG_END = re.compile(r"[.!?…。！？]+[»\"')」』]*$")   # end of a sentence
WEAK_END = re.compile(r"[,;:—\-、，；：]+$")             # a pause inside a sentence


def ends_sentence(text: str) -> bool:
    return STRONG_END.search(text) is not None


def ends_clause(text: str) -> bool:
    """True after any punctuation, strong or weak."""
    return ends_sentence(text) or WEAK_END.search(text) is not None


def build_cues(words: list[Word], layout: Layout) -> list[Cue]:
    """Group words into cues that fit on screen and break at natural points."""
    max_chars = layout.max_line_chars * MAX_LINES
    sep = layout.sep
    cues: list[Cue] = []
    cur: list[Word] = []

    def text_len(ws: list[Word]) -> int:
        return sum(len(x.text) for x in ws) + len(sep) * max(len(ws) - 1, 0)

    def flush(upto: int | None = None) -> None:
        """Emit cur[:upto] as a cue and keep the rest for the next one."""
        if upto is None:
            upto = len(cur)
        part, rest = cur[:upto], cur[upto:]
        if part:
            cues.append(Cue(part[0].start, part[-1].end, sep.join(w.text for w in part)))
        cur[:] = rest

    def last_good_break() -> int | None:
        """Index after the last punctuation-ended word, if it leaves a decent cue."""
        for i in range(len(cur) - 1, 0, -1):
            if ends_clause(cur[i - 1].text) and text_len(cur[:i]) >= MIN_BREAK_CHARS:
                return i
        return None

    for w in words:
        if cur:
            prev = cur[-1].text
            cur_len = text_len(cur)
            pause = w.start - cur[-1].end
            if pause > PAUSE_SPLIT_SECONDS:
                flush()                                  # natural break: silence
            elif ends_sentence(prev) and cur_len >= 15:
                flush()                                  # natural break: end of sentence
            elif ends_clause(prev) and cur_len >= max_chars * 0.7:
                flush()                                  # comma late in the cue
            elif cur_len + len(sep) + len(w.text) > max_chars or w.end - cur[0].start > MAX_CUE_SECONDS:
                flush(last_good_break())                 # forced: back up to punctuation
        cur.append(w)
    flush()

    # Timing clean-up: enforce a minimum display time, never overlap the next cue.
    for c, nxt in zip(cues, cues[1:] + [None]):
        if c.end - c.start < MIN_CUE_SECONDS:
            c.end = c.start + MIN_CUE_SECONDS
        if nxt is not None and c.end > nxt.start - CUE_GAP_SECONDS:
            c.end = max(nxt.start - CUE_GAP_SECONDS, c.start + 0.3)
    return cues


def wrap_lines(text: str, layout: Layout) -> str:
    """Split a cue into at most two lines of similar length, preferring a
    break after punctuation, so the second line is never a lonely word."""
    max_chars = layout.max_line_chars
    if len(text) <= max_chars:
        return text
    if layout.sep == " ":
        units = text.split()
        candidates = [(" ".join(units[:i]), " ".join(units[i:])) for i in range(1, len(units))]
    else:   # no spaces between words: any character boundary is a candidate
        candidates = [(text[:i], text[i:]) for i in range(1, len(text))]

    best, best_score = None, None
    for a, b in candidates:
        if len(a) > max_chars or len(b) > max_chars:
            continue
        score = abs(len(a) - len(b))
        if ends_clause(a):
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


def write_srt(cues: list[Cue], path: Path, layout: Layout) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i, c in enumerate(cues, 1):
            f.write(f"{i}\n{srt_time(c.start)} --> {srt_time(c.end)}\n"
                    f"{wrap_lines(c.text, layout)}\n\n")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def ensure_cuda_libs_on_path() -> None:
    """Make the pip-installed NVIDIA libraries (cuBLAS, cuDNN) findable.

    Linux: the loader only reads LD_LIBRARY_PATH at start-up, so if we had to
    add anything we re-launch this same process once with the new value.
    Windows: DLL directories can be added at runtime, no re-launch needed.
    """
    try:
        import nvidia  # the namespace package that holds nvidia/cublas, nvidia/cudnn, ...
    except ImportError:
        return
    roots = [Path(r) for r in nvidia.__path__]
    if sys.platform == "win32":
        for d in (d for root in roots for d in root.glob("*/bin") if d.is_dir()):
            os.add_dll_directory(str(d))
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
        return
    if os.environ.get("_SUBTITLE_REEXEC"):
        return
    lib_dirs = [str(d) for root in roots for d in root.glob("*/lib") if d.is_dir()]
    if not lib_dirs:
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if all(d in current.split(":") for d in lib_dirs):
        return
    os.environ["LD_LIBRARY_PATH"] = ":".join([*lib_dirs, current]).rstrip(":")
    os.environ["_SUBTITLE_REEXEC"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])


def process(media: Path, args: argparse.Namespace, ffmpeg: str, ffprobe: str) -> None:
    out_dir = args.output_dir or media.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== {media.name}")
    with tempfile.TemporaryDirectory(prefix="subs_") as tmp:
        wav = Path(tmp) / "audio.wav"
        print("[1/4] extracting audio")
        extract_audio(ffmpeg, media, wav)
        words, language = transcribe(wav, args.model, args.language, args.device,
                                     progress=not args.quiet)

    layout = layout_for(language)
    srt = out_dir / f"{media.stem}.{language}.srt"
    cues = build_cues(words, layout)
    write_srt(cues, srt, layout)
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
        attach_subtitles(ffmpeg, media, srt, out, language)
    print(f"      done: {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("media", nargs="+", help="video/audio file(s) or URL(s)")
    p.add_argument("--model", default="large-v3-turbo",
                   help="Whisper model: tiny, base, small, medium, large-v3, "
                        "large-v3-turbo (default)")
    p.add_argument("--language", default="auto",
                   help="spoken language code such as ru, en, ja; default auto = detect "
                        "from the first 30 seconds of speech")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--output-dir", type=Path,
                   help="where to write outputs (default: next to input; "
                        "for URLs the current directory)")
    p.add_argument("--max-height", type=int, default=1080,
                   help="highest video resolution to download for URLs (default 1080)")
    p.add_argument("--srt-only", action="store_true", help="only write the .srt file")
    p.add_argument("--burn", action="store_true",
                   help="burn subtitles into the picture (re-encodes video) "
                        "instead of adding a subtitle track")
    p.add_argument("--crf", type=int, default=20, help="x264 quality for --burn (lower=better)")
    p.add_argument("--quiet", action="store_true", help="no progress line")
    return p.parse_args()


def main() -> None:
    ensure_cuda_libs_on_path()
    args = parse_args()
    ffmpeg, ffprobe = find_tool("ffmpeg"), find_tool("ffprobe")

    failures = 0
    for item in args.media:
        if is_url(item):
            target = args.output_dir or Path.cwd()
            target.mkdir(parents=True, exist_ok=True)
            print(f"=== downloading {item}")
            try:
                files = download(item, target, ffmpeg, args.max_height, args.quiet)
            except Exception as e:  # yt-dlp raises many kinds; report and move on
                print(f"error: download failed: {e}", file=sys.stderr)
                failures += 1
                continue
        else:
            files = [Path(item)]
        for media in files:
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

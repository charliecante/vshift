#!/usr/bin/env python3
"""
vshift - a lightweight terminal voice changer.

Pipeline: mic -> faster-whisper (STT) -> piper (TTS, different voice) -> speakers.

Models load once and stay resident, so repeated conversions after the first
utterance are fast (no per-call reload).
"""
from __future__ import annotations

import argparse
import base64
import collections
import curses
import difflib
import hashlib
import json
import os
import queue
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import urllib.error
import urllib.request
from select import select
from pathlib import Path

# ---------------------------------------------------------------------------
# Friendly import guard - tell the user exactly what to install/activate.
# ---------------------------------------------------------------------------
try:
    import numpy as np
    import sounddevice as sd
    AUDIO_IMPORT_ERROR = None
except (ImportError, OSError) as e:
    # sounddevice raises OSError("PortAudio library not found") - not
    # ImportError - when the system PortAudio library is missing, so catch both.
    np = None
    sd = None
    AUDIO_IMPORT_ERROR = e



# ---------------------------------------------------------------------------
# Config / constants
# ---------------------------------------------------------------------------
DEFAULT_VOICES_DIR = Path.home() / ".local" / "share" / "vshift" / "voices"
STT_SAMPLE_RATE = 16000  # whisper's native rate
DEFAULT_STT_MODEL = "base.en"
DEFAULT_VOICE = "en_US-lessac-medium"
SETTINGS_PATH = Path.home() / ".config" / "vshift" / "settings.json"
VERSION = "1.0.7"
# Piper ids look like "en_US-lessac-medium"; Kokoro voices are namespaced as
# "kokoro:af_heart" so one setting/flag/key covers both engines.
VOICE_ID_RE = re.compile(r"^(?:kokoro:[a-z]{2}_[a-z0-9]+|[A-Za-z0-9][A-Za-z0-9_.-]*)$")
KOKORO_PREFIX = "kokoro:"
# Kokoro's English voices (a = American, b = British; f/m = female/male).
KOKORO_ENGLISH_VOICES = (
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole",
    "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx",
    "am_puck", "am_santa",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
)
KOKORO_BASE_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
# local file name -> (remote file name, exact size in bytes, sha256). Sizes and
# hashes are checked after download so a cut-off or corrupted file can never
# be loaded (that would sound like static, not an error).
KOKORO_FILES = {
    "kokoro-voices-v1.0.bin": ("voices-v1.0.bin", 28214398,
                               "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d"),
    "kokoro-v1.0.onnx": ("kokoro-v1.0.onnx", 325532387,
                         "7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5"),
    "kokoro-v1.0.int8.onnx": ("kokoro-v1.0.int8.onnx", 92361271,
                              "6e742170d309016e5891a994e1ce1559c702a2ccd0075e67ef7157974f6406cb"),
}

# faster-whisper/CTranslate2 has an open Python 3.14 semaphore-warning issue;
# vshift therefore isolates Whisper in a child process so a native crash cannot
# take down the main audio/UI process.

# A curated shortlist so `vshift voices` is actually useful without needing
# to browse the full catalog. Any valid piper voice id works with --voice,
# this is just a starting point covering a few genders/accents.
CURATED_VOICES = {
    "en_US-lessac-medium": "US, male, neutral/clear (default)",
    "en_US-amy-medium": "US, female, warm",
    "en_US-hfc_female-medium": "US, female, expressive",
    "en_US-hfc_male-medium": "US, male, expressive",
    "en_US-ryan-high": "US, male, high quality (slower to synthesize)",
    "en_US-kristin-medium": "US, female, news-anchor style",
    "en_GB-alan-medium": "British, male",
    "en_GB-alba-medium": "British, female, soft",
    "en_GB-northern_english_male-medium": "Northern English, male",
    # Kokoro voices: more natural, but a ~350 MB one-time download and slower
    # to synthesize. They only join the [ ] cycle once that download exists.
    "kokoro:af_heart": "Kokoro, US female (the Kokoro project's default voice)",
    "kokoro:af_bella": "Kokoro, US female",
    "kokoro:am_michael": "Kokoro, US male",
    "kokoro:am_adam": "Kokoro, US male",
    "kokoro:bf_emma": "Kokoro, British female",
    "kokoro:bm_george": "Kokoro, British male",
}
VOICE_SAMPLES_URL = "https://rhasspy.github.io/piper-samples/"

OFFICIAL_ENGLISH_VOICES = [
    "en_GB-alan-low", "en_GB-alan-medium", "en_GB-alba-medium",
    "en_GB-aru-medium", "en_GB-cori-medium", "en_GB-cori-high",
    "en_GB-jenny_dioco-medium", "en_GB-northern_english_male-medium",
    "en_GB-semaine-medium", "en_GB-southern_english_female-low",
    "en_GB-vctk-medium",
    "en_US-amy-low", "en_US-amy-medium", "en_US-arctic-medium",
    "en_US-bryce-medium", "en_US-danny-low", "en_US-hfc_female-medium",
    "en_US-hfc_male-medium", "en_US-joe-medium", "en_US-john-medium",
    "en_US-kathleen-low", "en_US-kristin-medium", "en_US-kusal-medium",
    "en_US-l2arctic-medium", "en_US-lessac-low", "en_US-lessac-medium",
    "en_US-lessac-high", "en_US-libritts-high", "en_US-libritts_r-medium",
    "en_US-ljspeech-medium", "en_US-ljspeech-high", "en_US-norman-medium",
    "en_US-reza_ibrahim-medium", "en_US-ryan-low", "en_US-ryan-medium",
    "en_US-ryan-high", "en_US-sam-medium",
]


# ---------------------------------------------------------------------------
# Terminal styling (plain ANSI, no extra dependency)
# ---------------------------------------------------------------------------
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    MAGENTA = "\033[35m"


if not sys.stdout.isatty() or os.environ.get("NO_COLOR") is not None:
    C.RESET = C.BOLD = C.DIM = C.CYAN = C.GREEN = C.YELLOW = C.RED = C.MAGENTA = ""


def say(msg: str, color: str = "") -> None:
    print(f"{color}{msg}{C.RESET}")


def err(msg: str) -> None:
    print(f"{C.RED}✗ {msg}{C.RESET}", file=sys.stderr)


def require_audio_dependencies() -> None:
    if AUDIO_IMPORT_ERROR is not None:
        if isinstance(AUDIO_IMPORT_ERROR, OSError):
            raise RuntimeError(
                f"The PortAudio system library is missing ({AUDIO_IMPORT_ERROR}). "
                "Install it: Arch/CachyOS: sudo pacman -S portaudio | "
                "Debian/Ubuntu: sudo apt install libportaudio2 | "
                "Fedora: sudo dnf install portaudio"
            )
        raise RuntimeError(
            f"Missing audio dependency ({AUDIO_IMPORT_ERROR}). "
            "Run ./install.sh or: pip install -r requirements.txt."
        )


# ---------------------------------------------------------------------------
# Persistent user settings
# ---------------------------------------------------------------------------
VOLUME_STEP = 0.1                # 10% per key press
VOLUME_MIN, VOLUME_MAX = 0.1, 2.0
VOLUME_DISTORT_WARN = 1.5        # above this, loud voices can clip


def apply_volume(audio: "np.ndarray", volume: float) -> "np.ndarray":
    """Scale int16 audio by `volume`, hard-clipping at the int16 limits.

    Done here, after synthesis, so it works for every voice and doesn't touch
    the Piper worker protocol. Above ~150% a loud voice can clip (it sounds
    crunchy); raising the Discord input volume or the system mixer is cleaner
    for that much boost.
    """
    if volume == 1.0 or audio.size == 0:
        return audio
    scaled = audio.astype(np.float32) * float(volume)
    return np.clip(scaled, -32768, 32767).astype(np.int16)


def _sanitize_settings(saved: object) -> dict:
    """Validate persisted settings so a hand-edited/corrupt file cannot crash startup."""
    out = {
        "voice": DEFAULT_VOICE,
        "stt_model": DEFAULT_STT_MODEL,
        "speed": None,
        "volume": 1.0,
        "speaker_id": None,
        "threshold": 400.0,
        "silence_ms": 700,
        # Off by default: looping the AI's own voice to your real speakers
        # while the mic is actively listening is a real feedback risk unless
        # you're on headphones. Turn it on with [H] or --monitor once you've
        # checked your setup won't just hear itself in a loop.
        "hear_generated": False,
        # Translate speech in any language into English before it is spoken.
        "translate": False,
        "source_language": None,   # Whisper language code; None = detect it
        "mic": None,               # PipeWire/Pulse source or device name; None = system default
    }
    if not isinstance(saved, dict):
        return out

    voice = saved.get("voice")
    if isinstance(voice, str) and VOICE_ID_RE.fullmatch(voice.strip()):
        out["voice"] = voice.strip()

    stt_model = saved.get("stt_model")
    if isinstance(stt_model, str) and stt_model.strip():
        out["stt_model"] = stt_model.strip()

    speed = saved.get("speed")
    if speed is None or isinstance(speed, (int, float)):
        try:
            speed_f = None if speed is None else float(speed)
            if speed_f is None or 0.25 <= speed_f <= 4.0:
                out["speed"] = speed_f
        except (TypeError, ValueError):
            pass

    volume = saved.get("volume")
    if isinstance(volume, (int, float)) and not isinstance(volume, bool):
        if VOLUME_MIN <= float(volume) <= VOLUME_MAX:
            out["volume"] = float(volume)

    speaker_id = saved.get("speaker_id")
    if speaker_id is None:
        out["speaker_id"] = None
    elif isinstance(speaker_id, int) and not isinstance(speaker_id, bool) and speaker_id >= 0:
        out["speaker_id"] = speaker_id

    try:
        threshold = float(saved.get("threshold", out["threshold"]))
        if 25.0 <= threshold <= 2000.0:
            out["threshold"] = threshold
    except (TypeError, ValueError):
        pass

    try:
        silence_ms = int(saved.get("silence_ms", out["silence_ms"]))
        if 200 <= silence_ms <= 3000:
            out["silence_ms"] = silence_ms
    except (TypeError, ValueError):
        pass

    if isinstance(saved.get("hear_generated"), bool):
        out["hear_generated"] = saved["hear_generated"]
    if isinstance(saved.get("translate"), bool):
        out["translate"] = saved["translate"]
    lang = saved.get("source_language")
    if isinstance(lang, str) and lang.strip().lower() in WHISPER_LANGUAGES:
        out["source_language"] = lang.strip().lower()
    mic = saved.get("mic")
    if isinstance(mic, str) and mic.strip() and len(mic) <= 300 and mic.isprintable():
        out["mic"] = mic.strip()
    return out


def load_settings() -> dict:
    try:
        with SETTINGS_PATH.open("r", encoding="utf-8") as f:
            return _sanitize_settings(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError, ValueError):
        return _sanitize_settings({})


def save_settings(values: dict) -> None:
    """Atomically persist user preferences so they survive restarts."""
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = _sanitize_settings(values)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=SETTINGS_PATH.parent,
            prefix="settings.", suffix=".tmp", delete=False
        ) as f:
            tmp_path = Path(f.name)
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, SETTINGS_PATH)
    except OSError:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Voice model management
# ---------------------------------------------------------------------------
def validate_voice_id(voice_id: str) -> str:
    voice_id = str(voice_id).strip()
    if not voice_id or not VOICE_ID_RE.fullmatch(voice_id):
        raise ValueError(
            "Invalid voice id. Use a Piper model id (letters, numbers, _, ., -) "
            "or a Kokoro voice like kokoro:af_heart."
        )
    return voice_id


def is_kokoro_voice(voice_id: str) -> bool:
    return str(voice_id).startswith(KOKORO_PREFIX)


def kokoro_model_filename() -> str:
    """fp32 by default; VSHIFT_KOKORO_MODEL=int8 picks the smaller (92 MB) model,
    which can be faster on CPUs with fast int8 maths. It was not faster on the
    one machine this was measured on, so it is opt-in rather than the default."""
    if os.environ.get("VSHIFT_KOKORO_MODEL", "").strip().lower() == "int8":
        return "kokoro-v1.0.int8.onnx"
    return "kokoro-v1.0.onnx"


def kokoro_paths(voices_dir: Path) -> tuple[Path, Path]:
    return voices_dir / kokoro_model_filename(), voices_dir / "kokoro-voices-v1.0.bin"


def kokoro_installed(voices_dir: Path) -> bool:
    return all(path.exists() for path in kokoro_paths(voices_dir))


def _download_verified(url: str, dest: Path, size: int, sha256: str,
                       label: str, announce: bool) -> None:
    """Download url to dest, atomically, only keeping it if size and hash match."""
    tmp = dest.with_name(dest.name + ".part")
    digest = hashlib.sha256()
    received = 0
    show_progress = announce and sys.stdout.isatty()
    try:
        request = urllib.request.Request(url, headers={"User-Agent": f"vshift/{VERSION}"})
        with urllib.request.urlopen(request, timeout=30) as response, open(tmp, "wb") as out:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                digest.update(chunk)
                received += len(chunk)
                if show_progress:
                    sys.stdout.write(f"\r  {label}: {received * 100 // size}% "
                                     f"({received // (1 << 20)} MB)")
                    sys.stdout.flush()
        if show_progress:
            sys.stdout.write("\n")
        if received != size or digest.hexdigest() != sha256:
            raise RuntimeError(
                f"{label} downloaded incompletely or corrupted "
                f"({received} of {size} bytes, or the checksum did not match). "
                "Try again; nothing was installed.")
        os.replace(tmp, dest)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise RuntimeError(
            f"Could not download {label}: {exc}. You can fetch {url} yourself "
            f"and save it as {dest}.") from exc
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def ensure_kokoro(voices_dir: Path, announce: bool = True) -> Path:
    """Make sure the Kokoro model + voices files exist and are intact; return the model path."""
    voices_dir.mkdir(parents=True, exist_ok=True)
    model_path, voices_path = kokoro_paths(voices_dir)
    for path in (voices_path, model_path):
        if path.exists():
            continue
        remote_name, size, sha256 = KOKORO_FILES[path.name]
        if announce:
            say(f"↓ Downloading Kokoro {'voices' if path is voices_path else 'model'} "
                f"({size // (1 << 20)} MB, one-time)...", C.CYAN)
        _download_verified(KOKORO_BASE_URL + remote_name, path, size, sha256, path.name, announce)
    return model_path


def ensure_voice(
    voice_id: str,
    voices_dir: Path,
    announce: bool = True,
) -> Path:
    """Return the .onnx path for voice_id, downloading it first if needed."""
    voice_id = validate_voice_id(voice_id)
    voices_dir.mkdir(parents=True, exist_ok=True)
    if is_kokoro_voice(voice_id):
        name = voice_id[len(KOKORO_PREFIX):]
        if name not in KOKORO_ENGLISH_VOICES:
            raise ValueError(
                f"Unknown Kokoro voice '{name}'. English voices: "
                + ", ".join(KOKORO_ENGLISH_VOICES))
        return ensure_kokoro(voices_dir, announce)
    onnx_path = voices_dir / f"{voice_id}.onnx"
    json_path = voices_dir / f"{voice_id}.onnx.json"
    if onnx_path.exists() and json_path.exists():
        return onnx_path

    if announce:
        say(f"↓ Downloading voice '{voice_id}' (one-time)...", C.CYAN)

    # Piper's downloader writes straight to the final file name, so an
    # interrupted download used to leave a cut-off .onnx that looked installed
    # and then failed to load forever. Download into a scratch folder and only
    # move complete files into place (the .onnx last, since both must exist).
    with tempfile.TemporaryDirectory(prefix=".download-", dir=voices_dir) as tmp:
        tmp_dir = Path(tmp)
        try:
            result = subprocess.run(
                [sys.executable, "-m", "piper.download_voices",
                 "--download-dir", str(tmp_dir), voice_id],
                capture_output=True, text=True, timeout=900,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"Voice download failed for '{voice_id}': {exc}") from exc

        tmp_onnx, tmp_json = tmp_dir / onnx_path.name, tmp_dir / json_path.name
        if result.returncode != 0 or not tmp_onnx.exists() or not tmp_json.exists():
            detail = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(
                f"Could not download voice '{voice_id}'. Check the id at "
                f"{VOICE_SAMPLES_URL} and your internet connection."
                + (f"\nDetails: {detail[-500:]}" if detail else "")
            )
        os.replace(tmp_json, json_path)
        os.replace(tmp_onnx, onnx_path)
    return onnx_path


def remove_voice_files(voice_id: str, voices_dir: Path) -> list[Path]:
    """Delete a Piper voice's downloaded files (for `vshift download --force`)."""
    voice_id = validate_voice_id(voice_id)
    removed = []
    if is_kokoro_voice(voice_id):
        paths = kokoro_paths(voices_dir)
    else:
        paths = (voices_dir / f"{voice_id}.onnx", voices_dir / f"{voice_id}.onnx.json")
    for path in paths:
        if path.exists():
            path.unlink()
            removed.append(path)
    return removed


_KOKORO_KINDS = {"af": "female", "am": "male", "bf": "female", "bm": "male"}


def voice_catalog(voices_dir: Path) -> list[dict]:
    """Every English voice vshift knows about, for the voice browser.

    Each entry: id, name, tag (engine + accent), desc, installed, search (the
    lowercase text a query is matched against). Downloaded voices come first,
    then curated ones, then the rest of the catalog.
    """
    voices_dir = Path(voices_dir)
    try:
        on_disk = {p.stem for p in voices_dir.glob("*.onnx")
                   if not p.name.startswith("kokoro-") and p.with_name(p.name + ".json").exists()}
    except OSError:
        on_disk = set()
    kokoro_ready = kokoro_installed(voices_dir)
    piper_ids = list(dict.fromkeys(
        [v for v in CURATED_VOICES if not is_kokoro_voice(v)] + OFFICIAL_ENGLISH_VOICES + sorted(on_disk)))
    entries = []
    for vid in piper_ids:
        lang, _, rest = vid.partition("-")
        quality = rest.rsplit("-", 1)[-1] if "-" in rest else ""
        accent = {"en_US": "US", "en_GB": "UK"}.get(lang, lang)
        desc = CURATED_VOICES.get(vid) or (f"{quality} quality" if quality else "")
        if vid not in OFFICIAL_ENGLISH_VOICES and vid not in CURATED_VOICES:
            desc = "downloaded (custom)"
        name, tag = _voice_parts(vid)
        entries.append(dict(
            id=vid, name=name, tag=tag, desc=desc, installed=vid in on_disk,
            curated=vid in CURATED_VOICES, engine="piper",
            search=f"{vid} {desc} piper {accent} {'british' if accent == 'UK' else 'american'}".lower()))
    for short in KOKORO_ENGLISH_VOICES:
        vid = KOKORO_PREFIX + short
        kind = _KOKORO_KINDS.get(short[:2], "")
        desc = CURATED_VOICES.get(vid, "").replace("Kokoro, ", "") or kind
        name, tag = _voice_parts(vid)
        entries.append(dict(
            id=vid, name=name, tag=tag, desc=desc, installed=kokoro_ready,
            curated=vid in CURATED_VOICES, engine="kokoro",
            search=(f"{vid} {desc} {kind} kokoro "
                    f"{'uk british' if short[0] == 'b' else 'us american'}").lower()))
    entries.sort(key=lambda e: (not e["installed"], e["engine"] != "piper", not e["curated"], e["id"]))
    return entries


def filter_voices(entries: list[dict], query: str) -> list[dict]:
    """Entries matching every word of query ('uk female', 'kokoro', 'amy').
    'male' must not match 'female', so words match whole words or id prefixes."""
    words = query.lower().split()
    if not words:
        return list(entries)

    def matches(entry: dict) -> bool:
        tokens = re.split(r"[^a-z0-9]+", entry["search"])
        for w in words:
            if w in ("male", "female"):
                if w not in tokens:
                    return False
            elif w not in entry["search"]:
                return False
        return True
    return [e for e in entries if matches(e)]


def voice_download_hint(entry: dict) -> str:
    if entry["engine"] == "kokoro":
        return "Not downloaded: Enter fetches Kokoro once (~350 MB, all 28 voices)"
    size = "~110 MB" if entry["id"].endswith("-high") else "~60 MB"
    return f"Not downloaded: Enter downloads it ({size}) and switches"


# ---------------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------------
def record_until_enter(device: int | None = None, mic: "MicChoice | None" = None) -> np.ndarray:
    """Record mono int16 audio from the mic until the user presses Enter."""
    frames: list[np.ndarray] = []
    stop_flag = threading.Event()
    if mic is not None:
        apply_mic_env(mic)
        device = mic.device

    def _reader():
        with sd.InputStream(
            samplerate=STT_SAMPLE_RATE, channels=1, dtype="int16", device=device
        ) as stream:
            while not stop_flag.is_set():
                data, _overflow = stream.read(1024)
                frames.append(data.copy())

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    say("  ● recording... press Enter to stop", C.RED)
    try:
        input()
    except KeyboardInterrupt:
        pass
    stop_flag.set()
    t.join(timeout=2)

    if not frames:
        return np.zeros(0, dtype=np.int16)
    return np.concatenate(frames, axis=0).reshape(-1)


def resample_int16(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Simple linear-interpolation resample, good enough for speech."""
    if audio.size == 0 or orig_sr == target_sr:
        return audio
    duration = audio.shape[0] / orig_sr
    target_len = max(1, int(round(duration * target_sr)))
    x_old = np.linspace(0, duration, num=audio.shape[0], endpoint=False)
    x_new = np.linspace(0, duration, num=target_len, endpoint=False)
    resampled = np.interp(x_new, x_old, audio.astype(np.float32))
    return resampled.astype(np.int16)


def play_int16(audio: np.ndarray, sample_rate: int, device: int | None = None) -> None:
    """Play audio, resampling as needed. Many virtual sinks (PipeWire null
    sinks in particular) only accept their own configured rate and raise
    PortAudioError('Invalid sample rate') for anything else, so we try the
    device's own default rate, then a couple of common hardware rates, before
    giving up."""
    if audio.size == 0:
        return

    candidates = _rate_candidates(sample_rate, device)

    last_err: Exception | None = None
    for sr in candidates:
        try:
            clip = audio if sr == sample_rate else resample_int16(audio, sample_rate, sr)
            sd.play(clip, samplerate=sr, device=device)
            sd.wait()
            return
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(
        f"Couldn't play audio on this output device at any sample rate ({last_err})"
    )


def _rate_candidates(sample_rate: int, device: int | None) -> list[int]:
    """Sample rates to try for an output device, best first."""
    candidates: list[int] = [sample_rate]
    if device is not None:
        try:
            info = sd.query_devices(device)
            dev_sr = int(round(info.get("default_samplerate") or 0))
            if dev_sr and dev_sr not in candidates:
                candidates.append(dev_sr)
        except Exception:
            pass
    for sr in (48000, 44100):
        if sr not in candidates:
            candidates.append(sr)
    return candidates


def _latency_from_env() -> "str | float | None":
    """VSHIFT_LATENCY=high|low|<seconds> -> PortAudio latency for the output
    stream; unset means PortAudio's default. A bigger buffer trades a little
    delay for fewer underruns (crackle) when the system is busy."""
    raw = os.environ.get("VSHIFT_LATENCY", "").strip().lower()
    if raw in ("low", "high"):
        return raw
    try:
        value = float(raw)
        return value if value > 0 else None
    except ValueError:
        return None


class Player:
    """Plays clips through one output stream that stays open between clips.

    sd.play() opens and closes a stream for every clip, which can leave small
    gaps between sentences (and pops on some sinks). This keeps the stream
    open and just writes the next clip into it. The stream is closed after a
    few idle seconds so vshift doesn't hold an output device open all day, and
    is reopened if the sample rate changes (e.g. after a voice switch) or a
    write fails. Only the playback thread should use an instance.
    """

    IDLE_CLOSE_S = 5.0

    def __init__(self, device: int | None = None):
        self.device = device
        self.latency = _latency_from_env()
        self.stream = None
        self.stream_rate = 0   # rate the open stream runs at
        self.source_rate = 0   # rate of the clips it was opened for
        self.last_used = 0.0

    def _write(self, audio: np.ndarray, sample_rate: int) -> None:
        clip = audio if sample_rate == self.stream_rate else resample_int16(
            audio, sample_rate, self.stream_rate)
        self.stream.write(np.ascontiguousarray(clip.reshape(-1, 1)))
        self.last_used = time.monotonic()

    def play(self, audio: np.ndarray, sample_rate: int) -> None:
        if audio.size == 0:
            return
        if self.stream is not None and self.source_rate == sample_rate:
            try:
                self._write(audio, sample_rate)
                return
            except Exception:
                self.close()  # stale/broken stream: fall through and reopen
        self.close()

        last_err: Exception | None = None
        for sr in _rate_candidates(sample_rate, self.device):
            try:
                extra = {} if self.latency is None else {"latency": self.latency}
                stream = sd.OutputStream(
                    samplerate=sr, channels=1, dtype="int16", device=self.device, **extra)
                stream.start()
            except Exception as e:
                last_err = e
                continue
            self.stream, self.stream_rate, self.source_rate = stream, sr, sample_rate
            try:
                self._write(audio, sample_rate)
                return
            except Exception as e:
                last_err = e
                self.close()
        raise RuntimeError(
            f"Couldn't play audio on this output device at any sample rate ({last_err})"
        )

    def drain(self) -> None:
        """Wait out the device buffer. write() returns once audio is buffered,
        not once it has been heard; call this when nothing else is queued so
        "playing" and the echo window end when the sound really does."""
        if self.stream is None:
            return
        try:
            time.sleep(min(max(float(self.stream.latency), 0.0), 0.5))
        except Exception:
            pass

    def close_if_idle(self) -> None:
        if self.stream is not None and time.monotonic() - self.last_used > self.IDLE_CLOSE_S:
            self.close()

    def close(self) -> None:
        stream, self.stream = self.stream, None
        if stream is None:
            return
        try:
            stream.stop()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass


def resolve_sink_output(sink_name: str) -> int | None:
    """Find the PortAudio output device index that plays into `sink_name`.

    PortAudio normally lists PulseAudio/PipeWire sinks by name, but a PortAudio
    built without a Pulse/PipeWire host API only lists ALSA devices, so the
    virtual sink never appears. In that case fall back to the ALSA "pulse"
    device and steer it to the sink with PULSE_SINK (which only affects
    playback, not the mic). Returns None if neither route works.
    """
    idx = find_device_index_by_name(sink_name, want_output=True)
    if idx is None:
        sd._terminate()
        sd._initialize()
        idx = find_device_index_by_name(sink_name, want_output=True)
    if idx is not None:
        return idx
    idx = find_device_index_by_name("pulse", want_output=True)
    if idx is not None:
        os.environ["PULSE_SINK"] = sink_name
        say(f"PortAudio doesn't list '{sink_name}'; routing through its 'pulse' "
            "device with PULSE_SINK instead.", C.YELLOW)
    return idx




# ---------------------------------------------------------------------------
# Voice-activity detection (energy-based) for hands-free "continuous" mode
# ---------------------------------------------------------------------------
_VAD_CHUNK_MS = 30
_VAD_CHUNK_SAMPLES = int(STT_SAMPLE_RATE * _VAD_CHUNK_MS / 1000)


class UtteranceSegmenter:
    """Streaming version of the old one-shot listen_for_utterance().

    Feed it fixed-size int16 chunks from a persistent InputStream callback;
    it returns a complete utterance once it sees speech followed by enough
    trailing silence, or None if there's nothing to emit yet.

    Because the mic stream that feeds this stays open for the whole session
    (see capture_loop below) instead of being closed while a reply is
    transcribed/synthesized/played, speech spoken mid-reply is captured and
    queued instead of being silently dropped.
    """

    def __init__(self, threshold: float, silence_ms: int,
                 min_speech_ms: int = 200, max_utterance_ms: int = 15000,
                 prebuffer_ms: int = 300):
        self.threshold = threshold
        self.silence_needed = max(1, silence_ms // _VAD_CHUNK_MS)
        self.min_speech_chunks = max(1, min_speech_ms // _VAD_CHUNK_MS)
        self.max_chunks = max(1, max_utterance_ms // _VAD_CHUNK_MS)
        self.ring: collections.deque = collections.deque(maxlen=max(1, prebuffer_ms // _VAD_CHUNK_MS))
        self.speaking = False
        self.utterance: list[np.ndarray] = []
        self.silence_run = 0
        self.speech_chunks = 0
        self.last_rms = 0.0  # level of the most recent chunk, for the TUI meter

    def reset(self) -> None:
        """Drop any half-collected utterance (used when the mic is paused, so
        speech from before the pause isn't glued to silence after it)."""
        self.ring.clear()
        self.speaking = False
        self.utterance = []
        self.silence_run = 0
        self.speech_chunks = 0
        self.last_rms = 0.0

    def update_params(self, threshold: float, silence_ms: int) -> None:
        """Live-tune sensitivity (e.g. from the TUI) without losing state."""
        self.threshold = threshold
        self.silence_needed = max(1, silence_ms // _VAD_CHUNK_MS)

    def feed(self, chunk: np.ndarray) -> np.ndarray | None:
        rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
        self.last_rms = rms

        if not self.speaking:
            self.ring.append(chunk)
            if rms > self.threshold:
                self.speaking = True
                self.utterance = list(self.ring)
                self.ring.clear()
                self.speech_chunks = 1
                self.silence_run = 0
            return None

        self.utterance.append(chunk)
        if rms > self.threshold:
            self.silence_run = 0
            self.speech_chunks += 1
        else:
            self.silence_run += 1

        if self.silence_run >= self.silence_needed and self.speech_chunks < self.min_speech_chunks:
            # Too short to be speech (a click, a cough, a key press). Drop it
            # instead of recording silence until max_utterance_ms and handing
            # Whisper 15 s of nothing, which it likes to hallucinate over.
            self.speaking = False
            self.utterance = []
            self.silence_run = 0
            self.speech_chunks = 0
            return None

        ended_by_silence = self.silence_run >= self.silence_needed
        ended_by_maxlen = len(self.utterance) >= self.max_chunks
        if not (ended_by_silence or ended_by_maxlen):
            return None

        result = np.concatenate(self.utterance, axis=0)
        self.speaking = False
        self.utterance = []
        self.silence_run = 0
        self.speech_chunks = 0
        return result


CALIBRATE_SECONDS = 2.0
_CALIBRATE_MIN_SAMPLES = 20      # ~0.6 s of 30 ms chunks; fewer means the mic isn't delivering
_CALIBRATE_MAX_NOISE = 600.0     # a noise floor this high means you were talking, not quiet
SPEED_STEP = 0.05                # Piper length_scale step per key press
SPEED_MIN, SPEED_MAX = 0.5, 2.0


def threshold_from_noise(levels: list[float]) -> tuple[float | None, str]:
    """Pick a speech threshold from mic levels measured while you stay quiet.

    Uses the 95th percentile (so one click or cough doesn't set it), doubled
    with a fixed minimum margin, rounded to the TUI's 25-step grid. Returns
    (threshold, "") or (None, reason) when the measurement can't be trusted.
    """
    if len(levels) < _CALIBRATE_MIN_SAMPLES:
        return None, "Calibration failed: not enough audio from the mic"
    floor = float(np.percentile(levels, 95))
    if floor > _CALIBRATE_MAX_NOISE:
        return None, "Too loud to calibrate: stay quiet and press C again"
    raw = max(floor * 2.0, floor + 100.0)
    return float(min(2000, max(25, int(round(raw / 25.0)) * 25))), ""


def capture_loop(
    state: "SharedState",
    in_dev: int | None,
    utt_queue: "queue.Queue[np.ndarray]",
    stop_event: threading.Event,
) -> None:
    """Owns the mic InputStream for the entire session. Keeps reading and
    segmenting continuously - including while a previous utterance is still
    being transcribed, synthesized, or played back - and pushes each
    completed utterance onto utt_queue. Retries on transient device errors
    instead of giving up."""
    segmenter = UtteranceSegmenter(state.threshold, state.silence_ms)
    while not stop_event.is_set():
        with state.lock:
            if state.quit:
                return
            mic, mic_generation = state.mic, state.mic_generation
        device = mic.device if mic is not None else in_dev
        try:
            if mic is not None:
                apply_mic_env(mic)
            with sd.InputStream(
                samplerate=STT_SAMPLE_RATE, channels=1, dtype="int16", device=device
            ) as stream:
                state.clear_error("mic")
                while not stop_event.is_set():
                    with state.lock:
                        if state.quit:
                            return
                        if state.mic_generation != mic_generation:
                            break  # mic switched ([M]): reopen on the new one
                        paused = state.paused
                        calibrating = state.calibrating_until is not None
                        segmenter.update_params(state.threshold, state.silence_ms)
                    data, _overflow = stream.read(_VAD_CHUNK_SAMPLES)
                    data = data.reshape(-1)
                    if paused:
                        segmenter.reset()
                        with state.lock:
                            state.level = 0.0
                            state.calibrating_until = None  # pausing cancels calibration
                        continue
                    if calibrating:
                        # Measure only; never emit utterances while calibrating.
                        segmenter.reset()
                        rms = float(np.sqrt(np.mean(data.astype(np.float32) ** 2)))
                        with state.lock:
                            state.level = rms
                        if state.calibration_sample(rms):
                            state.persist()
                        continue
                    utterance = segmenter.feed(data)
                    with state.lock:
                        state.level = segmenter.last_rms
                    if utterance is not None and utterance.size > 0:
                        utt_queue.put(utterance)
            segmenter.reset()  # don't glue speech from the old mic onto the new one
        except Exception as e:
            state.set_error("mic", f"Microphone: {e}")
            time.sleep(0.5)


class SharedState:
    """Thread-safe state shared between the audio worker, voice loader, and UI."""

    def __init__(
        self,
        threshold: float,
        silence_ms: int,
        voice: str,
        tts_client: "TTSClient",
        tts_voice_name: str,
        voices_dir: Path,
        speed: float | None,
        speaker_id: int | None,
        stt_model: str = DEFAULT_STT_MODEL,
        hear_generated: bool = False,
        settings_path: Path = SETTINGS_PATH,
        direct_output: bool = False,
        volume: float = 1.0,
        translate: bool = False,
        source_language: str | None = None,
    ):
        # RLock: error helpers below take the lock themselves and are also
        # called from code that already holds it.
        self.lock = threading.RLock()
        self.status = "starting"  # only meaningful value left: "starting" pre-pipeline; see status_label()
        self.stt_busy = False
        self.tts_busy = False
        self.playing = False
        self.queue_depth = 0
        self.text_queue: "queue.Queue[str] | None" = None  # set by worker_loop; typed text feeds into it
        self.last_heard = ""
        self.utterance_count = 0
        self.threshold = threshold
        self.silence_ms = silence_ms
        self.level = 0.0  # live mic RMS, for the TUI meter
        self.hear_generated = hear_generated
        self.speed = speed
        self.volume = volume
        self.speaker_id = speaker_id
        self.stt_model = stt_model
        self.stt_client: "STTClient | None" = None  # set by start_conversion_tui, for [L]
        self.previewer: "VoicePreviewer | None" = None  # voice browser samples, set likewise
        self.mic: "MicChoice | None" = None   # None: capture uses the in_dev it was given
        self.mic_generation = 0               # bumped on every switch so capture reopens
        self.mic_label = ""
        self.own_mic_names: set[str] = {DEFAULT_MIC_NAME}
        self.translate = translate
        self.source_language = source_language
        self.stt_loading = ""        # e.g. "Loading 'base' for translation…" while Whisper reloads
        self.last_language = ""      # language the last heard line was in (translate mode)
        self.settings_path = settings_path
        self.paused = False
        self.quit = False
        # Errors are tracked per source ("mic", "stt", "tts", "audio", "monitor")
        # so one stage succeeding can no longer erase another stage's error.
        self.errors: dict[str, str] = {}
        # True when audio goes straight to real speakers (no virtual mic), in
        # which case the H monitor toggle does not apply and the echo guard
        # is always on.
        self.direct_output = direct_output
        self.echo_entries: "collections.deque[dict]" = collections.deque(maxlen=16)
        self.last_spoken = ""                 # full text of the last line, for [R]epeat
        self.typed_history: list[str] = []    # this session only; Up/Down in the [T] prompt
        self.notice = ""                      # short transient message for the TUI
        self.notice_until = 0.0
        self.calibrating_until: float | None = None
        self.calib_levels: list[float] = []

        self.voice = voice
        self.tts_client = tts_client
        self.voices_dir = voices_dir
        self.pending_voice: str | None = None
        self.voice_loading = False
        self.voice_error = ""
        # Kokoro needs a ~350 MB download, so don't let a stray "]" trigger it:
        # its voices join the cycle once the files exist (or one is selected).
        self.refresh_voice_cycle()

    def refresh_voice_cycle(self) -> None:
        kokoro_ready = kokoro_installed(self.voices_dir)
        with self.lock:
            self.curated_voice_ids = tuple(
                v for v in CURATED_VOICES
                if not is_kokoro_voice(v) or kokoro_ready or v == self.voice)

    @property
    def error(self) -> str:
        with self.lock:
            return "; ".join(m for m in self.errors.values() if m)

    def set_error(self, key: str, message: str) -> None:
        with self.lock:
            if message:
                self.errors[key] = message
            else:
                self.errors.pop(key, None)

    def clear_error(self, key: str) -> None:
        self.set_error(key, "")

    def note_spoken(self, text: str) -> dict:
        """Remember what the pipeline is about to say, for the echo guard."""
        entry = {"text": _normalize_for_echo_compare(text), "ts": time.monotonic()}
        with self.lock:
            self.echo_entries.append(entry)
        return entry

    def _status_label(self) -> tuple[str, str]:
        """Compute an accurate (key, label) pair. Must be called with
        self.lock held. Capture/STT/TTS/playback run on independent threads
        and can genuinely all be busy at once - e.g. Whisper transcribing
        what you just said while Piper is still speaking the previous
        reply - so this is a live composite of what's actually happening
        right now, not a single string one stage overwrites another's."""
        if any(self.errors.get(k) for k in ("mic", "stt", "tts", "audio")):
            return "error", "Error — retrying"
        if self.status == "starting":
            return "starting", "Starting…"
        if self.stt_loading:
            return "starting", self.stt_loading
        if self.calibrating_until is not None:
            left = max(0.0, self.calibrating_until - time.monotonic())
            return "calibrating", f"Calibrating: stay quiet ({left:.0f}s)"
        if self.paused:
            return "paused", "Paused"
        parts = []
        if self.tts_busy or self.playing:
            parts.append("Speaking")
        if self.stt_busy:
            parts.append("Thinking")
        if not parts:
            parts.append("Listening")
        label = " + ".join(parts)
        if self.queue_depth > 0:
            label += f"  (+{self.queue_depth} queued)"
        key = "speaking" if "Speaking" in parts else ("thinking" if "Thinking" in parts else "listening")
        return key, label

    def snapshot(self) -> dict:
        with self.lock:
            status_key, status_label = self._status_label()
            return dict(
                status=status_key,
                status_label=status_label,
                last_heard=self.last_heard,
                utterance_count=self.utterance_count,
                threshold=self.threshold,
                silence_ms=self.silence_ms,
                level=self.level,
                notice=self.notice if time.monotonic() < self.notice_until else "",
                hear_generated=self.hear_generated,
                direct_output=self.direct_output,
                speed=self.speed,
                volume=self.volume,
                speaker_id=self.speaker_id,
                stt_model=self.stt_model,
                paused=self.paused,
                quit=self.quit,
                error=self.error,
                voice=self.voice,
                voice_loading=self.voice_loading,
                pending_voice=self.pending_voice,
                voice_error=self.voice_error,
                translate=self.translate,
                source_language=self.source_language,
                stt_loading=self.stt_loading,
                last_language=self.last_language,
                mic_label=self.mic_label,
            )

    def persist(self) -> None:
        snap = self.snapshot()
        try:
            save_settings({
                "voice": snap["voice"],
                "stt_model": snap.get("stt_model", DEFAULT_STT_MODEL),
                "speed": snap["speed"],
                "volume": snap["volume"],
                "speaker_id": snap["speaker_id"],
                "threshold": snap["threshold"],
                "silence_ms": snap["silence_ms"],
                "hear_generated": snap["hear_generated"],
                "translate": snap["translate"],
                "source_language": snap["source_language"],
                "mic": self.mic.saved_id if self.mic is not None else load_settings()["mic"],
            })
        except Exception:
            pass

    def set_notice(self, message: str, seconds: float = 4.0) -> None:
        with self.lock:
            self.notice = message
            self.notice_until = time.monotonic() + seconds

    def adjust_speed(self, direction: int) -> None:
        """direction +1 = faster, -1 = slower. Speed is Piper's length_scale
        (lower = faster); from "voice default" the first press starts at 1.0."""
        with self.lock:
            current = 1.0 if self.speed is None else self.speed
            self.speed = round(min(SPEED_MAX, max(SPEED_MIN, current - direction * SPEED_STEP)), 2)

    def reset_speed(self) -> None:
        with self.lock:
            self.speed = None

    def adjust_volume(self, direction: int) -> None:
        """direction +1 = louder, -1 = quieter, in 10% steps from 10% to 200%."""
        with self.lock:
            self.volume = round(min(VOLUME_MAX, max(VOLUME_MIN,
                                                    self.volume + direction * VOLUME_STEP)), 2)
            if self.volume > VOLUME_DISTORT_WARN:
                self.set_notice("Above 150% can distort. For more volume, raise your "
                                "Discord input volume or system mixer instead.", 6.0)

    def start_calibration(self) -> None:
        with self.lock:
            if self.paused:
                self.set_notice("Unpause (Space) before calibrating")
                return
            self.calib_levels = []
            self.calibrating_until = time.monotonic() + CALIBRATE_SECONDS

    def calibration_sample(self, rms: float) -> bool:
        """Feed one mic level while calibrating. Returns True when this sample
        finished the run and changed the threshold."""
        with self.lock:
            if self.calibrating_until is None:
                return False
            self.calib_levels.append(rms)
            if time.monotonic() < self.calibrating_until:
                return False
            levels, self.calib_levels = self.calib_levels, []
            self.calibrating_until = None
            threshold, why = threshold_from_noise(levels)
            if threshold is None:
                self.set_notice(why, 6.0)
                return False
            self.threshold = threshold
            self.set_notice(f"Threshold set to {threshold:.0f} from your room noise", 6.0)
            return True

    def set_mic(self, choice: "str | None") -> bool:
        """Switch microphone live. choice is a mic id from list_mics, or None
        for the system default. Returns False (with a notice) if it can't."""
        try:
            mic = resolve_mic(choice, self.own_mic_names)
        except (ValueError, RuntimeError) as exc:
            self.set_notice(str(exc), 8.0)
            return False
        with self.lock:
            self.mic, self.mic_label = mic, mic.label
            self.mic_generation += 1
        self.set_notice(mic.note or f"Listening to {mic.label}", 8.0 if mic.note else 4.0)
        self.persist()
        return True

    def remember_spoken(self, text: str) -> None:
        with self.lock:
            self.last_spoken = text

    def remember_typed(self, text: str) -> None:
        with self.lock:
            if text and (not self.typed_history or self.typed_history[-1] != text):
                self.typed_history.append(text)
                del self.typed_history[:-50]

    def repeat_last(self) -> bool:
        """Queue the last spoken/typed line to be spoken again."""
        with self.lock:
            text, tq = self.last_spoken, self.text_queue
        if not text or tq is None:
            return False
        for sentence in split_into_sentences(text):
            tq.put(sentence)
        return True

    def request_voice(self, voice_id: str) -> None:
        voice_id = voice_id.strip()
        if not voice_id:
            return
        with self.lock:
            if voice_id == self.voice and not self.voice_loading:
                return
            self.pending_voice = voice_id
            self.voice_error = ""

    def cycle_voice(self, direction: int) -> None:
        with self.lock:
            current = self.pending_voice or self.voice
            choices = self.curated_voice_ids
            if current in choices:
                index = choices.index(current)
            else:
                index = 0 if direction > 0 else len(choices) - 1
            self.pending_voice = choices[(index + direction) % len(choices)]
            self.voice_error = ""


def _load_requested_voice(state: SharedState, current_voice: str) -> str:
    """Load a requested Piper voice inside the isolated TTS worker."""
    with state.lock:
        requested = state.pending_voice
        if not requested:
            return current_voice
        state.voice_loading = True
        state.voice_error = ""

    try:
        state.tts_client.set_voice(requested)
    except Exception as e:
        with state.lock:
            state.voice_loading = False
            state.voice_error = str(e)
            state.pending_voice = None
        return current_voice

    with state.lock:
        state.pending_voice = None
        state.voice = requested
        state.voice_loading = False
        state.voice_error = ""
    if is_kokoro_voice(requested):
        state.refresh_voice_cycle()  # a first Kokoro voice may just have downloaded the model
    state.persist()
    return requested


_ECHO_GUARD_WINDOW_S = 6.0  # how long after speaking to still treat a close match as an echo
_ECHO_FUZZY_RATIO = 0.88
_ECHO_FUZZY_MIN_CHARS = 12  # shorter phrases must match exactly (no substring rule)
_PLAYBACK_GRACE_S = 0.15  # lets room/output reverb decay a little; doesn't block the pipeline


def _normalize_for_echo_compare(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", text.lower())).strip()


def _matches_spoken(heard: str, spoken: str) -> bool:
    if not heard or not spoken:
        return False
    if heard == spoken:
        return True
    shorter, longer = (heard, spoken) if len(heard) <= len(spoken) else (spoken, heard)
    # Short phrases only count as echoes when identical. The old "any 4+ char
    # substring" rule threw away genuine speech such as "good idea" right after
    # "I think that is a good idea".
    if len(shorter) < _ECHO_FUZZY_MIN_CHARS:
        return False
    # Mic caught most of it but clipped the start/end.
    if shorter in longer and len(shorter) >= 0.8 * len(longer):
        return True
    return difflib.SequenceMatcher(None, heard, spoken).ratio() >= _ECHO_FUZZY_RATIO


def _looks_like_echo(state: SharedState, heard_text: str) -> bool:
    """True if heard_text is plausibly the AI's own voice being picked back up.

    The guard is only active when there is an acoustic path from the generated
    voice back to the mic: local monitoring is on, or audio is played straight
    to real speakers (`vshift run`). With monitoring off in `vshift discord`
    the generated voice goes only to the virtual mic, so nothing can echo and
    the guard stays out of the way (it used to eat legitimate repeats).
    """
    with state.lock:
        active = state.hear_generated or state.direct_output
        entries = [(e["text"], e["ts"]) for e in state.echo_entries]
    if not active or not entries:
        return False
    heard = _normalize_for_echo_compare(heard_text)
    if not heard:
        return False
    now = time.monotonic()
    recent = [t for t, ts in entries if t and now - ts <= _ECHO_GUARD_WINDOW_S]
    if not recent:
        return False
    if any(_matches_spoken(heard, t) for t in recent):
        return True
    # The mic may have heard several consecutive sentences as one utterance.
    return len(recent) > 1 and _matches_spoken(heard, " ".join(recent))


def stt_stage(
    state: "SharedState",
    stt_model: "STTClient",
    utt_queue: "queue.Queue[np.ndarray]",
    text_queue: "queue.Queue[str]",
    stop_event: threading.Event,
) -> None:
    """Pops captured utterances and transcribes them. Runs in its own
    subprocess (STTClient), so this can transcribe utterance N+1 while
    utterance N is still being synthesized or played back."""
    while not stop_event.is_set():
        with state.lock:
            if state.quit:
                return
        try:
            audio = utt_queue.get(timeout=0.3)
        except queue.Empty:
            continue

        with state.lock:
            state.stt_busy = True

        text = transcribe(stt_model, audio)
        with state.lock:
            state.stt_busy = False
        state.set_error("stt", getattr(stt_model, "start_error", "") or "")
        # Drop Whisper's "okay" hallucinations here, on the heard path only, so
        # typing or repeating "ok" on purpose still gets spoken.
        text = normalize_tts_text(text)
        if not text:
            continue
        if _looks_like_echo(state, text):
            # Almost certainly the AI's own last reply being picked back up
            # (speaker bleed, monitoring enabled, room reverb). Discard it
            # silently rather than responding, which is what turns a single
            # echo into an escalating repeat loop.
            continue

        with state.lock:
            state.last_heard = text
            state.last_spoken = text
            state.utterance_count += 1
            state.last_language = getattr(stt_model, "last_language", "") if state.translate else ""
        for sentence in split_into_sentences(text):
            text_queue.put(sentence)


def tts_stage(
    state: "SharedState",
    text_queue: "queue.Queue[str]",
    audio_out_queue: "queue.Queue[tuple[np.ndarray, int, dict]]",
    stop_event: threading.Event,
) -> None:
    """Pops transcribed text and synthesizes it. Runs in its own subprocess
    (TTSClient), independent of both STT and playback."""
    with state.lock:
        current_voice = state.voice
    while not stop_event.is_set():
        with state.lock:
            if state.quit:
                return
        current_voice = _load_requested_voice(state, current_voice)
        try:
            text = text_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        current_voice = _load_requested_voice(state, current_voice)

        with state.lock:
            state.tts_busy = True
            user_speed = state.speed
            user_volume = state.volume
            user_speaker = state.speaker_id
        entry = state.note_spoken(text)

        effective_config = make_synthesis_config(user_speed, user_speaker)
        try:
            out_audio, out_rate = state.tts_client.synthesize(text, effective_config)
        except Exception as e:
            state.set_error("tts", f"TTS worker: {e}")
            continue
        finally:
            with state.lock:
                state.tts_busy = False
        state.clear_error("tts")
        if out_audio.size:
            audio_out_queue.put((apply_volume(out_audio, user_volume), out_rate, entry))


def playback_stage(
    state: "SharedState",
    out_dev: int | None,
    audio_out_queue: "queue.Queue[tuple[np.ndarray, int, dict]]",
    stop_event: threading.Event,
) -> None:
    """Pops synthesized audio and plays it. Kept on its own thread so a long
    reply playing out loud never blocks STT/TTS of what you say next.

    One output stream stays open across clips (see Player), so consecutive
    sentences play back to back instead of re-opening the device each time."""
    player = Player(out_dev)
    # Default: open a stream per clip (play_int16), as in 1.0.1. The persistent
    # Player is opt-in because on at least one real PipeWire setup it produced
    # static while this path was clean. VSHIFT_PERSISTENT_PLAYBACK=1 enables it.
    persistent = os.environ.get("VSHIFT_PERSISTENT_PLAYBACK", "").strip() not in ("", "0")
    try:
        while not stop_event.is_set():
            with state.lock:
                if state.quit:
                    return
            try:
                out_audio, out_rate, entry = audio_out_queue.get(timeout=0.3)
            except queue.Empty:
                player.close_if_idle()
                continue
            with state.lock:
                state.playing = True
            try:
                if persistent:
                    player.play(out_audio, out_rate)
                    if audio_out_queue.empty():
                        player.drain()
                else:
                    play_int16(out_audio, out_rate, device=out_dev)  # last clip: wait until it has actually been heard
            except Exception as e:
                state.set_error("audio", f"Audio output: {e}")
                continue
            finally:
                with state.lock:
                    state.playing = False
                    entry["ts"] = time.monotonic()  # echo window starts when playback ends
            state.clear_error("audio")
            # Let room/monitor reverb decay before the next clip, but only when
            # the voice can actually reach the mic. In plain `vshift discord`
            # (monitor off) there's no echo path, and this sleep would just be
            # a silent gap between sentences.
            with state.lock:
                echo_path = state.hear_generated or state.direct_output
            if echo_path:
                time.sleep(_PLAYBACK_GRACE_S)
    finally:
        player.close()


def worker_loop(
    state: "SharedState",
    stt_model: "STTClient",
    in_dev: int | None,
    out_dev: int | None,
    monitor_router: "MonitorRouter | None",
) -> None:
    """Runs the pipeline as 4 concurrent stages - capture+VAD, STT, TTS,
    playback - connected by queues. The mic stream (capture_loop) never
    closes, so anything said while a previous reply is still being
    transcribed, synthesized, or spoken is captured and queued rather than
    lost. Each stage uses a different resource (mic, STT subprocess, TTS
    subprocess, audio output), so different utterances can be at different
    stages of the pipeline at the same time."""
    utt_queue: "queue.Queue[np.ndarray]" = queue.Queue()
    text_queue: "queue.Queue[str]" = queue.Queue()
    audio_out_queue: "queue.Queue[tuple[np.ndarray, int, dict]]" = queue.Queue()
    stop_event = threading.Event()

    with state.lock:
        state.status = ""  # no longer "starting" - pipeline is up
        state.text_queue = text_queue  # lets the TUI's type-to-speak box feed in directly

    stages = [
        threading.Thread(target=capture_loop, args=(state, in_dev, utt_queue, stop_event), daemon=True),
        threading.Thread(target=stt_stage, args=(state, stt_model, utt_queue, text_queue, stop_event), daemon=True),
        threading.Thread(target=tts_stage, args=(state, text_queue, audio_out_queue, stop_event), daemon=True),
        threading.Thread(target=playback_stage, args=(state, out_dev, audio_out_queue, stop_event), daemon=True),
    ]
    for t in stages:
        t.start()

    try:
        while True:
            with state.lock:
                if state.quit:
                    return
                state.queue_depth = utt_queue.qsize() + text_queue.qsize() + audio_out_queue.qsize()
            time.sleep(0.1)
    finally:
        stop_event.set()
        with state.lock:
            state.text_queue = None
        for t in stages:
            t.join(timeout=2)


# ---------------------------------------------------------------------------
# TUI layout helpers
#
# The whole UI is one panel that is re-centered (horizontally AND vertically)
# from the live terminal size on every frame, so it stays in the middle when
# the window is resized, maximised or split. Its height is constant, so it
# never jumps around as messages appear and disappear.
# ---------------------------------------------------------------------------
_PANEL_MAX_W = 64
_PANEL_MIN_W = 50
_LABEL_W = 15  # width of the left "Voice / Threshold / ..." column


def _init_attrs() -> dict:
    """Curses attributes; degrades to bold/dim on terminals without colour."""
    attrs = {"plain": 0, "bold": curses.A_BOLD, "dim": curses.A_DIM,
             "good": curses.A_BOLD, "warn": curses.A_BOLD, "bad": curses.A_BOLD,
             "accent": curses.A_BOLD, "voice": curses.A_BOLD, "frame": curses.A_DIM,
             "key": curses.A_BOLD}
    try:
        if curses.has_colors():
            curses.start_color()
            try:
                curses.use_default_colors()
                bg = -1
            except curses.error:
                bg = curses.COLOR_BLACK
            for idx, (name, color) in enumerate(
                (("good", curses.COLOR_GREEN), ("warn", curses.COLOR_YELLOW),
                 ("bad", curses.COLOR_RED), ("accent", curses.COLOR_CYAN),
                 ("voice", curses.COLOR_MAGENTA)), start=1):
                curses.init_pair(idx, color, bg)
                attrs[name] = curses.color_pair(idx) | curses.A_BOLD
            attrs["frame"] = curses.color_pair(4) | curses.A_DIM   # cyan, muted
            attrs["key"] = curses.color_pair(4) | curses.A_BOLD
    except curses.error:
        pass
    return attrs


def _clip(text: str, width: int) -> str:
    text = " ".join(str(text).split())
    if width <= 0:
        return ""
    return text if len(text) <= width else text[: max(0, width - 1)] + "…"


def _wrap_lines(text: str, width: int, max_lines: int) -> list[str]:
    """Wrap text into exactly max_lines lines (padded / ellipsised)."""
    lines = textwrap.wrap(" ".join(str(text).split()), width=max(8, width)) if text else []
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = _clip(lines[-1] + "…", width)
    return lines + [""] * (max_lines - len(lines))


def _put(stdscr, y: int, x: int, text: str, attr: int = 0) -> None:
    """addstr that never raises and never writes outside the window."""
    rows, cols = stdscr.getmaxyx()
    if y < 0 or y >= rows or x >= cols or not text:
        return
    if x < 0:
        text, x = text[-x:], 0
    try:
        stdscr.addstr(y, x, text[: cols - x], attr)
    except curses.error:
        pass  # bottom-right cell quirk / tiny terminal


_METER_FULL_SCALE = 2000.0  # same ceiling as the threshold setting


def _meter_cells(level: float, threshold: float, width: int) -> list[tuple[str, str]]:
    """Mic-level bar as (char, kind) cells; kind is 'on', 'off' or 'mark'.

    Square-root scale so quiet speech is visible next to loud peaks. The
    threshold is drawn as a marker, so you can see whether your voice crosses it.
    """
    width = max(1, width)

    def pos(v: float) -> int:
        frac = max(0.0, min(1.0, v / _METER_FULL_SCALE)) ** 0.5
        return min(width - 1, int(frac * width))

    fill, mark = pos(level) + (1 if level > 0 else 0), pos(threshold)
    cells = []
    for i in range(width):
        if i == mark:
            cells.append(("┃", "mark"))
        elif i < fill:
            cells.append(("█", "on"))
        else:
            cells.append(("·", "off"))
    return cells


_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_STATUS_ICONS = {"listening": "●", "paused": "‖", "error": "✗", "starting": "…", "calibrating": "◌"}


def _status_icon(status: str) -> str:
    """Static dot when idle; a spinner while Whisper/the TTS engine is working,
    so a busy pipeline visibly moves instead of looking frozen."""
    if status in ("speaking", "thinking"):
        return _SPINNER[int(time.monotonic() * 10) % len(_SPINNER)]
    return _STATUS_ICONS.get(status, "●")


def _gauge(frac: float, width: int) -> tuple[str, str]:
    """A slim slider, returned as (filled, rest) so they can be coloured apart."""
    width = max(1, width)
    pos = int(round(max(0.0, min(1.0, frac)) * (width - 1)))
    return "━" * pos + "●", "─" * (width - 1 - pos)


def _voice_parts(voice_id: str) -> tuple[str, str]:
    """('lessac-medium', 'piper · en_US') style display for a voice id."""
    if is_kokoro_voice(voice_id):
        name = voice_id[len(KOKORO_PREFIX):]
        accent = {"a": "US", "b": "UK"}.get(name[:1], "")
        return name, f"kokoro {accent}".strip()
    lang, _, rest = voice_id.partition("-")
    if rest and "_" in lang:
        region = lang.split("_")[1]
        return rest, f"piper {'UK' if region == 'GB' else region}"
    return voice_id, "piper"


def _panel_rows(snap: dict, title: str, mic_label: str, attrs: dict, inner: int) -> list[tuple]:
    """Build the panel as a list of rows. Each row is ('rule',) or
    ('line', [(text, attr), ...], (right_text, right_attr) | None)."""
    rows: list[tuple] = []
    voice_attr = attrs.get("voice", attrs["accent"])
    key_attr = attrs.get("key", attrs["bold"])
    status_attr = {
        "listening": attrs["good"], "speaking": attrs["accent"], "thinking": attrs["warn"],
        "error": attrs["bad"], "paused": attrs["dim"], "starting": attrs["dim"],
        "calibrating": attrs["warn"],
    }.get(snap["status"], attrs["plain"])

    def line(left, right=None):
        rows.append(("line", left, right))

    def label_seg(label):
        return (f"{label:<{_LABEL_W}}", attrs["dim"])

    def value_width(hint):
        return inner - _LABEL_W - (len(hint) + 2 if hint else 0)

    def setting(label, value, hint="", value_attr=0):
        line([label_seg(label), (_clip(value, value_width(hint)), value_attr)],
             (hint, attrs["dim"]) if hint else None)

    # One slider width for every gauge row so they line up; leaves room for
    # the widest value ("voice default") even at the narrowest panel.
    gauge_w = max(4, min(12, inner - _LABEL_W - 8 - len("voice default") - 2))

    def gauge_setting(label, value, frac, hint, value_attr=0):
        """Slider followed by its value, e.g. '━━━━━●──────  120%'."""
        filled, rest = _gauge(frac, gauge_w)
        line([label_seg(label), (filled, attrs["accent"]), (rest, attrs["dim"]),
              ("  " + _clip(value, value_width(hint) - gauge_w - 2), value_attr)],
             (hint, attrs["dim"]))

    count = f"Utterances {snap['utterance_count']}"
    line([(f"{_status_icon(snap['status'])} ", status_attr),
          (_clip(snap["status_label"], inner - len(count) - 4), status_attr)],
         (count, attrs["dim"]))
    rows.append(("rule",))

    heard = snap["last_heard"]
    heard_lines = _wrap_lines(f"“{heard}”" if heard else "", inner - 2, 2)
    if not heard:
        heard_lines[0] = "(nothing yet — just talk)"
    for i, text in enumerate(heard_lines):
        line([("› " if i == 0 else "  ", attrs["accent"] if heard else attrs["dim"]),
              (text, attrs["dim"] if not heard else attrs["plain"])])
    rows.append(("rule",))

    # Voice: friendly name in colour, engine/accent tag dimmed after it.
    name, tag = _voice_parts(snap["voice"])
    if snap["voice_loading"]:
        target = snap["pending_voice"]
        name = f"{name} → {_voice_parts(target)[0] if target else 'loading'}"
        tag = "loading…"
    hint = "[ ]  V"
    room = value_width(hint)
    shown = _clip(name, room)
    voice_segs = [label_seg("Voice"), (shown, voice_attr)]
    if len(shown) + 2 + len(tag) <= room:
        voice_segs.append((f"  {tag}", attrs["warn"] if snap["voice_loading"] else attrs["dim"]))
    line(voice_segs, (hint, attrs["dim"]))

    setting("Microphone", snap["mic_label"] or "system default", "M")
    if snap["direct_output"]:
        setting("Output", mic_label)
    else:
        setting("Virtual mic", mic_label)
    setting("Threshold", f"{snap['threshold']:.0f}", "↑ ↓  C")

    # Mic level: dim below the threshold marker, green above it, yellow when
    # it's close to clipping; the live number sits at the right edge.
    level_txt = f"{snap['level']:>5.0f}"
    meter_w = inner - _LABEL_W - len(level_txt) - 1
    cells = _meter_cells(snap["level"], snap["threshold"], meter_w)
    mark = next((i for i, (_, k) in enumerate(cells) if k == "mark"), meter_w)
    hot = int(meter_w * 0.88)
    runs: list[tuple[str, int]] = []
    for i, (ch, kind) in enumerate(cells):
        if kind == "mark":
            attr = attrs["warn"]
        elif kind == "off":
            attr = attrs["dim"]
        elif i >= hot:
            attr = attrs["bad"]
        elif i > mark:
            attr = attrs["good"]
        else:
            attr = attrs["plain"]
        if runs and runs[-1][1] == attr:
            runs[-1] = (runs[-1][0] + ch, attr)
        else:
            runs.append((ch, attr))
    over = snap["level"] > snap["threshold"]
    line([label_seg("Mic level")] + runs,
         (level_txt, attrs["good"] if over else attrs["dim"]))

    setting("Silence gap", f"{snap['silence_ms']} ms", "← →")
    # Stored as Piper's length_scale (lower = faster); shown as a rate so + means faster.
    if snap["speed"] is None:
        gauge_setting("Speed", "voice default", (1.0 - 0.5) / 1.5, "- +  0", attrs["dim"])
    else:
        rate = 1.0 / float(snap["speed"])
        gauge_setting("Speed", f"{rate:.2f}x", (rate - 0.5) / 1.5, "- +  0")
    vol = snap["volume"]
    gauge_setting("Volume", f"{round(vol * 100)}%", (vol - VOLUME_MIN) / (VOLUME_MAX - VOLUME_MIN),
                  "< >", attrs["warn"] if vol > VOLUME_DISTORT_WARN else 0)
    if snap["direct_output"]:
        setting("Hear generated", "n/a — on your speakers", value_attr=attrs["dim"])
    else:
        on = snap["hear_generated"]
        setting("Hear generated", "● ON" if on else "○ OFF", "H", attrs["good"] if on else attrs["dim"])
    if snap["stt_loading"]:
        setting("Translate", "loading…", "L", attrs["warn"])
    elif snap["translate"]:
        src = snap["source_language"] or "auto"
        segs = [label_seg("Translate"), (f"● {src} → English", attrs["good"])]
        if snap["last_language"] and not snap["source_language"]:
            segs.append((f"  heard {snap['last_language']}", attrs["dim"]))
        line(segs, ("L", attrs["dim"]))
    else:
        setting("Translate", "○ OFF", "L", attrs["dim"])
    rows.append(("rule",))

    def keyline(pairs):
        segs: list[tuple[str, int]] = []
        for n, (k, what) in enumerate(pairs):
            if n:
                segs.append(("   ", 0))
            segs += [(k, key_attr), (f" {what}", attrs["dim"])]
        line(segs)

    if snap["direct_output"]:
        keyline([("Space", "pause"), ("C", "calibrate"), ("Q", "quit")])
    else:
        keyline([("Space", "pause"), ("H", "hear"), ("C", "calibrate"), ("Q", "quit")])
    keyline([("T", "type"), ("R", "repeat"), ("V", "voices"), ("L", "translate")])
    rows.append(("rule",))

    if snap["voice_error"]:
        msg, msg_attr = "✗ Voice: " + snap["voice_error"], attrs["bad"]
    elif snap["error"]:
        msg, msg_attr = "✗ " + snap["error"], attrs["bad"]
    elif snap["notice"]:
        msg, msg_attr = "› " + snap["notice"], attrs["accent"]
    elif snap["status"] == "calibrating":
        msg, msg_attr = "Stay quiet while vshift measures your room noise.", attrs["warn"]
    elif snap["direct_output"]:
        msg, msg_attr = "Tip: just talk — the converted voice plays on your speakers.", attrs["dim"]
    else:
        msg, msg_attr = f"Tip: pick “{mic_label}” as your microphone in Discord.", attrs["dim"]
    for text in _wrap_lines(msg, inner, 2):
        line([(text, msg_attr)])
    return rows


def _draw_panel(stdscr, snap: dict, title: str, mic_label: str, attrs: dict) -> None:
    """Draw the UI centered in the terminal (caller erases/refreshes)."""
    rows, cols = stdscr.getmaxyx()
    width = min(_PANEL_MAX_W, cols - 2)
    if width >= _PANEL_MIN_W:
        inner = width - 4
        body = _panel_rows(snap, title, mic_label, attrs, inner)
        height = len(body) + 2
        if rows >= height:
            top = (rows - height) // 2
            left = (cols - width) // 2
            frame = attrs.get("frame", attrs["dim"])
            version = f" v{VERSION} "
            head = f" {_clip(title, width - 10 - len(version))} "
            _put(stdscr, top, left, "┌─" + "─" * (width - 3) + "┐", frame)
            _put(stdscr, top, left + 2, head, attrs["accent"])
            _put(stdscr, top, left + width - 2 - len(version), version, attrs["dim"])
            for i, row in enumerate(body, start=1):
                y = top + i
                if row[0] == "rule":
                    _put(stdscr, y, left, "├" + "─" * (width - 2) + "┤", frame)
                    continue
                _put(stdscr, y, left, "│", frame)
                _put(stdscr, y, left + width - 1, "│", frame)
                x = left + 2
                for text, attr in row[1]:
                    room = left + width - 2 - x
                    if room <= 0:
                        break
                    _put(stdscr, y, x, text[:room], attr)
                    x += len(text)
                if row[2]:
                    text, attr = row[2]
                    _put(stdscr, y, left + width - 2 - len(text), text, attr)
            _put(stdscr, top + height - 1, left, "└" + "─" * (width - 2) + "┘", frame)
            return

    # Terminal too small for the full panel: compact view, still centered.
    text_lines = [
        f"vshift · {snap['status_label']}",
        f"Voice: {snap['voice']}",
        "Hear generated: n/a" if snap["direct_output"] else f"Hear generated: {'ON' if snap['hear_generated'] else 'OFF'}",
        "Space pause · T speak · [ ] voice · Q quit" if snap["direct_output"]
        else "Space pause · H hear generated · T speak · [ ] voice · Q quit",
    ]
    if snap["voice_error"] or snap["error"]:
        text_lines.append("! " + (snap["voice_error"] or snap["error"]))
    text_lines.append("(enlarge the terminal for the full view)")
    text_lines = [_clip(t, max(1, cols - 2)) for t in text_lines]
    top = max(0, (rows - len(text_lines)) // 2)
    block_w = max(len(t) for t in text_lines)
    left = max(0, (cols - block_w) // 2)
    for i, text in enumerate(text_lines):
        attr = attrs["bad"] if text.startswith("!") else (attrs["dim"] if i == len(text_lines) - 1 else 0)
        _put(stdscr, top + i, left, text, attr)


def _prompt_line(stdscr, state: "SharedState", redraw, title: str, label: str,
                 hint: str, maxlen: int, history: list[str] | None = None) -> str | None:
    """Centered single-line input box drawn over the live (still updating)
    panel. Returns the text on Enter, or None if cancelled with Esc/Ctrl-C.
    Scrolls horizontally, so long text is never cut off by the box width."""
    with state.lock:
        was_paused = state.paused
        state.paused = True
    buf = ""
    result: str | None = None
    history = list(history or [])
    hist_pos = len(history)  # == len(history) means "the line being typed"
    draft = ""
    try:
        try:
            curses.curs_set(1)
        except curses.error:
            pass
        while True:
            stdscr.erase()
            redraw()
            rows, cols = stdscr.getmaxyx()
            width = max(24, min(_PANEL_MAX_W, cols - 2))
            height = 6
            top = max(0, (rows - height) // 2)
            left = max(0, (cols - width) // 2)
            for y in range(height):  # blank the area so it sits cleanly on top
                _put(stdscr, top + y, left, " " * width)
            head = f" {_clip(title, width - 8)} "
            _put(stdscr, top, left, "┌─" + head + "─" * (width - 3 - len(head)) + "┐")
            for y in range(1, height - 1):
                _put(stdscr, top + y, left, "│")
                _put(stdscr, top + y, left + width - 1, "│")
            _put(stdscr, top + height - 1, left, "└" + "─" * (width - 2) + "┘")
            _put(stdscr, top + 1, left + 2, _clip(label, width - 4), curses.A_DIM)
            field_w = width - 6
            visible = buf[-field_w:]
            _put(stdscr, top + 2, left + 2, "› " + visible)
            _put(stdscr, top + 4, left + 2, _clip(hint, width - 4), curses.A_DIM)
            try:
                stdscr.move(top + 2, min(cols - 1, left + 4 + len(visible)))
            except curses.error:
                pass
            stdscr.refresh()

            try:
                ch = stdscr.get_wch()
            except curses.error:
                continue  # idle tick: keep the panel live
            except KeyboardInterrupt:
                break
            if isinstance(ch, int):
                if ch == curses.KEY_RESIZE:
                    curses.update_lines_cols()
                    stdscr.clear()
                elif ch == curses.KEY_BACKSPACE:
                    buf = buf[:-1]
                elif ch == curses.KEY_UP and hist_pos > 0:
                    if hist_pos == len(history):
                        draft = buf  # keep what was being typed so Down can bring it back
                    hist_pos -= 1
                    buf = history[hist_pos][:maxlen]
                elif ch == curses.KEY_DOWN and hist_pos < len(history):
                    hist_pos += 1
                    buf = draft if hist_pos == len(history) else history[hist_pos][:maxlen]
                elif ch == curses.KEY_ENTER:
                    result = buf.strip()
                    break
                continue
            if ch in ("\n", "\r"):
                result = buf.strip()
                break
            if ch == "\x1b":
                break
            if ch in ("\x7f", "\b"):
                buf = buf[:-1]
            elif ch == "\x15":  # Ctrl-U
                buf = ""
            elif ch.isprintable() and len(buf) < maxlen:
                buf += ch
    finally:
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        with state.lock:
            state.paused = was_paused
    return result or None


_BROWSER_KEYS = "↑↓ move · Enter use · Tab preview · Esc close"


def _browser_rows(entries: list[dict], query: str, current: str) -> list[dict]:
    """Filtered entries, plus a 'use this id' row when the query is a valid
    voice id that isn't in the catalog (so any Piper voice is still reachable)."""
    shown = filter_voices(entries, query)
    typed = query.strip()
    if typed and VOICE_ID_RE.fullmatch(typed) and not any(e["id"] == typed for e in entries):
        shown.append(dict(id=typed, name=typed, tag="custom id", desc="use this voice id",
                          installed=False, curated=False, engine="custom", search=typed.lower()))
    return shown


def _browse_voices(stdscr, state: "SharedState", redraw, attrs: dict | None = None) -> str | None:
    """Searchable voice list drawn over the live panel. Type to filter, arrows
    to move, Enter to switch, Tab (or →) to hear a sample on your own speakers.
    Returns the chosen voice id, or None. The mic is paused while it's open."""
    attrs = attrs or {"plain": 0, "bold": curses.A_BOLD, "dim": curses.A_DIM,
                      "good": curses.A_BOLD, "warn": curses.A_BOLD, "bad": curses.A_BOLD,
                      "accent": curses.A_BOLD}
    voice_attr = attrs.get("voice", attrs["accent"])
    frame = attrs.get("frame", attrs["dim"])
    with state.lock:
        was_paused = state.paused
        state.paused = True
        current = state.voice
        previewer = getattr(state, "previewer", None)
    entries = voice_catalog(state.voices_dir)
    query, sel, scroll = "", 0, 0
    # Start on the voice in use.
    sel = next((i for i, e in enumerate(entries) if e["id"] == current), 0)
    result: str | None = None
    message, message_attr = "", attrs["dim"]
    try:
        while True:
            shown = _browser_rows(entries, query, current)
            sel = max(0, min(sel, len(shown) - 1))
            stdscr.erase()
            redraw()
            rows, cols = stdscr.getmaxyx()
            width = max(30, min(_PANEL_MAX_W + 6, cols - 2))
            height = max(10, min(24, rows - 2))
            top, left = max(0, (rows - height) // 2), max(0, (cols - width) // 2)
            list_h = height - 7
            scroll = min(max(scroll, sel - list_h + 1), sel)
            inner = width - 4

            for y in range(height):
                _put(stdscr, top + y, left, " " * width)
            count = f" {len(shown)} of {len(entries)} "
            _put(stdscr, top, left, "┌" + "─" * (width - 2) + "┐", frame)
            _put(stdscr, top, left + 2, " Voices ", attrs["accent"])
            _put(stdscr, top, left + width - 2 - len(count), count, attrs["dim"])
            for y in range(1, height - 1):
                _put(stdscr, top + y, left, "│", frame)
                _put(stdscr, top + y, left + width - 1, "│", frame)
            for y in (2, height - 4):
                _put(stdscr, top + y, left, "├" + "─" * (width - 2) + "┤", frame)
            _put(stdscr, top + height - 1, left, "└" + "─" * (width - 2) + "┘", frame)

            field = query[-(inner - 4):]
            _put(stdscr, top + 1, left + 2, "⌕ ", attrs["accent"])
            if field:
                _put(stdscr, top + 1, left + 4, field)
            else:
                _put(stdscr, top + 1, left + 4, "type to search: amy, uk female, kokoro…", attrs["dim"])

            name_w, tag_w = 18, 10
            for i in range(list_h):
                idx = scroll + i
                y = top + 3 + i
                if idx >= len(shown):
                    if not shown and i == 0:
                        _put(stdscr, y, left + 2, "no voices match", attrs["dim"])
                    continue
                e = shown[idx]
                mark, mark_attr = (("●", attrs["good"]) if e["id"] == current else
                                   ("✓", attrs["good"]) if e["installed"] else
                                   ("+", attrs["accent"]) if e["engine"] == "custom" else
                                   ("↓", attrs["dim"]))
                text = (f"{_clip(e['name'], name_w):<{name_w}} {_clip(e['tag'], tag_w):<{tag_w}} "
                        f"{_clip(e['desc'], inner - name_w - tag_w - 4)}")
                if idx == sel:
                    _put(stdscr, y, left + 1, " " * (width - 2), curses.A_REVERSE)
                    _put(stdscr, y, left + 2, mark + " ", curses.A_REVERSE | curses.A_BOLD)
                    _put(stdscr, y, left + 4, text[:inner - 2], curses.A_REVERSE | curses.A_BOLD)
                else:
                    _put(stdscr, y, left + 2, mark + " ", mark_attr)
                    _put(stdscr, y, left + 4, _clip(e["name"], name_w),
                         voice_attr if e["installed"] or e["id"] == current else 0)
                    _put(stdscr, y, left + 4 + name_w + 1,
                         text[name_w + 1:inner - 2], attrs["dim"])

            # Detail line: preview progress/errors win, then what Enter would do.
            if previewer is not None and (previewer.status or previewer.error):
                detail, detail_attr = ((previewer.status, attrs["accent"]) if previewer.status
                                       else (previewer.error, attrs["bad"]))
            elif message:
                detail, detail_attr = message, message_attr
            elif not shown:
                detail, detail_attr = "Backspace to widen the search", attrs["dim"]
            else:
                e = shown[sel]
                if e["id"] == current:
                    detail, detail_attr = "In use now", attrs["good"]
                elif e["engine"] == "custom":
                    detail, detail_attr = "Enter tries this id (downloads it if it exists)", attrs["dim"]
                elif e["installed"]:
                    detail, detail_attr = "Downloaded · Enter switches · Tab plays a sample", attrs["dim"]
                else:
                    detail, detail_attr = voice_download_hint(e), attrs["warn"]
            _put(stdscr, top + height - 3, left + 2, _clip(detail, inner), detail_attr)
            _put(stdscr, top + height - 2, left + 2, _clip(_BROWSER_KEYS, inner), attrs["dim"])
            stdscr.refresh()

            try:
                ch = stdscr.get_wch()
            except curses.error:
                continue  # idle tick: keeps the panel and preview status live
            except KeyboardInterrupt:
                break
            message = ""
            if isinstance(ch, int):
                if ch == curses.KEY_RESIZE:
                    curses.update_lines_cols()
                    stdscr.clear()
                elif ch == curses.KEY_UP:
                    sel -= 1
                elif ch == curses.KEY_DOWN:
                    sel += 1
                elif ch == curses.KEY_PPAGE:
                    sel -= list_h
                elif ch == curses.KEY_NPAGE:
                    sel += list_h
                elif ch == curses.KEY_HOME:
                    sel = 0
                elif ch == curses.KEY_END:
                    sel = len(shown) - 1
                elif ch == curses.KEY_BACKSPACE:
                    query, sel, scroll = query[:-1], 0, 0
                elif ch == curses.KEY_RIGHT:
                    ch = "\t"
                elif ch == curses.KEY_ENTER:
                    ch = "\n"
                if isinstance(ch, int):
                    continue
            if ch in ("\n", "\r"):
                if shown:
                    result = shown[sel]["id"]
                    break
                continue
            if ch == "\x1b":
                break
            if ch == "\t":
                if not shown:
                    continue
                e = shown[sel]
                if previewer is None:
                    message, message_attr = "Previews aren't available here", attrs["warn"]
                elif not e["installed"]:
                    message, message_attr = "Download it first (Enter) to hear a sample", attrs["warn"]
                else:
                    with state.lock:
                        speed, volume = state.speed, state.volume
                    if not previewer.preview(e["id"], speed, volume):
                        message, message_attr = "Still playing the last sample…", attrs["dim"]
            elif ch in ("\x7f", "\b"):
                query, sel, scroll = query[:-1], 0, 0
            elif ch == "\x15":  # Ctrl-U
                query, sel, scroll = "", 0, 0
            elif ch.isprintable() and len(query) < 60:
                query, sel, scroll = query + ch, 0, 0
    finally:
        if previewer is not None:
            previewer.stop()   # don't let the mic hear a sample once it un-pauses
        with state.lock:
            state.paused = was_paused
    return result


def _pick_mic(stdscr, state: "SharedState", redraw, attrs: dict | None = None) -> str | None:
    """List microphones over the live panel. Returns the chosen mic id, ""
    for the system default, or None if cancelled. The mic is paused while open."""
    attrs = attrs or {"plain": 0, "bold": curses.A_BOLD, "dim": curses.A_DIM,
                      "good": curses.A_BOLD, "warn": curses.A_BOLD, "bad": curses.A_BOLD,
                      "accent": curses.A_BOLD}
    frame = attrs.get("frame", attrs["dim"])
    with state.lock:
        was_paused = state.paused
        state.paused = True
        current = state.mic.saved_id if state.mic is not None else None
        own = set(state.own_mic_names)
    mics = list_mics(own)
    default_label = next((m["label"] for m in mics if m["default"]), "")
    items = [dict(id="", label="System default", desc=default_label or "whatever your desktop uses")]
    for m in mics:
        tag = {"hardware": "", "virtual": "virtual", "device": "device"}.get(m["kind"], "")
        items.append(dict(id=m["id"], label=m["label"], desc=tag))
    sel = next((i for i, it in enumerate(items) if it["id"] == (current or "")), 0)
    result: str | None = None
    try:
        while True:
            stdscr.erase()
            redraw()
            rows, cols = stdscr.getmaxyx()
            width = max(30, min(_PANEL_MAX_W, cols - 2))
            list_h = max(1, min(len(items), rows - 8))
            height = list_h + 6
            top, left = max(0, (rows - height) // 2), max(0, (cols - width) // 2)
            inner = width - 4
            scroll = max(0, min(sel - list_h + 1, len(items) - list_h)) if sel >= list_h else 0
            for y in range(height):
                _put(stdscr, top + y, left, " " * width)
            _put(stdscr, top, left, "┌" + "─" * (width - 2) + "┐", frame)
            _put(stdscr, top, left + 2, " Microphone ", attrs["accent"])
            for y in range(1, height - 1):
                _put(stdscr, top + y, left, "│", frame)
                _put(stdscr, top + y, left + width - 1, "│", frame)
            _put(stdscr, top + 1, left + 2, _clip("Which mic should vshift listen to?", inner), attrs["dim"])
            _put(stdscr, top + 2, left, "├" + "─" * (width - 2) + "┤", frame)
            for i in range(list_h):
                idx = scroll + i
                if idx >= len(items):
                    break
                it = items[idx]
                y = top + 3 + i
                mark = "●" if it["id"] == (current or "") else " "
                desc = it["desc"]
                text = f"{mark} {it['label']}"
                room = inner - len(desc) - 2
                text = text[:room] + ("…" if len(text) > room else "")
                if idx == sel:
                    _put(stdscr, y, left + 1, " " * (width - 2), curses.A_REVERSE)
                    _put(stdscr, y, left + 2, text, curses.A_REVERSE | curses.A_BOLD)
                    _put(stdscr, y, left + width - 2 - len(desc), desc, curses.A_REVERSE)
                else:
                    _put(stdscr, y, left + 2, text, attrs["good"] if mark == "●" else 0)
                    _put(stdscr, y, left + width - 2 - len(desc), desc, attrs["dim"])
            _put(stdscr, top + height - 3, left, "├" + "─" * (width - 2) + "┤", frame)
            _put(stdscr, top + height - 2, left + 2, _clip("↑↓ move · Enter listen · Esc cancel", inner),
                 attrs["dim"])
            _put(stdscr, top + height - 1, left, "└" + "─" * (width - 2) + "┘", frame)
            stdscr.refresh()

            try:
                ch = stdscr.get_wch()
            except curses.error:
                continue
            except KeyboardInterrupt:
                break
            if ch == curses.KEY_RESIZE:
                curses.update_lines_cols()
                stdscr.clear()
            elif ch == curses.KEY_UP:
                sel = max(0, sel - 1)
            elif ch == curses.KEY_DOWN:
                sel = min(len(items) - 1, sel + 1)
            elif ch in ("\n", "\r", curses.KEY_ENTER):
                result = items[sel]["id"]
                break
            elif ch in ("\x1b", "q", "Q"):
                break
    finally:
        with state.lock:
            state.paused = was_paused
    return result


def _prompt_type_to_speak(stdscr, state: "SharedState", redraw) -> None:
    """Type free text and have it spoken through the current TTS voice, out
    the same route (virtual mic / speakers) as converted speech - lets you
    say something without touching your mic at all."""
    with state.lock:
        history = list(state.typed_history)
    text = _prompt_line(stdscr, state, redraw, "Type to speak", "Say this with the current voice",
                        "Enter speak · ↑ history · Esc cancel", 500, history=history)
    if not text:
        return
    state.remember_typed(text)
    with state.lock:
        state.last_heard = f"(typed) {text}"
        state.last_spoken = text
        state.utterance_count += 1
        tq = state.text_queue
    if tq is not None:
        for sentence in split_into_sentences(text):
            tq.put(sentence)


def _apply_monitor(state: SharedState, monitor_router: "MonitorRouter | None", want: bool) -> None:
    """Turn local monitoring on/off and make the UI reflect what actually happened.

    Runs pactl (slow, can fail), so it must be called WITHOUT holding state.lock.
    Previously the toggle flipped to "ON" even when the loopback failed to load,
    and the failure was swallowed - that is why "Hear generated" looked broken.
    """
    if monitor_router is None:
        with state.lock:
            state.hear_generated = False
        state.set_error("monitor", "Monitoring isn't available in this mode")
        return
    monitor_router.set_enabled(want)
    actual = monitor_router.enabled
    with state.lock:
        state.hear_generated = actual
    if want and not actual:
        state.set_error("monitor", monitor_router.last_error or "Couldn't enable monitoring")
    else:
        state.clear_error("monitor")


def _set_translate(state: SharedState, want: bool) -> threading.Thread | None:
    """Turn translation on/off. Whisper may need another model for it (and a
    one-time download), so the reload runs on a background thread; the panel
    shows "Loading…" meanwhile and speech said in between waits in the queue."""
    with state.lock:
        client = state.stt_client
        if client is None:
            state.set_notice("Translation isn't available here")
            return None
        if state.stt_loading:
            state.set_notice("Still loading the speech model…")
            return None
        lang = state.source_language
        model = translation_model_for(client.model_name) if want else client.model_name
        if model != client.active_model:
            first = "" if _stt_model_cached(model) else " (first time: downloading, can take minutes)"
            state.stt_loading = f"Loading Whisper '{model}'{first}…"

    def work() -> None:
        error = client.configure(want, lang)
        if error and want:
            client.configure(False, lang)  # back to the model that worked
        with state.lock:
            state.stt_loading = ""
            state.translate = want and not error
            state.last_language = ""
        if error:
            state.set_notice(f"Translation off: {error}", 10.0)
        else:
            state.set_notice("Translating your speech into English" if want
                             else "Translation off: speak English")
            state.persist()

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    return worker


def _handle_key(ch: int, state: SharedState, monitor_router: "MonitorRouter | None" = None) -> str | None:
    changed = False
    monitor_want: bool | None = None
    with state.lock:
        if ch in (ord("q"), ord("Q")):
            state.quit = True
            return None
        if ch == ord(" "):
            state.paused = not state.paused
            changed = True
        elif ch in (ord("h"), ord("H")):
            if not state.direct_output:
                monitor_want = not state.hear_generated
        elif ch in (ord("v"), ord("V")):
            return "voice_prompt"
        elif ch in (ord("t"), ord("T")):
            return "type_prompt"
        elif ch in (ord("m"), ord("M")):
            return "mic_prompt"
        elif ch == curses.KEY_UP:
            state.threshold = min(2000.0, state.threshold + 25.0)
            changed = True
        elif ch == curses.KEY_DOWN:
            state.threshold = max(25.0, state.threshold - 25.0)
            changed = True
        elif ch == curses.KEY_RIGHT:
            state.silence_ms = min(3000, state.silence_ms + 100)
            changed = True
        elif ch == curses.KEY_LEFT:
            state.silence_ms = max(200, state.silence_ms - 100)
            changed = True

    if monitor_want is not None:
        _apply_monitor(state, monitor_router, monitor_want)
        changed = True

    if ch == ord("["):
        state.cycle_voice(-1)
        changed = True
    elif ch == ord("]"):
        state.cycle_voice(1)
        changed = True
    elif ch in (ord("-"), ord("_")):
        state.adjust_speed(-1)   # slower
        changed = True
    elif ch in (ord("+"), ord("=")):
        state.adjust_speed(+1)   # faster ("=" so you don't need Shift)
        changed = True
    elif ch == ord("0"):
        state.reset_speed()
        changed = True
    elif ch in (ord(","), ord("<")):
        state.adjust_volume(-1)  # quieter
        changed = True
    elif ch in (ord("."), ord(">")):
        state.adjust_volume(+1)  # louder ("." so you don't need Shift)
        changed = True
    elif ch in (ord("r"), ord("R")):
        if not state.repeat_last():
            state.set_notice("Nothing to repeat yet")
    elif ch in (ord("c"), ord("C")):
        state.start_calibration()
    elif ch in (ord("l"), ord("L")):
        with state.lock:
            want = not state.translate
        _set_translate(state, want)

    if changed:
        state.persist()
    return None


def _draw_tui(stdscr, state: SharedState, title: str, mic_label: str, monitor_router: "MonitorRouter | None" = None) -> None:
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    stdscr.keypad(True)
    stdscr.timeout(100)
    attrs = _init_attrs()

    def redraw() -> None:
        _draw_panel(stdscr, state.snapshot(), title, mic_label, attrs)

    while True:
        stdscr.erase()
        redraw()  # geometry is recomputed from the live size every frame => always centered
        stdscr.refresh()

        ch = stdscr.getch()
        if ch == -1:
            continue
        if ch == curses.KEY_RESIZE:
            curses.update_lines_cols()
            stdscr.clear()  # wipe leftovers from the old size
            continue
        action = _handle_key(ch, state, monitor_router)
        if action == "mic_prompt":
            picked = _pick_mic(stdscr, state, redraw, attrs)
            if picked is not None:
                state.set_mic(picked or None)  # "" = system default
        if action == "voice_prompt":
            voice_id = _browse_voices(stdscr, state, redraw, attrs)
            if voice_id:
                state.request_voice(voice_id)
        elif action == "type_prompt":
            _prompt_type_to_speak(stdscr, state, redraw)
        with state.lock:
            if state.quit:
                return


def start_conversion_tui(
    title: str,
    voice: str,
    mic_label: str,
    stt_model: STTClient,
    tts_client: "TTSClient",
    in_dev: int | None,
    out_dev: int | None,
    monitor_router: "MonitorRouter | None",
    threshold: float,
    silence_ms: int,
    voices_dir: Path,
    speed: float | None,
    speaker_id: int | None,
    hear_generated: bool = False,
    direct_output: bool = False,
    volume: float = 1.0,
    mic: "MicChoice | None" = None,
    own_mic_names: "set[str] | None" = None,
) -> None:
    """Run the fixed-layout TUI with live generated-audio monitoring and voice switching."""
    translate = bool(getattr(stt_model, "translate", False))
    state = SharedState(
        threshold=threshold,
        silence_ms=silence_ms,
        voice=voice,
        tts_client=tts_client,
        tts_voice_name=voice,
        voices_dir=voices_dir,
        speed=speed,
        speaker_id=speaker_id,
        stt_model=stt_model.model_name,
        hear_generated=hear_generated,
        direct_output=direct_output,
        volume=volume,
        translate=translate,
        source_language=getattr(stt_model, "language", None),
    )
    if isinstance(stt_model, STTClient):
        state.stt_client = stt_model
    if own_mic_names:
        state.own_mic_names |= set(own_mic_names)
    if mic is not None:
        state.mic, state.mic_label = mic, mic.label
        if mic.note:
            state.set_notice(mic.note, 12.0)
    state.previewer = VoicePreviewer(
        voices_dir,
        local_sink=monitor_router.real_output_sink if monitor_router is not None else None,
        out_dev=out_dev, direct_output=direct_output)
    if monitor_router is not None:
        if monitor_router.enabled != hear_generated:
            monitor_router.set_enabled(hear_generated)
        if hear_generated and not monitor_router.enabled:
            # Saved preference said "on" but the loopback couldn't be created:
            # show the real state and the reason instead of a false "ON".
            state.hear_generated = False
            state.set_error("monitor", monitor_router.last_error or "Couldn't enable monitoring")

    worker = threading.Thread(
        target=worker_loop,
        args=(state, stt_model, in_dev, out_dev, monitor_router),
        daemon=True,
    )
    worker.start()

    os.environ.setdefault("ESCDELAY", "25")  # make Esc respond instantly
    try:
        curses.wrapper(_draw_tui, state, title, mic_label, monitor_router)
    finally:
        with state.lock:
            state.quit = True
        state.previewer.close()
        worker.join(timeout=2)


# ---------------------------------------------------------------------------
# Virtual microphone management (PipeWire/PulseAudio via pactl)
# ---------------------------------------------------------------------------
def _pactl_list_sinks() -> str:
    result = subprocess.run(
        ["pactl", "list", "short", "sinks"], capture_output=True, text=True
    )
    return result.stdout


def sink_exists(name: str) -> bool:
    return any((line.split()[1] == name) for line in _pactl_list_sinks().splitlines() if len(line.split()) >= 2)


def create_null_sink(name: str) -> None:
    subprocess.run(
        [
            "pactl", "load-module", "module-null-sink",
            f"sink_name={name}",
            f"sink_properties=device.description={name}",
        ],
        check=True, capture_output=True, text=True,
    )


def _module_line_has(line: str, key: str, value: str) -> bool:
    """Exact key=value token match (so 'vshift_mic' does not match 'vshift_mic2')."""
    return re.search(rf"(?:^|\s){re.escape(key)}={re.escape(value)}(?:\s|$)", line) is not None


def remove_null_sink(name: str) -> None:
    result = subprocess.run(
        ["pactl", "list", "short", "modules"], capture_output=True, text=True
    )
    for line in result.stdout.splitlines():
        if "module-null-sink" in line and _module_line_has(line, "sink_name", name):
            mod_id = line.split()[0]
            subprocess.run(["pactl", "unload-module", mod_id], capture_output=True)


def _pactl_list_sources() -> str:
    result = subprocess.run(
        ["pactl", "list", "short", "sources"], capture_output=True, text=True
    )
    return result.stdout


def source_exists(name: str) -> bool:
    return any((line.split()[1] == name) for line in _pactl_list_sources().splitlines() if len(line.split()) >= 2)


def create_remap_source(sink_name: str, source_name: str) -> None:
    """Wrap a null sink's monitor in a plain (non-monitor) source.

    Chromium-based apps (Discord included) deliberately hide '.monitor'
    sources from their microphone list. A remap-source is a normal input
    device backed by that monitor, so it shows up like any other mic.
    """
    subprocess.run(
        [
            "pactl", "load-module", "module-remap-source",
            f"master={sink_name}.monitor",
            f"source_name={source_name}",
            f"source_properties=device.description={source_name}",
        ],
        check=True, capture_output=True, text=True,
    )


def remove_remap_source(source_name: str) -> None:
    result = subprocess.run(
        ["pactl", "list", "short", "modules"], capture_output=True, text=True
    )
    for line in result.stdout.splitlines():
        if "module-remap-source" in line and _module_line_has(line, "source_name", source_name):
            mod_id = line.split()[0]
            subprocess.run(["pactl", "unload-module", mod_id], capture_output=True)


def find_device_index_by_name(name_substr: str, want_output: bool = True) -> int | None:
    devices = list(sd.query_devices())
    key = "max_output_channels" if want_output else "max_input_channels"
    needle = name_substr.lower()
    for i, d in enumerate(devices):  # exact match wins
        if d[key] > 0 and d["name"].lower() == needle:
            return i
    for i, d in enumerate(devices):
        if d[key] > 0 and needle in d["name"].lower():
            return i
    return None


# ---------------------------------------------------------------------------
# Choosing the microphone vshift listens to
#
# PortAudio builds without a PulseAudio host API (common on Arch/CachyOS) only
# list raw ALSA hardware, so PipeWire's microphones never show up by name, and
# opening hw:X directly would lock the mic away from Discord. Instead vshift
# lists PipeWire/Pulse sources with pactl and records through PortAudio's
# "pulse" ALSA device, steered to the chosen source with PULSE_SOURCE (or
# "pipewire" with PIPEWIRE_NODE): the same trick as PULSE_SINK for output.
# ---------------------------------------------------------------------------
DEFAULT_SINK_NAME = "vshift_mic"
DEFAULT_MIC_NAME = "vshift_mic_input"
_ORIGINAL_MIC_ENV = {k: os.environ.get(k) for k in ("PULSE_SOURCE", "PIPEWIRE_NODE")}


class MicChoice:
    """A resolved microphone: PortAudio device index, the env that steers it,
    a human label, and the id to save (None = system default)."""

    def __init__(self, device: int | None, env: dict, label: str, saved_id: str | None,
                 note: str = ""):
        self.device, self.env, self.label, self.saved_id, self.note = device, env, label, saved_id, note

    def __repr__(self) -> str:
        return f"MicChoice({self.device!r}, {self.env!r}, {self.label!r}, {self.saved_id!r})"


def apply_mic_env(mic: "MicChoice | None") -> None:
    """Point PortAudio's pulse/pipewire device at the chosen mic. Must run
    right before the InputStream opens (the plugin reads it at open time)."""
    env = mic.env if mic is not None else {}
    for key, original in _ORIGINAL_MIC_ENV.items():
        value = env.get(key, original)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _parse_pactl_sources_text(text: str) -> list[dict]:
    """Fallback for pactl older than 16 (no --format=json)."""
    sources, cur = [], None
    for raw in text.splitlines():
        line = raw.strip()
        if raw.startswith("Source #"):
            cur = {"name": "", "description": "", "properties": {}}
            sources.append(cur)
        elif cur is None:
            continue
        elif line.startswith("Name:"):
            cur["name"] = line.split(":", 1)[1].strip()
        elif line.startswith("Description:"):
            cur["description"] = line.split(":", 1)[1].strip()
        elif " = " in line:
            key, _, value = line.partition(" = ")
            cur["properties"][key.strip()] = value.strip().strip('"')
    return sources


def list_pulse_sources() -> list[dict]:
    """PipeWire/Pulse sources as dicts: name, description, kind (hardware,
    virtual or monitor). Empty when pactl is missing or the server is down."""
    if shutil.which("pactl") is None:
        return []
    try:
        r = subprocess.run(["pactl", "--format=json", "list", "sources"],
                           capture_output=True, text=True, timeout=5)
        raw = json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else None
        if raw is None:
            r = subprocess.run(["pactl", "list", "sources"], capture_output=True, text=True, timeout=5)
            raw = _parse_pactl_sources_text(r.stdout) if r.returncode == 0 else []
    except (OSError, subprocess.SubprocessError, ValueError):
        return []
    out = []
    for src in raw:
        name = str(src.get("name") or "")
        if not name:
            continue
        props = src.get("properties") or {}
        dev_class = str(props.get("device.class", ""))
        if name.endswith(".monitor") or dev_class == "monitor":
            kind = "monitor"
        elif dev_class == "filter" or not props.get("device.api"):
            kind = "virtual"
        else:
            kind = "hardware"
        out.append({"name": name, "description": str(src.get("description") or name), "kind": kind})
    return out


def default_pulse_source() -> str | None:
    try:
        r = subprocess.run(["pactl", "get-default-source"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    name = r.stdout.strip()
    return name if r.returncode == 0 and name else None


def list_mics(own_names: "set[str] | None" = None) -> list[dict]:
    """Microphones vshift can listen to, real ones first: dicts with id,
    label, kind ('hardware', 'virtual' or 'device') and default (bool).

    Monitors (what a speaker is playing) and vshift's own virtual mic are left
    out: listening to either would make vshift hear itself. Falls back to
    PortAudio's input devices when PipeWire/Pulse isn't reachable.
    """
    own = {DEFAULT_MIC_NAME} | set(own_names or ())
    sources = [m for m in list_pulse_sources() if m["kind"] != "monitor" and m["name"] not in own]
    if sources:
        default = default_pulse_source()
        mics = [dict(id=m["name"], label=m["description"], kind=m["kind"], default=m["name"] == default)
                for m in sources]
        mics.sort(key=lambda m: m["kind"] != "hardware")
        return mics
    if sd is None:
        return []
    try:
        devices = list(sd.query_devices())
        default_in = sd.default.device[0]
    except Exception:
        return []
    return [dict(id=d["name"], label=d["name"], kind="device", default=i == default_in, index=i)
            for i, d in enumerate(devices) if d.get("max_input_channels", 0) > 0]


def _pulse_capture_route(source: str) -> tuple[int, dict]:
    """PortAudio device + env that records from a PipeWire/Pulse source."""
    for device_name, env_key in (("pulse", "PULSE_SOURCE"), ("pipewire", "PIPEWIRE_NODE")):
        idx = find_device_index_by_name(device_name, want_output=False)
        if idx is not None and sd.query_devices(idx)["name"].lower() == device_name:
            return idx, {env_key: source}
    raise RuntimeError(
        "PortAudio can't reach PipeWire/PulseAudio microphones (it has no 'pulse' or "
        "'pipewire' device). Install pulseaudio-alsa or pipewire-alsa, or pick a device "
        "number from `vshift devices` with --mic.")


def resolve_mic(choice: "str | int | None", own_names: "set[str] | None" = None) -> MicChoice:
    """Turn a --mic value / saved setting into something capture can open.

    choice may be None or "default" (system default), a PortAudio device
    number, a PipeWire/Pulse source name, or part of a mic's name or
    description ("fifine"). Raises ValueError when nothing matches.
    """
    own = {DEFAULT_MIC_NAME} | set(own_names or ())
    text = "" if choice is None else str(choice).strip()
    if text.lower() in ("", "default"):
        default = default_pulse_source()
        if default and (default in own or default.endswith(".monitor")):
            # The system default is vshift's own output (PipeWire can make a
            # new virtual mic the default): listening to it would loop the
            # generated voice back in. Use a real mic instead.
            real = [m for m in list_mics(own) if m["kind"] == "hardware"]
            if real:
                mic = resolve_mic(real[0]["id"], own)
                mic.saved_id = None
                mic.note = (f"Your default mic is '{default}' (vshift's own output), so vshift "
                            f"listens to '{mic.label}' instead. Pick another with M or --mic.")
                return mic
        label = next((m["label"] for m in list_mics(own) if m["default"]), "system default")
        return MicChoice(None, {}, f"{label} (default)" if label != "system default" else label, None)

    if text.isdigit():
        idx = int(text)
        try:
            info = sd.query_devices(idx)
        except Exception as exc:
            raise ValueError(f"No audio device number {idx}. See `vshift devices`.") from exc
        if info.get("max_input_channels", 0) <= 0:
            raise ValueError(f"Device {idx} ({info['name']}) has no inputs. See `vshift devices`.")
        return MicChoice(idx, {}, info["name"], None)

    if text in own:
        raise ValueError(f"'{text}' is vshift's own virtual mic; listening to it would loop "
                         "the generated voice back in. Pick your real microphone.")
    mics = list_mics(own)
    needle = text.lower()
    matches = ([m for m in mics if m["id"] == text]
               or [m for m in mics if m["label"].lower() == needle]
               or [m for m in mics if needle in m["id"].lower() or needle in m["label"].lower()])
    if not matches:
        names = ", ".join(m["label"] for m in mics) or "none found"
        raise ValueError(f"No microphone matches '{text}'. Available: {names}. "
                         "See `vshift mics`.")
    if len(matches) > 1:
        # "mic" matching "fifine Microphone" and "Voice_Changer_Mic": prefer real hardware.
        matches = [m for m in matches if m["kind"] in ("hardware", "device")] or matches
    if len(matches) > 1:
        names = ", ".join(f"'{m['label']}'" for m in matches)
        raise ValueError(f"'{text}' matches several microphones ({names}); be more specific.")
    mic = matches[0]
    if mic["kind"] == "device":
        return MicChoice(mic["index"], {}, mic["label"], mic["id"])
    idx, env = _pulse_capture_route(mic["id"])
    return MicChoice(idx, env, mic["label"], mic["id"])


# ---------------------------------------------------------------------------
# STT isolation
# ---------------------------------------------------------------------------
STT_READY_TIMEOUT_CACHED_S = 60.0     # model already on disk: just loading it
STT_READY_TIMEOUT_DOWNLOAD_S = 900.0  # first run: model has to be downloaded first


def _stt_model_cached(model_name: str) -> bool:
    """Best-effort check that a faster-whisper model is already downloaded."""
    try:
        if Path(model_name).expanduser().exists():
            return True
        hub = os.environ.get("HF_HUB_CACHE")
        if hub:
            hub_dir = Path(hub)
        else:
            hub_dir = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
        snaps = hub_dir / f"models--Systran--faster-whisper-{model_name}" / "snapshots"
        return snaps.is_dir() and any(snaps.iterdir())
    except OSError:
        return False


# Whisper's language codes (faster_whisper.tokenizer._LANGUAGE_CODES), kept
# here so the parent process can validate --source-lang without importing it.
WHISPER_LANGUAGES = frozenset(
    "af am ar as az ba be bg bn bo br bs ca cs cy da de el en es et eu fa fi fo fr gl gu "
    "ha haw he hi hr ht hu hy id is it ja jw ka kk km kn ko la lb ln lo lt lv mg mi mk ml "
    "mn mr ms mt my ne nl nn no oc pa pl ps pt ro ru sa sd si sk sl sn so sq sr su sv sw "
    "ta te tg th tk tl tr tt uk ur uz vi yi yo yue zh".split())


def translation_model_for(model_name: str) -> str:
    """The multilingual model to translate with. English-only models (".en",
    and the distil-* family) cannot translate, so use their multilingual twin:
    base.en -> base, distil-small.en -> small, distil-large-v3 -> large-v3."""
    name = model_name.strip()
    if Path(name).expanduser().exists():
        return name  # a local model folder: trust the user
    if name.startswith("distil-"):
        name = name[len("distil-"):]
    if name.endswith(".en"):
        name = name[:-3]
    return name


class STTClient:
    """Run faster-whisper in a child process so native crashes cannot kill vshift.

    This is especially useful with Python 3.14/CTranslate2 combinations that have
    been reported to emit leaked-semaphore warnings or crash during transcription.
    The child owns the WhisperModel and is automatically restarted after a crash.

    With translate on, speech in any language comes back as English text, using
    the multilingual twin of model_name (see translation_model_for). model_name
    stays what the user chose, so it is what gets saved.
    """

    def __init__(self, model_name: str, translate: bool = False, language: str | None = None):
        self.model_name = model_name
        self.translate = translate
        self.language = language       # source language when translating; None = detect
        self.multilingual = False      # reported by the worker once it is ready
        self.last_language = ""        # language Whisper detected/used for the last result
        self.lock = threading.Lock()
        self.proc: subprocess.Popen[bytes] | None = None
        self.start_error = ""
        self.log_path = Path.home() / ".cache" / "vshift" / "stt_worker.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_handle = None
        if not _stt_model_cached(self.active_model):
            say(f"First run: downloading the '{self.active_model}' speech model (one-time, "
                "~150 MB). This can take a few minutes...", C.YELLOW)
        self._start()

    @property
    def active_model(self) -> str:
        return translation_model_for(self.model_name) if self.translate else self.model_name

    def configure(self, translate: bool, language: str | None = None) -> str:
        """Switch translation on/off, reloading Whisper only if the model
        changes. Blocks while the model loads (or downloads, the first time),
        so call it off the UI thread. Returns an error message, or ""."""
        with self.lock:
            before = self.active_model
            self.translate, self.language = translate, language
            if self.active_model != before or self.proc is None or self.proc.poll() is not None:
                self._start()
            if not self.start_error and translate and not self.multilingual:
                return (f"'{self.active_model}' is an English-only model and can't translate. "
                        "Use a multilingual one, e.g. --stt-model small")
            return self.start_error

    def _start(self) -> None:
        self._stop()
        model = self.active_model
        # A cold start has to download the model first, which easily exceeds 60 s
        # on a slow connection and used to make the very first launch fail.
        ready_timeout = (
            STT_READY_TIMEOUT_CACHED_S if _stt_model_cached(model)
            else STT_READY_TIMEOUT_DOWNLOAD_S
        )
        worker_script = Path(__file__).with_name("stt_worker.py")
        env = os.environ.copy()
        env["PYTHONWARNINGS"] = "default"
        try:
            self.log_handle = open(self.log_path, "a", encoding="utf-8")
            self.log_handle.write(f"\n--- STT worker start {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            self.log_handle.flush()
            self.proc = subprocess.Popen(
                [sys.executable, str(worker_script), model],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.log_handle,
                env=env,
                bufsize=0,
            )
            ok, why = self._wait_ready(timeout=ready_timeout)
            if not ok:
                raise RuntimeError(why)
            self.start_error = ""
        except Exception as e:
            # Kill the half-started worker instead of leaking it, and keep the
            # real reason (e.g. "model load failed: ...") for the UI.
            self._stop()
            self.start_error = str(e)

    def _wait_ready(self, timeout: float) -> tuple[bool, str]:
        proc = self.proc
        if proc is None or proc.stdout is None:
            return False, "Whisper worker did not start"
        ready, _, _ = select([proc.stdout], [], [], timeout)
        if not ready:
            return False, f"Whisper worker did not become ready within {int(timeout)} seconds (see {self.log_path})"
        line = proc.stdout.readline()
        if not line:
            try:
                code = proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                code = None
            return False, f"Whisper worker exited during startup (code {code}); see {self.log_path}"
        try:
            msg = json.loads(line.decode("utf-8"))
        except Exception:
            return False, "Whisper worker sent an unreadable startup message"
        if msg.get("type") == "ready":
            self.multilingual = bool(msg.get("multilingual", False))
            return True, ""
        return False, str(msg.get("message") or "Whisper worker failed to start")

    def _stop(self) -> None:
        proc = self.proc
        self.proc = None
        if proc is None:
            return
        try:
            if proc.stdin:
                try:
                    proc.stdin.write(b"\xff\xff\xff\xff")
                except Exception:
                    pass
                try:
                    proc.stdin.close()
                except Exception:
                    pass
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.terminate()
                proc.wait(timeout=1.0)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=1.0)
            except Exception:
                pass
        finally:
            if self.log_handle is not None:
                try:
                    self.log_handle.close()
                except OSError:
                    pass
                self.log_handle = None

    def transcribe(self, audio_int16: np.ndarray, timeout: float = 45.0) -> str:
        if audio_int16.size == 0:
            return ""
        payload = np.asarray(audio_int16, dtype=np.float32).astype(np.float32) / 32768.0
        raw = payload.tobytes()

        with self.lock:
            for attempt in range(2):
                proc = self.proc
                if proc is None or proc.poll() is not None:
                    self._start()
                    proc = self.proc
                if proc is None or proc.stdin is None or proc.stdout is None:
                    self.start_error = self.start_error or "Whisper worker could not be started"
                    return ""

                if self.translate:
                    opts = {"task": "translate", "language": self.language}
                else:
                    opts = {"task": "transcribe", "language": "en"}
                header = json.dumps(opts).encode("utf-8")
                try:
                    proc.stdin.write(struct.pack("!I", len(header)) + header)
                    proc.stdin.write(struct.pack("!I", len(raw)))
                    proc.stdin.write(raw)
                    proc.stdin.flush()

                    ready, _, _ = select([proc.stdout], [], [], timeout)
                    if not ready:
                        raise TimeoutError("Whisper worker timed out")
                    line = proc.stdout.readline()
                    if not line:
                        raise BrokenPipeError("Whisper worker exited during transcription")
                    msg = json.loads(line.decode("utf-8"))
                    if msg.get("type") == "error":
                        raise RuntimeError(msg.get("message", "Whisper worker error"))
                    self.last_language = str(msg.get("language") or "")
                    return str(msg.get("text", "")).strip()
                except Exception as e:
                    self.start_error = f"Whisper worker restarted: {e}"
                    self._stop()
                    if attempt == 0:
                        continue
                    return ""
        return ""

    def close(self) -> None:
        with self.lock:
            self._stop()


def transcribe(model: STTClient, audio_int16: np.ndarray) -> str:
    """Transcribe one utterance through the isolated Whisper worker."""
    return model.transcribe(audio_int16)


# ---------------------------------------------------------------------------
# STT / TTS
# ---------------------------------------------------------------------------
def normalize_tts_text(text: str) -> str:
    """Reject common one-word Whisper hallucinations/repetitions."""
    cleaned = re.sub(r"\s+", " ", text.strip())
    if not cleaned:
        return ""
    words = cleaned.lower().split()
    if len(words) <= 2 and words and all(w.strip('.,!?;:\'"') in {"ok", "okay"} for w in words):
        return ""
    if len(words) >= 3:
        compact = [w.strip('.,!?;:\'"').lower() for w in words]
        if len(set(compact)) == 1 and compact[0] in {"ok", "okay"}:
            return ""
    return cleaned


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_ABBREVIATION_RE = re.compile(
    r"\b(mr|mrs|ms|dr|prof|sr|jr|st|vs|etc|approx|e\.g|i\.e)\.$", re.IGNORECASE
)


def split_into_sentences(text: str) -> list[str]:
    """Split one transcribed utterance into sentence-ish chunks.

    The TTS/playback stages treat each returned piece as an independent
    pipeline item. That means a multi-sentence utterance starts playing its
    first sentence while later ones are still being synthesized, instead of
    the whole reply waiting on one big Piper call - the same overlap the
    4-stage pipeline already gives across utterances, now within one.

    Deliberately simple (splits after . ! ? followed by whitespace, with a
    small guard against common abbreviations like "Dr." or "e.g."). It
    doesn't need to be linguistically perfect - a slightly wrong split just
    changes where a pause falls, whereas failing to split at all is the
    behavior this replaces. Falls back to the whole text as one chunk when
    there's nothing to split.
    """
    text = text.strip()
    if not text:
        return []
    raw_parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text) if p.strip()]
    if len(raw_parts) <= 1:
        return [text]

    merged: list[str] = []
    for part in raw_parts:
        if merged and _ABBREVIATION_RE.search(merged[-1]):
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return merged or [text]


def make_synthesis_config(speed: float | None, speaker_id: int | None) -> dict:
    """Build only explicitly configured Piper options.

    Current Piper releases resolve None for length/noise settings internally,
    but volume is a concrete float and should not be overridden with None.
    Omitting unset values also keeps the voice model's own defaults intact.
    """
    config = {}
    if speed is not None:
        config["length_scale"] = float(speed)
    if speaker_id is not None:
        config["speaker_id"] = int(speaker_id)
    return config


class TTSClient:
    """Isolate the TTS engine (Piper or Kokoro, chosen by voice id) in a child process.

    A native Piper/ONNX abort cannot corrupt the parent voice-changer process.
    The worker also owns all voice model switching, so changing voices is
    serialized with synthesis.
    """

    def __init__(self, voices_dir: Path, voice_id: str, speed: float | None, speaker_id: int | None):
        self.voices_dir = voices_dir
        self.voice_id = voice_id
        self.speed = speed
        self.speaker_id = speaker_id
        self.lock = threading.Lock()
        self.proc: subprocess.Popen[str] | None = None
        self.start_error = ""
        self.log_path = Path.home() / ".cache" / "vshift" / "tts_worker.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_handle = None
        self._start()

    def _start(self) -> None:
        self._stop()
        worker_script = Path(__file__).with_name("tts_worker.py")
        self.voice_id = self.voice_id.strip()
        try:
            self.log_handle = open(self.log_path, "a", encoding="utf-8")
            self.log_handle.write(f"\n--- TTS worker start {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            self.log_handle.flush()
            env = os.environ.copy()
            env.setdefault("OMP_NUM_THREADS", "1")
            env.setdefault("OPENBLAS_NUM_THREADS", "1")
            env.setdefault("MKL_NUM_THREADS", "1")
            env.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")
            self.proc = subprocess.Popen(
                [sys.executable, str(worker_script),
                 "--voices-dir", str(self.voices_dir),
                 "--voice", self.voice_id,
                 "--speed", "" if self.speed is None else str(self.speed),
                 "--speaker-id", "" if self.speaker_id is None else str(self.speaker_id),
                 "--kokoro-model", str(kokoro_paths(self.voices_dir)[0]),
                 "--kokoro-voices", str(kokoro_paths(self.voices_dir)[1])],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.log_handle,
                text=True,
                bufsize=1,
                env=env,
            )
            if self.proc.stdout is None:
                raise RuntimeError("TTS worker has no output pipe")
            # Bounded wait: a hung voice load used to block startup forever.
            ready, _, _ = select([self.proc.stdout], [], [], 90.0)
            if not ready:
                raise RuntimeError(f"TTS worker did not become ready within 90 seconds; see {self.log_path}")
            line = self.proc.stdout.readline()
            if not line:
                code = self.proc.poll()
                raise RuntimeError(f"TTS worker exited during startup (code {code}); see {self.log_path}")
            msg = json.loads(line)
            if msg.get("type") != "ready":
                raise RuntimeError(msg.get("message", "TTS worker failed to start"))
            self.start_error = ""
        except Exception as e:
            self._stop()  # don't leak a half-started worker
            self.start_error = str(e)

    def _stop(self) -> None:
        proc = self.proc
        self.proc = None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.write(json.dumps({"type": "quit"}) + "\n")
                proc.stdin.flush()
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            try:
                proc.terminate()
                proc.wait(timeout=1.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        finally:
            if self.log_handle is not None:
                try:
                    self.log_handle.close()
                except OSError:
                    pass
                self.log_handle = None

    def _request(self, payload: dict, timeout: float = 60.0) -> dict:
        with self.lock:
            for attempt in range(2):
                proc = self.proc
                if proc is None or proc.poll() is not None:
                    self._start()
                    proc = self.proc
                if proc is None or proc.stdin is None or proc.stdout is None:
                    raise RuntimeError(self.start_error or "TTS worker is unavailable")
                try:
                    proc.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
                    proc.stdin.flush()
                    ready, _, _ = select([proc.stdout], [], [], timeout)
                    if not ready:
                        raise TimeoutError("TTS worker timed out")
                    line = proc.stdout.readline()
                    if not line:
                        raise BrokenPipeError("TTS worker exited")
                    msg = json.loads(line)
                    if msg.get("type") == "error":
                        raise RuntimeError(msg.get("message", "TTS worker error"))
                    return msg
                except Exception as e:
                    self._stop()
                    if attempt == 0:
                        continue
                    raise RuntimeError(f"TTS worker crashed: {e}; see {self.log_path}") from e
        raise RuntimeError("TTS worker unavailable")

    def set_voice(self, voice_id: str) -> None:
        voice_id = voice_id.strip()
        if not voice_id:
            return
        ensure_voice(voice_id, self.voices_dir, announce=False)
        msg = self._request({"type": "set_voice", "voice": voice_id}, timeout=90.0)
        if msg.get("type") != "voice":
            raise RuntimeError(msg.get("message", "Voice switch failed"))
        self.voice_id = voice_id

    def synthesize(self, text: str, config: dict) -> tuple[np.ndarray, int]:
        clean = " ".join(text.split())
        if not clean:
            return np.zeros(0, dtype=np.int16), 22050
        msg = self._request({
            "type": "synthesize",
            "text": clean,
            "speaker_id": config.get("speaker_id"),
            "length_scale": config.get("length_scale"),
            "noise_scale": config.get("noise_scale"),
            "noise_w_scale": config.get("noise_w_scale"),
        }, timeout=60.0)
        raw = base64.b64decode(msg.get("audio", ""))
        rate = int(msg.get("sample_rate", 22050))
        return np.frombuffer(raw, dtype=np.int16).copy(), rate

    def close(self) -> None:
        with self.lock:
            self._stop()


class MonitorRouter:
    """Play the virtual sink's monitor on a real output so you can hear the
    generated voice locally (the "Hear generated" / H toggle).

    Implemented as a PulseAudio/PipeWire `module-loopback` from
    `<sink>.monitor` to your real output. Failures are recorded in
    `last_error` (and ~/.cache/vshift/monitor.log) instead of being swallowed,
    and loading falls back through progressively simpler argument sets because
    PulseAudio and pipewire-pulse don't accept exactly the same options.
    """

    def __init__(self, source_sink: str, enabled: bool = True):
        self.source_sink = source_sink
        self.enabled = False
        self.module_id: int | None = None
        self.last_error = ""
        self.physical_sink: str | None = None
        if enabled:
            self.set_enabled(True)

    # -- pactl helpers ------------------------------------------------------
    @staticmethod
    def _pactl(*args: str, timeout: float = 5.0) -> subprocess.CompletedProcess:
        return subprocess.run(["pactl", *args], capture_output=True, text=True, timeout=timeout)

    def _log(self, message: str) -> None:
        try:
            path = Path.home() / ".cache" / "vshift" / "monitor.log"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")
        except OSError:
            pass

    @staticmethod
    def _default_sink() -> str | None:
        try:
            r = MonitorRouter._pactl("get-default-sink")
            name = r.stdout.strip()
            if r.returncode == 0 and name:
                return name
            # Older pactl has no get-default-sink; fall back to `pactl info`.
            r = MonitorRouter._pactl("info")
            for line in r.stdout.splitlines():
                if line.lower().startswith("default sink:"):
                    name = line.split(":", 1)[1].strip()
                    return name or None
        except (OSError, subprocess.SubprocessError):
            pass
        return None

    def _sink_names(self) -> list[str]:
        try:
            r = self._pactl("list", "short", "sinks")
        except (OSError, subprocess.SubprocessError):
            return []
        names = []
        for line in r.stdout.splitlines():
            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) >= 2:
                names.append(parts[1])
        return names

    def _pick_output_sink(self) -> str | None:
        """The real output to monitor through - never the virtual sink itself
        (looping it into itself gives silence or a feedback howl)."""
        def usable(name: str | None) -> bool:
            return bool(name) and name != self.source_sink and not name.endswith(".monitor")

        default = self._default_sink()
        if usable(default):
            return default
        for name in self._sink_names():
            if usable(name):
                return name
        return None

    def _loopback_ids(self) -> list[int]:
        try:
            r = self._pactl("list", "short", "modules")
        except (OSError, subprocess.SubprocessError):
            return []
        ids = []
        for line in r.stdout.splitlines():
            parts = line.split(None, 2)
            if len(parts) < 3 or parts[1] != "module-loopback":
                continue
            if _module_line_has(parts[2], "source", f"{self.source_sink}.monitor"):
                try:
                    ids.append(int(parts[0]))
                except ValueError:
                    pass
        return ids

    def _find_existing(self) -> int | None:
        ids = self._loopback_ids()
        return ids[0] if ids else None

    def _module_alive(self, module_id: int) -> bool:
        try:
            r = self._pactl("list", "short", "modules")
        except (OSError, subprocess.SubprocessError):
            return False
        return any(line.split(None, 1)[:1] == [str(module_id)] for line in r.stdout.splitlines())

    # -- public API ------------------------------------------------------------
    def set_enabled(self, enabled: bool) -> None:
        if not enabled:
            for mid in self._loopback_ids() or ([self.module_id] if self.module_id is not None else []):
                try:
                    self._pactl("unload-module", str(mid))
                except (OSError, subprocess.SubprocessError):
                    pass
            self.module_id = None
            self.enabled = False
            return

        if self.enabled and self.module_id is not None and self._module_alive(self.module_id):
            return  # already running
        self.enabled = False
        self.module_id = None
        self.last_error = ""

        sink = self._pick_output_sink()
        if sink is None:
            self.last_error = "Monitor: no real output device found to play through"
            self._log(self.last_error)
            return
        self.physical_sink = sink

        existing = self._find_existing()
        if existing is not None:
            self.module_id = existing
            self.enabled = True
            return

        base = ["load-module", "module-loopback", f"source={self.source_sink}.monitor", f"sink={sink}"]
        tag = "source_output_properties=application.name=vshift-monitor"
        attempts = [
            ["latency_msec=20", "source_dont_move=true", "sink_dont_move=true", tag],
            ["latency_msec=30", tag],
            ["latency_msec=50"],
            [],
        ]
        errors: list[str] = []
        for extra in attempts:
            args = base + extra
            try:
                r = self._pactl(*args)
            except FileNotFoundError:
                self.last_error = "Monitor: 'pactl' not found (install pipewire-pulse or pulseaudio-utils)"
                self._log(self.last_error)
                return
            except subprocess.SubprocessError as e:
                errors.append(f"{type(e).__name__}: {e}")
                self._log(f"pactl {' '.join(args)} -> {errors[-1]}")
                continue
            if r.returncode != 0:
                errors.append((r.stderr or r.stdout).strip() or f"pactl exited with code {r.returncode}")
                self._log(f"pactl {' '.join(args)} -> {errors[-1]}")
                continue
            try:
                module_id = int(r.stdout.split()[0])
            except (ValueError, IndexError):
                errors.append(f"unexpected pactl output: {r.stdout!r}")
                self._log(f"pactl {' '.join(args)} -> {errors[-1]}")
                continue
            time.sleep(0.15)
            if not self._module_alive(module_id):
                errors.append("loopback module unloaded itself right after loading")
                self._log(f"pactl {' '.join(args)} -> {errors[-1]}")
                continue
            self.module_id = module_id
            self.enabled = True
            self._log(f"monitor on: {self.source_sink}.monitor -> {sink} (module {module_id}, args: {' '.join(extra) or 'minimal'})")
            return

        self.last_error = f"Monitor loopback failed: {errors[-1] if errors else 'unknown error'}"
        self._log(self.last_error)

    def toggle(self) -> bool:
        self.set_enabled(not self.enabled)
        return self.enabled

    def real_output_sink(self) -> str | None:
        """Your speakers/headphones (never the virtual sink), e.g. for previews."""
        return self._pick_output_sink()

    def close(self) -> None:
        self.set_enabled(False)


PREVIEW_TEXT = "Hi there! This is how I sound. Pretty good, right?"


def play_on_local_sink(audio: np.ndarray, sample_rate: int, sink: str | None) -> subprocess.Popen:
    """Start playing audio on a real PulseAudio/PipeWire sink (never the
    virtual mic): used for voice previews, which must not reach Discord.
    Returns the player process (already fed and closed on stdin)."""
    env = {k: v for k, v in os.environ.items() if k != "PULSE_SINK"}
    raw = np.ascontiguousarray(audio, dtype=np.int16).tobytes()
    if shutil.which("paplay"):
        cmd = ["paplay", "--raw", "--format=s16le", f"--rate={sample_rate}", "--channels=1"]
        if sink:
            cmd.append(f"--device={sink}")
    elif shutil.which("pw-play"):
        cmd = ["pw-play", "--rate", str(sample_rate), "--channels", "1", "--format", "s16"]
        if sink:
            cmd += ["--target", sink]
        cmd.append("-")
    else:
        raise RuntimeError("previews need paplay (libpulse / pulseaudio-utils) or pw-play")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, env=env)
    try:
        proc.stdin.write(raw)
        proc.stdin.close()
    except BrokenPipeError:
        pass
    return proc


class VoicePreviewer:
    """Speaks a sample sentence in another voice without touching the live one.

    Uses its own TTS worker (started on first use, reused for later previews,
    closed with the browser) so the voice in your call doesn't change. In
    `vshift discord` the sample plays on your real speakers/headphones through
    paplay, never into the virtual mic; in `vshift run` it uses the normal output.
    """

    def __init__(self, voices_dir: Path, local_sink=None, out_dev: int | None = None,
                 direct_output: bool = False):
        self.voices_dir = voices_dir
        self.local_sink = local_sink        # callable -> sink name, for discord mode
        self.out_dev = out_dev
        self.direct_output = direct_output
        self.lock = threading.Lock()
        self.client: "TTSClient | None" = None
        self.thread: threading.Thread | None = None
        self.player: subprocess.Popen | None = None
        self.status = ""
        self.error = ""

    @property
    def busy(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def preview(self, voice_id: str, speed: float | None = None, volume: float = 1.0) -> bool:
        if self.busy:
            return False
        self.error = ""
        self.status = f"Loading {_voice_parts(voice_id)[0]}…"
        self.thread = threading.Thread(target=self._run, args=(voice_id, speed, volume), daemon=True)
        self.thread.start()
        return True

    def _run(self, voice_id: str, speed: float | None, volume: float) -> None:
        try:
            with self.lock:
                if self.client is None:
                    self.client = TTSClient(self.voices_dir, voice_id, None, None)
                    if self.client.start_error:
                        error, self.client = self.client.start_error, None
                        raise RuntimeError(error)
                elif self.client.voice_id != voice_id:
                    self.client.set_voice(voice_id)
                client = self.client
            self.status = f"Previewing {_voice_parts(voice_id)[0]}…"
            audio, rate = client.synthesize(PREVIEW_TEXT, make_synthesis_config(speed, None))
            audio = apply_volume(audio, volume)
            if self.direct_output:
                player = Player(self.out_dev)  # its own stream: never clashes with sd.play
                try:
                    player.play(audio, rate)  # blocks until (almost) all of it is out
                    player.drain()
                finally:
                    player.close()
            else:
                self.player = play_on_local_sink(audio, rate, self.local_sink() if self.local_sink else None)
                _, err_out = self.player.communicate(timeout=60)
                if self.player.returncode not in (0, None, -15):
                    raise RuntimeError((err_out or b"").decode("utf-8", "replace").strip()
                                       or f"player exited with {self.player.returncode}")
            self.status = ""
        except Exception as exc:
            self.status = ""
            self.error = f"Preview failed: {exc}"
        finally:
            self.player = None

    def stop(self, wait: float = 4.0) -> None:
        """Cut a playing preview short (so the mic doesn't hear it once it
        un-pauses) and wait briefly for the preview thread."""
        player = self.player
        if player is not None and player.poll() is None:
            try:
                player.terminate()
            except OSError:
                pass
        if self.thread is not None:
            self.thread.join(timeout=wait)

    def close(self) -> None:
        self.stop()
        with self.lock:
            if self.client is not None:
                self.client.close()
                self.client = None


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_version(_args: argparse.Namespace) -> None:
    print(f"vshift {VERSION}")


def cmd_config(args: argparse.Namespace) -> None:
    if args.reset:
        try:
            SETTINGS_PATH.unlink()
            say(f"✓ Reset settings: {SETTINGS_PATH}", C.GREEN)
        except FileNotFoundError:
            say("Settings are already at defaults.", C.DIM)
        except OSError as exc:
            sys.exit(f"Could not reset settings: {exc}")
        return
    prefs = load_settings()
    if args.json:
        print(json.dumps(prefs, indent=2, sort_keys=True))
        return
    say(f"Settings: {SETTINGS_PATH}", C.BOLD)
    for key, value in prefs.items():
        print(f"  {key:<16} {value}")


def cmd_doctor(args: argparse.Namespace) -> None:
    say("vshift diagnostics", C.BOLD)
    checks = []

    py_ok = sys.version_info >= (3, 12) and sys.version_info < (3, 14)
    checks.append((py_ok, f"Python {sys.version_info.major}.{sys.version_info.minor}"))
    pactl = shutil.which("pactl")
    checks.append((bool(pactl), "pactl" if pactl else "pactl not found"))

    if AUDIO_IMPORT_ERROR is None:
        checks.append((True, "numpy + sounddevice (PortAudio found)"))
    else:
        checks.append((False, f"audio dependencies: {AUDIO_IMPORT_ERROR}"))

    if pactl:
        default_sink = MonitorRouter._default_sink()
        sink_arg = getattr(args, "sink_name", None) or "vshift_mic"
        if not default_sink:
            checks.append((False, "default output sink (needed for 'Hear generated'): none found"))
        elif default_sink == sink_arg:
            checks.append((False, f"default output sink is the virtual sink '{default_sink}' - "
                                  "switch your normal speakers/headphones back to default"))
        else:
            checks.append((True, f"default output sink: {default_sink}"))
        default_source = default_pulse_source()
        if default_source and (default_source.startswith("vshift_mic") or default_source.endswith(".monitor")):
            checks.append((False, f"default microphone is '{default_source}', not a real mic - set your "
                                  "real mic as default, or pick one with --mic (see `vshift mics`)"))
        elif default_source:
            checks.append((True, f"default microphone: {default_source}"))

    for name in ("faster_whisper", "piper"):
        try:
            __import__(name)
            checks.append((True, name))
        except Exception as exc:
            checks.append((False, f"{name}: {exc}"))

    prefs = load_settings()
    voices_dir = Path(args.voices_dir).expanduser()
    using_kokoro = is_kokoro_voice(prefs["voice"])
    try:
        __import__("kokoro_onnx")
        checks.append((True, "kokoro_onnx"))
    except Exception as exc:
        # Only a hard failure if you actually use a Kokoro voice.
        hint = f"{exc}. Install it: {sys.executable} -m pip install 'kokoro-onnx>=0.6,<1'"
        if using_kokoro:
            checks.append((False, f"kokoro_onnx (needed for {prefs['voice']}): {hint}"))
        else:
            say(f"  - kokoro_onnx not installed (optional, for Kokoro voices): {hint}", C.DIM)
    checks.append((True, f"settings: {SETTINGS_PATH}"))
    if using_kokoro:
        model_path, voices_path = kokoro_paths(voices_dir)
        checks.append((model_path.exists() and voices_path.exists(),
                       f"voice cached: {prefs['voice']} ({model_path.name} + {voices_path.name} "
                       f"in {voices_dir})"))
    else:
        voice_path = voices_dir / f"{prefs['voice']}.onnx"
        checks.append((voice_path.exists(), f"voice cached: {prefs['voice']} ({voices_dir})"))

    failed = 0
    for ok, label in checks:
        if ok:
            say(f"  ✓ {label}", C.GREEN)
        else:
            failed += 1
            say(f"  ✗ {label}", C.RED)
    if failed:
        say("\nSome checks need attention.", C.YELLOW)
        raise SystemExit(1)
    say("\nAll basic checks passed.", C.GREEN)


def cmd_voices(args: argparse.Namespace) -> None:
    prefs = load_settings()
    say(f"\nSaved voice: {prefs['voice']}\n", C.BOLD)
    voices_dir = Path(args.voices_dir).expanduser()
    if args.installed:
        paths = sorted(p for p in voices_dir.glob("*.onnx") if not p.name.startswith("kokoro-"))
        kokoro_ready = kokoro_installed(voices_dir)
        if not paths and not kokoro_ready:
            say(f"No downloaded voices in {voices_dir}", C.YELLOW)
        else:
            say("Downloaded voices:", C.BOLD)
            for path in paths:
                ok = path.with_name(path.name + ".json").exists()
                marker = "✓" if ok else "!"
                print(f"  {marker} {path.stem}")
            if kokoro_ready:
                print(f"  ✓ Kokoro model ({kokoro_model_filename()}): "
                      f"all {len(KOKORO_ENGLISH_VOICES)} English voices available as kokoro:<name>")
        return
    say("Curated starter voices (use with --voice <id>):\n", C.BOLD)
    for vid, desc in CURATED_VOICES.items():
        print(f"  {C.GREEN}{vid:<38}{C.RESET} {desc}")
    say("\nOfficial English Piper catalog IDs:", C.BOLD)
    for vid in OFFICIAL_ENGLISH_VOICES:
        print(f"  {vid}")
    say("\nKokoro voices (use as --voice kokoro:<name>):", C.BOLD)
    print("  " + ", ".join(KOKORO_ENGLISH_VOICES))
    say("  The first Kokoro voice you use downloads ~350 MB once; after that you can also\n"
        "  cycle through the Kokoro voices with [ and ]. Run `vshift tts-bench` to check\n"
        "  whether your CPU keeps up with it.", C.DIM)
    say(f"\nFull Piper catalog with audio samples: {VOICE_SAMPLES_URL}", C.DIM)


_BENCH_SENTENCES = (
    "Sounds good to me.",
    "I think we should meet at the usual place around seven tonight.",
    "Honestly, I was not expecting that to work, but it sounds pretty good to me.",
)


def cmd_tts_bench(args: argparse.Namespace) -> None:
    """Time each voice's engine on this machine, through the same worker vshift uses."""
    require_audio_dependencies()
    voices_dir = Path(args.voices_dir).expanduser()
    voices = list(dict.fromkeys(args.voice or [load_settings()["voice"], "kokoro:af_heart"]))
    say(f"\nTTS speed on this machine ({os.cpu_count()} CPU threads)\n", C.BOLD)
    say("RTF (real-time factor) = seconds to synthesize / seconds of speech. Below 1.0\n"
        "keeps up with speech; lower leaves more headroom. The delay you feel for a\n"
        "sentence is the 'time to speak' column.\n", C.DIM)
    for voice in voices:
        say(f"{voice}", C.BOLD)
        try:
            ensure_voice(voice, voices_dir)
        except (ValueError, RuntimeError) as exc:
            say(f"  ✗ {exc}", C.RED)
            continue
        client = TTSClient(voices_dir, voice, None, None)
        try:
            if client.start_error:
                say(f"  ✗ {client.start_error}", C.RED)
                continue
            try:
                client.synthesize("Warm up.", {})  # the first call is slower than the rest
                rows = []
                for text in _BENCH_SENTENCES:
                    started = time.perf_counter()
                    audio, rate = client.synthesize(text, {})
                    took = time.perf_counter() - started
                    rows.append((text, audio.size / rate if rate else 0.0, took))
            except RuntimeError as exc:
                say(f"  ✗ {exc}", C.RED)
                continue
        finally:
            client.close()
        for text, seconds, took in rows:
            rtf = took / seconds if seconds else float("inf")
            print(f"  {len(text):3d} chars  {seconds:5.2f}s of speech  "
                  f"{took:5.2f}s to speak  RTF {rtf:4.2f}")
        total_audio = sum(r[1] for r in rows)
        total_time = sum(r[2] for r in rows)
        overall = total_time / total_audio if total_audio else float("inf")
        typical = 3.0 * overall  # seconds to produce a typical 3-second sentence
        short_time = rows[0][2]
        if overall >= 1.0:
            verdict, colour = "slower than real time: it will fall behind. Use a lighter voice.", C.RED
        elif overall >= 0.5 or short_time > 1.0:
            verdict, colour = (f"keeps up, but a typical 3-second sentence takes about "
                               f"{typical:.1f}s to produce (a short one: {short_time:.1f}s), "
                               "on top of transcription. Expect a noticeable delay.", C.YELLOW)
        else:
            verdict, colour = (f"fast: a typical 3-second sentence takes about {typical:.1f}s "
                               f"(a short one: {short_time:.1f}s).", C.GREEN)
        say(f"  overall RTF {overall:.2f}: {verdict}\n", colour)


def cmd_devices(args: argparse.Namespace) -> None:
    require_audio_dependencies()
    devices = sd.query_devices()
    if args.json:
        print(json.dumps([dict(d) for d in devices], indent=2, default=str))
        return
    say("\nAudio devices (use the index with --input-device / --output-device):\n", C.BOLD)
    for i, d in enumerate(devices):
        ins = int(d.get("max_input_channels", 0))
        outs = int(d.get("max_output_channels", 0))
        print(f"  {i:>3}  in:{ins:<2} out:{outs:<2}  {d['name']}")
    try:
        default_in, default_out = sd.default.device
        say(f"\nCurrent defaults -> input: {default_in}, output: {default_out}", C.DIM)
    except Exception:
        pass


def cmd_mics(args: argparse.Namespace) -> None:
    mics = list_mics()
    saved = load_settings()["mic"]
    if args.json:
        print(json.dumps([{**m, "saved": m["id"] == saved} for m in mics], indent=2))
        return
    if not mics:
        say("No microphones found. Is PipeWire/PulseAudio running? Try `vshift devices`.", C.YELLOW)
        return
    say("\nMicrophones (use any part of a name with --mic, or press M in vshift):\n", C.BOLD)
    for m in mics:
        marks = [w for w, on in (("saved", m["id"] == saved), ("system default", m["default"]),
                                 (m["kind"], m["kind"] != "hardware")) if on]
        bullet = f"{C.GREEN}●{C.RESET}" if m["id"] == saved else " "
        print(f"  {bullet} {m['label']:<34} {C.DIM}{', '.join(marks)}{C.RESET}")
        if m["id"] != m["label"]:
            print(f"      {C.DIM}{m['id']}{C.RESET}")
    if saved is None:
        say("\nvshift uses your system default mic (no --mic saved).", C.DIM)
    default = default_pulse_source()
    if default in (DEFAULT_MIC_NAME,):
        say(f"\nYour system default mic is '{default}', vshift's own output. vshift will listen to "
            "a real mic instead, but other apps may be using the wrong one: set your real mic as "
            "default in your sound settings, or pass --mic.", C.YELLOW)


def cmd_download(args: argparse.Namespace) -> None:
    voices_dir = Path(args.voices_dir).expanduser()
    try:
        if args.force:
            for path in remove_voice_files(args.voice, voices_dir):
                say(f"Removed {path}", C.DIM)
        path = ensure_voice(args.voice, voices_dir)
    except (ValueError, RuntimeError) as exc:
        sys.exit(str(exc))
    say(f"✓ Voice ready at {path}", C.GREEN)


def cmd_test_mic(args: argparse.Namespace) -> None:
    """Sanity-check the virtual mic path (sink -> remap-source -> app) with a
    plain tone, before bringing STT/TTS into the picture at all. If Discord's
    input meter (or `pavucontrol`'s Recording tab) doesn't move while this
    plays, the problem is in the audio routing, not vshift's conversion."""
    require_audio_dependencies()
    if shutil.which("pactl") is None:
        sys.exit(
            "'pactl' not found. This command needs PipeWire or PulseAudio "
            "(Linux only). On CachyOS/Arch: sudo pacman -S pipewire-pulse"
        )

    if args.seconds <= 0:
        raise ValueError("--seconds must be greater than 0")
    sink_name = args.sink_name
    mic_name = args.mic_name
    created_sink = False
    created_source = False

    if not sink_exists(sink_name):
        say(f"Creating virtual sink '{sink_name}'...", C.CYAN)
        create_null_sink(sink_name)
        created_sink = True
        time.sleep(0.4)
    if not source_exists(mic_name):
        say(f"Creating virtual mic '{mic_name}'...", C.CYAN)
        create_remap_source(sink_name, mic_name)
        created_source = True
        time.sleep(0.4)

    try:
        out_idx = resolve_sink_output(sink_name)
        if out_idx is None:
            sys.exit(f"Couldn't find '{sink_name}' as an audio device. Try `vshift devices`. "
                     "On Arch/CachyOS, installing pulseaudio-alsa may add the 'pulse' device "
                     "vshift can route through.")

        say(f"\nSelect \"{mic_name}\" as the mic in Discord/pavucontrol now.", C.BOLD)
        say(f"Playing a {args.seconds:g}s test tone — watch the input level meter...\n", C.CYAN)

        sr = 48000
        t = np.linspace(0, args.seconds, int(sr * args.seconds), endpoint=False)
        tone = (np.sin(2 * np.pi * 440 * t) * 10000).astype(np.int16)
        play_int16(tone, sr, device=out_idx)

        say("Done. If the meter moved, routing is fine — the issue is upstream", C.GREEN)
        say("(STT/TTS). If it didn't move, check that you picked the right", C.GREEN)
        say(f"device (\"{mic_name}\", not \"{sink_name}\") and that it's not muted.", C.GREEN)
    finally:
        # Previously an early exit or a playback error left the virtual devices behind.
        if not args.keep_sink:
            if created_source:
                remove_remap_source(mic_name)
            if created_sink:
                remove_null_sink(sink_name)


def _resolve_mic_args(args: argparse.Namespace, prefs: dict, own: set[str]) -> "MicChoice":
    """--mic (or the older --input-device), else the saved mic, else the default.
    A mic named on the command line is saved for next time; a saved mic that
    has been unplugged falls back to the default with a warning."""
    if args.mic is not None:
        mic = resolve_mic(args.mic, own)  # ValueError -> clean exit in main()
        args.mic = mic.saved_id
    elif args.input_device is not None:
        mic = resolve_mic(str(args.input_device), own)
        args.mic = prefs.get("mic")
    else:
        args.mic = prefs.get("mic")
        try:
            mic = resolve_mic(args.mic, own)
        except (ValueError, RuntimeError) as exc:
            say(f"Saved microphone unavailable ({exc}); using the default.", C.YELLOW)
            mic = resolve_mic(None, own)
    if mic.note:
        say(mic.note, C.YELLOW)
    say(f"Listening to: {mic.label}", C.CYAN)
    return mic


def _resolve_translate_args(args: argparse.Namespace, prefs: dict) -> None:
    """--translate/--no-translate/--source-lang, falling back to saved settings."""
    if args.translate is None:
        args.translate = bool(prefs.get("translate", False))
    if args.source_lang is None:
        args.source_lang = prefs.get("source_language")
    elif args.source_lang == "auto":
        args.source_lang = None


def _start_stt(args: argparse.Namespace) -> "STTClient":
    stt = STTClient(args.stt_model, translate=args.translate, language=args.source_lang)
    if args.translate and not stt.start_error:
        say(f"Translating speech ({args.source_lang or 'language auto-detected'}) into English "
            f"with Whisper '{stt.active_model}'.", C.CYAN)
        if not stt.multilingual:
            say(f"  '{stt.active_model}' is English-only, so it can't translate; "
                "pass a multilingual model such as --stt-model small.", C.YELLOW)
    return stt


def cli_language(value: str) -> str:
    code = value.strip().lower()
    if code != "auto" and code not in WHISPER_LANGUAGES:
        raise argparse.ArgumentTypeError(
            "use a Whisper language code such as de, es, fr, ja, or 'auto'")
    return code


def add_translate_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--translate", dest="translate", action="store_true",
                       help="Speak in any language, come out in English (toggle live with L). "
                            "Uses the multilingual twin of --stt-model, e.g. base for base.en.")
    group.add_argument("--no-translate", dest="translate", action="store_false",
                       help="Turn translation off (the default)")
    parser.set_defaults(translate=None)
    parser.add_argument("--source-lang", type=cli_language, default=None, metavar="CODE",
                        help="Language you speak when translating, e.g. de or ja "
                             "(default: auto-detect; 'auto' resets a saved choice)")


def cmd_discord(args: argparse.Namespace) -> None:
    """One command: create/reuse a virtual mic sink, then run continuous
    real-time conversion straight into it, so Discord/games can pick it up
    as an input device."""
    require_audio_dependencies()
    prefs = load_settings()
    if args.voice is None:
        args.voice = prefs.get("voice", DEFAULT_VOICE)
    if args.stt_model is None:
        args.stt_model = prefs.get("stt_model", DEFAULT_STT_MODEL)
    if args.speed is None:
        args.speed = prefs.get("speed")
    if args.speaker_id is None:
        args.speaker_id = prefs.get("speaker_id")
    if args.threshold is None:
        args.threshold = float(prefs.get("threshold", 400.0))
    if args.silence_ms is None:
        args.silence_ms = int(prefs.get("silence_ms", 700))
    _resolve_translate_args(args, prefs)
    if args.hear_generated is None:
        args.hear_generated = bool(prefs.get("hear_generated", False))
    if shutil.which("pactl") is None:
        sys.exit(
            "'pactl' not found. This command needs PipeWire or PulseAudio "
            "(Linux only). On CachyOS/Arch: sudo pacman -S pipewire-pulse"
        )

    sink_name = args.sink_name
    mic_name = args.mic_name
    created_sink = False
    created_source = False
    own_mics = {mic_name, DEFAULT_MIC_NAME}
    mic = _resolve_mic_args(args, prefs, own_mics)

    if sink_exists(sink_name):
        say(f"Using existing virtual sink '{sink_name}'", C.DIM)
    else:
        say(f"Creating virtual sink '{sink_name}'...", C.CYAN)
        try:
            create_null_sink(sink_name)
        except subprocess.CalledProcessError as e:
            sys.exit(f"Failed to create virtual sink: {e.stderr.strip()}")
        created_sink = True
        time.sleep(0.4)  # give pipewire a moment to register the new device

    # Discord and other Chromium-based apps deliberately hide ".monitor"
    # sources from their mic list, so we also expose the same audio as a
    # plain remap-source, which shows up as a normal input device.
    if source_exists(mic_name):
        say(f"Using existing virtual mic '{mic_name}'", C.DIM)
    else:
        say(f"Creating virtual mic '{mic_name}'...", C.CYAN)
        try:
            create_remap_source(sink_name, mic_name)
        except subprocess.CalledProcessError as e:
            if created_sink:
                remove_null_sink(sink_name)
            sys.exit(f"Failed to create virtual mic: {e.stderr.strip()}")
        created_source = True
        time.sleep(0.4)

    out_idx = resolve_sink_output(sink_name)
    if out_idx is None:
        if created_source:
            remove_remap_source(mic_name)
        if created_sink:
            remove_null_sink(sink_name)
        sys.exit(
            f"Created sink '{sink_name}' but couldn't find it as an audio "
            "device. Try `vshift devices` to inspect what's available. "
            "On Arch/CachyOS, installing pulseaudio-alsa may add the 'pulse' "
            "device vshift can route through."
        )

    voices_dir = Path(args.voices_dir).expanduser()
    try:
        ensure_voice(args.voice, voices_dir)
    except (ValueError, RuntimeError) as exc:
        if created_source:
            remove_remap_source(mic_name)
        if created_sink:
            remove_null_sink(sink_name)
        sys.exit(str(exc))
    save_settings({
        **prefs,
        "voice": args.voice, "stt_model": args.stt_model, "speed": args.speed,
        "speaker_id": args.speaker_id, "threshold": args.threshold,
        "silence_ms": args.silence_ms, "hear_generated": args.hear_generated,
        "translate": args.translate, "source_language": args.source_lang,
        "mic": args.mic,
    })

    stt_model = None
    tts_client = None
    monitor_router = None
    try:
        say(f"Loading STT model '{args.stt_model}'...", C.CYAN)
        stt_model = _start_stt(args)
        if stt_model.start_error:
            raise RuntimeError(f"Could not start isolated Whisper worker: {stt_model.start_error}")
        say(f"Loading isolated TTS voice '{args.voice}'...", C.CYAN)
        tts_client = TTSClient(voices_dir, args.voice, args.speed, args.speaker_id)
        if tts_client.start_error:
            raise RuntimeError(f"Could not start isolated Piper worker: {tts_client.start_error}")
        monitor_router = MonitorRouter(sink_name, enabled=bool(args.hear_generated))

        say("\nIn Discord's mic dropdown (Settings > Voice & Video > Input", C.DIM)
        say(f"Device), pick \"{mic_name}\". Starting in 2s...", C.DIM)
        time.sleep(2)

        start_conversion_tui(
            title="vshift discord",
            voice=args.voice,
            mic_label=mic_name,
            stt_model=stt_model,
            tts_client=tts_client,
            in_dev=mic.device,
            out_dev=out_idx,
            monitor_router=monitor_router,
            threshold=args.threshold,
            silence_ms=args.silence_ms,
            voices_dir=voices_dir,
            speed=args.speed,
            speaker_id=args.speaker_id,
            hear_generated=bool(args.hear_generated),
            volume=prefs["volume"],
            mic=mic,
            own_mic_names=own_mics,
        )
    finally:
        if monitor_router is not None:
            monitor_router.close()
        if tts_client is not None:
            tts_client.close()
        if stt_model is not None:
            stt_model.close()
        if not args.keep_sink:
            if created_source:
                say(f"Removing virtual mic '{mic_name}'...", C.DIM)
                remove_remap_source(mic_name)
            if created_sink:
                say(f"Removing virtual sink '{sink_name}'...", C.DIM)
                remove_null_sink(sink_name)
        say("bye 👋", C.DIM)


def cmd_run(args: argparse.Namespace) -> None:
    require_audio_dependencies()
    prefs = load_settings()
    if args.voice is None:
        args.voice = prefs.get("voice", DEFAULT_VOICE)
    if args.stt_model is None:
        args.stt_model = prefs.get("stt_model", DEFAULT_STT_MODEL)
    if args.speed is None:
        args.speed = prefs.get("speed")
    if args.speaker_id is None:
        args.speaker_id = prefs.get("speaker_id")
    if args.threshold is None:
        args.threshold = float(prefs.get("threshold", 400.0))
    if args.silence_ms is None:
        args.silence_ms = int(prefs.get("silence_ms", 700))
    _resolve_translate_args(args, prefs)
    mic = _resolve_mic_args(args, prefs, {DEFAULT_MIC_NAME})
    voices_dir = Path(args.voices_dir).expanduser()
    ensure_voice(args.voice, voices_dir)
    save_settings({
        **prefs,
        "voice": args.voice, "stt_model": args.stt_model, "speed": args.speed,
        "speaker_id": args.speaker_id, "threshold": args.threshold,
        "silence_ms": args.silence_ms,
        "translate": args.translate, "source_language": args.source_lang,
        "mic": args.mic,
    })

    say(f"Loading STT model '{args.stt_model}'...", C.CYAN)
    stt_model = _start_stt(args)
    if stt_model.start_error:
        sys.exit(f"Could not start isolated Whisper worker: {stt_model.start_error}")

    say(f"Loading isolated TTS voice '{args.voice}'...", C.CYAN)
    tts_client = TTSClient(voices_dir, args.voice, args.speed, args.speaker_id)
    if tts_client.start_error:
        stt_model.close()
        sys.exit(f"Could not start isolated Piper worker: {tts_client.start_error}")
    in_dev = mic.device
    out_dev = args.output_device

    try:
        if args.continuous:
            out_label = f"device {out_dev}" if out_dev is not None else "default output"
            start_conversion_tui(
                title="vshift run",
                voice=args.voice,
                mic_label=out_label,
                stt_model=stt_model,
                tts_client=tts_client,
                in_dev=in_dev,
                out_dev=out_dev,
                monitor_router=None,
                threshold=args.threshold,
                silence_ms=args.silence_ms,
                voices_dir=voices_dir,
                speed=args.speed,
                speaker_id=args.speaker_id,
                hear_generated=False,
                volume=prefs["volume"],
                mic=mic,
                direct_output=True,  # audio goes straight to speakers: echo guard on, H not applicable
            )
            say("bye 👋", C.DIM)
            return

        say("\n" + "─" * 52, C.DIM)
        say(f" vshift  ·  voice: {C.MAGENTA}{args.voice}{C.RESET}", C.BOLD)
        say(" Enter = start/stop recording  ·  q + Enter = quit", C.DIM)
        say("─" * 52 + "\n", C.DIM)

        while True:
            try:
                cmd = input(f"{C.CYAN}▶{C.RESET} press Enter to talk ('q' to quit): ")
            except (EOFError, KeyboardInterrupt):
                break
            if cmd.strip().lower() in ("q", "quit", "exit"):
                break

            audio = record_until_enter(mic=mic)
            if audio.size == 0:
                say("  (nothing recorded)", C.DIM)
                continue

            t0 = time.time()
            text = transcribe(stt_model, audio)
            if not text:
                say("  (didn't catch any speech)", C.DIM)
                continue
            say(f"  heard: \"{text}\"  ({time.time() - t0:.1f}s)", C.YELLOW)

            t0 = time.time()
            try:
                out_audio, out_rate = tts_client.synthesize(
                    text, make_synthesis_config(args.speed, args.speaker_id)
                )
                say(f"  synthesized ({time.time() - t0:.1f}s), playing...", C.GREEN)
                play_int16(apply_volume(out_audio, prefs["volume"]), out_rate, device=out_dev)
            except RuntimeError as exc:
                err(f"  {exc}")
    finally:
        stt_model.close()
        tts_client.close()

    say("\nbye 👋", C.DIM)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def cli_float_range(min_value: float, max_value: float):
    def parse(value: str) -> float:
        try:
            result = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("must be a number") from exc
        if not min_value <= result <= max_value:
            raise argparse.ArgumentTypeError(f"must be between {min_value:g} and {max_value:g}")
        return result
    return parse


def cli_int_range(min_value: int, max_value: int):
    def parse(value: str) -> int:
        try:
            result = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("must be an integer") from exc
        if not min_value <= result <= max_value:
            raise argparse.ArgumentTypeError(f"must be between {min_value} and {max_value}")
        return result
    return parse


def cmd_cleanup(args: argparse.Namespace) -> None:
    """Remove only the virtual audio objects created by vshift."""
    if shutil.which("pactl") is None:
        raise RuntimeError("pactl not found; this command needs PipeWire/PulseAudio")
    removed_source = False
    removed_sink = False
    if source_exists(args.mic_name):
        remove_remap_source(args.mic_name)
        removed_source = True
    if sink_exists(args.sink_name):
        remove_null_sink(args.sink_name)
        removed_sink = True
    # Also remove a stale generated-voice monitor loopback for this sink.
    temp_router = MonitorRouter(args.sink_name, enabled=False)
    stale = temp_router._find_existing()
    if stale is not None:
        subprocess.run(["pactl", "unload-module", str(stale)], capture_output=True)
        temp_router.module_id = None
    say(f"✓ Cleanup complete: mic={'removed' if removed_source else 'not found'}, "
        f"sink={'removed' if removed_sink else 'not found'}", C.GREEN)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vshift",
        description="Linux terminal voice changer: microphone → Whisper → Piper → virtual mic.",
        epilog="Try: vshift discord  |  vshift devices  |  vshift voices --installed  |  vshift doctor",
    )
    p.add_argument("--version", action="version", version=f"vshift {VERSION}")
    sub = p.add_subparsers(dest="command", required=True)

    p_version = sub.add_parser("version", help="Show the vshift version")
    p_version.set_defaults(func=cmd_version)

    p_config = sub.add_parser("config", help="Show or reset saved preferences")
    p_config.add_argument("--json", action="store_true", help="Print settings as JSON")
    p_config.add_argument("--reset", action="store_true", help="Delete saved settings")
    p_config.set_defaults(func=cmd_config)

    p_doctor = sub.add_parser("doctor", help="Check dependencies, audio tools, and saved voice")
    p_doctor.add_argument("--voices-dir", default=str(DEFAULT_VOICES_DIR))
    p_doctor.set_defaults(func=cmd_doctor)

    p_run = sub.add_parser("run", help="Start the interactive voice changer")
    p_run.add_argument("--voice", default=None,
                        help=f"Voice id (default: {DEFAULT_VOICE}): a Piper id, or kokoro:<name> "
                             "such as kokoro:af_heart. See `vshift voices`.")
    p_run.add_argument("--stt-model", default=None,
                        help=f"faster-whisper model size (default: {DEFAULT_STT_MODEL}). "
                             "tiny.en is faster/less accurate, small.en is slower/more accurate.")
    p_run.add_argument("--speed", type=cli_float_range(0.25, 4.0), default=None,
                        help="Piper length_scale: >1 slower, <1 faster (default: voice default)")
    p_run.add_argument("--speaker-id", type=cli_int_range(0, 1024), default=None,
                        help="Speaker id for multi-speaker voices (default: 0)")
    p_run.add_argument("--voices-dir", default=str(DEFAULT_VOICES_DIR),
                        help=f"Where voice models are stored (default: {DEFAULT_VOICES_DIR})")
    p_run.add_argument("--mic", default=None, metavar="NAME",
                        help="Microphone to listen to: part of its name from `vshift mics` "
                             "(e.g. fifine), a device number, or 'default'. Saved for next time.")
    p_run.add_argument("--input-device", type=int, default=None,
                        help="Input device index, see `vshift devices`")
    p_run.add_argument("--output-device", type=int, default=None,
                        help="Output device index, see `vshift devices` "
                             "(point this at a virtual mic sink to use vshift in calls/games)")
    p_run.add_argument("--continuous", action="store_true",
                        help="Hands-free mode: auto-detect speech instead of pressing Enter")
    p_run.add_argument("--threshold", type=cli_float_range(25.0, 2000.0), default=None,
                        help="Continuous mode: mic RMS level that counts as speech (default: 400)")
    p_run.add_argument("--silence-ms", type=cli_int_range(200, 3000), default=None,
                        help="Continuous mode: ms of silence that ends an utterance (default: 700)")
    add_translate_args(p_run)
    p_run.set_defaults(func=cmd_run)

    p_discord = sub.add_parser(
        "discord",
        help="One-command real-time mode: auto-creates a virtual mic and streams into it",
    )
    p_discord.add_argument("--voice", default=None,
                            help=f"Voice id (default: {DEFAULT_VOICE}): a Piper id, or kokoro:<name> "
                             "such as kokoro:af_heart. See `vshift voices`.")
    p_discord.add_argument("--stt-model", default=None,
                            help=f"faster-whisper model size (default: {DEFAULT_STT_MODEL})")
    p_discord.add_argument("--speed", type=cli_float_range(0.25, 4.0), default=None,
                            help="Piper length_scale: >1 slower, <1 faster")
    p_discord.add_argument("--speaker-id", type=cli_int_range(0, 1024), default=None,
                            help="Speaker id for multi-speaker voices")
    p_discord.add_argument("--voices-dir", default=str(DEFAULT_VOICES_DIR))
    p_discord.add_argument("--mic", default=None, metavar="NAME",
                        help="Microphone to listen to: part of its name from `vshift mics` "
                             "(e.g. fifine), a device number, or 'default'. Saved for next time.")
    p_discord.add_argument("--input-device", type=int, default=None,
                            help="Real mic device index, see `vshift devices` (default: system default)")
    p_discord.add_argument("--sink-name", default="vshift_mic",
                            help="Name of the virtual sink vshift plays into (default: vshift_mic)")
    p_discord.add_argument("--mic-name", default="vshift_mic_input",
                            help="Name of the virtual mic Discord/games see as an input "
                                 "device (default: vshift_mic_input)")
    p_discord.add_argument("--keep-sink", action="store_true",
                            help="Don't remove the virtual sink/mic on exit")
    monitor_group = p_discord.add_mutually_exclusive_group()
    monitor_group.add_argument("--monitor", dest="hear_generated", action="store_true", help="Hear the generated voice locally")
    monitor_group.add_argument("--no-monitor", dest="hear_generated", action="store_false", help="Disable local generated-voice monitoring")
    p_discord.set_defaults(hear_generated=None)
    p_discord.add_argument("--threshold", type=cli_float_range(25.0, 2000.0), default=None,
                            help="Mic RMS level that counts as speech (default: 400)")
    p_discord.add_argument("--silence-ms", type=cli_int_range(200, 3000), default=None,
                            help="ms of silence that ends an utterance (default: 700)")
    add_translate_args(p_discord)
    p_discord.set_defaults(func=cmd_discord)

    p_test = sub.add_parser(
        "test-mic",
        help="Play a test tone through the virtual mic to verify routing, no STT/TTS involved",
    )
    p_test.add_argument("--sink-name", default="vshift_mic")
    p_test.add_argument("--mic-name", default="vshift_mic_input")
    p_test.add_argument("--keep-sink", action="store_true",
                         help="Don't remove the virtual sink/mic on exit")
    p_test.add_argument("--seconds", type=cli_float_range(0.1, 120.0), default=3.0,
                         help="Length of the test tone (default: 3s)")
    p_test.set_defaults(func=cmd_test_mic)

    p_cleanup = sub.add_parser("cleanup", help="Remove vshift's virtual mic, sink, and stale monitor")
    p_cleanup.add_argument("--sink-name", default="vshift_mic")
    p_cleanup.add_argument("--mic-name", default="vshift_mic_input")
    p_cleanup.set_defaults(func=cmd_cleanup)

    p_voices = sub.add_parser("voices", help="List curated voices or downloaded models")
    p_voices.add_argument("--installed", action="store_true", help="List voices already downloaded")
    p_voices.add_argument("--voices-dir", default=str(DEFAULT_VOICES_DIR))
    p_voices.set_defaults(func=cmd_voices)

    p_bench = sub.add_parser(
        "tts-bench", help="Measure how fast each voice engine runs on this machine")
    p_bench.add_argument("--voice", action="append", default=None,
                         help="Voice id to test (repeat for several). Default: your saved "
                              "voice and kokoro:af_heart.")
    p_bench.add_argument("--voices-dir", default=str(DEFAULT_VOICES_DIR))
    p_bench.set_defaults(func=cmd_tts_bench)

    p_mics = sub.add_parser("mics", help="List the microphones vshift can listen to")
    p_mics.add_argument("--json", action="store_true", help="Print as JSON")
    p_mics.set_defaults(func=cmd_mics)

    p_devices = sub.add_parser("devices", help="List audio input/output devices")
    p_devices.add_argument("--json", action="store_true", help="Print device data as JSON")
    p_devices.set_defaults(func=cmd_devices)

    p_dl = sub.add_parser("download", help="Pre-download a voice without starting the CLI")
    p_dl.add_argument("voice", help="Piper voice id, e.g. en_US-amy-medium")
    p_dl.add_argument("--voices-dir", default=str(DEFAULT_VOICES_DIR))
    p_dl.add_argument("--force", action="store_true",
                      help="Delete the existing files and download again (fixes a damaged voice)")
    p_dl.set_defaults(func=cmd_download)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if sys.version_info >= (3, 14):
        say(
            "Warning: Python 3.14 is known to trigger semaphore cleanup warnings "
            "with some faster-whisper/CTranslate2 builds. vshift isolates Whisper in "
            "a worker process; Python 3.13/3.12 is still recommended.",
            C.YELLOW,
        )
    try:
        args.func(args)
    except KeyboardInterrupt:
        print()
        sys.exit(130)
    except (ValueError, RuntimeError, OSError) as exc:
        err(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Isolated TTS worker used by vshift (Piper and Kokoro engines).

All TTS/ONNX Runtime work happens here so a native abort cannot take down
vshift's microphone/UI process. Requests are JSON lines; audio is base64 in the
response to keep the IPC framing simple and robust.

Voice ids pick the engine: "kokoro:af_heart" is a Kokoro voice, anything else
is a Piper voice id such as "en_US-lessac-medium". Engine libraries are
imported lazily, so a Piper-only session never loads Kokoro and vice versa.
"""
from __future__ import annotations

import argparse
import base64
import gc
import traceback
import json
import os
import sys
from pathlib import Path

# Reduce native thread pressure. These are inherited from the parent too, but
# setting defaults here also protects direct launches of this worker.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")

KOKORO_PREFIX = "kokoro:"
# Only the English voices: Kokoro's other languages need phonemizers vshift
# doesn't ship, and its speech-to-text side is English.
KOKORO_ENGLISH_PREFIXES = ("af_", "am_", "bf_", "bm_")
KOKORO_MIN_SPEED, KOKORO_MAX_SPEED = 0.5, 2.0  # kokoro-onnx rejects values outside this
KOKORO_RATE = 24000


def send(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Piper
# ---------------------------------------------------------------------------
class PiperEngine:
    kind = "piper"

    def __init__(self, voices_dir: Path, voice_id: str):
        if not voice_id or any(ch in voice_id for ch in "/\\") or ".." in voice_id:
            raise ValueError("Invalid Piper voice id")
        path = voices_dir / f"{voice_id}.onnx"
        if not path.exists():
            raise FileNotFoundError(f"Voice model not found: {path}")
        from piper import PiperVoice
        try:
            self.voice = PiperVoice.load(str(path))
        except Exception as exc:
            if "protobuf" in str(exc).lower():
                # A cut-off download (older vshift versions could leave one).
                raise RuntimeError(
                    f"the voice file {path.name} is damaged or incomplete. "
                    f"Re-download it with: vshift download {voice_id} --force") from exc
            raise

    def synthesize(self, text, speaker_id, length_scale, noise_scale, noise_w_scale, volume):
        from piper import SynthesisConfig
        kwargs = {}
        if speaker_id is not None:
            kwargs["speaker_id"] = int(speaker_id)
        if length_scale is not None:
            kwargs["length_scale"] = float(length_scale)
        if noise_scale is not None:
            kwargs["noise_scale"] = float(noise_scale)
        if noise_w_scale is not None:
            kwargs["noise_w_scale"] = float(noise_w_scale)
        if volume is not None:
            kwargs["volume"] = float(volume)
        config = SynthesisConfig(**kwargs)
        chunks = []
        for chunk in self.voice.synthesize(text, syn_config=config):
            chunks.append(chunk)
        if not chunks:
            return b"", 22050
        rate = int(chunks[0].sample_rate)
        raw = b"".join(c.audio_int16_bytes for c in chunks)
        return raw, rate


# ---------------------------------------------------------------------------
# Kokoro
# ---------------------------------------------------------------------------
def _ort_threads() -> int:
    """Threads for Kokoro's ONNX session. Speech-to-text runs at the same time
    in another process, so don't take every core by default."""
    raw = os.environ.get("VSHIFT_TTS_THREADS", "").strip()
    if raw.isdigit() and int(raw) >= 1:
        return int(raw)
    return max(1, min(4, os.cpu_count() or 1))


class KokoroEngine:
    kind = "kokoro"

    def __init__(self, model_path: Path, voices_path: Path, voice_name: str):
        for p in (model_path, voices_path):
            if not p.exists():
                raise FileNotFoundError(f"Kokoro file not found: {p}")
        try:
            import onnxruntime as ort
            from kokoro_onnx import Kokoro
        except ImportError as exc:
            raise RuntimeError(
                f"the kokoro-onnx package is not installed ({exc}). Install it with: "
                f"{sys.executable} -m pip install 'kokoro-onnx>=0.6,<1'"
            ) from exc
        options = ort.SessionOptions()
        options.intra_op_num_threads = _ort_threads()
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
        self.kokoro = Kokoro.from_session(session, str(voices_path))
        self.voice = ""
        self.lang = "en-us"
        self.set_voice(voice_name)

    def set_voice(self, name: str) -> None:
        """Switching Kokoro voices only swaps a style vector: no model reload."""
        if not name.startswith(KOKORO_ENGLISH_PREFIXES):
            raise ValueError(f"'{name}' is not an English Kokoro voice "
                             f"(expected a name starting with af_, am_, bf_ or bm_)")
        if name not in self.kokoro.get_voices():
            raise ValueError(f"Kokoro voice '{name}' not found")
        self.voice = name
        self.lang = "en-gb" if name.startswith("b") else "en-us"

    def synthesize(self, text, speaker_id, length_scale, noise_scale, noise_w_scale, volume):
        # vshift's speed is Piper's length_scale (lower = faster); Kokoro's is a
        # rate (higher = faster). speaker_id/noise/volume don't apply here.
        speed = 1.0
        if length_scale is not None and float(length_scale) > 0:
            speed = 1.0 / float(length_scale)
        speed = min(KOKORO_MAX_SPEED, max(KOKORO_MIN_SPEED, speed))
        import numpy as np
        if not any(ch.isalnum() for ch in text):
            # Punctuation-only (Whisper artefacts like "..." or "?!") would
            # otherwise be voiced as a stray blip in the call.
            return b"", KOKORO_RATE
        try:
            samples, rate = self.kokoro.create(
                text, voice=self.voice, speed=speed, lang=self.lang)
        except ValueError as exc:
            # Whisper sometimes hands us punctuation or whitespace only.
            if "no phonemes" in str(exc):
                return b"", KOKORO_RATE
            raise
        pcm = np.clip(np.asarray(samples, dtype=np.float32) * 32767.0, -32768, 32767)
        return pcm.astype(np.int16).tobytes(), int(rate)


# ---------------------------------------------------------------------------
# Engine selection
# ---------------------------------------------------------------------------
def load_engine(voice_id: str, voices_dir: Path, kokoro_model: Path | None,
                kokoro_voices: Path | None, previous=None):
    if voice_id.startswith(KOKORO_PREFIX):
        name = voice_id[len(KOKORO_PREFIX):]
        if isinstance(previous, KokoroEngine):
            previous.set_voice(name)  # raises before changing anything if invalid
            return previous
        if kokoro_model is None or kokoro_voices is None:
            raise FileNotFoundError("Kokoro model paths were not provided")
        return KokoroEngine(kokoro_model, kokoro_voices, name)
    return PiperEngine(voices_dir, voice_id)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voices-dir", required=True)
    parser.add_argument("--voice", required=True)
    parser.add_argument("--speed", default="")
    parser.add_argument("--speaker-id", default="")
    parser.add_argument("--kokoro-model", default="")
    parser.add_argument("--kokoro-voices", default="")
    args = parser.parse_args()

    voices_dir = Path(args.voices_dir)
    kokoro_model = Path(args.kokoro_model) if args.kokoro_model else None
    kokoro_voices = Path(args.kokoro_voices) if args.kokoro_voices else None
    speaker_id = int(args.speaker_id) if args.speaker_id else None
    speed = float(args.speed) if args.speed else None

    try:
        engine = load_engine(args.voice, voices_dir, kokoro_model, kokoro_voices)
    except Exception as exc:
        send({"type": "error", "message": f"voice load failed: {exc}"})
        return 1

    current_voice = args.voice
    send({"type": "ready", "voice": current_voice})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            kind = req.get("type")

            if kind == "quit":
                return 0

            if kind == "set_voice":
                requested = str(req.get("voice", "")).strip()
                if not requested:
                    send({"type": "error", "message": "empty voice id"})
                    continue
                new_engine = load_engine(
                    requested, voices_dir, kokoro_model, kokoro_voices, previous=engine)
                if new_engine is not engine:
                    old_engine = engine
                    engine = new_engine
                    del old_engine
                    gc.collect()
                current_voice = requested
                send({"type": "voice", "voice": current_voice})
                continue

            if kind == "synthesize":
                text = str(req.get("text", "")).strip()
                if not text:
                    send({"type": "audio", "audio": "", "sample_rate": 22050})
                    continue
                req_speaker = req.get("speaker_id")
                req_speed = req.get("length_scale")
                req_noise = req.get("noise_scale")
                req_noise_w = req.get("noise_w_scale")
                req_volume = req.get("volume")
                req_speaker = int(req_speaker) if req_speaker is not None else speaker_id
                req_speed = float(req_speed) if req_speed is not None else speed
                req_noise = float(req_noise) if req_noise is not None else None
                req_noise_w = float(req_noise_w) if req_noise_w is not None else None
                req_volume = float(req_volume) if req_volume is not None else None
                raw, rate = engine.synthesize(
                    text, req_speaker, req_speed, req_noise, req_noise_w, req_volume)
                send({
                    "type": "audio",
                    "audio": base64.b64encode(raw).decode("ascii"),
                    "sample_rate": rate,
                    "voice": current_voice,
                })
                continue

            send({"type": "error", "message": f"unknown request: {kind}"})
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            send({"type": "error", "message": f"{type(exc).__name__}: {exc}"})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

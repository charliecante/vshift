#!/usr/bin/env python3
"""Isolated faster-whisper worker used by vshift."""
from __future__ import annotations

import json
import struct
import sys

import numpy as np
from faster_whisper import WhisperModel


def send(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _detect_repeated_phrase_len(norm_words: list[str]) -> int | None:
    """Whisper occasionally decodes into a loop and outputs the same phrase
    back-to-back many times (a well-known degenerate-decode failure mode,
    distinct from the person actually repeating themselves). Detect the
    shortest repeating unit that accounts for most of the transcript so it can
    be collapsed to a single occurrence.

    Deliberately conservative so real speech is left alone: a single word must
    repeat 8+ times ("no no no no" and laughter like "ha ha ha ha" pass
    through untouched); a multi-word phrase must repeat 4+ times.
    """
    n = len(norm_words)
    if n < 4:
        return None
    for phrase_len in range(1, n // 4 + 1):
        min_reps = 8 if phrase_len == 1 else 4
        reps = n // phrase_len
        if reps < min_reps:
            continue
        phrase = norm_words[:phrase_len]
        matched = 0
        for r in range(reps):
            if norm_words[r * phrase_len:(r + 1) * phrase_len] == phrase:
                matched += 1
            else:
                break
        covered = matched * phrase_len
        if matched >= min_reps and covered / n >= 0.7:
            return phrase_len
    return None


def transcribe(model: WhisperModel, audio: np.ndarray, task: str = "transcribe",
               language: str | None = "en") -> tuple[str, str]:
    """Return (text, language). task "translate" turns speech in any language
    into English text (multilingual models only); language None means detect it."""
    segments, info = model.transcribe(
        audio,
        beam_size=3,
        task=task,
        language=language,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 150},
        condition_on_previous_text=False,
        no_speech_threshold=0.75,
        log_prob_threshold=-0.5,
        compression_ratio_threshold=2.4,
        temperature=0.0,
    )
    texts: list[str] = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        no_speech_prob = float(getattr(seg, "no_speech_prob", 0.0) or 0.0)
        avg_logprob = float(getattr(seg, "avg_logprob", 0.0) or 0.0)
        compression_ratio = float(getattr(seg, "compression_ratio", 0.0) or 0.0)
        if no_speech_prob >= 0.75 and avg_logprob < -1.0:
            continue
        # Whisper's own garbage/repetition heuristic: a segment whose text
        # compresses unusually well is almost always the same tokens looping.
        if compression_ratio > 2.4:
            continue
        texts.append(text)
    detected = str(getattr(info, "language", "") or language or "")
    result = " ".join(texts).strip()
    if not result:
        return "", detected

    raw_words = result.split()
    norm_words = [w.strip(".,!?;:'\"").lower() for w in raw_words]

    phrase_len = _detect_repeated_phrase_len(norm_words)
    if phrase_len is not None:
        raw_words = raw_words[:phrase_len]
        norm_words = norm_words[:phrase_len]
        result = " ".join(raw_words)

    if norm_words and all(w in {"ok", "okay"} for w in norm_words):
        return "", detected
    return result, detected


def read_exact(source, size: int) -> bytes | None:
    data = source.read(size)
    return data if len(data) == size else None


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: stt_worker.py MODEL", file=sys.stderr)
        return 2
    model_name = sys.argv[1]
    try:
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
    except Exception as e:
        send({"type": "error", "message": f"model load failed: {e}"})
        return 1

    # Translation needs a multilingual model; tell vshift so it can say so.
    send({"type": "ready", "multilingual": bool(getattr(model.model, "is_multilingual", False))})
    source = sys.stdin.buffer
    # Each request: u32 length + JSON options ({"task", "language"}), then
    # u32 length + float32 mono 16 kHz audio. A length of 0xFFFFFFFF quits.
    while True:
        header = source.read(4)
        if not header or header == b"\xff\xff\xff\xff":
            return 0
        if len(header) != 4:
            return 1
        (opt_size,) = struct.unpack("!I", header)
        raw_opts = read_exact(source, opt_size)
        size_raw = read_exact(source, 4) if raw_opts is not None else None
        if size_raw is None:
            return 1
        (size,) = struct.unpack("!I", size_raw)
        payload = read_exact(source, size)
        if payload is None:
            return 1
        try:
            opts = json.loads(raw_opts.decode("utf-8") or "{}")
            task = "translate" if opts.get("task") == "translate" else "transcribe"
            language = opts.get("language") or None
            audio = np.frombuffer(payload, dtype=np.float32)
            text, detected = transcribe(model, audio, task, language)
            send({"type": "result", "text": text, "language": detected})
        except Exception as e:
            send({"type": "error", "message": str(e)})

if __name__ == "__main__":
    raise SystemExit(main())

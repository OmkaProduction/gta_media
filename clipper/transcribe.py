"""Word-level transcription of short audio windows with faster-whisper."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass
class Word:
    start: float  # absolute seconds in the VOD
    end: float
    text: str


@lru_cache(maxsize=1)
def _model():
    from faster_whisper import WhisperModel

    size = os.environ.get("WHISPER_MODEL", "small")
    return WhisperModel(size, device="cpu", compute_type="int8")


def transcribe(audio: Path, offset: float, language: str = "en") -> list[Word]:
    segments, _ = _model().transcribe(
        str(audio), language=language, word_timestamps=True, vad_filter=True, beam_size=1,
    )
    words = []
    for seg in segments:
        for w in seg.words or []:
            text = w.word.strip()
            if text:
                words.append(Word(start=offset + w.start, end=offset + w.end, text=text))
    return words


def words_text(words: list[Word], start: float, end: float) -> str:
    return " ".join(w.text for w in words if w.end > start and w.start < end)

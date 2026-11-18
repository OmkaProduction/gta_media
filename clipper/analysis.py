"""Finds loud / excited moments in a long audio track.

Streamer highlights (laughing, shouting, sudden reactions) are louder than the
stream's own baseline, so we score every second by how far its loudness is above
the rolling median of the surrounding few minutes and keep the strongest peaks.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 8000


@dataclass
class Candidate:
    peak: float    # second of the strongest moment
    start: float   # window we transcribe and offer to the selector
    end: float
    score: float


def loudness_per_second(audio: Path) -> np.ndarray:
    """dB loudness of every second of audio, decoded in chunks via ffmpeg."""
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(audio), "-ac", "1", "-ar", str(SAMPLE_RATE),
         "-f", "s16le", "-"],
        stdout=subprocess.PIPE,
    )
    values = []
    chunk = SAMPLE_RATE * 2 * 60  # one minute of 16-bit samples
    leftover = b""
    assert proc.stdout is not None
    while True:
        data = proc.stdout.read(chunk)
        if not data:
            break
        data = leftover + data
        usable = len(data) - len(data) % (SAMPLE_RATE * 2)
        leftover = data[usable:]
        samples = np.frombuffer(data[:usable], dtype=np.int16).astype(np.float32) / 32768.0
        if samples.size:
            rms = np.sqrt(np.mean(samples.reshape(-1, SAMPLE_RATE) ** 2, axis=1))
            values.append(20 * np.log10(rms + 1e-6))
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed to decode {audio}")
    return np.concatenate(values) if values else np.zeros(0, dtype=np.float32)


def _rolling_median(x: np.ndarray, window: int) -> np.ndarray:
    # Median over coarse 10 s blocks keeps this fast on 10+ hour streams.
    block = 10
    n_blocks = int(np.ceil(len(x) / block))
    padded = np.pad(x, (0, n_blocks * block - len(x)), mode="edge")
    block_med = np.median(padded.reshape(n_blocks, block), axis=1)
    half = max(1, window // block // 2)
    out = np.empty(n_blocks)
    for i in range(n_blocks):
        out[i] = np.median(block_med[max(0, i - half): i + half + 1])
    return np.repeat(out, block)[: len(x)]


def find_candidates(
    loudness: np.ndarray,
    max_candidates: int = 25,
    min_gap: float = 90.0,
    before: float = 45.0,
    after: float = 25.0,
    baseline_window: int = 300,
    min_relative: float = 0.2,
) -> list[Candidate]:
    """Top loudness peaks, at least `min_gap` seconds apart, as transcription windows."""
    n = len(loudness)
    if n == 0:
        return []
    # Silence (muted mic, BRB screens) must not count as a baseline drop.
    clipped = np.maximum(loudness, -60.0)
    excess = clipped - _rolling_median(clipped, baseline_window)
    # A 4-second moving sum rewards sustained reactions over single clicks.
    kernel = np.ones(4)
    smooth = np.convolve(np.maximum(excess, 0), kernel, mode="same")

    order = np.argsort(smooth)[::-1]
    # Ignore background noise: a candidate must reach a fraction of the strongest peak.
    floor = smooth[order[0]] * min_relative if len(order) else 0.0
    chosen: list[int] = []
    for idx in order:
        if smooth[idx] <= 0 or smooth[idx] < floor or len(chosen) >= max_candidates:
            break
        if all(abs(idx - c) >= min_gap for c in chosen):
            chosen.append(int(idx))

    return sorted(
        (Candidate(peak=float(c), start=max(0.0, c - before), end=min(float(n), c + after),
                   score=float(smooth[c])) for c in chosen),
        key=lambda c: c.start,
    )


def cut_audio(audio: Path, start: float, end: float, dest: Path) -> Path:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.2f}", "-t", f"{end - start:.2f}",
         "-i", str(audio), "-ac", "1", "-ar", "16000", str(dest)],
        check=True,
    )
    return dest

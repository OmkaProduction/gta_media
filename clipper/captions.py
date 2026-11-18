"""Builds an ASS subtitle file: hook, streamer credit and word-by-word captions for a 1080x1920 clip."""
from __future__ import annotations

from .transcribe import Word

HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,DejaVu Sans,78,&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,6,2,2,80,80,560,1
Style: Hook,DejaVu Sans,70,&H0000F0FF,&H00FFFFFF,&H00000000,&HA0000000,-1,0,0,0,100,100,0,0,3,18,0,8,70,70,260,1
Style: Credit,DejaVu Sans,40,&H00FFFFFF,&H00FFFFFF,&H00000000,&H78000000,-1,0,0,0,100,100,0,0,3,10,0,8,60,60,120,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _ts(seconds: float) -> str:
    seconds = max(0.0, seconds)
    cs = int(round(seconds * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "(").replace("}", ")").replace("\n", " ")


def group_words(words: list[Word], max_words: int = 3, max_gap: float = 0.6, max_len: float = 1.6) -> list[list[Word]]:
    """Short caption chunks: a new chunk on a pause, after max_words or max_len seconds."""
    groups: list[list[Word]] = []
    for w in words:
        if (
            groups
            and len(groups[-1]) < max_words
            and w.start - groups[-1][-1].end <= max_gap
            and w.end - groups[-1][0].start <= max_len
        ):
            groups[-1].append(w)
        else:
            groups.append([w])
    return groups


def build_ass(words: list[Word], clip_start: float, clip_end: float, hook: str, credit: str,
              hook_seconds: float = 2.5) -> str:
    duration = clip_end - clip_start
    lines = [HEADER]
    lines.append(f"Dialogue: 2,{_ts(0)},{_ts(duration)},Credit,,0,0,0,,{_escape(credit)}")
    if hook:
        lines.append(f"Dialogue: 3,{_ts(0)},{_ts(min(hook_seconds, duration))},Hook,,0,0,0,,{_escape(hook.upper())}")
    inside = [w for w in words if w.end > clip_start and w.start < clip_end]
    groups = group_words(inside)
    for i, g in enumerate(groups):
        start = g[0].start - clip_start
        # keep each chunk on screen until the next one starts (no flicker), max +0.5 s
        nxt = groups[i + 1][0].start - clip_start if i + 1 < len(groups) else duration
        end = min(nxt, g[-1].end - clip_start + 0.5, duration)
        text = " ".join(w.text for w in g).upper()
        lines.append(f"Dialogue: 1,{_ts(start)},{_ts(end)},Caption,,0,0,0,,{_escape(text)}")
    return "\n".join(lines) + "\n"

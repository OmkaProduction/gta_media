"""Picks the best clips among loudness candidates.

With ANTHROPIC_API_KEY set, Claude reads each candidate's transcript and the
campaign rules and returns exact clip boundaries, a hook and a title. Without a
key (or if the call fails) we fall back to the loudest candidates.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

from .analysis import Candidate
from .config import Streamer
from .transcribe import Word, words_text

MODEL = os.environ.get("CLIPPER_MODEL", "claude-opus-5-5")


@dataclass
class ClipPlan:
    start: float
    end: float
    hook: str    # on-screen text for the first seconds
    title: str   # post title / first caption line
    reason: str


SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["clips"],
    "properties": {
        "clips": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["candidate", "start", "end", "hook", "title", "reason"],
                "properties": {
                    "candidate": {"type": "integer"},
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "hook": {"type": "string"},
                    "title": {"type": "string"},
                    "reason": {"type": "string"},
                },
            },
        }
    },
}

SYSTEM = """You are the editor of a short-form channel that clips GTA 6 livestreams for paid clipping campaigns.
You get loud moments from one stream, each with a timestamped transcript. Pick the moments most likely to go viral as 15-60 second vertical clips.

What works: a clear setup and payoff, a strong genuine reaction, a funny line, a fail or glitch, a surprising game detail, an exchange with chat.
Skip: moments that need context the viewer does not have, dead air, menu screens, long story cutscenes, ending or major story spoilers, NSFW content, and anything that makes the streamer look bad.

For each clip you choose:
- start/end: absolute seconds within the candidate's window. Start 1-3 seconds before the setup so the clip opens on action, and end right after the payoff.
- hook: on-screen text shown for the first 2 seconds, max 8 words, English, names the streamer, says what happens without lying about it.
- title: the post title, max 90 characters, English, no hashtags.
- reason: one short sentence on why it will hold viewers.
Never invent dialogue or events that are not in the transcript. Return fewer clips rather than weak ones."""


def _prompt(streamer: Streamer, candidates: list[Candidate], words: list[Word], count: int) -> str:
    parts = [
        f"Streamer: {streamer.channel} ({streamer.platform})",
        f"Campaign rules: {streamer.notes or 'none given'}",
        f"Clip length: {streamer.min_clip_seconds}-{streamer.max_clip_seconds} seconds.",
        f"Choose up to {count} clips.",
        "",
    ]
    for i, c in enumerate(candidates):
        lines = []
        for w in words:
            if c.start <= w.start < c.end:
                lines.append(f"[{w.start:.1f}] {w.text}")
        transcript = " ".join(lines) or "(no speech detected)"
        parts.append(
            f"Candidate {i}: window {c.start:.1f}-{c.end:.1f}s, loudest at {c.peak:.1f}s, "
            f"loudness score {c.score:.1f}\n{transcript}\n"
        )
    return "\n".join(parts)


def _clamp(plan: ClipPlan, cand: Candidate, streamer: Streamer) -> ClipPlan | None:
    start = max(cand.start, min(plan.start, cand.end))
    end = max(cand.start, min(plan.end, cand.end))
    if end - start > streamer.max_clip_seconds:
        end = start + streamer.max_clip_seconds
    if end - start < streamer.min_clip_seconds:
        # widen around the middle, staying inside the window where possible
        mid = (start + end) / 2
        half = streamer.min_clip_seconds / 2
        start, end = max(0.0, mid - half), mid + half
    return ClipPlan(start=start, end=end, hook=plan.hook.strip(), title=plan.title.strip(), reason=plan.reason)


def fallback_plans(streamer: Streamer, candidates: list[Candidate], words: list[Word], count: int) -> list[ClipPlan]:
    best = sorted(candidates, key=lambda c: c.score, reverse=True)[:count]
    plans = []
    for c in best:
        start = max(c.start, c.peak - 25)
        end = min(c.end, start + min(40, streamer.max_clip_seconds))
        text = words_text(words, start, end)
        plans.append(ClipPlan(
            start=start, end=end,
            hook=f"{streamer.channel} in GTA 6",
            title=(text[:80] + "...") if len(text) > 80 else (text or f"{streamer.channel} GTA 6 moment"),
            reason="loudest moment (no AI selection)",
        ))
    return sorted(plans, key=lambda p: p.start)


def select_clips(streamer: Streamer, candidates: list[Candidate], words: list[Word], count: int) -> list[ClipPlan]:
    if not candidates:
        return []
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set: using loudness-only selection")
        return fallback_plans(streamer, candidates, words, count)

    import anthropic

    client = anthropic.Anthropic()
    try:
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content": _prompt(streamer, candidates, words, count)}],
        )
    except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError) as exc:
        print(f"Claude unavailable ({exc}); using loudness-only selection")
        return fallback_plans(streamer, candidates, words, count)

    if response.stop_reason == "refusal":
        print("Claude declined this transcript; using loudness-only selection")
        return fallback_plans(streamer, candidates, words, count)
    text = "".join(b.text for b in response.content if b.type == "text")
    try:
        raw = json.loads(text)["clips"]
    except (json.JSONDecodeError, KeyError) as exc:
        print(f"Could not parse Claude's answer ({exc}); using loudness-only selection")
        return fallback_plans(streamer, candidates, words, count)

    plans = []
    for item in raw[:count]:
        idx = item.get("candidate")
        if not isinstance(idx, int) or not 0 <= idx < len(candidates):
            continue
        plan = _clamp(ClipPlan(item["start"], item["end"], item["hook"], item["title"], item["reason"]),
                      candidates[idx], streamer)
        if plan:
            plans.append(plan)
    return sorted(plans, key=lambda p: p.start)

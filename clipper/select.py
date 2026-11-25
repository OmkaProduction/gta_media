"""Picks the best clips among loudness candidates.

With ANTHROPIC_API_KEY set, Claude reads each candidate's transcript and the
campaign rules and returns exact clip boundaries, a hook and a title. Without a
key (or if the call fails) we fall back to the loudest candidates.
"""
from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path

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
    punch: float | None = None   # absolute VOD second of the strongest reaction (the edit zooms in there)


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
                "required": ["candidate", "start", "end", "punch", "hook", "title", "reason"],
                "properties": {
                    "candidate": {"type": "integer"},
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "punch": {"type": "number"},
                    "hook": {"type": "string"},
                    "title": {"type": "string"},
                    "reason": {"type": "string"},
                },
            },
        }
    },
}

SYSTEM = """You are the editor of a short-form channel that clips GTA 6 livestreams for paid clipping campaigns.
You get the moments from one stream where the streamer got loud and/or chat exploded, each with a timestamped transcript.
A chat spike (many more messages than usual) means viewers found the moment exciting: prefer those moments. Pick the moments most likely to go viral as 15-60 second vertical clips.

Every clip must be GTA 6 content: GTA 6 gameplay on screen, or the streamer reacting to what just happened in GTA 6.
Each candidate comes with a still frame from its loudest second. Skip any candidate whose frame shows something else
(a browser, a sports game, another game, a just-chatting or loading screen) or whose talk is clearly not about GTA 6,
however funny it is. Campaigns only pay for GTA 6 clips, so return no clip rather than an off-topic one.

Hundreds of other clippers cut the same streams for the same campaigns, and YouTube buries near-identical uploads.
So do not just take the single biggest, most obvious moment of the stream that everyone else will clip too.
Prefer moments with their own angle: a funny exchange with chat, an odd game detail, a running joke, a small fail
with a great reaction, a clever or surprising line. A chat spike still matters: it shows viewers cared.

What works: a clear setup and payoff, a strong genuine reaction, a funny line, a fail or glitch, a surprising game detail, an exchange with chat.

Reference for the kind of moment we want most (a GTA 6 reel that got 1.8M views in a day): a car drives toward a
"tunnel" under a bridge, the tunnel turns out to be a painted mural, and the car crashes into the wall like in a
cartoon. Caption: "Nice Work, Rockstar!!! You even made the tunnels untrustworthy....." It works because the game
itself does something funny that a viewer gets in one second without any context: a visual gag, a fail, absurd physics,
a game detail so realistic or so silly it makes you ask "did Rockstar really do that?". The payoff is visible on screen,
not only in what the streamer says, and the whole thing fits in about 15-25 seconds. Rank this kind of on-screen moment
(with the streamer's reaction to it) above moments that are funny only because of talk with chat.

Skip: moments that need context the viewer does not have, dead air, menu screens, long story cutscenes, ending or major story spoilers, NSFW content, and anything that makes the streamer look bad.

For each clip you choose:
- start/end: absolute seconds within the candidate's window. Start 1-3 seconds before the setup so the clip opens on action, and end right after the payoff.
- punch: the absolute second of the strongest reaction or payoff inside the clip; the edit zooms in on it.
- hook: on-screen text shown for the first 2 seconds, max 7 words, English. It must make a viewer stay: a question,
  a stake or a tease of the payoff in our own words (not a plain description), names the streamer, never lies.
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
            f"score {c.score:.1f}"
            + (f", chat activity {c.chat:.1f}x the usual rate" if c.chat else "")
            + f"\n{transcript}\n"
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
    punch = plan.punch if plan.punch is not None and start < plan.punch < end else cand.peak
    punch = punch if start < punch < end else None
    return ClipPlan(start=start, end=end, hook=plan.hook.strip(), title=plan.title.strip(), reason=plan.reason,
                    punch=punch)


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
            punch=c.peak if start < c.peak < end else None,
        ))
    return sorted(plans, key=lambda p: p.start)


def _content(prompt: str, frames: list[Path | None] | None) -> list[dict] | str:
    if not frames or not any(frames):
        return prompt
    blocks: list[dict] = [{"type": "text", "text": prompt}]
    for i, frame in enumerate(frames):
        if frame is None:
            blocks.append({"type": "text", "text": f"Candidate {i}: no frame available."})
            continue
        blocks.append({"type": "text", "text": f"Candidate {i}: frame at its loudest second."})
        blocks.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/jpeg",
            "data": base64.b64encode(Path(frame).read_bytes()).decode("ascii"),
        }})
    return blocks


def select_clips(streamer: Streamer, candidates: list[Candidate], words: list[Word], count: int,
                 frames: list[Path | None] | None = None) -> list[ClipPlan]:
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
            messages=[{"role": "user", "content": _content(_prompt(streamer, candidates, words, count), frames)}],
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
        plan = _clamp(ClipPlan(item["start"], item["end"], item["hook"], item["title"], item["reason"],
                                item.get("punch")),
                      candidates[idx], streamer)
        if plan:
            plans.append(plan)
    return sorted(plans, key=lambda p: p.start)

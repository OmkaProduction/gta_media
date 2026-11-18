"""Post captions and the review sheet for each batch."""
from __future__ import annotations

from .config import Streamer
from .select import ClipPlan


def caption(streamer: Streamer, plan: ClipPlan) -> str:
    """Caption text to paste when posting. Campaign hashtags go first so they are never cut off."""
    tags = " ".join(streamer.hashtags)
    return f"{plan.title}\n\n{tags}\n{streamer.cta}"


def _hms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def review_entry(streamer: Streamer, vod_url: str, plan: ClipPlan, filename: str) -> str:
    return (
        f"## {filename}\n\n"
        f"- Streamer: {streamer.channel} ({streamer.platform})\n"
        f"- Source: {vod_url} at {_hms(plan.start)} - {_hms(plan.end)} ({plan.end - plan.start:.0f} s)\n"
        f"- Why: {plan.reason}\n\n"
        f"Caption to paste:\n\n```\n{caption(streamer, plan)}\n```\n\n"
        "Checklist before posting: streamer in focus, no story spoiler, no NSFW, captions match the audio.\n"
        "After posting: submit the post link in Content Rewards within 30 minutes.\n"
    )

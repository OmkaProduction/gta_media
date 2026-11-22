"""Chat activity over a VOD: how many chat messages per second viewers sent at each moment.

The most interesting moments of a stream are when chat explodes. Neither platform offers a
cheap per-second count, so we sample the chat replay every SAMPLE_EVERY seconds and measure
the message rate there:
- Twitch: the public GQL chat-replay query (the one the Twitch player uses), from an offset.
- Kick: the channel messages endpoint, which returns the messages of a few seconds after a time.
Every failure is soft: without chat data the clip picker falls back to loudness alone.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .sources import USER_AGENT, Vod

SAMPLE_EVERY = 20          # seconds between samples
TWITCH_CLIENT_ID = "kimne78kx3ncx6brgo4mv6wki5h1ko"
TWITCH_COMMENTS_HASH = "b70a3591ff0f4e0313d126c6a1502d79a1c02baebb288227c582044aa76adf6a"
KICK_WINDOW = 5.0          # the Kick endpoint returns about 5 seconds of messages


def _post_json(url: str, body, headers: dict) -> object:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"User-Agent": USER_AGENT, "Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.load(resp)


def _get_json(url: str) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.load(resp)


def twitch_rate(video_id: str, offset: float, fetch=_post_json) -> float:
    """Messages per second in the chat replay right after `offset` seconds."""
    answer = fetch("https://gql.twitch.tv/gql", [{
        "operationName": "VideoCommentsByOffsetOrCursor",
        "variables": {"videoID": video_id, "contentOffsetSeconds": int(offset)},
        "extensions": {"persistedQuery": {"version": 1, "sha256Hash": TWITCH_COMMENTS_HASH}},
    }], {"Client-Id": TWITCH_CLIENT_ID})
    edges = answer[0]["data"]["video"]["comments"]["edges"]
    times = [e["node"]["contentOffsetSeconds"] for e in edges]
    times = [t for t in times if t >= offset]
    if not times:
        return 0.0
    # One page holds a fixed number of messages: the busier chat is, the shorter the span.
    span = max(max(times) - offset, 1.0)
    if len(times) < 5:          # sparse chat: the page reaches far ahead
        span = max(span, SAMPLE_EVERY)
    return len(times) / span


def kick_rate(channel_id: int, vod_start: dt.datetime, offset: float, fetch=_get_json) -> float:
    at = (vod_start + dt.timedelta(seconds=offset)).astimezone(dt.timezone.utc)
    answer = fetch(f"https://kick.com/api/v2/channels/{channel_id}/messages"
                   f"?start_time={at.strftime('%Y-%m-%dT%H:%M:%S.000Z')}")
    messages = ((answer or {}).get("data") or {}).get("messages") or []
    return len(messages) / KICK_WINDOW


def _rate_function(vod: Vod):
    if vod.chat.get("kick_channel_id") and vod.chat.get("start_time"):
        start = dt.datetime.fromisoformat(str(vod.chat["start_time"]).replace("Z", "+00:00"))
        if start.tzinfo is None:
            start = start.replace(tzinfo=dt.timezone.utc)
        return lambda t: kick_rate(int(vod.chat["kick_channel_id"]), start, t)
    m = re.search(r"twitch\.tv/videos/(\d+)", vod.url) or re.fullmatch(r"v?(\d+)", vod.vod_id)
    if "twitch" in vod.url and m:
        return lambda t: twitch_rate(m.group(1), t)
    return None


def activity(vod: Vod, start: float, seconds: int, rate=None, workers: int = 8) -> np.ndarray | None:
    """Chat messages per second for each second of [start, start + seconds), or None."""
    rate = rate or _rate_function(vod)
    if rate is None or seconds <= 0:
        return None
    offsets = np.arange(0, seconds, SAMPLE_EVERY, dtype=float)

    def one(t):
        try:
            return rate(start + t)
        except Exception:
            return None

    with ThreadPoolExecutor(workers) as pool:
        values = list(pool.map(one, offsets))
    ok = [v is not None for v in values]
    if sum(ok) < max(3, 0.7 * len(values)):
        print(f"chat replay unavailable ({sum(ok)}/{len(values)} samples); using loudness only")
        return None
    xs = offsets[ok]
    ys = np.array([v for v in values if v is not None], dtype=float)
    if not ys.any():
        print("chat replay is empty; using loudness only")
        return None
    return np.interp(np.arange(seconds, dtype=float), xs + SAMPLE_EVERY / 2, ys)

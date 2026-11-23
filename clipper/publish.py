"""Queues finished clips in Buffer: python -m clipper.publish OUT_DIR FOLDER

OUT_DIR holds the rendered clips (<date>-<streamer>-NN-<slug>.mp4) and their captions (.txt).
FOLDER is where scripts/push_clips.sh put them on the `clips` branch (run-<run id>), so the
public URL of a clip is <base_url>/<FOLDER>/<file>.
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path

import yaml

from .config import ROOT

STATE = ROOT / "state" / "processed.json"

CONFIG = ROOT / "config" / "publish.yaml"
API_URL = "https://api.buffer.com"
GAMING_CATEGORY = "20"  # YouTube category "Gaming"

MUTATION = """
mutation CreatePost($input: CreatePostInput!) {
  createPost(input: $input) {
    ... on PostActionSuccess { post { id } }
    ... on MutationError { message }
  }
}
"""


def load_config(path: Path = CONFIG) -> dict:
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("buffer", {})


def streamer_of(filename: str) -> str:
    m = re.match(r"\d{8}-([a-z0-9_]+)-\d{2}-", filename)
    return m.group(1) if m else ""


def post_input(channel: dict, video_url: str, caption: str, due_at: str | None = None) -> dict:
    title = caption.splitlines()[0].strip() if caption.strip() else "GTA 6 clip"
    service = channel["service"]
    metadata: dict = {}
    if service == "youtube":
        metadata["youtube"] = {"title": title[:100], "categoryId": GAMING_CATEGORY,
                               "privacy": "public", "madeForKids": False}
    elif service == "instagram":
        metadata["instagram"] = {"type": "reel", "shouldShareToFeed": True}
    elif service == "facebook":
        metadata["facebook"] = {"type": "reel"}
    elif service == "tiktok":
        metadata["tiktok"] = {"title": title[:90]}
    data = {
        "channelId": channel["id"],
        "text": caption,
        "schedulingType": "automatic",
        "mode": channel.get("mode") or "addToQueue",
        "assets": [{"video": {"url": video_url}}],
    }
    if metadata:
        data["metadata"] = metadata
    if due_at:
        data["mode"] = "customScheduled"
        data["dueAt"] = due_at
    return data


def _create(api_key: str, data: dict) -> str:
    body = json.dumps({"query": MUTATION, "variables": {"input": data}}).encode()
    req = urllib.request.Request(API_URL, data=body, method="POST", headers={
        "Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        answer = json.load(resp)
    if answer.get("errors"):
        raise RuntimeError(answer["errors"][0].get("message", answer["errors"]))
    result = (answer.get("data") or {}).get("createPost") or {}
    if "post" not in result:
        raise RuntimeError(result.get("message") or f"unexpected answer {answer}")
    return result["post"]["id"]


def rank(clip: Path, priority: list[str] | None = None) -> tuple:
    """Priority streamers first (campaigns that pay per view), then best clips first:
    the clip number is the picker's rank within its VOD."""
    m = re.match(r"\d{8}-[a-z0-9_]+-(\d{2})-", clip.name)
    first = 0 if priority and streamer_of(clip.name) in priority else 1
    return (first, int(m.group(1)) if m else 99, clip.name)


def _load_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8") or "{}")
    except FileNotFoundError:
        return {}


def free_slots(config: dict, state: dict, now: datetime) -> list[datetime]:
    """Posting times from config `times` (local `timezone`) that are still free, over the
    next `days_ahead` days. A slot counts as taken once state holds `slot:<local time>`."""
    tz = ZoneInfo(config.get("timezone") or "Europe/Warsaw")
    local_now = now.astimezone(tz)
    earliest = local_now + timedelta(minutes=int(config.get("min_lead_minutes", 20)))
    slots = []
    for day in range(int(config.get("days_ahead", 2)) + 1):
        date = (local_now + timedelta(days=day)).date()
        for t in config.get("times") or []:
            hh, mm = (int(x) for x in t.split(":"))
            at = datetime(date.year, date.month, date.day, hh, mm, tzinfo=tz)
            if at >= earliest and f"slot:{at:%Y-%m-%dT%H:%M}" not in state:
                slots.append(at)
    return sorted(slots)


def publish(out_dir: Path, folder: str, config: dict | None = None, api_key: str | None = None,
            create=_create, state_path: Path = STATE, now: datetime | None = None) -> int:
    config = load_config() if config is None else config
    api_key = api_key if api_key is not None else os.environ.get("BUFFER_API_KEY", "")
    channels = config.get("channels") or []
    if not config.get("enabled") or not channels:
        print("Buffer publishing is off (config/publish.yaml)")
        return 0
    if not api_key:
        print("BUFFER_API_KEY not set: not publishing")
        return 0
    base = config["base_url"].rstrip("/")
    clips = sorted((c for c in out_dir.glob("*.mp4") if c.stat().st_size > 0), key=lambda c: rank(c, config.get("priority")))
    # With `times`, each clip gets the next free fixed posting time, so posts go out at the
    # same hours every day and the backlog never grows past `days_ahead` days.
    state = _load_state(state_path)
    slots = free_slots(config, state, now or datetime.now(timezone.utc)) if config.get("times") else None
    if slots is not None:
        if len(clips) > len(slots):
            print(f"{len(slots)} free posting times: queueing {len(slots)} of {len(clips)} clips")
        clips = clips[:len(slots)]
    queued = failed = 0
    for i, clip in enumerate(clips):
        txt = clip.with_suffix(".txt")
        caption = txt.read_text(encoding="utf-8").strip() if txt.exists() else ""
        streamer = streamer_of(clip.name)
        url = f"{base}/{folder}/{clip.name}"
        due = slots[i] if slots is not None else None
        for ch in channels:
            if ch.get("streamers") and streamer not in ch["streamers"]:
                continue
            try:
                post_id = create(api_key, post_input({**ch, "mode": ch.get("mode") or config.get("mode")},
                                                     url, caption,
                                                     due.isoformat() if due else None))
                print(f"queued {clip.name} on {ch['service']} for {due or 'the queue'} ({post_id})")
                queued += 1
            except Exception as exc:
                print(f"could not queue {clip.name} on {ch['service']}: {exc}", file=sys.stderr)
                failed += 1
        if due:
            state[f"slot:{due:%Y-%m-%dT%H:%M}"] = clip.name
    if slots is not None and clips:
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{queued} posts queued in Buffer, {failed} failed")
    return 1 if failed and not queued else 0


if __name__ == "__main__":
    raise SystemExit(publish(Path(sys.argv[1]), sys.argv[2]))

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
from pathlib import Path

import yaml

from .config import ROOT

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


def post_input(channel: dict, video_url: str, caption: str) -> dict:
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


def publish(out_dir: Path, folder: str, config: dict | None = None, api_key: str | None = None,
            create=_create) -> int:
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
    queued = failed = 0
    for clip in sorted(out_dir.glob("*.mp4")):
        txt = clip.with_suffix(".txt")
        caption = txt.read_text(encoding="utf-8").strip() if txt.exists() else ""
        streamer = streamer_of(clip.name)
        url = f"{base}/{folder}/{clip.name}"
        for ch in channels:
            if ch.get("streamers") and streamer not in ch["streamers"]:
                continue
            try:
                post_id = create(api_key, post_input({**ch, "mode": ch.get("mode") or config.get("mode")},
                                                     url, caption))
                print(f"queued {clip.name} on {ch['service']} ({post_id})")
                queued += 1
            except Exception as exc:
                print(f"could not queue {clip.name} on {ch['service']}: {exc}", file=sys.stderr)
                failed += 1
    print(f"{queued} posts queued in Buffer, {failed} failed")
    return 1 if failed and not queued else 0


if __name__ == "__main__":
    raise SystemExit(publish(Path(sys.argv[1]), sys.argv[2]))

"""Loads config/streamers.yaml into Streamer objects."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "streamers.yaml"


@dataclass
class Streamer:
    id: str
    platform: str
    channel: str
    videos_url: str
    since: dt.date
    credit: str
    hashtags: list[str]
    cta: str
    notes: str = ""
    language: str = "en"
    layout: str = "fit"
    facecam: dict | None = None
    clips_per_vod: int = 6
    min_clip_seconds: int = 15
    max_clip_seconds: int = 60
    enabled: bool = True
    extra: dict = field(default_factory=dict)


def _as_date(value) -> dt.date:
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value))


def load_streamers(path: Path = DEFAULT_CONFIG) -> list[Streamer]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    defaults = data.get("defaults", {})
    known = set(Streamer.__dataclass_fields__)
    out = []
    for raw in data["streamers"]:
        merged = {**defaults, **raw}
        merged["since"] = _as_date(merged["since"])
        merged["notes"] = " ".join(str(merged.get("notes", "")).split())
        if merged.get("platform") not in {"twitch", "kick", "youtube"}:
            raise ValueError(f"{merged.get('id')}: unknown platform {merged.get('platform')!r}")
        if merged.get("layout", "fit") == "split" and not merged.get("facecam"):
            raise ValueError(f"{merged['id']}: layout 'split' needs a facecam box")
        extra = {k: v for k, v in merged.items() if k not in known}
        kwargs = {k: v for k, v in merged.items() if k in known}
        out.append(Streamer(**kwargs, extra=extra))
    ids = [s.id for s in out]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate streamer ids in config")
    return out

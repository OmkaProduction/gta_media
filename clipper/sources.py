"""Finding and downloading VODs with yt-dlp (Kick listing goes through Kick's public API)."""
from __future__ import annotations

import datetime as dt
import json
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .config import Streamer

USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130 Safari/537.36"


@dataclass
class Vod:
    streamer_id: str
    vod_id: str
    url: str
    title: str
    published: dt.date | None
    duration: float | None
    is_live: bool = False
    media_url: str | None = None  # direct HLS playlist when the platform gives one (Kick)

    @property
    def download_url(self) -> str:
        return self.media_url or self.url

    @property
    def key(self) -> str:
        return f"{self.streamer_id}:{self.vod_id}"


def _run_json(cmd: list[str]) -> list[dict]:
    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return [json.loads(line) for line in res.stdout.splitlines() if line.strip()]


def _parse_date(value) -> dt.date | None:
    if not value:
        return None
    s = str(value)
    try:
        if len(s) == 8 and s.isdigit():  # yt-dlp upload_date YYYYMMDD
            return dt.date(int(s[:4]), int(s[4:6]), int(s[6:]))
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _from_timestamp(ts) -> dt.date | None:
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).date() if ts else None


def _list_with_ytdlp(streamer: Streamer, limit: int) -> list[Vod]:
    entries = _run_json([
        "yt-dlp", "--flat-playlist", "--dump-json", "--playlist-end", str(limit),
        "--extractor-args", "youtubetab:approximate_date", streamer.videos_url,
    ])
    vods = []
    for e in entries:
        url = e.get("url") or e.get("webpage_url")
        if not url:
            continue
        if streamer.platform == "youtube" and not url.startswith("http"):
            url = f"https://www.youtube.com/watch?v={url}"
        vods.append(Vod(
            streamer_id=streamer.id,
            vod_id=str(e.get("id")),
            url=url,
            title=e.get("title") or "",
            published=_parse_date(e.get("upload_date")) or _from_timestamp(e.get("timestamp")),
            duration=e.get("duration"),
            is_live=e.get("live_status") in {"is_live", "is_upcoming"},
        ))
    return vods


def _list_kick(streamer: Streamer, limit: int) -> list[Vod]:
    req = urllib.request.Request(
        f"https://kick.com/api/v2/channels/{streamer.channel}/videos",
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        items = json.load(resp)
    vods = []
    for item in items[:limit]:
        video = item.get("video") or {}
        uuid = video.get("uuid") or item.get("uuid")
        if not uuid:
            continue
        vods.append(Vod(
            streamer_id=streamer.id,
            vod_id=str(uuid),
            url=f"https://kick.com/{streamer.channel}/videos/{uuid}",
            title=item.get("session_title") or "",
            published=_parse_date(item.get("start_time") or item.get("created_at")),
            duration=(item.get("duration") or 0) / 1000 or None,
            is_live=bool(item.get("is_live")),
            # The channel API already returns the VOD's HLS playlist; using it directly avoids
            # yt-dlp's kick:vod extractor, which fails behind Kick's Cloudflare.
            media_url=item.get("source") or video.get("source") or None,
        ))
    return vods


def list_new_vods(streamer: Streamer, processed: set[str], limit: int = 10) -> list[Vod]:
    """Finished VODs published on/after `since` that were not processed yet, oldest first."""
    if streamer.platform == "kick":
        try:
            vods = _list_kick(streamer, limit)
        except Exception as exc:  # Kick API is behind Cloudflare; fall back to yt-dlp
            print(f"[{streamer.id}] Kick API failed ({exc}); trying yt-dlp")
            vods = _list_with_ytdlp(streamer, limit)
    else:
        vods = _list_with_ytdlp(streamer, limit)
    keywords = [k.lower() for k in streamer.extra.get("title_keywords", [])]
    fresh = [
        v for v in vods
        if (not keywords or any(k in v.title.lower() for k in keywords))

        and v.key not in processed
        and (v.published is None or v.published >= streamer.since)
    ]
    return sorted(fresh, key=lambda v: v.published or dt.date.min)


@dataclass
class AudioChunk:
    path: Path
    start: float      # VOD second where this audio begins
    end: float        # VOD second where it ends
    finished: bool    # the whole VOD is covered (broadcast over)
    empty: bool = False  # nothing left to process


MIN_LIVE_CHUNK = 30 * 60   # wait for at least 30 new minutes of a live VOD
LIVE_SAFETY = 120          # stay 2 min behind the live edge
MAX_CHUNK = 6 * 3600       # at most 6 h of audio per run, so one run fits the job's time limit


def download_audio(vod_url: str, dest: Path, start: float = 0.0) -> AudioChunk:
    """Audio track from `start` to the end of the VOD (or to near the live edge).

    Kick and Twitch VODs have no audio-only stream, and a 1080p marathon VOD can be
    50+ GB, more than a GitHub runner's disk. So we take the smallest rendition
    (or an audio-only one when it exists) and let ffmpeg keep just the audio,
    without ever storing the video.

    A VOD whose broadcast is still live (24/7 marathons) is processed in chunks:
    each run takes what was added since the last run, if that is at least 30 min.
    Long VODs (marathons, finished or not) are taken at most 6 h per run; the next run
    continues where this one stopped.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest = dest.with_suffix(".m4a")
    res = subprocess.run(
        ["yt-dlp", "-f", "bestaudio/worst", "-g", vod_url],
        check=True, capture_output=True, text=True,
    )
    stream_url = res.stdout.strip().splitlines()[-1]
    finished, duration = hls_status(stream_url)
    if finished and duration and start >= duration - 60:
        # everything was processed already (e.g. the saved offset reached the end)
        return AudioChunk(path=dest, start=start, end=max(start, duration), finished=True, empty=True)
    end = None
    if not finished:
        end = duration - LIVE_SAFETY
        if end - start < MIN_LIVE_CHUNK:
            raise StillLive(vod_url)
    if duration and (end if end is not None else duration) - start > MAX_CHUNK:
        end = start + MAX_CHUNK
        finished = False  # more of this VOD is left for the next run
    cmd = ["ffmpeg", "-v", "error", "-y"]
    if start:
        cmd += ["-ss", f"{start:.2f}"]
    if end is not None:
        cmd += ["-t", f"{end - start:.2f}"]
    cmd += ["-i", stream_url, "-vn", "-c:a", "aac", "-b:a", "64k", "-ac", "1", str(dest)]
    subprocess.run(cmd, check=True, timeout=4 * 3600)
    if end is None:
        end = start + _probe_duration(dest)
    return AudioChunk(path=dest, start=start, end=end, finished=finished)


def _probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return float(out or 0)


class StillLive(Exception):
    """The broadcast is live and not enough new footage has been added since the last run."""


def hls_status(url: str) -> tuple[bool, float]:
    """(finished, seconds listed so far) for an HLS media playlist.

    A playlist without #EXT-X-ENDLIST is still growing (the broadcast is live).
    Non-HLS URLs and unreadable playlists count as finished.
    """
    if ".m3u8" not in url or not url.startswith("http"):
        return True, 0.0
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=30) as resp:
            text = resp.read().decode("utf-8", "replace")
    except Exception as exc:
        print(f"could not check playlist ({exc}); assuming it is finished")
        return True, 0.0
    duration = 0.0
    for line in text.splitlines():
        if line.startswith("#EXTINF:"):
            try:
                duration += float(line[8:].split(",")[0])
            except ValueError:
                pass
    return "#EXT-X-ENDLIST" in text, duration


def is_unfinished_hls(url: str) -> bool:
    return not hls_status(url)[0]


def grab_frames(vod_url: str, times: list[float], dest_dir: Path) -> list[Path | None]:
    """One small still per time (VOD seconds), so the clip picker can see what is on screen.

    Frames that cannot be fetched are None; selection then relies on the transcript alone.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        res = subprocess.run(
            ["yt-dlp", "-f", "worst[height>=360][vcodec!=none]/worst[vcodec!=none]/worst", "-g", vod_url],
            check=True, capture_output=True, text=True, timeout=120,
        )
        stream_url = res.stdout.strip().splitlines()[0]
    except Exception as exc:
        print(f"could not get a video stream for frames ({exc})")
        return [None] * len(times)
    frames: list[Path | None] = []
    for i, t in enumerate(times):
        out = dest_dir / f"frame{i:02d}.jpg"
        try:
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", stream_url,
                 "-frames:v", "1", "-vf", "scale=640:-2", "-q:v", "5", str(out)],
                check=True, timeout=90,
            )
            frames.append(out if out.exists() and out.stat().st_size else None)
        except Exception as exc:
            print(f"no frame at {t:.0f}s ({exc})")
            frames.append(None)
    return frames


def download_section(vod_url: str, start: float, end: float, dest: Path) -> Path:
    """Download only [start, end] of the VOD in up to 1080p."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "yt-dlp", "-f", "bv*[height<=1080]+ba/b[height<=1080]/b",
        "--download-sections", f"*{start:.2f}-{end:.2f}", "--force-keyframes-at-cuts",
        "--merge-output-format", "mp4", "--concurrent-fragments", "8",
        "-o", str(dest), vod_url,
    ], check=True)
    return dest

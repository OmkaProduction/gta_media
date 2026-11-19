"""Prints the latest VOD titles of channels, to check who streams GTA 6.

Usage: python scripts/peek_vods.py twitch:ewroon kick:nitro youtube:@KanalZero ...
"""
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clipper import sources  # noqa: E402
from clipper.config import Streamer  # noqa: E402

URLS = {
    "twitch": "https://www.twitch.tv/{c}/videos?filter=archives&sort=time",
    "kick": "https://kick.com/{c}/videos",
    "youtube": "https://www.youtube.com/{c}/streams",
}

for arg in sys.argv[1:]:
    platform, channel = arg.split(":", 1)
    s = Streamer(id=channel.lower().strip("@"), platform=platform, channel=channel,
                 videos_url=URLS[platform].format(c=channel), since=dt.date(2026, 1, 1),
                 credit="", hashtags=[], cta="")
    try:
        vods = sources._list_kick(s, 5) if platform == "kick" else sources._list_with_ytdlp(s, 5)
        gta = any(k in v.title.lower() for v in vods for k in ("gta 6", "gta vi", "gta6", "gtavi"))
        print(f"== {arg}: {'GTA 6' if gta else 'no GTA 6'} in last {len(vods)} VODs")
        for v in vods:
            print(f"   {v.published} {'LIVE ' if v.is_live else ''}{(v.duration or 0) / 3600:.1f}h  {v.title[:100]}")
    except Exception as exc:
        print(f"== {arg}: could not list ({str(exc)[:200]})")

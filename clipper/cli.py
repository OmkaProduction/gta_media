"""Entry point: python -m clipper [--streamer ID] [--vod-url URL] [--max-vods N] [--dry-run]

For every enabled streamer: find finished VODs we have not processed, find the loud
moments, transcribe them, pick the best clips, cut them to 9:16 with captions and
write everything (mp4 + review sheet) to out/.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import sys
from pathlib import Path

from . import analysis, captions, chat, metadata, render, select, sources, transcribe
from .config import ROOT, Streamer, load_streamers

STATE_FILE = ROOT / "state" / "processed.json"


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8") or "{}")
    return {}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "clip"


def process_vod(streamer: Streamer, vod: sources.Vod, work: Path, out: Path, dry_run: bool,
                start: float = 0.0) -> tuple[list[str], sources.AudioChunk]:
    tag = f"{streamer.id}-{_slug(vod.vod_id)}"
    vod_dir = work / tag
    print(f"[{streamer.id}] {vod.title!r} ({vod.url}) from {start / 3600:.1f} h")

    chunk = sources.download_audio(vod.download_url, vod_dir / "audio.m4a", start=start)
    if chunk.empty:
        print(f"[{streamer.id}] nothing left after {start / 3600:.1f} h; marking the VOD done")
        return [], chunk
    audio = chunk.path
    loud = analysis.loudness_per_second(audio)
    activity = chat.activity(vod, chunk.start, len(loud))
    if activity is not None:
        print(f"[{streamer.id}] chat replay: {activity.mean():.1f} messages/s on average, peak {activity.max():.1f}")
    candidates = analysis.find_candidates(loud, max_candidates=max(12, streamer.clips_per_vod * 4), chat=activity)
    print(f"[{streamer.id}] {len(loud) / 3600:.1f} h of audio "
          f"({'whole VOD' if chunk.finished else 'live, chunk so far'}), {len(candidates)} candidate moments")

    words: list[transcribe.Word] = []
    for i, c in enumerate(candidates):
        piece = analysis.cut_audio(audio, c.start, c.end, vod_dir / f"cand{i:02d}.wav")
        words += transcribe.transcribe(piece, offset=c.start + chunk.start, language=streamer.language)
    # from here on all times are VOD seconds
    candidates = [analysis.Candidate(c.peak + chunk.start, c.start + chunk.start, c.end + chunk.start, c.score, c.chat)
                  for c in candidates]

    frames = sources.grab_frames(vod.download_url, [c.peak for c in candidates], vod_dir / "frames")
    print(f"[{streamer.id}] {sum(f is not None for f in frames)}/{len(frames)} frames for the clip picker")
    plans = select.select_clips(streamer, candidates, words, streamer.clips_per_vod, frames=frames)
    print(f"[{streamer.id}] selected {len(plans)} clips")

    written = []
    for n, plan in enumerate(plans, 1):
        name = f"{dt.date.today():%Y%m%d}-{streamer.id}-{n:02d}-{_slug(plan.title)}"
        if dry_run:
            print(f"  would cut {plan.start:.0f}-{plan.end:.0f}s: {plan.title}")
            continue
        section = sources.download_section(vod.download_url, plan.start, plan.end, vod_dir / f"{name}.src.mp4")
        ass = vod_dir / f"{name}.ass"
        ass.write_text(captions.build_ass(words, plan.start, plan.end, plan.hook, streamer.credit), encoding="utf-8")
        render.render_vertical(section, ass, out / f"{name}.mp4", streamer.layout, streamer.facecam,
                               duration=plan.end - plan.start)
        (out / f"{name}.txt").write_text(metadata.caption(streamer, plan) + "\n", encoding="utf-8")
        written.append(metadata.review_entry(streamer, vod.url, plan, f"{name}.mp4"))
    shutil.rmtree(vod_dir, ignore_errors=True)
    return written, chunk


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="clipper")
    p.add_argument("--streamer", help="only this streamer id")
    p.add_argument("--vod-url", help="process this one VOD (needs --streamer)")
    p.add_argument("--max-vods", type=int, default=1, help="VODs per streamer per run")
    p.add_argument("--out", default=str(ROOT / "out"))
    p.add_argument("--work", default=str(ROOT / "work"))
    p.add_argument("--dry-run", action="store_true", help="select clips but do not download or render video")
    args = p.parse_args(argv)

    streamers = [s for s in load_streamers() if s.enabled or s.id == args.streamer]
    if args.streamer:
        streamers = [s for s in streamers if s.id == args.streamer]
        if not streamers:
            p.error(f"unknown streamer {args.streamer!r}")
    if args.vod_url and len(streamers) != 1:
        p.error("--vod-url needs --streamer")

    out, work = Path(args.out), Path(args.work)
    out.mkdir(parents=True, exist_ok=True)
    state = load_state()
    review: list[str] = []
    failures = 0

    for streamer in streamers:
        if args.vod_url:
            vods = [sources.Vod(streamer.id, _slug(args.vod_url)[-30:], args.vod_url, args.vod_url, None, None)]
        else:
            try:
                finished = {k for k, v in state.items() if isinstance(v, dict) and v.get("finished", True)}
                vods = sources.list_new_vods(streamer, finished)
            except Exception as exc:
                print(f"[{streamer.id}] could not list VODs: {exc}", file=sys.stderr)
                failures += 1
                continue
            if not vods:
                print(f"[{streamer.id}] no new finished VODs")
        done = 0
        for vod in vods:
            if done >= args.max_vods:
                break
            try:
                start = float(state.get(vod.key, {}).get("offset", 0.0))
                written, chunk = process_vod(streamer, vod, work, out, args.dry_run, start=start)
                review += written
            except sources.StillLive:
                print(f"[{streamer.id}] {vod.url} is live with under 30 new minutes; next run")
                continue
            except Exception as exc:
                print(f"[{streamer.id}] failed on {vod.url}: {exc}", file=sys.stderr)
                failures += 1
                continue
            done += 1
            if not args.dry_run and not args.vod_url:
                state[vod.key] = {"url": vod.url, "processed": dt.datetime.now(dt.timezone.utc).isoformat(),
                                  "offset": round(chunk.end, 1), "finished": chunk.finished}
                save_state(state)

    if review:
        sheet = out / "REVIEW.md"
        sheet.write_text("# Clips to review\n\n" + "\n".join(review), encoding="utf-8")
        print(f"{len(review)} clips written to {out}")
    return 1 if failures and not review else 0


if __name__ == "__main__":
    raise SystemExit(main())

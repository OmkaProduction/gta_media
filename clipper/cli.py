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

from . import analysis, captions, metadata, render, select, sources, transcribe
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


def process_vod(streamer: Streamer, vod: sources.Vod, work: Path, out: Path, dry_run: bool) -> list[str]:
    tag = f"{streamer.id}-{_slug(vod.vod_id)}"
    vod_dir = work / tag
    print(f"[{streamer.id}] {vod.title!r} ({vod.url})")

    audio = sources.download_audio(vod.url, vod_dir / "audio.m4a")
    loud = analysis.loudness_per_second(audio)
    candidates = analysis.find_candidates(loud, max_candidates=max(12, streamer.clips_per_vod * 4))
    print(f"[{streamer.id}] {len(loud) / 3600:.1f} h of audio, {len(candidates)} candidate moments")

    words: list[transcribe.Word] = []
    for i, c in enumerate(candidates):
        piece = analysis.cut_audio(audio, c.start, c.end, vod_dir / f"cand{i:02d}.wav")
        words += transcribe.transcribe(piece, offset=c.start, language=streamer.language)

    plans = select.select_clips(streamer, candidates, words, streamer.clips_per_vod)
    print(f"[{streamer.id}] selected {len(plans)} clips")

    written = []
    for n, plan in enumerate(plans, 1):
        name = f"{dt.date.today():%Y%m%d}-{streamer.id}-{n:02d}-{_slug(plan.title)}"
        if dry_run:
            print(f"  would cut {plan.start:.0f}-{plan.end:.0f}s: {plan.title}")
            continue
        section = sources.download_section(vod.url, plan.start, plan.end, vod_dir / f"{name}.src.mp4")
        ass = vod_dir / f"{name}.ass"
        ass.write_text(captions.build_ass(words, plan.start, plan.end, plan.hook, streamer.credit), encoding="utf-8")
        render.render_vertical(section, ass, out / f"{name}.mp4", streamer.layout, streamer.facecam,
                               duration=plan.end - plan.start)
        (out / f"{name}.txt").write_text(metadata.caption(streamer, plan) + "\n", encoding="utf-8")
        written.append(metadata.review_entry(streamer, vod.url, plan, f"{name}.mp4"))
    shutil.rmtree(vod_dir, ignore_errors=True)
    return written


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
                vods = sources.list_new_vods(streamer, set(state))[: args.max_vods]
            except Exception as exc:
                print(f"[{streamer.id}] could not list VODs: {exc}", file=sys.stderr)
                failures += 1
                continue
            if not vods:
                print(f"[{streamer.id}] no new finished VODs")
        for vod in vods:
            try:
                review += process_vod(streamer, vod, work, out, args.dry_run)
            except Exception as exc:
                print(f"[{streamer.id}] failed on {vod.url}: {exc}", file=sys.stderr)
                failures += 1
                continue
            if not args.dry_run and not args.vod_url:
                state[vod.key] = {"url": vod.url, "processed": dt.datetime.now(dt.timezone.utc).isoformat()}
                save_state(state)

    if review:
        sheet = out / "REVIEW.md"
        sheet.write_text("# Clips to review\n\n" + "\n".join(review), encoding="utf-8")
        print(f"{len(review)} clips written to {out}")
    return 1 if failures and not review else 0


if __name__ == "__main__":
    raise SystemExit(main())

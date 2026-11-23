import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from clipper import analysis, captions, cli, metadata, render, select
from clipper.analysis import Candidate
from clipper.config import load_streamers
from clipper.transcribe import Word


@pytest.fixture(scope="module")
def streamers():
    return {s.id: s for s in load_streamers()}


def test_config_loads_campaign_rules(streamers):
    assert {"dom", "lacy", "clix", "tomo"} <= set(streamers)
    assert "#dasgasdom3" in streamers["dom"].hashtags
    assert "#LACY" in streamers["lacy"].hashtags
    assert "#Clix" in streamers["clix"].hashtags
    assert streamers["tomo"].enabled is False
    assert streamers["dom"].platform == "kick"


def test_find_candidates_picks_loud_bursts_and_respects_gap():
    rng = np.random.default_rng(0)
    loud = -35 + rng.normal(0, 1.5, 1800)
    for t in (300, 310, 900, 1500):
        loud[t:t + 6] = -8
    cands = analysis.find_candidates(loud, max_candidates=5, min_gap=90)
    peaks = sorted(round(c.peak) for c in cands)
    assert any(abs(p - 305) < 20 for p in peaks)
    assert any(abs(p - 903) < 10 for p in peaks)
    assert any(abs(p - 1503) < 10 for p in peaks)
    # 300 and 310 are one moment, not two
    assert sum(1 for p in peaks if 280 < p < 330) == 1
    for c in cands:
        assert c.start <= c.peak <= c.end


def test_find_candidates_ignores_background_noise():
    rng = np.random.default_rng(1)
    loud = -35 + rng.normal(0, 1.0, 1200)
    loud[600:605] = -5
    cands = analysis.find_candidates(loud, max_candidates=10)
    assert len(cands) == 1 and abs(cands[0].peak - 602) < 6


def test_loudness_per_second_from_real_audio(tmp_path):
    wav = tmp_path / "a.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    "aevalsrc='if(between(t,20,23),0.9,0.01)*sin(2*PI*440*t)':s=16000:d=40", str(wav)], check=True)
    loud = analysis.loudness_per_second(wav)
    assert len(loud) == 40
    assert loud[21] > loud[5] + 30
    top = analysis.find_candidates(loud, max_candidates=1, baseline_window=30)[0]
    assert 19 <= top.peak <= 24


def test_group_words_splits_on_pauses():
    words = [Word(0, .3, "no"), Word(.35, .6, "way"), Word(.65, .9, "bro"), Word(.95, 1.2, "look"),
             Word(3.0, 3.3, "what")]
    groups = captions.group_words(words)
    assert [len(g) for g in groups] == [3, 1, 1]


def test_build_ass_is_relative_to_clip_and_escaped():
    words = [Word(100.0, 100.4, "WHAT"), Word(100.5, 100.9, "{is}"), Word(130.0, 130.5, "outside")]
    ass = captions.build_ass(words, 99.0, 110.0, "Lacy finds a glitch", "LACY on Twitch")
    assert "0:00:01.00" in ass  # first word 1 s into the clip
    assert "OUTSIDE" not in ass  # word after the clip end is dropped
    assert "{is}" not in ass and "(IS)" in ass
    assert "LACY FINDS A GLITCH" in ass and "LACY on Twitch" in ass


def test_fallback_selection_without_api_key(monkeypatch, streamers):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cands = [Candidate(peak=500, start=455, end=525, score=10), Candidate(peak=900, start=855, end=925, score=30)]
    plans = select.select_clips(streamers["lacy"], cands, [Word(880, 881, "bro")], 1)
    assert len(plans) == 1
    p = plans[0]
    assert 855 <= p.start < p.end <= 925
    assert 15 <= p.end - p.start <= 60


class _Block:
    def __init__(self, text):
        self.type, self.text = "text", text


class _Resp:
    def __init__(self, text, stop="end_turn"):
        self.content, self.stop_reason = [_Block(text)], stop


def _fake_client(answer, calls):
    class Messages:
        def create(self, **kw):
            calls.append(kw)
            return answer

    class Beta:
        messages = Messages()

    class Client:
        beta = Beta()

    return lambda: Client()


def test_claude_selection_is_clamped_and_validated(monkeypatch, streamers):
    import anthropic

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    answer = _Resp(json.dumps({"clips": [
        {"candidate": 0, "start": 440, "end": 600, "hook": " Dom crashes ", "title": "Dom vs the bridge", "reason": "fail"},
        {"candidate": 7, "start": 1, "end": 20, "hook": "x", "title": "bad index", "reason": "x"},
    ]}))
    calls = []
    monkeypatch.setattr(anthropic, "Anthropic", _fake_client(answer, calls))
    cands = [Candidate(peak=500, start=455, end=525, score=10)]
    plans = select.select_clips(streamers["dom"], cands, [], 3)
    assert len(plans) == 1
    assert plans[0].start == 455 and plans[0].end == 515  # clamped to the window and 60 s
    assert plans[0].hook == "Dom crashes"
    kw = calls[0]
    assert kw["model"] == "claude-opus-5-5"
    assert kw["output_config"]["format"]["type"] == "json_schema"
    assert "dasgasd3m" in kw["messages"][0]["content"]


def test_claude_refusal_falls_back(monkeypatch, streamers):
    import anthropic

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(anthropic, "Anthropic", _fake_client(_Resp("", stop="refusal"), []))
    cands = [Candidate(peak=500, start=455, end=525, score=10)]
    plans = select.select_clips(streamers["dom"], cands, [], 2)
    assert plans and plans[0].reason.startswith("loudest")


def test_caption_puts_campaign_hashtags_and_cta(streamers):
    plan = select.ClipPlan(10, 40, "hook", "Dom finds the secret room", "x")
    text = metadata.caption(streamers["dom"], plan)
    assert text.startswith("Dom finds the secret room")
    assert "#dasgasdom3" in text and "kick.com/dasgasd3m" in text


@pytest.mark.parametrize("layout,facecam", [("fit", None), ("split", {"x": 0.75, "y": 0.0, "w": 0.25, "h": 0.25})])
def test_render_vertical_produces_1080x1920(tmp_path, layout, facecam):
    src = tmp_path / "src.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=1920x1080:d=4:r=30",
                    "-f", "lavfi", "-i", "sine=f=300:d=4", "-shortest", str(src)], check=True)
    ass = tmp_path / "c.ass"
    ass.write_text(captions.build_ass([Word(1, 1.5, "hello")], 0, 4, "hook", "credit"))
    out = render.render_vertical(src, ass, tmp_path / "out.mp4", layout, facecam, duration=4)
    probe = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(out)],
        check=True, capture_output=True, text=True).stdout)
    video = next(s for s in probe["streams"] if s["codec_type"] == "video")
    assert (video["width"], video["height"]) == (1080, 1920)
    assert any(s["codec_type"] == "audio" for s in probe["streams"])


def test_end_to_end_with_local_vod(tmp_path, monkeypatch):
    """Full run on a synthetic 'stream' with two loud reactions; downloads are replaced by local files."""
    vod = tmp_path / "vod.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=1280x720:d=300:r=15",
        "-f", "lavfi", "-i",
        "aevalsrc='(0.02+0.9*between(t,100,104)+0.9*between(t,220,224))*sin(2*PI*330*t)':s=16000:d=300",
        "-shortest", "-c:v", "libx264", "-preset", "ultrafast", str(vod)], check=True)

    def fake_audio(url, dest, start=0.0):
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(vod), "-vn", "-c:a", "aac", str(dest)], check=True)
        return cli.sources.AudioChunk(dest, 0.0, 300.0, True)

    def fake_section(url, start, end, dest):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-t", str(end - start),
                        "-i", str(vod), "-c:v", "libx264", "-preset", "ultrafast", str(dest)], check=True)
        return dest

    monkeypatch.setattr(cli.sources, "download_audio", fake_audio)
    monkeypatch.setattr(cli.sources, "download_section", fake_section)
    monkeypatch.setattr(cli.transcribe, "transcribe",
                        lambda path, offset, language: [Word(offset + 40, offset + 41, "no way")])
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(cli, "STATE_FILE", tmp_path / "state.json")

    out = tmp_path / "out"
    rc = cli.main(["--streamer", "lacy", "--vod-url", "https://example.invalid/vod", "--out", str(out),
                   "--work", str(tmp_path / "work")])
    assert rc == 0
    clips = sorted(out.glob("*.mp4"))
    assert len(clips) == 2
    review = (out / "REVIEW.md").read_text()
    assert "#LACY" in review and "Content Rewards" in review
    starts = sorted(int(line.split(" at ")[1].split(":")[2][:2]) + 60 * int(line.split(" at ")[1].split(":")[1])
                    for line in review.splitlines() if line.startswith("- Source:"))
    assert any(70 <= s <= 104 for s in starts) and any(190 <= s <= 224 for s in starts)


def test_kick_listing_uses_direct_hls_source(monkeypatch, streamers):
    import io
    from clipper import sources

    payload = json.dumps([
        {"session_title": "GTA 6 day 1", "is_live": False, "start_time": "2026-11-18 20:00:00",
         "duration": 3600000, "source": "https://stream.kick.com/x/master.m3u8", "video": {"uuid": "abc"}},
        {"session_title": "live now", "is_live": True, "start_time": "2026-11-19 10:00:00",
         "duration": 0, "source": "https://stream.kick.com/y/master.m3u8", "video": {"uuid": "def"}},
        {"session_title": "old", "is_live": False, "start_time": "2026-11-10 10:00:00",
         "duration": 1000, "video": {"uuid": "old"}},
    ]).encode()
    monkeypatch.setattr(sources.urllib.request, "urlopen", lambda req, timeout: io.BytesIO(payload))
    vods = sources.list_new_vods(streamers["dom"], processed=set())
    assert [v.vod_id for v in vods] == ["abc"]
    assert vods[0].download_url == "https://stream.kick.com/x/master.m3u8"
    assert vods[0].url == "https://kick.com/dasgasd3m/videos/abc"
    assert sources.list_new_vods(streamers["dom"], processed={"dom:abc"}) == []


def test_unfinished_hls_is_detected(monkeypatch):
    import io
    from clipper import sources

    live = b"#EXTM3U\n#EXTINF:10,\nseg1.ts\n"
    done = live + b"#EXT-X-ENDLIST\n"
    monkeypatch.setattr(sources.urllib.request, "urlopen", lambda req, timeout: io.BytesIO(live))
    assert sources.is_unfinished_hls("https://x/v.m3u8") is True
    monkeypatch.setattr(sources.urllib.request, "urlopen", lambda req, timeout: io.BytesIO(done))
    assert sources.is_unfinished_hls("https://x/v.m3u8") is False
    assert sources.is_unfinished_hls("https://x/v.mp4") is False


def test_live_vod_is_skipped_and_not_marked(monkeypatch, tmp_path):
    from clipper import sources

    vods = [sources.Vod("lacy", "1", "https://t/1", "live", None, None),
            sources.Vod("lacy", "2", "https://t/2", "done", None, None)]
    monkeypatch.setattr(cli.sources, "list_new_vods", lambda s, processed: vods)
    calls = []

    def fake_process(streamer, vod, work, out, dry_run, start=0.0):
        calls.append(vod.vod_id)
        if vod.vod_id == "1":
            raise sources.StillLive(vod.url)
        return [], sources.AudioChunk(Path("x"), start, 100.0, True)

    monkeypatch.setattr(cli, "process_vod", fake_process)
    monkeypatch.setattr(cli, "STATE_FILE", tmp_path / "state.json")
    assert cli.main(["--streamer", "lacy", "--out", str(tmp_path / "o")]) == 0
    assert calls == ["1", "2"]
    state = json.loads((tmp_path / "state.json").read_text())
    assert list(state) == ["lacy:2"]


def test_live_marathon_is_processed_in_chunks(monkeypatch, tmp_path):
    from clipper import sources

    vod = sources.Vod("lacy", "9", "https://t/9", "24/7 GTA 6 marathon", None, None)
    seen = {"lists": [], "starts": []}

    def fake_list(s, processed):
        seen["lists"].append(set(processed))
        return [vod]

    def fake_process(streamer, v, work, out, dry_run, start=0.0):
        seen["starts"].append(start)
        return [], sources.AudioChunk(Path("x"), start, start + 3600, False)

    monkeypatch.setattr(cli.sources, "list_new_vods", fake_list)
    monkeypatch.setattr(cli, "process_vod", fake_process)
    monkeypatch.setattr(cli, "STATE_FILE", tmp_path / "state.json")
    cli.main(["--streamer", "lacy", "--out", str(tmp_path / "o")])
    cli.main(["--streamer", "lacy", "--out", str(tmp_path / "o")])
    assert seen["starts"] == [0.0, 3600.0]          # second run continues where the first stopped
    assert "lacy:9" not in seen["lists"][1]         # unfinished VOD is offered again
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["lacy:9"]["finished"] is False and state["lacy:9"]["offset"] == 7200.0


def test_hls_status_sums_duration(monkeypatch):
    import io
    from clipper import sources

    text = b"#EXTM3U\n#EXTINF:10.0,\na.ts\n#EXTINF:5.5,\nb.ts\n"
    monkeypatch.setattr(sources.urllib.request, "urlopen", lambda req, timeout: io.BytesIO(text))
    assert sources.hls_status("https://x/v.m3u8") == (False, 15.5)


def test_title_filter_skips_non_gta6_vods(monkeypatch, streamers):
    from clipper import sources

    entries = [sources.Vod("lacy", "a", "u", "BIKER DAY joining a biker gang", None, None),
               sources.Vod("lacy", "b", "u", "GTA 6 EARLY ACCESS marathon", None, None)]
    monkeypatch.setattr(sources, "_list_with_ytdlp", lambda s, limit: entries)
    assert [v.vod_id for v in sources.list_new_vods(streamers["lacy"], set())] == ["b"]


def test_long_vod_is_cut_into_6h_chunks(monkeypatch, tmp_path):
    from clipper import sources

    class R:
        stdout = "https://x/v.m3u8\n"

    cmds = []
    monkeypatch.setattr(sources.subprocess, "run", lambda cmd, **kw: (cmds.append(cmd), R())[1])
    monkeypatch.setattr(sources, "hls_status", lambda url: (True, 20 * 3600.0))  # finished 20 h VOD
    probed = iter([6 * 3600.0, 2 * 3600.0])
    monkeypatch.setattr(sources, "_probe_duration", lambda path: next(probed))
    chunk = sources.download_audio("https://t/1", tmp_path / "a.m4a", start=3600.0)
    assert (chunk.start, chunk.end, chunk.finished) == (3600.0, 3600.0 + 6 * 3600, False)
    ff = cmds[-1]
    assert ff[ff.index("-t") + 1] == f"{6 * 3600:.2f}"

    chunk = sources.download_audio("https://t/1", tmp_path / "a.m4a", start=18 * 3600.0)
    assert chunk.finished is True and "-t" not in cmds[-1]   # last piece runs to the end


def test_frames_are_sent_to_claude(monkeypatch, streamers, tmp_path):
    import anthropic

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    calls = []
    monkeypatch.setattr(anthropic, "Anthropic", _fake_client(_Resp(json.dumps({"clips": []})), calls))
    frame = tmp_path / "f.jpg"
    frame.write_bytes(b"\xff\xd8jpeg")
    cands = [Candidate(peak=500, start=455, end=525, score=10), Candidate(peak=900, start=860, end=930, score=8)]
    assert select.select_clips(streamers["lacy"], cands, [], 2, frames=[frame, None]) == []
    content = calls[0]["messages"][0]["content"]
    assert [b["type"] for b in content].count("image") == 1
    assert "no frame available" in content[-1]["text"]
    assert "GTA 6 content" in calls[0]["system"]


def test_vod_already_processed_to_the_end(monkeypatch, tmp_path):
    from clipper import sources

    class R:
        stdout = "https://x/v.m3u8\n"

    monkeypatch.setattr(sources.subprocess, "run", lambda cmd, **kw: R())
    monkeypatch.setattr(sources, "hls_status", lambda url: (True, 5 * 3600.0))
    chunk = sources.download_audio("https://t/1", tmp_path / "a.m4a", start=6 * 3600.0)
    assert chunk.empty and chunk.finished


def test_publish_queues_each_clip_on_matching_channels(tmp_path):
    from clipper import publish

    (tmp_path / "20261119-dom-01-shark.mp4").write_bytes(b"x")
    (tmp_path / "20261119-dom-01-shark.txt").write_text("Dom vs shark\n\n#dasgasdom3 #GTA6\nWatch Dom live", encoding="utf-8")
    (tmp_path / "20261119-lacy-01-x.mp4").write_bytes(b"x")
    config = {"enabled": True, "base_url": "https://raw.example/clips/", "mode": "addToQueue", "channels": [
        {"id": "a" * 24, "service": "youtube"},
        {"id": "b" * 24, "service": "tiktok", "streamers": ["dom"]},
    ]}
    sent = []
    assert publish.publish(tmp_path, "run-1", config, "key", create=lambda k, d: sent.append(d) or "p1") == 0
    assert len(sent) == 3   # dom on both channels, lacy only on YouTube
    yt = sent[0]
    assert yt["assets"][0]["video"]["url"] == "https://raw.example/clips/run-1/20261119-dom-01-shark.mp4"
    assert yt["metadata"]["youtube"] == {"title": "Dom vs shark", "categoryId": "20",
                                         "privacy": "public", "madeForKids": False}
    assert yt["text"].startswith("Dom vs shark") and yt["mode"] == "addToQueue"
    assert sent[1]["metadata"]["tiktok"]["title"] == "Dom vs shark"


def test_publish_is_off_without_config_or_key(tmp_path):
    from clipper import publish

    (tmp_path / "20261119-dom-01-x.mp4").write_bytes(b"x")
    boom = lambda k, d: (_ for _ in ()).throw(AssertionError("should not post"))
    assert publish.publish(tmp_path, "r", {"enabled": False, "channels": [{"id": "a", "service": "youtube"}]}, "k", create=boom) == 0
    assert publish.publish(tmp_path, "r", {"enabled": True, "base_url": "x", "channels": [{"id": "a", "service": "youtube"}]}, "", create=boom) == 0
    assert publish.load_config()["channels"][0]["service"] == "youtube"


def test_live_hls_is_read_from_the_start_and_short_audio_is_trusted(monkeypatch, tmp_path):
    from clipper import sources

    class R:
        stdout = "https://x/live.m3u8\n"

    cmds = []
    monkeypatch.setattr(sources.subprocess, "run", lambda cmd, **kw: (cmds.append(cmd), R())[1])
    monkeypatch.setattr(sources, "hls_status", lambda url: (False, 10 * 3600.0))  # live, 10 h so far
    monkeypatch.setattr(sources, "_probe_duration", lambda path: 3 * 3600.0)     # only 3 h came down
    chunk = sources.download_audio("https://t/9", tmp_path / "a.m4a")
    ff = cmds[-1]
    assert ff[ff.index("-live_start_index") + 1] == "0" and ff.index("-live_start_index") < ff.index("-i")
    assert (chunk.start, chunk.end, chunk.finished) == (0.0, 3 * 3600.0, False)


def test_publish_fills_fixed_posting_times(tmp_path):
    import json
    from datetime import datetime, timezone
    from clipper import publish
    out = tmp_path / "out"
    out.mkdir()
    for name in ["20261120-dom-02-b", "20261120-lacy-01-a", "20261120-dom-01-c", "20261120-lacy-03-d"]:
        (out / f"{name}.mp4").write_bytes(b"x")
    state = tmp_path / "state.json"
    state.write_text('{"lacy:v1": {"finished": true}, "slot:2026-11-21T12:00": "old.mp4"}')
    sent = []
    config = {"enabled": True, "base_url": "https://x/clips", "timezone": "Europe/Warsaw",
              "times": ["12:00", "21:00"], "days_ahead": 1,
              "channels": [{"id": "a" * 24, "service": "youtube"}]}
    create = lambda key, data: sent.append((data["assets"][0]["video"]["url"], data["mode"], data["dueAt"])) or "id"
    now = datetime(2026, 11, 20, 20, 50, tzinfo=timezone.utc)  # 21:50 in Warsaw: today's slots are gone
    publish.publish(out, "run-1", config=config, api_key="k", create=create, state_path=state, now=now)
    assert sent == [("https://x/clips/run-1/20261120-dom-01-c.mp4", "customScheduled", "2026-11-21T21:00:00+01:00")]
    saved = json.loads(state.read_text())
    assert saved["slot:2026-11-21T21:00"] == "20261120-dom-01-c.mp4" and saved["lacy:v1"] == {"finished": True}
    publish.publish(out, "run-2", config=config, api_key="k", create=create, state_path=state, now=now)
    assert len(sent) == 1


def test_publish_entries_in_state_do_not_break_vod_listing(monkeypatch, tmp_path):
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"published:2026-11-20": 2, "slot:2026-11-21T12:00": "a.mp4",
                                 "lacy:1": {"finished": True}}))
    seen = []
    monkeypatch.setattr(cli.sources, "list_new_vods", lambda s, processed: seen.append(set(processed)) or [])
    monkeypatch.setattr(cli, "STATE_FILE", state)
    assert cli.main(["--streamer", "lacy", "--out", str(tmp_path / "o")]) == 0
    assert seen == [{"lacy:1"}]


def test_chat_spike_becomes_a_candidate_even_when_audio_is_flat():
    rng = np.random.default_rng(1)
    loud = -30 + rng.normal(0, 0.5, 3600)
    loud[600:606] += 15                       # one loud moment
    rate = np.full(3600, 2.0)
    rate[2008:2030] = 25.0                    # chat explodes ~8 s after a quiet moment at 2000 s
    cands = analysis.find_candidates(loud, max_candidates=2, chat=rate)
    peaks = sorted(c.peak for c in cands)
    assert any(abs(p - 603) < 10 for p in peaks)
    chat_cand = [c for c in cands if abs(c.peak - 2005) < 15]
    assert chat_cand and chat_cand[0].chat > 5


def test_chat_rates_from_twitch_and_kick_replays():
    import datetime as dt
    from clipper import chat
    page = [{"data": {"video": {"comments": {"edges": [
        {"node": {"contentOffsetSeconds": 100 + i * 0.5}} for i in range(40)]}}}}]
    assert abs(chat.twitch_rate("1", 100, fetch=lambda url, body, headers: page) - 40 / 19.5) < 1e-6
    seen = []
    kick = lambda url: seen.append(url) or {"data": {"messages": [{}] * 10}}
    start = dt.datetime(2026, 11, 20, 12, 0, tzinfo=dt.timezone.utc)
    assert chat.kick_rate(42, start, 65, fetch=kick) == 2.0
    assert seen == ["https://kick.com/api/v2/channels/42/messages?start_time=2026-11-20T12:01:05.000Z"]


def test_chat_activity_is_none_when_the_replay_fails():
    from clipper import chat, sources
    vod = sources.Vod("lacy", "v1", "https://www.twitch.tv/videos/1", "t", None, None)
    def broken(t):
        raise OSError("blocked")
    assert chat.activity(vod, 0, 600, rate=broken) is None
    got = chat.activity(vod, 0, 600, rate=lambda t: 3.0 if 200 <= t < 240 else 1.0)
    assert got.shape == (600,) and got.max() == 3.0


def test_subscribe_button_shows_after_hook_and_at_the_end():
    ass = captions.build_ass([], 0, 30, "hook", "credit")
    subs = [l for l in ass.splitlines() if ",Subscribe," in l]
    assert len(subs) == 2 and "SUBSCRIBE" in subs[0]
    assert subs[0].split(",")[1] == "0:00:04.00" and subs[1].split(",")[2] == "0:00:30.00"
    assert ",Subscribe," not in captions.build_ass([], 0, 30, "hook", "credit", subscribe=False)

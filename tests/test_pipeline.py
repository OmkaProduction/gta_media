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

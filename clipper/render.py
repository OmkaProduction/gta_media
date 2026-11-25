"""Turns a 16:9 stream section into a 1080x1920 vertical clip with burned-in subtitles."""
from __future__ import annotations

import subprocess
from pathlib import Path

W, H = 1080, 1920


def zoom_expr(duration: float | None, punch: float | None) -> str:
    """Scale factor over time: a slow push-in across the clip, plus a quick punch-in on the reaction."""
    parts = ["1"]
    if duration:
        parts.append(f"0.06*t/{duration:.2f}")
    if punch is not None:
        # ease in over 0.15 s, hold ~1.2 s, ease out over 0.3 s
        p = punch
        parts.append(f"0.14*clip((t-{p - 0.15:.2f})/0.15,0,1)*clip(({p + 1.5:.2f}-t)/0.3,0,1)")
    return "+".join(parts)


def progress_bar(duration: float | None) -> str:
    if not duration:
        return ""
    return f",drawbox=x=0:y=ih-16:w='iw*t/{duration:.2f}':h=16:color=0xFF2D2D@0.95:t=fill"


def filter_graph(layout: str, facecam: dict | None, ass_path: Path,
                 duration: float | None = None, punch: float | None = None) -> str:
    z = zoom_expr(duration, punch)
    bar = progress_bar(duration)
    subs = "subtitles=filename='{}'".format(str(ass_path).replace("\\", "/").replace("'", r"\'").replace(":", r"\:"))
    if layout == "split" and facecam:
        # facecam (top 640 px) above a centre crop of the gameplay (bottom 1280 px)
        x, y, w, h = (facecam[k] for k in ("x", "y", "w", "h"))
        cam = f"crop=iw*{w}:ih*{h}:iw*{x}:ih*{y},scale={W}:640:force_original_aspect_ratio=increase,crop={W}:640"
        game = f"scale=w='trunc(2276*({z})/2)*2':h=-2:eval=frame,crop={W}:1280"
        return (
            f"[0:v]split=2[a][b];[a]{cam}[cam];[b]{game}[game];"
            f"[cam][game]vstack=inputs=2,setsar=1,{subs}{bar}[v]"
        )
    # fit: full gameplay frame across the width, blurred zoomed copy behind it
    return (
        f"[0:v]split=2[a][b];"
        f"[a]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=24:2,eq=brightness=-0.15[bg];"
        f"[b]scale=w='trunc({W}*({z})/2)*2':h=-2:eval=frame,crop='min(iw,{W})':ih[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1,{subs}{bar}[v]"
    )


def render_vertical(source: Path, ass_path: Path, dest: Path, layout: str = "fit",
                    facecam: dict | None = None, duration: float | None = None,
                    punch: float | None = None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-v", "error", "-y", "-i", str(source),
        "-filter_complex", filter_graph(layout, facecam, ass_path, duration, punch),
        "-map", "[v]", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-r", "30",
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
        "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
        "-movflags", "+faststart",
    ]
    if duration:
        cmd += ["-t", f"{duration:.2f}"]
    cmd.append(str(dest))
    subprocess.run(cmd, check=True)
    return dest

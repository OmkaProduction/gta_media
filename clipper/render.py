"""Turns a 16:9 stream section into a 1080x1920 vertical clip with burned-in subtitles."""
from __future__ import annotations

import subprocess
from pathlib import Path

W, H = 1080, 1920


def filter_graph(layout: str, facecam: dict | None, ass_path: Path) -> str:
    subs = "subtitles=filename='{}'".format(str(ass_path).replace("\\", "/").replace("'", r"\'").replace(":", r"\:"))
    if layout == "split" and facecam:
        # facecam (top 640 px) above a centre crop of the gameplay (bottom 1280 px)
        x, y, w, h = (facecam[k] for k in ("x", "y", "w", "h"))
        cam = f"crop=iw*{w}:ih*{h}:iw*{x}:ih*{y},scale={W}:640:force_original_aspect_ratio=increase,crop={W}:640"
        game = f"scale=-2:1280,crop={W}:1280"
        return (
            f"[0:v]split=2[a][b];[a]{cam}[cam];[b]{game}[game];"
            f"[cam][game]vstack=inputs=2,setsar=1,{subs}[v]"
        )
    # fit: full gameplay frame across the width, blurred zoomed copy behind it
    return (
        f"[0:v]split=2[a][b];"
        f"[a]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=24:2,eq=brightness=-0.15[bg];"
        f"[b]scale={W}:-2[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1,{subs}[v]"
    )


def render_vertical(source: Path, ass_path: Path, dest: Path, layout: str = "fit",
                    facecam: dict | None = None, duration: float | None = None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-v", "error", "-y", "-i", str(source),
        "-filter_complex", filter_graph(layout, facecam, ass_path),
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

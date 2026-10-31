"""Technical QC for 16:9 Long-Form Hindi Storytelling Videos (`1920x1080`)."""
import json
import os
import subprocess
from pathlib import Path


def _run(args, timeout=90):
    p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError(p.stderr[-1600:])
    return p.stdout.strip()


def validate_rendered_video(video_path: Path, scenes: list, report_path: Path) -> dict:
    if not video_path.exists() or video_path.stat().st_size < 150_000:
        raise RuntimeError("QC: rendered MP4 is missing or implausibly small.")

    raw = _run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration:stream=codec_type,width,height,avg_frame_rate",
        "-of", "json", str(video_path),
    ])
    data = json.loads(raw)
    duration = float(data.get("format", {}).get("duration", 0))
    v = next((x for x in data.get("streams", []) if x.get("codec_type") == "video"), {})
    a = next((x for x in data.get("streams", []) if x.get("codec_type") == "audio"), None)

    if (v.get("width"), v.get("height")) != (1920, 1080):
        raise RuntimeError(f"QC: expected 16:9 widescreen 1920x1080, got {v.get('width')}x{v.get('height')}.")
    if not a:
        raise RuntimeError("QC: no audio stream.")
    if duration < 10.0:
        raise RuntimeError(f"QC: duration {duration:.1f}s is too short (< 10s).")
    if not scenes:
        raise RuntimeError("QC: no scenes were rendered.")

    black = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(video_path), "-vf", "blackdetect=d=1.5:pix_th=0.04:pic_th=0.98", "-an", "-f", "null", "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=120,
    )
    if black.returncode not in (0, 1):
        raise RuntimeError("QC: FFmpeg black-frame analysis failed.")
    if "black_start:" in black.stderr:
        raise RuntimeError("QC: sustained black frames detected.")

    voices_used = sorted({str(s.get("voice_used", "")) for s in scenes if s.get("voice_used")})
    female_scenes = sum(1 for s in scenes if str(s.get("speaker_gender", "")).lower() == "female")

    report = {
        "passed": True,
        "aspect_ratio": "16:9",
        "width": 1920,
        "height": 1080,
        "duration_seconds": round(duration, 2),
        "has_audio": True,
        "scene_count": len(scenes),
        "voices_used": voices_used,
        "female_character_scenes": female_scenes,
        "upload_eligible": True,
        "dry_run": os.getenv("DRY_RUN", "false").strip().lower() == "true",
        "notes": ["16:9 widescreen dimensions (1920x1080), Hindi audio mix, and visual checks passed."],
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


# Backward-compatible alias
validate_rendered_short = validate_rendered_video

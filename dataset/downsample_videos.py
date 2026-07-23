#!/usr/bin/env python3
"""
Downsample all .mp4 videos under train/validation/test to a maximum
vertical resolution of 480p while preserving aspect ratio.

Safety rules:
- Original files are only replaced after the output is verified.
- Verification checks: readable by ffprobe, has a video stream,
  height <= 480, duration within tolerance of original.
- On any failure the temporary output is removed and the original is kept.
- Progress is logged to downsample_log.json so reruns skip completed work.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
SPLIT_FOLDERS = ["Train", "Validation", "Test"]
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv"}
MAX_HEIGHT = 480
DURATION_TOLERANCE = 0.05        # 5% duration difference allowed
MIN_FREE_MB = 512                # require at least 512 MB free on output drive

LOG_FILE = ROOT_DIR / "downsample_log.json"

# Encoder preference for H.264 output. The script auto-selects the first
# encoder that is available in the local ffmpeg build.
ENCODER_OPTIONS = [
    ("libx264", ["-preset", "medium", "-crf", "23"]),
    ("libopenh264", ["-b:v", "2M"]),
    ("h264_nvenc", ["-preset", "p4", "-cq", "23"]),
    ("h264_amf", ["-usage", "transcoding", "-qp_i", "23", "-qp_p", "23"]),
    ("h264_qsv", ["-preset", "medium", "-global_quality", "23"]),
    ("h264_mf", ["-rate_control", "quality", "-quality", "80"]),
    ("h264_vulkan", ["-quality_level", "2"]),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[{Path(__file__).name}] {msg}", flush=True)


def run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    """Run a command and capture stdout/stderr."""
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        **kwargs,
    )


_SELECTED_ENCODER: tuple[str, list[str]] | None = None


def select_h264_encoder() -> tuple[str, list[str]]:
    """Return the best available H.264 encoder and its recommended arguments."""
    global _SELECTED_ENCODER
    if _SELECTED_ENCODER is not None:
        return _SELECTED_ENCODER

    result = run(["ffmpeg", "-hide_banner", "-encoders"])
    if result.returncode != 0:
        raise RuntimeError("cannot list ffmpeg encoders")
    encoder_list = result.stdout

    for encoder, args in ENCODER_OPTIONS:
        # ffmpeg encoder lines look like " V....D libx264 ..."
        if f" {encoder} " in encoder_list or f" {encoder}\n" in encoder_list:
            _SELECTED_ENCODER = (encoder, args)
            log(f"Using H.264 encoder: {encoder}")
            return _SELECTED_ENCODER

    raise RuntimeError("no usable H.264 encoder found in ffmpeg")


def ffprobe_json(path: Path) -> dict[str, Any] | None:
    """Return ffprobe JSON for the first video stream, or None on failure."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,duration,nb_frames,codec_name:stream_disposition=:format=duration",
        "-of", "json",
        str(path),
    ]
    result = run(cmd)
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


def get_video_info(path: Path) -> dict[str, Any] | None:
    """Extract useful video metadata, or None if the file is not a valid video."""
    data = ffprobe_json(path)
    if not data or "streams" not in data or not data["streams"]:
        return None
    stream = data["streams"][0]
    fmt = data.get("format", {})

    width = stream.get("width")
    height = stream.get("height")
    codec = stream.get("codec_name")

    def _float(value: Any) -> float | None:
        try:
            return float(value) if value is not None else None
        except (ValueError, TypeError):
            return None

    duration = _float(stream.get("duration")) or _float(fmt.get("duration"))
    nb_frames = _float(stream.get("nb_frames"))

    if width is None or height is None:
        return None

    return {
        "width": int(width),
        "height": int(height),
        "duration": duration,
        "nb_frames": int(nb_frames) if nb_frames else None,
        "codec": codec,
    }


def has_enough_disk_space(path: Path, required_mb: int = MIN_FREE_MB) -> bool:
    """Check that the drive containing *path* has enough free space."""
    try:
        stat = shutil.disk_usage(path.resolve())
        free_mb = stat.free / (1024 * 1024)
        return free_mb >= required_mb
    except OSError:
        return False


def is_already_processed(video_path: Path, log_data: dict[str, Any]) -> bool:
    """Check whether a video has already been successfully downsampled."""
    key = str(video_path.resolve())
    entry = log_data.get(key)
    if not entry or entry.get("status") != "success":
        return False
    # If the file is newer than the log entry, it may have been overwritten.
    try:
        mtime = video_path.stat().st_mtime
        if mtime > entry.get("timestamp", 0):
            return False
    except OSError:
        pass
    return True


def verify_output(
    original: Path,
    output: Path,
    original_info: dict[str, Any],
) -> tuple[bool, str]:
    """Verify that *output* is a valid downsampled version of *original*."""
    if not output.exists():
        return False, "output file does not exist"
    if output.stat().st_size == 0:
        return False, "output file is empty"

    output_info = get_video_info(output)
    if output_info is None:
        return False, "output is not a readable video"

    if output_info["height"] > MAX_HEIGHT:
        return False, f"output height {output_info['height']} exceeds {MAX_HEIGHT}"

    orig_dur = original_info.get("duration")
    out_dur = output_info.get("duration")
    if orig_dur and out_dur and orig_dur > 0:
        ratio = abs(orig_dur - out_dur) / orig_dur
        if ratio > DURATION_TOLERANCE:
            return False, (
                f"duration mismatch: original={orig_dur:.3f}s output={out_dur:.3f}s "
                f"({ratio*100:.1f}% difference)"
            )

    orig_frames = original_info.get("nb_frames")
    out_frames = output_info.get("nb_frames")
    if orig_frames and out_frames and orig_frames > 0:
        frame_ratio = abs(orig_frames - out_frames) / orig_frames
        if frame_ratio > 0.05:
            return False, (
                f"frame count mismatch: original={orig_frames} output={out_frames}"
            )

    return True, "ok"


def downsample_video(
    video_path: Path,
    log_data: dict[str, Any],
) -> dict[str, Any]:
    """Downsample a single video if needed, replacing the original on success."""
    result: dict[str, Any] = {
        "path": str(video_path),
        "status": "pending",
        "message": "",
        "timestamp": 0,
    }

    if is_already_processed(video_path, log_data):
        result["status"] = "skipped"
        result["message"] = "already processed"
        return result

    log(f"Processing {video_path}")

    # 1. Probe original.
    original_info = get_video_info(video_path)
    if original_info is None:
        result["status"] = "failed"
        result["message"] = "cannot probe original video"
        log(f"  FAILED: {result['message']}")
        return result

    result["original_info"] = original_info

    # 2. Skip if already at or below target height.
    if original_info["height"] <= MAX_HEIGHT:
        result["status"] = "skipped"
        result["message"] = f"already {original_info['height']}p"
        log(f"  SKIPPED: {result['message']}")
        return result

    # 3. Check disk space.
    if not has_enough_disk_space(video_path.parent):
        result["status"] = "failed"
        result["message"] = "insufficient disk space"
        log(f"  FAILED: {result['message']}")
        return result

    # 4. Build ffmpeg command.
    #    scale=-2:480 keeps width divisible by 2 and sets height to 480.
    tmp_fd = None
    tmp_path: Path | None = None
    try:
        tmp_fd, tmp_name = tempfile.mkstemp(
            suffix=video_path.suffix,
            prefix=video_path.stem + "_downsample_",
            dir=video_path.parent,
        )
        os.close(tmp_fd)
        tmp_path = Path(tmp_name)

        encoder, encoder_args = select_h264_encoder()
        cmd = [
            "ffmpeg",
            "-y",                       # overwrite temp output if needed
            "-hide_banner",
            "-loglevel", "error",
            "-i", str(video_path),
            "-vf", f"scale=-2:{MAX_HEIGHT},format=yuv420p",
            "-c:v", encoder,
            *encoder_args,
            "-c:a", "copy",             # keep audio untouched
            "-movflags", "+faststart",
            str(tmp_path),
        ]

        log(f"  Running: {' '.join(cmd)}")
        proc = run(cmd)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()}")

        # 5. Verify output.
        ok, msg = verify_output(video_path, tmp_path, original_info)
        if not ok:
            raise RuntimeError(f"verification failed: {msg}")

        # 6. Replace original with verified output.
        backup_path = video_path.with_suffix(video_path.suffix + ".backup")
        shutil.move(str(video_path), str(backup_path))
        try:
            shutil.move(str(tmp_path), str(video_path))
        except Exception:
            # Restore original if replacement fails.
            shutil.move(str(backup_path), str(video_path))
            raise

        # 7. Remove backup after successful replacement.
        try:
            backup_path.unlink()
        except OSError:
            pass

        new_info = get_video_info(video_path)
        result["status"] = "success"
        result["message"] = f"downsampled to {new_info['height']}p"
        result["new_info"] = new_info
        log(f"  SUCCESS: {result['message']}")

    except Exception as exc:
        result["status"] = "failed"
        result["message"] = str(exc)
        log(f"  FAILED: {result['message']}")
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass

    result["timestamp"] = video_path.stat().st_mtime if video_path.exists() else 0
    return result


def collect_videos(root: Path) -> list[Path]:
    """Collect all video files under the configured split folders."""
    videos: list[Path] = []
    for split in SPLIT_FOLDERS:
        split_path = root / split
        if not split_path.is_dir():
            log(f"Folder not found: {split_path}")
            continue
        for path in split_path.rglob("*"):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                videos.append(path)
    videos.sort()
    return videos


def load_log() -> dict[str, Any]:
    if LOG_FILE.exists():
        try:
            with LOG_FILE.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_log(log_data: dict[str, Any]) -> None:
    with LOG_FILE.open("w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    log("Starting video downsampling")
    log(f"Root directory: {ROOT_DIR}")
    log(f"Target max height: {MAX_HEIGHT}p")

    log_data = load_log()
    videos = collect_videos(ROOT_DIR)
    log(f"Found {len(videos)} video(s)")

    stats = {"success": 0, "failed": 0, "skipped": 0}

    try:
        for idx, video_path in enumerate(videos, start=1):
            log(f"[{idx}/{len(videos)}] {video_path}")
            result = downsample_video(video_path, log_data)
            log_data[str(video_path.resolve())] = result
            save_log(log_data)

            status = result["status"]
            if status in stats:
                stats[status] += 1
    except KeyboardInterrupt:
        log("Interrupted by user. Progress has been saved.")
        return 1

    log("Done.")
    log(f"Stats: {stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

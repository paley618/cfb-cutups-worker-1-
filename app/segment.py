import asyncio
import logging
import os
import signal

logger = logging.getLogger(__name__)

_CANCEL_EVENT: asyncio.Event | None = None


def ffmpeg_set_cancel(ev):  # called by runner
    global _CANCEL_EVENT
    _CANCEL_EVENT = ev


async def _run(cmd: list[str]):
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    while True:
        if _CANCEL_EVENT is not None and _CANCEL_EVENT.is_set():
            try:
                proc.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=0.5)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg error ({proc.returncode}): {err.decode(errors='ignore')[:400]}"
                )
            return out, err
        except asyncio.TimeoutError:
            continue


def _ensure_parent(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


async def cut_clip(src: str, dst: str, start: float, end: float):
    """Cut a clip from video with diagnostic logging.

    Args:
        src: Source video path
        dst: Destination clip path
        start: Start timestamp in seconds
        end: End timestamp in seconds
    """
    _ensure_parent(dst)
    duration = max(0.01, end - start)

    logger.info(f"[CLIP GENERATION] Creating clip")
    logger.info(f"  Source: {src}")
    logger.info(f"  Output: {dst}")
    logger.info(f"  Window: {start:.1f}s - {end:.1f}s")
    logger.info(f"  Duration: {duration:.1f}s")

    fast = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.3f}",
        "-i",
        src,
        "-t",
        f"{duration:.3f}",
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        dst,
    ]
    try:
        logger.info(f"  Attempting fast copy mode (no re-encoding)...")
        await _run(fast)
        logger.info(f"  ✓ Clip created successfully using fast copy mode")
        return
    except Exception as e:  # noqa: BLE001
        logger.warning(f"  ⚠️  Fast copy failed: {str(e)[:100]}")
        logger.info(f"  Falling back to re-encoding mode...")

    accurate = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.3f}",
        "-i",
        src,
        "-t",
        f"{duration:.3f}",
        "-vf",
        "scale=w='if(gt(iw,1920),1920,iw)':h=-2",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-movflags",
        "+faststart",
        dst,
    ]
    try:
        await _run(accurate)
        logger.info(f"  ✓ Clip created successfully using re-encoding mode")
    except Exception as e:
        logger.error(f"  ❌ Clip creation failed: {str(e)[:200]}")
        raise


async def make_thumb(src: str, t: float, dst: str):
    _ensure_parent(dst)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{t:.3f}",
        "-i",
        src,
        "-frames:v",
        "1",
        "-q:v",
        "2",
        dst,
    ]
    await _run(cmd)


async def concatenate_clips(filelist_path: str, dst: str):
    _ensure_parent(dst)
    copy_cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        filelist_path,
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        dst,
    ]
    try:
        await _run(copy_cmd)
        return
    except Exception:  # noqa: BLE001
        pass
    reencode_cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        filelist_path,
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "20",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        dst,
    ]
    await _run(reencode_cmd)

import asyncio
import os


async def _run(cmd: list[str]):
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    out, err = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg error ({proc.returncode}): {err.decode(errors='ignore')[:400]}"
        )
    return out, err


def _ensure_parent(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


async def cut_clip(src: str, dst: str, start: float, end: float):
    _ensure_parent(dst)
    duration = max(0.01, end - start)
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
        await _run(fast)
        return
    except Exception:
        pass
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
    await _run(accurate)


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
    except Exception:
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

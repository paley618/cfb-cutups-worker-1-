from fastapi import APIRouter, Query

router = APIRouter()


@router.get("/api/util/video-meta")
def video_meta(videoUrl: str = Query(...)):
    # minimal stub – real impl may call ffprobe in your worker
    # for now just return the URL so the orchestrator has a field
    return {
        "status": "VIDEO_META_STUB",
        "videoUrl": videoUrl,
        "durationSeconds": None,
    }

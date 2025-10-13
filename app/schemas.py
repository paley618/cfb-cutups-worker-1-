from typing import Optional

from pydantic import BaseModel, HttpUrl, model_validator


class Options(BaseModel):
    play_padding_pre: float = 3.0
    play_padding_post: float = 5.0
    scene_thresh: float = 0.30
    min_duration: float = 4.0
    max_duration: float = 20.0


class JobSubmission(BaseModel):
    video_url: Optional[HttpUrl] = None
    upload_id: Optional[str] = None
    presigned_url: Optional[HttpUrl] = None
    webhook_url: Optional[HttpUrl] = None
    options: Options = Options()

    @model_validator(mode="after")
    def _require_source(self):
        if not (self.video_url or self.upload_id or self.presigned_url):
            raise ValueError("Provide either video_url, upload_id, or presigned_url.")
        return self

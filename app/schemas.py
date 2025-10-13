from typing import Optional
from pydantic import BaseModel, HttpUrl, model_validator


class Options(BaseModel):
    play_padding_pre: float = 3.0
    play_padding_post: float = 5.0


class JobSubmission(BaseModel):
    # Provide EITHER video_url OR upload_id OR presigned_url
    video_url: Optional[HttpUrl] = None          # e.g., YouTube/Vimeo/Drive/Dropbox/etc.
    upload_id: Optional[str] = None              # server-side uploaded file handle
    presigned_url: Optional[HttpUrl] = None      # direct download (S3/Cloud Storage)
    webhook_url: Optional[HttpUrl] = None
    options: Options = Options()

    @model_validator(mode="after")
    def _require_source(self):
        if not (self.video_url or self.upload_id or self.presigned_url):
            raise ValueError("Provide either video_url, upload_id, or presigned_url.")
        return self

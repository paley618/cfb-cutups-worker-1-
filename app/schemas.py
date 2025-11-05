from typing import Optional

from pydantic import AliasChoices, BaseModel, Field, HttpUrl, model_validator


class Options(BaseModel):
    play_padding_pre: float = 3.0
    play_padding_post: float = 5.0
    scene_thresh: float = 0.30
    min_duration: float = 4.0
    max_duration: float = 20.0


class CFBDInput(BaseModel):
    use_cfbd: bool = False
    game_id: Optional[int] = None
    season: Optional[int] = Field(
        default=None, validation_alias=AliasChoices("season", "year", "cfbd_year")
    )
    week: Optional[int] = Field(default=None, validation_alias=AliasChoices("week", "cfbd_week"))
    team: Optional[str] = None
    season_type: Optional[str] = "regular"

    @property
    def year(self) -> Optional[int]:
        return self.season


class JobSubmission(BaseModel):
    video_url: Optional[HttpUrl] = None
    upload_id: Optional[str] = None
    presigned_url: Optional[HttpUrl] = None
    webhook_url: Optional[HttpUrl] = None
    options: Options = Options()
    cfbd: Optional[CFBDInput] = None

    @model_validator(mode="after")
    def _require_source(self):
        if not (self.video_url or self.upload_id or self.presigned_url):
            raise ValueError("Provide either video_url, upload_id, or presigned_url.")
        return self

from typing import Dict, List, Literal, Optional, TypedDict

from pydantic import AliasChoices, BaseModel, Field, HttpUrl, model_validator


class Options(BaseModel):
    play_padding_pre: float = 3.0
    play_padding_post: float = 5.0
    scene_thresh: float = 0.30
    min_duration: float = 4.0
    max_duration: float = 20.0


class CFBDInput(BaseModel):
    use_cfbd: bool = False
    require_cfbd: bool = False
    game_id: Optional[int] = None
    season: Optional[int] = Field(
        default=None, validation_alias=AliasChoices("season", "year", "cfbd_year")
    )
    week: Optional[int] = Field(default=None, validation_alias=AliasChoices("week", "cfbd_week"))
    team: Optional[str] = None
    season_type: Optional[str] = Field(
        default="regular",
        validation_alias=AliasChoices("season_type", "seasonType"),
    )
    home_team: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("home_team", "homeTeam"),
    )
    away_team: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("away_team", "awayTeam"),
    )

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


BucketName = Literal["team_offense", "opp_offense", "special_teams"]


class ClipItem(TypedDict):
    id: str
    start: float
    end: float
    duration: float
    file: str
    thumb: str
    bucket: BucketName
    score: float


class Manifest(TypedDict, total=False):
    job_id: str
    source_url: str
    source: Dict[str, object]
    clips: List[Dict[str, object]]
    buckets: Dict[BucketName, List[ClipItem]]
    bucket_counts: Dict[BucketName, int]
    detector_meta: Dict[str, object]

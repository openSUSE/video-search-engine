from typing import List, Optional

from pydantic import BaseModel, confloat, conint, constr, field_validator, model_validator

from api.search_utils import is_usable_video_filename_scope

QueryString = constr(strip_whitespace=True, min_length=1, max_length=1000)
VideoFilenameString = constr(strip_whitespace=True, min_length=1, max_length=512)
SearchLimit = conint(strict=True, ge=1, le=50)
ResultString = constr(strip_whitespace=True, min_length=1)
SpeakersString = constr(strip_whitespace=True)
NonNegativeFiniteFloat = confloat(ge=0, allow_inf_nan=False)


class SearchQuery(BaseModel):
    query: QueryString
    top_k: SearchLimit = 5
    video_filename: Optional[VideoFilenameString] = None

    @field_validator("video_filename")
    @classmethod
    def validate_video_filename_scope(cls, value):
        if value is not None and not is_usable_video_filename_scope(value):
            raise ValueError("video_filename must be a filename")
        return value


class SearchResult(BaseModel):
    id: ResultString
    score: NonNegativeFiniteFloat
    start_time: NonNegativeFiniteFloat
    end_time: NonNegativeFiniteFloat
    title: ResultString
    summary: ResultString
    video_filename: ResultString
    speakers: SpeakersString

    @field_validator("score", "start_time", "end_time", mode="before")
    @classmethod
    def reject_bool_numbers(cls, value, info):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{info.field_name} must be a number")
        return value

    @field_validator("video_filename")
    @classmethod
    def validate_video_filename_scope(cls, value):
        if not is_usable_video_filename_scope(value):
            raise ValueError("video_filename must be a filename")
        return value

    @model_validator(mode="after")
    def validate_time_range(self):
        if self.end_time < self.start_time:
            raise ValueError("end_time must be greater than or equal to start_time")
        return self


class SearchResponse(BaseModel):
    results: List[SearchResult]

from typing import List, Optional

from pydantic import BaseModel, conint, constr

NonEmptyString = constr(strip_whitespace=True, min_length=1)
SearchLimit = conint(ge=1, le=50)


class SearchQuery(BaseModel):
    query: NonEmptyString
    top_k: SearchLimit = 5
    video_filename: Optional[NonEmptyString] = None


class SearchResult(BaseModel):
    id: str
    score: float
    start_time: float
    end_time: float
    title: str
    summary: str
    video_filename: str
    speakers: str


class SearchResponse(BaseModel):
    results: List[SearchResult]

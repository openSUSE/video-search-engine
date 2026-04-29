from typing import List, Optional

from pydantic import BaseModel, conint, constr

QueryString = constr(strip_whitespace=True, min_length=1, max_length=1000)
VideoFilenameString = constr(strip_whitespace=True, min_length=1, max_length=512)
SearchLimit = conint(ge=1, le=50)


class SearchQuery(BaseModel):
    query: QueryString
    top_k: SearchLimit = 5
    video_filename: Optional[VideoFilenameString] = None


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

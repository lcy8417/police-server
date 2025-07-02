from typing import List, Tuple, Optional
from pydantic import BaseModel


class SegmentRequest(BaseModel):
    image: str
    polygon: List[Tuple[float, float]]


class BinarizationRequest(BaseModel):
    image: str
    threshold: int = 127
    type: str = "standard"


class PatternsExtractRequest(BaseModel):
    image: str
    type: str = "crime"
    line_ys: List[int]
    render_size: List[float]


class CrimeSearchRequest(BaseModel):
    image: str
    top: Optional[List[str]] = []
    mid: Optional[List[str]] = []
    bottom: Optional[List[str]] = []
    outline: Optional[List[str]] = []

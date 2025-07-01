from typing import List, Tuple
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
    top: List[str] = []
    mid: List[str] = []
    bottom: List[str] = []
    outline: List[str] = []


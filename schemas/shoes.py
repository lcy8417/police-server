from pydantic import BaseModel, Field
from typing import List, Optional, Tuple


class ShoesRequest(BaseModel):
    findLocation: str = Field(None, description="수집장소")
    manufacturer: str = Field(None, description="제조사")
    emblem: str = Field(None, description="상표명")
    modelNumber: str = Field(..., description="모델번호")
    findYear: int = Field(None, description="수집연도")
    top: List[str] = Field(None, description="신발 상단 폴리곤 좌표")
    mid: List[str] = Field(None, description="신발 중단 폴리곤 좌표")
    bottom: List[str] = Field(None, description="신발 하단 폴리곤 좌표")
    outline: List[str] = Field(None, description="신발 외곽 폴리곤 좌표")
    image: str = Field(..., description="신발 이미지 (base64 인코딩된 문자열)")

class ShoesUpdate(BaseModel):
    findLocation: str = Field(None, description="수집장소")
    manufacturer: str = Field(None, description="제조사")
    emblem: str = Field(None, description="상표명")
    findYear: int = Field(None, description="수집연도")
    top: List[str] = Field(None, description="신발 상단 폴리곤 좌표")
    mid: List[str] = Field(None, description="신발 중단 폴리곤 좌표")
    bottom: List[str] = Field(None, description="신발 하단 폴리곤 좌표")
    outline: List[str] = Field(None, description="신발 외곽 폴리곤 좌표")

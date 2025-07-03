from pydantic import BaseModel, Field
from typing import List, Optional, Tuple, Optional


class ShoesRequest(BaseModel):
    findLocation: str = Field(None, description="수집장소")
    manufacturer: str = Field(None, description="제조사")
    emblem: str = Field(None, description="상표명")
    modelNumber: str = Field(..., description="모델번호")
    findYear: int = Field(None, description="수집연도")
    top: Optional[List[str]] = Field([], description="신발 상단 폴리곤 좌표")
    mid: Optional[List[str]] = Field([], description="신발 중단 폴리곤 좌표")
    bottom: Optional[List[str]] = Field([], description="신발 하단 폴리곤 좌표")
    outline: Optional[List[str]] = Field([], description="신발 외곽 폴리곤 좌표")
    image: str = Field(..., description="신발 이미지 (base64 인코딩된 문자열)")


class ShoesUpdate(BaseModel):
    findLocation: Optional[str] = Field(None, description="수집장소")
    manufacturer: Optional[str] = Field(None, description="제조사")
    emblem: Optional[str] = Field(None, description="상표명")
    findYear: Optional[int] = Field(0, description="수집연도")
    top: Optional[List[str]] = Field([], description="신발 상단 폴리곤 좌표")
    mid: Optional[List[str]] = Field([], description="신발 중단 폴리곤 좌표")
    bottom: Optional[List[str]] = Field([], description="신발 하단 폴리곤 좌표")
    outline: Optional[List[str]] = Field([], description="신발 외곽 폴리곤 좌표")

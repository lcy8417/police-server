from typing import List, Optional, Tuple, Union
from pydantic import BaseModel, Field
from datetime import datetime


# ✅ 요청(Request) 모델
class RegisterForm(BaseModel):
    image: str
    crimeNumber: str
    imageNumber: Optional[str] = None
    crimeName: Optional[str] = None
    findTime: Optional[str] = None
    requestOffice: Optional[str] = None
    findMethod: Optional[str] = None
    state: int = 0
    ranking: int = 0
    matchingShoes: Optional[str] = None


# ✅ 요청(Request) 모델
class CrimeHistory(BaseModel):
    image: str
    registerTime: datetime = datetime.now()
    top: List[List[Union[str, int]]]
    mid: List[List[Union[str, int]]]
    bottom: List[List[Union[str, int]]]
    outline: List[List[Union[str, int]]]
    ranking: int = 0
    matchingShoes: str = ""
    # zoom: Optional[int] = 0
    # contrast: Optional[int] = 0
    # saturation: Optional[int] = 0
    # brightness: Optional[int] = 0
    # rotate: Optional[int] = 0


class PatternUpdate(BaseModel):
    top: List[Tuple[str, int]] = []
    mid: List[Tuple[str, int]] = []
    bottom: List[Tuple[str, int]] = []
    outline: List[Tuple[str, int]] = []


class EditImageInsert(BaseModel):
    image: str = None

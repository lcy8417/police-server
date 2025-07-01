from typing import List, Optional, Tuple
from pydantic import BaseModel
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
    top: List[Tuple[str, int]] = []
    mid: List[Tuple[str, int]] = []
    bottom: List[Tuple[str, int]] = []
    outline: List[Tuple[str, int]] = []
    # zoom: Optional[int] = 0
    # contrast: Optional[int] = 0
    # saturation: Optional[int] = 0
    # brightness: Optional[int] = 0
    # rotate: Optional[int] = 0

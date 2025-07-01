from fastapi import APIRouter, Path

router = APIRouter(prefix="/crime", tags=["Search"])


@router.post("/{id}/search")
async def search_similar_shoes(id: int):
    # 필수 문양, 이미지, 페이지 수로 검색
    pass


@router.post("/{id}/search/{sid}")
async def get_search_result(id: int, sid: int):
    # 신발의 디테일 검색 결과
    pass

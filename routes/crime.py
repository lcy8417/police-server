from fastapi import APIRouter, Depends, status, Body, Path, Query
from fastapi.exceptions import HTTPException
from fastapi.responses import FileResponse, JSONResponse

from sqlalchemy import text, Connection
from sqlalchemy.exc import SQLAlchemyError
from db.database import context_get_conn

from typing import Optional, List
from pydantic import BaseModel
from datetime import datetime
import base64
import os.path as osp
import os


router = APIRouter(prefix="/crime", tags=["Crime"])


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


# 등록 로직
@router.post("/register")
async def register_crime(
    data: RegisterForm,
    conn: Connection = Depends(context_get_conn),
):
    try:

        query = f"""
            INSERT INTO crimeData (crimeNumber, imageNumber, crimeName, findTime, requestOffice)
            VALUES (:crimeNumber, :imageNumber, :crimeName, :findTime, :requestOffice)
        """

        # 인코딩된 이미지 데이터를 base64로 디코딩하여 저장할 수 있습니다.
        image_data = base64.b64decode(data.image)
        save_path = f"static/crime_images/{data.crimeNumber}.png"

        with open(save_path, "wb") as f:
            f.write(image_data)

        conn.execute(
            text(query),
            {
                "crimeNumber": data.crimeNumber,
                "crimeName": data.crimeName,
                "imageNumber": data.imageNumber,
                "findTime": data.findTime,
                "requestOffice": data.requestOffice,
            },
        )
        conn.commit()

        return FileResponse(save_path, media_type="image/png")

    except SQLAlchemyError as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="요청하신 서비스가 잠시 내부적으로 문제가 발생하였습니다.",
        )
    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="알수없는 이유로 서비스 오류가 발생하였습니다",
        )


# 전체 검색 로직. 스키마에 따른 필터링을 하이퍼 파라미터로 줄 수 있음.
@router.get("/")
async def get_crimes(conn: Connection = Depends(context_get_conn)):
    try:

        query = f"""
            SELECT * FROM crimeData
        """

        result = conn.execute(text(query))

        if result.rowcount == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="조회된 사건이 없습니다.",
            )

        img_root = "http://localhost:8000/crime_images"

        rows = result.fetchall()
        data = [dict(row._mapping) for row in rows]
        data = [
            {
                **dict(row._mapping),
                "image": osp.join(img_root, f"{row._mapping['crimeNumber']}.png"),
            }
            for row in rows
        ]
        result.close()

        return JSONResponse(content=data)

    except SQLAlchemyError as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="요청하신 서비스가 잠시 내부적으로 문제가 발생하였습니다.",
        )


# crimeNumber에서 검색한 이력
@router.get("/{crimeNumber}")
async def get_crime_detail(
    crimeNumber: str, conn: Connection = Depends(context_get_conn)
):
    query = """
    SELECT crimeNumber, ranking, registerTime FROM crimeDataHistory WHERE crimeNumber = :crimeNumber ORDER BY id
    """

    try:
        result = conn.execute(text(query), {"crimeNumber": crimeNumber})
        rows = result.fetchall()
        data = [dict(row._mapping) for row in rows]
        result.close()

        # datetime 처리
        for row in data:
            if isinstance(row.get("registerTime"), datetime):
                row["registerTime"] = row["registerTime"].strftime("%Y-%m-%d %H:%M:%S")

        return JSONResponse(content=data)
    except SQLAlchemyError as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="요청하신 서비스가 잠시 내부적으로 문제가 발생하였습니다.",
        )
    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="알수없는 이유로 서비스 오류가 발생하였습니다",
        )


class CrimeHistory(BaseModel):
    image: str
    zoom: Optional[int] = 0
    contrast: Optional[int] = 0
    saturation: Optional[int] = 0
    brightness: Optional[int] = 0
    rotate: Optional[int] = 0


# 원본 이미지에서 편집된 이미지 등록.
@router.post("/{crimeNumber}")
async def update_crime(
    crimeNumber: str = Path(...),
    registerTime: datetime = datetime.now(),
    ranking: int = 0,
    data: CrimeHistory = Body(...),
    conn: Connection = Depends(context_get_conn),
):

    # 기존에 있는 데이터 조회
    existing_query = """
    SELECT * FROM crimeDataHistory WHERE crimeNumber = :crimeNumber ORDER BY id DESC LIMIT 1
    """

    # 검색 히스토리 등록
    query = """
    INSERT INTO crimeDataHistory (crimeNumber, registerTime, ranking)
    VALUES (:crimeNumber, :registerTime, :ranking)
    """

    try:

        existing_result = conn.execute(
            text(existing_query), {"crimeNumber": crimeNumber}
        )
        count = len(existing_result.fetchall())

        existing_result.close()

        root = "static/crime_history/"
        folder_path = osp.join(root, crimeNumber)

        if not osp.exists(folder_path):
            os.makedirs(folder_path)

        # 인코딩된 image를 base64로 디코딩하여 저장
        image_data = base64.b64decode(data.image)
        save_path = osp.join(folder_path, f"{count + 1}.png")
        with open(save_path, "wb") as f:
            f.write(image_data)

        conn.execute(
            text(query),
            {
                "crimeNumber": crimeNumber,
                "registerTime": registerTime,
                "ranking": ranking,
            },
        )
        conn.commit()

    except SQLAlchemyError as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="요청하신 서비스가 잠시 내부적으로 문제가 발생하였습니다.",
        )
    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="알수없는 이유로 서비스 오류가 발생하였습니다",
        )

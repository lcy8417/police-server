from fastapi import APIRouter, Depends, status, Body, Path, Query
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse

from sqlalchemy import text, Connection
from sqlalchemy.exc import SQLAlchemyError
from db.database import context_get_conn

from typing import Optional, List, Tuple
from datetime import datetime
import base64
import os.path as osp
import os
from ast import literal_eval
import json

from schemas.shoes import ShoesRequest, ShoesUpdate

# 주석

router = APIRouter(prefix="/shoes", tags=["Shoes"])


@router.post("/register")
async def register_shoes(
    data: ShoesRequest = Body(..., description="신발 등록 정보"),
    conn: Connection = Depends(context_get_conn),
):
    try:
        query = """
        INSERT INTO shoesData (
            findLocation, manufacturer, emblem, modelNumber, findYear, top, mid, bottom
            ) VALUES (
            :findLocation, :manufacturer, :emblem, :modelNumber, :findYear, :top, :mid, :bottom
        )
        """
        # 데이터베이스에 신발 정보 저장

        conn.execute(
            text(query),
            {
                "findLocation": data.findLocation,
                "manufacturer": data.manufacturer,
                "emblem": data.emblem,
                "modelNumber": data.modelNumber,
                "findYear": data.findYear,
                "top": json.dumps(data.top) if data.top else json.dumps([]),
                "mid": json.dumps(data.mid) if data.mid else json.dumps([]),
                "bottom": json.dumps(data.bottom) if data.bottom else json.dumps([]),
            },
        )
        conn.commit()

        # 인코딩된 문자열 image를 디코딩하여 파일로 저장
        if data.image:
            image_data = base64.b64decode(data.image)
            image_path = osp.join("static/shoes_images/B", f"{data.modelNumber}.png")

            with open(image_path, "wb") as image_file:
                image_file.write(image_data)

        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="이미지 데이터가 제공되지 않았습니다.",
            )

        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={"message": "신발 정보가 성공적으로 등록되었습니다."},
        )

    except SQLAlchemyError as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="요청하신 서비스가 잠시 내부적으로 문제가 발생하였습니다.",
        )


@router.put("/{modelNumber}")
async def shoes_info_update(
    modelNumber: str = Path(..., description="수정할 신발의 모델 번호"),
    data: ShoesUpdate = Body(..., description="신발 수정 정보"),
    conn: Connection = Depends(context_get_conn),
):
    try:
        query = """
        UPDATE shoesData
        SET modelNumber = :modelNumber, findLocation = :findLocation, manufacturer = :manufacturer,
            emblem = :emblem, top = :top, mid = :mid, bottom = :bottom, outline = :outline
        
        WHERE modelNumber = :modelNumber
        """

        conn.execute(
            text(query),
            {
                "modelNumber": str(modelNumber),
                "findLocation": data.findLocation,
                "manufacturer": data.manufacturer,
                "findYear": data.findYear,
                "emblem": data.emblem,
                "top": json.dumps(data.top),
                "mid": json.dumps(data.mid),
                "bottom": json.dumps(data.bottom),
                "outline": json.dumps(data.outline),
            },
        )
        conn.commit()

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"message": "신발 정보가 성공적으로 업데이트되었습니다."},
        )

    except SQLAlchemyError as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="요청하신 서비스가 잠시 내부적으로 문제가 발생하였습니다.",
        )


@router.get("/")
async def get_all_shoes(
    page: int = Query(0, ge=0), conn: Connection = Depends(context_get_conn)
):
    try:
        query = """
        SELECT modelNumber, findLocation, manufacturer, findYear, emblem, top, mid, bottom FROM shoesData
        ORDER BY id DESC
        LIMIT 10 OFFSET :offset
        """
        offset = page * 10
        result = conn.execute(text(query), {"offset": offset})
        rows = result.fetchall()

        if not rows:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"message": "등록된 신발 정보가 없습니다."},
            )
        img_root = "http://localhost:8000/shoes_images/B"

        data = [dict(row._mapping) for row in rows]
        data = [
            {
                **dict(row._mapping),
                "image": osp.join(img_root, f"{row._mapping['modelNumber']}.png"),
                "top": (
                    json.loads(row._mapping["top"]) if row._mapping["top"] else None
                ),
                "mid": (
                    json.loads(row._mapping["mid"]) if row._mapping["mid"] else None
                ),
                "bottom": (
                    json.loads(row._mapping["bottom"])
                    if row._mapping["bottom"]
                    else None
                ),
            }
            for row in rows
        ]
        result.close()

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=data,
        )

    except SQLAlchemyError as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="요청하신 서비스가 잠시 내부적으로 문제가 발생하였습니다.",
        )


@router.get("/{id}")
async def get_shoe_detail(id: int):
    pass


@router.put("/{id}/edit")
async def edit_shoe(id: int, data: dict):
    pass


@router.get("/{id}/extract")
async def extract_shoe_pattern(id: int):
    pass

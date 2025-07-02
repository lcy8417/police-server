from fastapi import APIRouter, Depends, status, Body, Path, Query
from fastapi.exceptions import HTTPException
from fastapi.responses import FileResponse, JSONResponse

from sqlalchemy import text, Connection
from sqlalchemy.exc import SQLAlchemyError
from db.database import context_get_conn

import json
from datetime import datetime
import base64
import os.path as osp
import os
from utils.json_utils import safe_json_loads

from schemas.crime import RegisterForm, CrimeHistory, PatternUpdate, EditImageInsert


router = APIRouter(prefix="/crime", tags=["Crime"])


# 등록 로직
@router.post("/register")
async def register_crime(
    data: RegisterForm,
    conn: Connection = Depends(context_get_conn),
):
    try:

        query = f"""
            INSERT INTO crime_data (crime_number, image_number, crime_name, find_time, request_office)
            VALUES (:crime_number, :image_number, :crime_name, :find_time, :request_office)
        """

        # 인코딩된 이미지 데이터를 base64로 디코딩하여 저장할 수 있습니다.
        image_data = base64.b64decode(data.image)
        save_path = f"static/crime_images/{data.crimeNumber}.png"

        if osp.exists(save_path):
            os.remove(save_path)

        with open(save_path, "wb") as f:
            f.write(image_data)

        conn.execute(
            text(query),
            {
                "crime_number": data.crimeNumber,
                "crime_name": data.crimeName,
                "image_number": data.imageNumber,
                "find_time": data.findTime,
                "request_office": data.requestOffice,
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
            SELECT * FROM crime_data
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
                "image": osp.join(img_root, f"{row._mapping['crime_number']}.png"),
                "top": safe_json_loads(row._mapping["top"]),
                "mid": safe_json_loads(row._mapping["mid"]),
                "bottom": safe_json_loads(row._mapping["bottom"]),
                "outline": safe_json_loads(row._mapping["outline"]),
                "editImage": (
                    ("data:image/png;base64," + row._mapping["edit_image"])
                    if row._mapping["edit_image"]
                    else None
                ),
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


# crime_number에서의 히스토리 데이터 조회
@router.get("/{crime_number}")
async def get_crime_detail(
    crime_number: str, conn: Connection = Depends(context_get_conn)
):
    query = """
    SELECT id, register_time, ranking, matching_shoes FROM crime_data_history WHERE crime_number = :crime_number ORDER BY id
    """

    try:
        result = conn.execute(text(query), {"crime_number": crime_number})
        rows = result.fetchall()
        data = [dict(row._mapping) for row in rows]
        result.close()

        # datetime 처리
        for row in data:
            if isinstance(row.get("register_time"), datetime):
                row["register_time"] = row["register_time"].strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

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


# 현재 편집 이미지인지 조회
@router.get("/{crime_number}/image_load")
async def get_crime_detail(
    crime_number: str = Path(..., description="조회할 사건의 번호"),
    edit: bool = Query(False, description="편집된 이미지 여부"),
    conn: Connection = Depends(context_get_conn),
):
    column = "edit_image" if edit else "crime_number"

    query = f"""
    SELECT {column} FROM crime_data WHERE crime_number = :crime_number 
    """

    try:
        result = conn.execute(text(query), {"crime_number": crime_number})
        row = result.fetchone()

        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="해당 사건의 이미지가 없습니다.",
            )

        image_data = row[0]

        return JSONResponse(
            content={
                "image": ("data:image/png;base64," + image_data if edit else image_data)
            }
        )
    except SQLAlchemyError as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="요청하신 서비스가 잠시 내부적으로 문제가 발생하였습니다.",
        )


# crime_number의 history에서 id로 데이터 조회
@router.get("/history/{id}")
async def get_crime_detail(
    id: int = Path(..., description="조회할 히스토리의 ID"),
    conn: Connection = Depends(context_get_conn),
):
    query = """
    SELECT edit_image, matching_shoes, top, mid, bottom, outline FROM crime_data_history WHERE  id = :id
    """

    try:
        result = conn.execute(text(query), {"id": id})
        data = dict(result.fetchall()[0]._mapping)
        for key in ["top", "mid", "bottom", "outline"]:
            if data[key] is not None:
                data[key] = json.loads(data[key])
            else:
                data[key] = []

        result.close()

        if data.get("edit_image"):
            data["edit_image"] = "data:image/png;base64," + data.get("edit_image", "")

        return JSONResponse(content=data)
    except SQLAlchemyError as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="요청하신 서비스가 잠시 내부적으로 문제가 발생하였습니다.",
        )


# 원본 이미지에서 편집된 정보 등록.
@router.post("/{crime_number}")
async def update_crime(
    crime_number: str = Path(...),
    data: CrimeHistory = Body(...),
    conn: Connection = Depends(context_get_conn),
):

    # 기존에 있는 데이터 조회
    existing_query = """
    SELECT * FROM crime_data_history WHERE crime_number = :crime_number ORDER BY id DESC LIMIT 1
    """

    # 검색 히스토리 등록
    query = """
    INSERT INTO crime_data_history (crime_number, edit_image, register_time, ranking, top, mid, bottom ,outline, matching_shoes)
    VALUES ( :crime_number, :edit_image, :register_time, :ranking, :top, :mid, :bottom, :outline, :matching_shoes)
    """

    try:

        existing_result = conn.execute(
            text(existing_query), {"crime_number": crime_number}
        )
        count = len(existing_result.fetchall())

        existing_result.close()

        # root = "static/crime_history/"
        # folder_path = osp.join(root, crime_number)

        # if not osp.exists(folder_path):
        #     os.makedirs(folder_path)

        # print(data.image)
        # # 인코딩된 image를 base64로 디코딩하여 저장
        # image_data = base64.b64decode(data.image)
        # save_path = osp.join(folder_path, f"{count + 1}.png")
        # with open(save_path, "wb") as f:
        #     f.write(image_data)

        conn.execute(
            text(query),
            {
                "crime_number": crime_number,
                "register_time": data.registerTime,
                "ranking": data.ranking,
                "top": json.dumps(data.top),
                "mid": json.dumps(data.mid),
                "bottom": json.dumps(data.bottom),
                "outline": json.dumps(data.outline),
                "matching_shoes": data.matchingShoes,
                "edit_image": data.editImage,
            },
        )
        conn.commit()

    except SQLAlchemyError as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="요청하신 서비스가 잠시 내부적으로 문제가 발생하였습니다.",
        )


@router.put("/{crime_number}")
async def patterns_info_update(
    crime_number: str = Path(..., description="패턴을 입력할 사건의 번호"),
    data: PatternUpdate = Body(..., description="패턴 입력 정보"),
    conn: Connection = Depends(context_get_conn),
):
    try:
        query = """
        UPDATE crime_data
        SET top = :top, mid = :mid, bottom = :bottom, outline = :outline WHERE crime_number = :crime_number
        """

        conn.execute(
            text(query),
            {
                "crime_number": crime_number,
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


@router.put("/edit_image/{crime_number}")
async def edit_image_update(
    crime_number: str = Path(..., description="패턴을 입력할 사건의 번호"),
    data: EditImageInsert = Body(..., description="패턴 입력 정보"),
    conn: Connection = Depends(context_get_conn),
):
    try:
        query = """
        UPDATE crime_data
        SET edit_image = :edit_image WHERE crime_number = :crime_number
        """

        conn.execute(
            text(query),
            {
                "crime_number": crime_number,
                "edit_image": data.image if data.image else None,
            },
        )
        conn.commit()

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"message": "편집 이미지가 성공적으로 업로드 되었습니다."},
        )

    except SQLAlchemyError as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="요청하신 서비스가 잠시 내부적으로 문제가 발생하였습니다.",
        )

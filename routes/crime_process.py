from fastapi import APIRouter, UploadFile, File, Path, Depends, status, Query, Body
from typing import Optional, List, Tuple
from sqlalchemy import text, Connection
from db.database import context_get_conn
from sqlalchemy.exc import SQLAlchemyError
from fastapi.exceptions import HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse
import cv2
import base64
from fastapi.responses import JSONResponse
import os.path as osp
from utils.utils import path_to_seg, bytes_to_np
import numpy as np
from models import models, query_cache
from utils.utils import (
    name_to_pil,
    bytes_to_pil,
    pil_to_bytes,
    inpainting_prepaire,
    np_to_bytes,
    patterns_prepaire,
    search_prepare,
)
from models.denoising.pipline import pipline_retinex
from schemas.crime_process import (
    SegmentRequest,
    BinarizationRequest,
    PatternsExtractRequest,
    CrimeSearchRequest,
)
import json
import torch
from models.image_search.image_search_adapter import ImageSearchAdapter
from db.database import context_get_conn


router = APIRouter(prefix="/crime", tags=["Crime Processing"])


@router.post("/{crime_number}/segmentation")
async def segment_image(
    crime_number: str = Path(...),
    render_size: List[float] = Query(...),
    data: SegmentRequest = Body(...),
):

    try:
        seg_image = path_to_seg(data.polygon, render_size, data.image)

        return JSONResponse(content={"image": seg_image})

    except SQLAlchemyError as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="요청하신 서비스가 잠시 내부적으로 문제가 발생하였습니다.",
        )


@router.post("/{crime_number}/denoising")
async def denoise_image(crime_number: str = Path(...), image_data: str = Body(...)):
    try:
        model = models.get("denoising")

        # 이미지 데이터가 base64로 인코딩된 경우
        if not image_data.startswith("data:image"):
            img = name_to_pil("static/crime_images/", image_data)  # 예시
        else:
            img = bytes_to_pil(image_data.split(",")[1])

        res_img, *_ = pipline_retinex(model, img)

        encoded_image = f"data:image/png;base64,{pil_to_bytes(res_img)}"
        return JSONResponse(content={"image": encoded_image})
    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="이미지 처리 중 오류가 발생했습니다.",
        )


@router.post("/{crime_number}/inpainting")
async def inpaint_image(
    crime_number: str = Path(...),
    render_size: List[float] = Query(...),
    data: SegmentRequest = Body(...),
):
    model = models.get("inpainting")
    src_img, mask_img = inpainting_prepaire(
        polygon=data.polygon,
        render_size=render_size,
        image_data=data.image,
    )

    res_img = model.predict(src_img, mask_img)

    return JSONResponse(
        content={"image": f"data:image/png;base64,{np_to_bytes(res_img)}"}
    )

import time

@router.post("/{crime_number}/binarization")
async def binarization_image(
    crime_number: str = Path(...),
    data: BinarizationRequest = Body(...),
):
    start_time = time.time()

    print(query_cache, crime_number)
    if crime_number not in query_cache:
        if data.image.startswith("data:image"):
            # base64 인코딩된 이미지 데이터 처리
            query_cache[crime_number] = bytes_to_np(data.image.split(",")[1], read_type="gray")
        else:
            query_cache[crime_number] = cv2.imread(
                osp.join("static/crime_images", crime_number + ".png"),
                cv2.IMREAD_GRAYSCALE,
            )


    img = query_cache[crime_number]

    max_value = 255
    threshold = data.threshold
    t_type = data.type.lower()  # 소문자로 통일

    # 다양한 이진화 방식
    if t_type == "standard":
        _, binary_img = cv2.threshold(img, threshold, max_value, cv2.THRESH_BINARY)
    elif t_type == "standard_inv":
        _, binary_img = cv2.threshold(
            img, threshold, max_value, cv2.THRESH_BINARY_INV
        )
    elif t_type == "trunc":
        _, binary_img = cv2.threshold(img, threshold, max_value, cv2.THRESH_TRUNC)
    elif t_type == "tozero":
        _, binary_img = cv2.threshold(img, threshold, max_value, cv2.THRESH_TOZERO)
    elif t_type == "tozero_inv":
        _, binary_img = cv2.threshold(
            img, threshold, max_value, cv2.THRESH_TOZERO_INV
        )
    else:
        raise ValueError(f"지원하지 않는 이진화 타입입니다: {data.type}")

    end_time = time.time() - start_time

    print(end_time)
    return JSONResponse(
        content={
            "image": f"data:image/png;base64,{np_to_bytes(binary_img)}",
        }
    )


@router.post("/{crime_number}/patterns_extract")
async def extract_patterns(
    crime_number: str = Path(...),
    data: PatternsExtractRequest = Body(...),
):

    model = models.get("pattern_extract")

    top, mid, bottom, all = patterns_prepaire(
        line_ys=data.line_ys, render_size=data.render_size, image_data=data.image
    )

    return JSONResponse(
        content={
            "top": model.predict(top, type=data.type, part="top"),
            "mid": model.predict(mid, type=data.type, part="mid"),
            "bottom": model.predict(bottom, type=data.type, part="bottom"),
            "outline": model.predict(all, type=data.type, part="all"),
        }
    )


@torch.no_grad()
@router.post("/{crime_number}/search")
async def search_crime(
    crime_number: str = Path(...),
    page: int = Query(0),
    binary: str = Query('original'),
    data: CrimeSearchRequest = Body(...),
    conn: Connection = Depends(context_get_conn),
):

    try:
        #### 1. 🔍 유사 이미지 검색 모델 준비 (모델 로드 및 추론 준비)

        model = models.get("image_search").to("cuda")
        model.eval()

        query_image = search_prepare(
            image_data=data.image, image_size=1024, device="cuda"
        )
        page_images = ImageSearchAdapter.calculate_distances(
            query_image=query_image, model=model
        )

        #### 2. 📦 전체 신발 문양 정보 조회 (DB에서 정보 가져오기)
        # TODO 추후 캐싱 필요: 신발이 업데이트 될 때만 로드되도록 최적화 필요
        # query = """
        # SELECT model_number, top, mid, bottom, outline FROM shoes_data
        # """

        # result = conn.execute(text(query)).fetchall()

        # if not result:
        #     return JSONResponse(content={"result": []})

        result = ImageSearchAdapter.load_patterns_info()

        if not result:
            return JSONResponse(content={"result": []})

        filtered_images = ImageSearchAdapter.essential_patterns_filter(
            page_images=page_images,
            result=result,
            data=data,
        )

        total_data = list(
            map(
                lambda x: {"image": x, "similarity": "95%"},
                filtered_images[page * 50 : (page + 1) * 50],
            )
        )

        return JSONResponse(
            content={"result": total_data, "total": len(filtered_images)}
        )

    except SQLAlchemyError as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="데이터베이스 쿼리 중 오류가 발생했습니다.",
        )

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
import numpy as np
from utils.utils import path_to_seg_base64

router = APIRouter(prefix="/crime", tags=["Crime Processing"])


import numpy as np


class SegmentRequest(BaseModel):
    image: str
    polygon: List[Tuple[float, float]]


@router.post("/{crimeNumber}/segmentation")
async def segment_image(
    crimeNumber: str = Path(...),
    render_size: List[float] = Query(...),
    conn: Connection = Depends(context_get_conn),
    data: SegmentRequest = Body(...),
):

    root = "static/crime_images/"

    try:
        if "crime_images" in data.image:  # 인코딩된 이미지가 아닐 때: 결과 이미지 경로
            seg_image = path_to_seg_base64(
                data.polygon, render_size, osp.join(root, crimeNumber + ".png")
            )

        return JSONResponse(content={"image": seg_image})

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


@router.post("/{id}/binarization")
async def binarize_image(id: int, threshold: int = 127):

    # binarization logic
    pass


@router.post("/{id}/denoising")
async def denoise_image(id: int):
    pass


@router.post("/{id}/inpainting")
async def inpaint_image(id: int):
    pass


@router.post("/{id}/extract")
async def extract_patterns(id: int):
    pass

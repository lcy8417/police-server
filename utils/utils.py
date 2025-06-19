import os.path as osp
import base64
from typing import Optional, List, Tuple
import cv2
import numpy as np


# Only png images are supported
def path_to_seg_base64(
    polygon: List[Tuple[float, float]], render_size: str, file_path: str
) -> Optional[str]:
    image = cv2.imread(file_path)

    rend_width, rend_height = render_size
    ori_width, ori_height = image.shape[1], image.shape[0]

    width_scale = ori_width / rend_width
    height_scale = ori_height / rend_height

    if image is None:
        raise FileNotFoundError("이미지를 불러올 수 없습니다.")

    if polygon[0] != polygon[-1]:
        polygon.append(polygon[0])

    polygon = list(map(lambda x: (x[0] * width_scale, x[1] * height_scale), polygon))

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)

    masked_image = cv2.bitwise_and(image, image, mask=mask)

    x, y, w, h = cv2.boundingRect(pts)
    cropped = masked_image[y : y + h, x : x + w]

    _, buffer = cv2.imencode(".png", cropped)
    encoded_image = base64.b64encode(buffer).decode("utf-8")
    encoded_image = f"data:image/png;base64,{encoded_image}"

    return encoded_image

import os.path as osp
from typing import Optional, List, Tuple, Union
import cv2
import numpy as np
from PIL import Image
import os
from io import BytesIO
from base64 import b64encode, b64decode
import torch
import torchvision.transforms as transforms

crime_root = "static/crime_images/"
shoes_root = "static/shoes_images/B"


# Only png images are supported
def path_to_seg(
    polygon: List[Tuple[float, float]],
    render_size: str,
    image_data: str,
    return_type: str = "base64",
) -> Union[str, np.ndarray]:

    is_encoded = image_data.startswith("data:image")

    if is_encoded:
        image = bytes_to_np(image_data.split(",")[1])
    else:
        image_path = osp.join(crime_root, image_data + ".png")
        if not osp.exists(image_path):
            raise FileNotFoundError(f"파일이 존재하지 않습니다: {image_path}")
        image = cv2.imread(image_path)

    rend_width, rend_height = render_size
    ori_width, ori_height = image.shape[1], image.shape[0]

    print(
        f"렌더링 크기: {rend_width}x{rend_height}, 원본 크기: {ori_width}x{ori_height}"
    )
    # 경로일때만 실제 크기와 렌더링 크기를 비교
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

    if return_type == "mask":
        return mask

    masked_image = cv2.bitwise_and(image, image, mask=mask)

    x, y, w, h = cv2.boundingRect(pts)
    cropped = masked_image[y : y + h, x : x + w]

    _, buffer = cv2.imencode(".png", cropped)
    encoded_image = b64encode(buffer).decode("utf-8")
    encoded_image = f"data:image/png;base64,{encoded_image}"

    return encoded_image


def name_to_pil(crime_root: str, image_id: str) -> Image.Image:
    image_id = image_id.split(".")[0] + ".png"
    path = os.path.join(crime_root, image_id)
    src_img = Image.open(path).convert("RGB")
    return src_img


def bytes_to_pil(byte_image: str) -> Image.Image:
    image_pil = Image.open(BytesIO(b64decode(byte_image)))
    return image_pil


def bytes_to_np(image_data: str, read_type: str = "color") -> np.ndarray:
    return cv2.imdecode(
        np.frombuffer(b64decode(image_data), np.uint8),
        cv2.IMREAD_COLOR if read_type == "color" else cv2.IMREAD_GRAYSCALE,
    )


def pil_to_bytes(pil_image: Image.Image, format: str = "PNG") -> str:
    buffered = BytesIO()
    pil_image.save(buffered, format=format)
    img_bytes = buffered.getvalue()
    base64_str = b64encode(img_bytes).decode("utf-8")
    return base64_str


def np_to_bytes(np_image: np.ndarray, format: str = "PNG") -> str:
    _, buffer = cv2.imencode(f".{format.lower()}", np_image)
    base64_str = b64encode(buffer).decode("utf-8")
    return base64_str


def pil_to_tensor(image_data: Image.Image, image_size=1024) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return transform(image_data)


def np_to_pil(np_image: np.ndarray) -> Image.Image:
    """Convert a NumPy array to a PIL Image."""
    if np_image.ndim == 2:  # Grayscale image
        return Image.fromarray(np_image, mode="L")
    elif np_image.ndim == 3 and np_image.shape[2] == 3:  # RGB image
        return Image.fromarray(np_image, mode="RGB")
    elif np_image.ndim == 3 and np_image.shape[2] == 4:  # RGBA image
        return Image.fromarray(np_image, mode="RGBA")
    else:
        raise ValueError("Unsupported image format")


def inpainting_prepaire(
    polygon: List[Tuple[float, float]],
    render_size: str,
    image_data: str,
    fixed_size: List[int] = [512, 512],
) -> Tuple[np.ndarray, np.ndarray]:
    mask_img = path_to_seg(
        polygon=polygon,
        render_size=render_size,
        image_data=image_data,
        return_type="mask",
    )

    if image_data.startswith("data:image"):
        src_img = bytes_to_np(image_data.split(",")[1])
    else:
        src_img = cv2.imread(osp.join(crime_root, image_data + ".png"))

    # if fixed_size:
    #     mask_img = cv2.resize(
    #         mask_img, tuple(fixed_size), interpolation=cv2.INTER_NEAREST
    #     )
    #     src_img = cv2.resize(src_img, tuple(fixed_size), interpolation=cv2.INTER_LINEAR)

    return src_img, mask_img


def patterns_prepaire(
    line_ys: List[int],
    render_size: str,
    image_data: str,
    return_type: str = "base64",
) -> Union[str, np.ndarray]:

    is_encoded = image_data.startswith("data:image")

    if is_encoded:
        image = bytes_to_np(image_data.split(",")[1])
    else:

        shoes_path = osp.join(shoes_root, image_data + ".png")
        crime_path = osp.join(crime_root, image_data + ".png")
        if osp.exists(shoes_path):
            image = cv2.imread(shoes_path)
        elif osp.exists(crime_path):
            image = cv2.imread(crime_path)
        else:
            raise FileNotFoundError(
                f"파일이 존재하지 않습니다: {shoes_path} 또는 {crime_path}"
            )

    rend_height = render_size[1]
    ori_height = image.shape[0]

    # 경로일때만 실제 크기와 렌더링 크기를 비교
    height_scale = 1 if is_encoded else (ori_height / rend_height)

    if image is None:
        raise FileNotFoundError("이미지를 불러올 수 없습니다.")

    line_ys = list(map(lambda x: int(x * height_scale), line_ys))

    top, mid, bottom, all = (
        image[: line_ys[0], :],
        image[line_ys[0] : line_ys[1], :],
        image[line_ys[1] :, :],
        image[:, :],
    )

    return [np_to_pil(top), np_to_pil(mid), np_to_pil(bottom), np_to_pil(all)]


def search_prepare(
    image_data: str,
    image_size: int = 1024,
    device: str = "cpu",
):
    if image_data.startswith("data:image"):
        query_image = bytes_to_pil(image_data.split(",")[1])
    else:
        query_image = name_to_pil(crime_root, image_data)

    query_image = (
        pil_to_tensor(query_image, image_size=image_size).unsqueeze(0).to(device)
    )

    return query_image

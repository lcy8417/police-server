import cv2
import numpy as np
import os
from datetime import datetime, timezone, timedelta


class BackgroundDeletionModel:

    def __init__(self):
        self.image_id = None
        self.image_src = None
        self.mask_polygons = None
        self.image_mask = None
        self.image_with_mask = None
        self.image_dst = None
        self.h, self.w, self.c = None, None, None
        self.timeline = None

        self.save_path = {
            # 'src': 'data/preprocessing/background/src',
            "src": "data/images/samples/PL_query",
            "mask": "data/preprocessing/background/mask",
            "dst": "data/preprocessing/background/dst",
        }

    def set_image(
        self, image_id, polygons, image_data=None, scale=1.0, max_wh=[10000, 10000]
    ):
        self.image_id = image_id

        if image_data is None:  # 이미지 불러오기
            full_path = os.path.join(self.save_path["src"], f"{self.image_id}.png")
            img_array = np.fromfile(full_path, np.uint8)
            self.image_src = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        else:
            self.image_src = image_data

        if max_wh and max_wh[0] != 10000:
            self.image_src = cv2.resize(self.image_src, max_wh)

        original_h, original_w = self.image_src.shape[:2]

        scaled_w = int(original_w * scale)
        scaled_h = int(original_h * scale)

        if scale > 1:
            scaled_img = cv2.resize(
                self.image_src, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR
            )

            #   확대된 이미지를 원본 크기만큼 가운데서 잘라냄
            start_x = (scaled_w - max_wh[0]) // 2
            start_y = (scaled_h - max_wh[1]) // 2

            if start_x < 0:
                start_x = 0
            if start_y < 0:
                start_y = 0

            # 원본 크기만큼 가운데를 잘라냄
            end_x = start_x + max_wh[0]
            end_y = start_y + max_wh[1]
            if end_x > scaled_w:
                end_x = scaled_w
            if end_y > scaled_h:
                end_y = scaled_h

            self.image_src = scaled_img[start_y:end_y, start_x:end_x]

        # if scale < 1:
        #     self.image_src = cv2.resize(self.image_src, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

        self.h, self.w, self.c = self.image_src.shape

        self.timeline = datetime.now(tz=timezone(timedelta(hours=9))).strftime(
            "%Y%m%d_%H%M%S"
        )

        for i, pos in enumerate(polygons):
            max_value = self.w if i % 2 == 0 else self.h
            polygons[i] = max(min(polygons[i], max_value), 0)

        cnt_x, cnt_y = 0, 0
        self.mask_polygons = []
        for idx in range(0, len(polygons), 2):
            self.mask_polygons.append(
                np.array((int(polygons[idx]), int(polygons[idx + 1])))
            )
            cnt_x = max(cnt_x, int(polygons[idx]))
            cnt_y = max(cnt_y, int(polygons[idx + 1]))
        self.mask_polygons = [np.array(self.mask_polygons, dtype=np.int32)]

    def delete_background(
        self, is_patch=False
    ):  # 원래 이미지에 대해서, 폴리곤을 마스크 따고 저장하는 것(바이너리와 이미지에 크롭)
        if self.mask_polygons is not None and self.image_src is not None:
            mask = np.zeros(shape=(self.h, self.w), dtype=np.uint8)
            print("폴리곤", self.mask_polygons, is_patch, self.get_patch_image_coords())
            cv2.fillPoly(mask, self.mask_polygons, 255)
            self.image_mask = cv2.merge([mask, mask, mask])
            self.image_dst = cv2.bitwise_and(
                self.image_src, self.image_mask
            )  # 까만 배경을 생성하는 것

            print(self.image_src.shape, self.image_mask.shape, self.image_dst.shape)
            if is_patch:
                x1, y1, x2, y2 = self.get_patch_image_coords()
                self.image_dst = self.image_dst[
                    y1:y2, x1:x2, :
                ]  # 까만 배경을 폴리곤 쉐잎만큼 자름

            image_mask_path = os.path.join(
                self.save_path["mask"], f"{self.image_id}.png"
            )
            self.write_image(self.image_mask, image_mask_path)

            image_dst_path = os.path.join(self.save_path["dst"], f"{self.image_id}.png")

            self.write_image(self.image_dst, image_dst_path)

            return self.image_dst

    def get_patch_image_coords(self):
        if not self.mask_polygons:
            return None
        x1 = min(self.mask_polygons[0][:, 0])
        y1 = min(self.mask_polygons[0][:, 1])
        x2 = max(self.mask_polygons[0][:, 0])
        y2 = max(self.mask_polygons[0][:, 1])

        patch_coords = [x1, y1, x2, y2]
        return patch_coords

    def reset(self):
        self.image_id = None
        self.image_src = None
        self.mask_polygons = None
        self.image_mask = None
        self.image_with_mask = None
        self.image_dst = None

        self.h, self.w, self.c = None, None, None
        self.timeline = None

    @staticmethod
    def write_image(image, path):
        if os.path.exists(path):
            os.remove(path)

        ext = ".png"

        result, encoded_img = cv2.imencode(ext, image)

        with open(path, "wb") as f:
            f.write(encoded_img)

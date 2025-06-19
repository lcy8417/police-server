import cv2
import numpy as np
import torch
import os
import yaml
from omegaconf import OmegaConf
from datetime import datetime, timezone, timedelta

from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.trainers import load_checkpoint


class InpaintingModel:

    def __init__(self):
        self.image_id = None
        self.src_img = None
        self.mask_polygons = None
        self.mask_img = None
        self.src_with_mask_img = None
        self.dst_img = None

        self.fix_w, self.fix_h = 512, 512
        self.h, self.w, self.c = None, None, None
        self.timeline = None
        self.script_path = "./bin/predict3.py"
        self.model_path = "weights/preprocessing/inpainting/"
        self.config_path = "weights/preprocessing/inpainting/config.yaml"
        self.ckpt_path = "weights/preprocessing/inpainting/models/best.ckpt"

        self.save_path = {
            # 'src': 'data/preprocessing/inpainting/src/',
            "src": "data/images/samples/PL_query",
            "mask": "data/preprocessing/inpainting/mask/",
            "inpaint": "data/preprocessing/inpainting/inpaint/",
            "temp": "data/preprocessing/inpainting/temp/",
        }

        # --- model config
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            with open(self.config_path, mode="r") as rf:
                train_config = OmegaConf.create(yaml.safe_load(rf))

            train_config.training_model.predict_only = True
            train_config.visualizer.kind = "noop"

            self.model = load_checkpoint(
                train_config, self.ckpt_path, strict=False, map_location=self.device
            )
            self.model.freeze()
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print("Failed Loading Inpainting Model.. ", e)

    def set_image(self, image_id, image_data=None, scale=1.0, max_wh=[10000, 10000]):
        self.image_id = image_id

        if image_data is None:
            full_path = os.path.join(self.save_path["src"], f"{self.image_id}.png")
            img_array = np.fromfile(full_path, np.uint8)
            self.src_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            # self.src_img = cv2.imread(os.path.join(self.save_path['src'], f'{self.image_id}.png'))
        else:
            self.src_img = image_data

        if max_wh and max_wh[0] != 10000:
            self.src_img = cv2.resize(self.src_img, max_wh)

        original_h, original_w = self.src_img.shape[:2]

        scaled_w = int(original_w * scale)
        scaled_h = int(original_h * scale)

        scaled_img = cv2.resize(
            self.src_img, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR
        )

        if scale > 1:
            original_h, original_w = self.src_img.shape[:2]

            scaled_w = int(original_w * scale)
            scaled_h = int(original_h * scale)

            scaled_img = cv2.resize(
                self.src_img, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR
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

            self.src_img = scaled_img[start_y:end_y, start_x:end_x]

        # if scale < 1:
        #     self.src_img = cv2.resize(self.src_img, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

        self.timeline = datetime.now(tz=timezone(timedelta(hours=9))).strftime(
            "%Y%m%d_%H%M%S"
        )

        self.h, self.w, self.c = self.src_img.shape
        # 임시 폴더 생성
        os.mkdir(os.path.join(self.save_path["temp"], self.timeline))

        src_resize = cv2.resize(
            self.src_img, (self.fix_w, self.fix_h), interpolation=cv2.INTER_NEAREST
        )
        self.write_image(
            src_resize,
            os.path.join(self.save_path["temp"], self.timeline, "input_image.png"),
        )
        print()

    def generate_mask_image(self, polygons, only_mask=True):
        if not polygons:
            print("polygons are not exist.")
            return

        self.mask_polygons = []
        for idx in range(0, len(polygons), 2):
            self.mask_polygons.append(
                np.array((int(polygons[idx]), int(polygons[idx + 1])))
            )
        self.mask_polygons = [np.array(self.mask_polygons, dtype=np.int32)]

        if self.mask_polygons is not None and self.src_img is not None:
            mask_img = np.zeros(shape=(self.h, self.w), dtype=np.uint8)
            cv2.fillPoly(mask_img, self.mask_polygons, color=255)
            self.mask_img = cv2.merge([mask_img, mask_img, mask_img])

            if only_mask:
                self.src_with_mask_img = cv2.bitwise_and(self.src_img, self.mask_img)

            self.write_image(
                self.mask_img,
                os.path.join(self.save_path["mask"], f"{self.image_id}.png"),
            )

            mask_resize = cv2.resize(
                self.mask_img, (self.fix_w, self.fix_h), interpolation=cv2.INTER_NEAREST
            )
            self.write_image(
                mask_resize,
                os.path.join(self.save_path["temp"], self.timeline, "mask_image.png"),
            )

    def predict(self):
        if self.src_img is None:
            print("src img not exist. please set src_img first.")
            return

        if self.mask_img is None:
            print("mask img not exist. please generate mask_img first.")
            return

        if not os.path.exists(self.script_path):
            print("please check script path")
            return

        if not os.path.exists(self.config_path):
            print("please check config_path")
            return

        # --- Model Predict

        full_path = os.path.join(
            self.save_path["temp"], self.timeline, f"input_image.png"
        )
        img_array = np.fromfile(full_path, np.uint8)
        src_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        full_path = os.path.join(
            self.save_path["temp"], self.timeline, f"mask_image.png"
        )
        img_array = np.fromfile(full_path, np.uint8)
        mask_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)

        batch = {
            "image": torch.from_numpy(src_img.transpose((2, 0, 1))).float().unsqueeze(0)
            / 255.0,
            "mask": torch.from_numpy(mask_img).float().unsqueeze(0).unsqueeze(0)
            / 255.0,
        }

        with torch.no_grad():
            batch = move_to_device(batch, self.device)
            batch["mask"] = (batch["mask"] > 0) * 1
            batch = self.model(batch)
            cur_res = batch["inpainted"][0].permute(1, 2, 0).detach().cpu().numpy()

        output_image = np.clip(cur_res * 255, 0, 255).astype("uint8")
        output_image_path = os.path.join(
            self.save_path["temp"], self.timeline, f"output_image.png"
        )
        self.write_image(output_image, output_image_path)

        # image resize 를 통해 원본 이미지 크기로 변환 (512,512) to (org_w, org_h)
        output_image = cv2.resize(
            output_image, (self.w, self.h), interpolation=cv2.INTER_NEAREST
        )
        self.dst_img = output_image

        # dst_image 저장
        dst_image_path = os.path.join(self.save_path["inpaint"], f"{self.image_id}.png")
        self.write_image(self.dst_img, dst_image_path)

        # 작업 끝나면 이미지 초기화
        self.reset()

    def write_images(self, src=None, mask=None, dst=None):
        ext = ".png"

        if src is not None and self.src_img is not None:
            path = os.path.join(
                self.save_path["src"], self.image_id, f"_{self.timeline}.png"
            )

            result, encoded_img = cv2.imencode(ext, self.src_img)

            with open(path, "wb") as f:
                f.write(encoded_img)

            # cv2.imwrite(path, self.src_img)

        if mask is not None and self.mask_img is not None:
            path = os.path.join(
                self.save_path["mask"], self.image_id, f"_{self.timeline}.png"
            )

            result, encoded_img = cv2.imencode(ext, self.mask_img)

            with open(path, "wb") as f:
                f.write(encoded_img)

            # cv2.imwrite(path, self.mask_img)

        if dst is not None and self.dst_img is not None:
            path = os.path.join(
                self.save_path["inpaint"], self.image_id, f"_{self.timeline}.png"
            )

            result, encoded_img = cv2.imencode(ext, self.dst_img)

            with open(path, "wb") as f:
                f.write(encoded_img)

            # cv2.imwrite(path, self.dst_img)

    def reset(self):
        self.image_id = None
        self.src_img = None
        self.mask_polygons = None
        self.mask_img = None
        self.src_with_mask_img = None
        self.dst_img = None

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

        # cv2.imwrite(path, image)


# if __name__ == '__main__':
#     model = InpaintingModel()
#     model.set_image('00001-21-0017_1')
#     model.generate_mask_image([207, 18, 165, 34,
#                                151, 78, 167, 113,
#                                202, 141, 259, 124,
#                                271, 79, 246, 33, 207, 18])
#     model.predict()
#     model.reset()

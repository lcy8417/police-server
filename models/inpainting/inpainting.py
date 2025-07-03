import cv2
import numpy as np
import torch
import os
import yaml
from omegaconf import OmegaConf
from datetime import datetime, timezone, timedelta

from .saicinpainting.evaluation.utils import move_to_device
from .saicinpainting.training.trainers import load_checkpoint


class InpaintingModel:

    def __init__(self):

        self.config_path = "weights/preprocessing/inpainting/config.yaml"
        self.ckpt_path = "weights/preprocessing/inpainting/models/best.ckpt"

        # --- model config
        try:
            self.device = (
                "cuda"
                if torch.backends.mps.is_available()
                else "cuda" if torch.cuda.is_available() else "cpu"
            )

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

    def predict(self, src_img, mask_img):

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

        return output_image

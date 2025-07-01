#!/usr/bin/env python3

import logging
import os
import sys
import traceback

# 현재 파일의 디렉토리를 기준으로 lama 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
lama_main_path = os.path.abspath(os.path.join(current_dir, '..'))
saicinpainting_path = os.path.abspath(os.path.join(current_dir, '..', 'saicinpainting'))
src_path = os.path.abspath(os.path.join(current_dir, 'src'))

sys.path.append(lama_main_path)
sys.path.append(saicinpainting_path)
sys.path.append(src_path)
sys.path.append(current_dir)

# 디버깅을 위해 sys.path 출력
print("sys.path:", sys.path)

# 모듈 임포트
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.evaluation.refinement import refine_predict

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import cv2
import hydra
import numpy as np
import torch
import tqdm
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.utils import register_debug_signal_handlers

LOGGER = logging.getLogger(__name__)

def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

@hydra.main(config_path=os.path.join(current_dir, '..', 'configs', 'prediction'), config_name='default.yaml')
def main(predict_config: OmegaConf):
    try:
        if sys.platform != 'win32':
            register_debug_signal_handlers()

        # 디버깅을 위해 predict_config 내용 출력
        print("predict_config:", predict_config)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 절대 경로로 설정
        train_config_path = os.path.abspath(os.path.join(current_dir, '..', 'big-lama', 'config.yaml'))
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        out_ext = predict_config.get('out_ext', '.png')

        # 절대 경로로 설정
        checkpoint_path = os.path.abspath(os.path.join(current_dir, '..', 'big-lama', 'models', 'best.ckpt'))
        model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location=device)
        model.freeze()
        model.to(device)
        model.eval()

        input_image_path = os.path.join(predict_config.indir, 'input_image.png')
        mask_image_path = os.path.join(predict_config.indir, 'mask_image.png')

        if not os.path.exists(input_image_path):
            LOGGER.error(f"Input image not found: {input_image_path}")
            sys.exit(1)
        if not os.path.exists(mask_image_path):
            LOGGER.error(f"Mask image not found: {mask_image_path}")
            sys.exit(1)

        ensure_dir_exists(predict_config.outdir)
        cur_out_fname = os.path.join(predict_config.outdir, 'output_image.png')

        image = cv2.imread(input_image_path)
        mask = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            LOGGER.error(f"Image or mask loading failed. Image: {input_image_path}, Mask: {mask_image_path}")
            sys.exit(1)

        # Ensure mask size matches the image size
        if mask.shape != image.shape[:2]:
            LOGGER.warning("Mask size does not match image size, resizing mask to match image size.")
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(mask_image_path, mask)  # Save the resized mask

        batch = {
            'image': torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0,
            'mask': torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0) / 255.0
        }

        with torch.no_grad():
            batch = move_to_device(batch, device)
            batch['mask'] = (batch['mask'] > 0) * 1
            batch = model(batch)
            cur_res = batch[predict_config.out_key][0].permute(1, 2, 0).detach().cpu().numpy()

        cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
        cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
        cv2.imwrite(cur_out_fname, cur_res)

    except KeyboardInterrupt:
        LOGGER.warning('Interrupted by user')
    except Exception as ex:
        LOGGER.critical(f'Prediction failed due to {ex}:\n{traceback.format_exc()}')
        sys.exit(1)

if __name__ == '__main__':
    main()

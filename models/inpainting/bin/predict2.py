import argparse
import os
import cv2
import torch
from torchvision import transforms
from PIL import Image
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.utils import register_debug_signal_handlers
from omegaconf import OmegaConf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to the config file.')
    parser.add_argument('indir', type=str, help='Input directory.')
    parser.add_argument('outdir', type=str, help='Output directory.')
    parser.add_argument('--maskdir', type=str, help='Mask directory.')
    parser.add_argument('--checkpoint', type=str, help='Path to the checkpoint file.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for inference.')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load config
    config_path = args.config
    with open(config_path, 'r') as f:
        config = OmegaConf.load(f)

    # Load checkpoint
    checkpoint = args.checkpoint if args.checkpoint else config.model.checkpoint
    model = load_checkpoint(config, checkpoint, strict=False, map_location='cpu')
    model.eval()

    # Load image and mask
    device = torch.device(args.device)
    model = model.to(device)

    transform = transforms.ToTensor()
    
    for img_name in os.listdir(args.indir):
        img_path = os.path.join(args.indir, img_name)
        mask_path = os.path.join(args.maskdir, img_name) if args.maskdir else None

        img = Image.open(img_path).convert('RGB')
        img = transform(img).unsqueeze(0).to(device)
        
        if mask_path and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
            mask = transform(mask).unsqueeze(0).to(device)
        else:
            mask = torch.zeros_like(img)[:, 0:1, :, :]

        with torch.no_grad():
            result = model(img, mask)

        out_img = result[0].cpu()
        out_img = transforms.ToPILImage()(out_img)
        out_img.save(os.path.join(args.outdir, img_name))

if __name__ == '__main__':
    register_debug_signal_handlers()
    main()

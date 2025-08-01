import os
import numpy as np
import cv2
import torch
import torch.optim as optim
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import torch.nn.init as init
from tqdm import tqdm
from .loss.loss_functions import (
    reconstruction_loss,
    illumination_smooth_loss,
    reflectance_smooth_loss,
    noise_loss,
    normalize01,
)
from .conf import conf
from .model import RRDNet


def pipline_retinex(net, img):
    img_tensor = transforms.ToTensor()(img)  # [c, h, w]
    img_tensor = img_tensor.to(conf.device)
    img_tensor = img_tensor.unsqueeze(0)  # [1, c, h, w]

    optimizer = optim.Adam(net.parameters(), lr=conf.lr)
    net = net.to(conf.device)
    # iterations
    for i in range(conf.iterations + 1):
        # forward
        if img_tensor.shape[1] == 1:
            img_tensor = img_tensor.expand(-1, 3, -1, -1)

        illumination, reflectance, noise = net(img_tensor)  # [1, c, h, w]
        # loss computing
        loss_recons = reconstruction_loss(img_tensor, illumination, reflectance, noise)
        loss_illu = illumination_smooth_loss(img_tensor, illumination)
        loss_reflect = reflectance_smooth_loss(img_tensor, illumination, reflectance)
        loss_noise = noise_loss(img_tensor, illumination, reflectance, noise)

        loss = (
            loss_recons
            + conf.illu_factor * loss_illu
            + conf.reflect_factor * loss_reflect
            + conf.noise_factor * loss_noise
        )

        # backward
        net.zero_grad()
        loss.backward()
        optimizer.step()

        # log
        if i % 100 == 0:
            print(
                "iter:",
                i,
                "  reconstruction loss:",
                float(loss_recons.data),
                "  illumination loss:",
                float(loss_illu.data),
                "  reflectance loss:",
                float(loss_reflect.data),
                "  noise loss:",
                float(loss_noise.data),
            )

    # adjustment
    adjust_illu = torch.pow(illumination, conf.gamma)
    res_image = adjust_illu * ((img_tensor - noise) / illumination)
    res_image = torch.clamp(res_image, min=0, max=1)

    if conf.device != "cpu":
        res_image = res_image.cpu()
        illumination = illumination.cpu()
        adjust_illu = adjust_illu.cpu()
        reflectance = reflectance.cpu()
        noise = noise.cpu()

    res_img = transforms.ToPILImage()(res_image.squeeze(0))
    illum_img = transforms.ToPILImage()(illumination.squeeze(0))
    adjust_illu_img = transforms.ToPILImage()(adjust_illu.squeeze(0))
    reflect_img = transforms.ToPILImage()(reflectance.squeeze(0))
    noise_img = transforms.ToPILImage()(normalize01(noise.squeeze(0)))

    return res_img, illum_img, adjust_illu_img, reflect_img, noise_img


if __name__ == "__main__":

    # Init Model
    net = RRDNet()
    net = net.to(conf.device)

    # Test
    file_list = os.listdir(conf.test_image_path)
    for i in tqdm(file_list):

        img = Image.open(conf.test_image_path + i).convert("RGB")

        res_img, illum_img, adjust_illu_img, reflect_img, noise_img = pipline_retinex(
            net, img
        )
        res_img.save(conf.save_result + i)
        illum_img.save(conf.save__illumination + i)
        adjust_illu_img.save(conf.save_adjust_illumination + i)
        reflect_img.save(conf.save_reflect + i)
        noise_img.save(conf.save_noise + i)

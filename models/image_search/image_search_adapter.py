import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F

from .dataset import ImageFromList
from glob import glob

import numpy as np
import os.path as osp
import os
from tqdm import tqdm

root = "static/crime_images"


class ImageSearchAdapter:

    ref_root_path = "static/shoes_images/B"
    ref_memory = "data/ref_vecs.npy"
    device = "mps"
    ref_names = list(map(lambda x: x.split(".")[0], os.listdir(ref_root_path)))
    ref_images = list(map(lambda x: osp.join(root, x) + ".png", ref_names))

    ref_vecs = np.load(ref_memory) if osp.exists(ref_memory) else None
    # ref_vecs 정렬 코드 넣어줘야함.
    image_size = 1024

    @classmethod
    def get_ref_vectors(cls, model, image_size=1024):

        if cls.ref_vecs is None:
            cls.ref_loader = DataLoader(
                ImageFromList(Image_paths=cls.ref_images, imsize=image_size),
                batch_size=4,
                shuffle=False,
            )

            cls.ref_vecs = cls.extract_vectors(
                model,
                cls.ref_loader,
                device=cls.device,
                img_size=len(cls.ref_images),
                batch_size=4,
            ).numpy()

            # memory에 저장
            np.save(cls.ref_memory, cls.ref_vecs)

    @classmethod
    def calculate_distances(cls, query_image, model, page=0):
        if cls.ref_vecs is None:
            cls.get_ref_vectors(model, image_size=cls.image_size)

        query_vector = (
            model.forward_test(query_image).detach().cpu().numpy()
        )  # shape: (1, D)
        ref_vectors = cls.ref_vecs

        dists = np.linalg.norm(ref_vectors - query_vector, axis=1)

        # 정렬된 인덱스
        indices = np.argsort(dists)
        page_data = list(
            map(lambda idx: cls.ref_names[idx], indices[page * 50 : (page + 1) * 50])
        )

        return page_data

    @torch.no_grad()
    def extract_vectors(
        net, loader, ms=[1], device=torch.device("mps"), img_size=0, batch_size=1
    ):
        net.eval()
        vecs = torch.zeros(img_size, net.outputdim)
        ms = [1]
        if len(ms) == 1:
            for i, input in enumerate(loader):
                vecs[i * batch_size : (i + 1) * batch_size, :] = (
                    net.forward_test(input.to(device)).cpu().data.squeeze()
                )
                print("\r>>>> {}/{} done...".format(i + 1, len(loader)), end="")
        else:
            for i, input in enumerate(
                tqdm(loader, desc="Processing", total=len(loader))
            ):
                vec = torch.zeros(input.shape[0], net.outputdim)
                for s in ms:
                    if s == 1:
                        input_ = input.clone()
                    else:
                        input_ = F.interpolate(
                            input, scale_factor=s, mode="bilinear", align_corners=False
                        )
                    vec += net.forward_test(input_.to(device)).cpu().data.squeeze()
                vec /= len(ms)
                vecs[i * batch_size : (i + 1) * batch_size, :] = F.normalize(
                    vec, p=2, dim=0
                )
                print("\r>>>> {}/{} done...".format(i + 1, len(loader)), end="")

        return vecs

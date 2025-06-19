import torch.utils.data as data
import torchvision.transforms as transforms
# import torchvision.transforms.functional as F_t
from PIL import Image, ImageFile
from src.server.databases.leaving.leaving_retrieval_image import pil_loader
from io import BytesIO
from base64 import b64decode
# import ipdb
# import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

def imthumbnail(img: Image, imsize: int):
    img.thumbnail((imsize, imsize), Image.ANTIALIAS)
    return img

class ImageFromList(data.Dataset):
    def __init__(self, Image_paths=None, imsize=None, bbox=None, loader=pil_loader):
        super(ImageFromList, self).__init__()
        self.Image_paths = Image_paths
        self.transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.bbox = bbox
        self.imsize = imsize
        self.loader = loader
        self.len = len(Image_paths)

    def __getitem__(self, index):
        path = self.Image_paths[index]
        img = self.loader(path)
        imfullsize = max(img.size)

        if self.bbox is not None:
            img = img.crop(self.bbox[index])

        if self.imsize is not None:
            if self.bbox is not None:
                img = imthumbnail(img, self.imsize * max(img.size) / imfullsize)
            else:
                img = imthumbnail(img, self.imsize)

        if self.transforms is not None:
            img = self.transforms(img)

        return img

    def __len__(self):
        return self.len


class ImageFromEncoding(data.Dataset):
    def __init__(self, encode_image=None, imsize=None, loader=pil_loader):
        super(ImageFromEncoding, self).__init__()
        self.encode_image = [Image.open(BytesIO(b64decode(encode_image))).convert('RGB')]
        self.transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.imsize = imsize
        self.loader = loader

    def __getitem__(self, index):
        img = self.encode_image[index]
        img = imthumbnail(img, self.imsize)

        if self.transforms is not None:
            img = self.transforms(img)

        return img

    def __len__(self):
        return 1
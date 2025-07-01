import torch.utils.data as data
import torchvision.transforms as transforms

# import torchvision.transforms.functional as F_t
from PIL import Image, ImageFile

from io import BytesIO
from base64 import b64decode

ImageFile.LOAD_TRUNCATED_IMAGES = True


def pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def imresize(img, imsize):
    img = transforms.Resize(imsize)(img)
    return img


class ImageFromList(data.Dataset):
    def __init__(
        self,
        Image_paths=None,
        imsize=None,
        bbox=None,
        loader=pil_loader,
    ):
        super(ImageFromList, self).__init__()
        self.Image_paths = Image_paths
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
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
            img = imresize(img, (self.imsize, self.imsize))

        if self.transforms is not None:
            img = self.transforms(img)

        return img

    def __len__(self):
        return self.len

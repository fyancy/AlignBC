# import torch
from torchvision.transforms import functional as F
# from torchvision.transforms import Compose, ToTensor, Normalize


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    def __call__(self, image, target):  # whc to chw
        return F.to_tensor(image), target


class Normalize:
    def __init__(self, mean, std, to_bgr=False):
        self.mean = mean
        self.std = std
        self.to_bgr = to_bgr

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if self.to_bgr:
            image = image[[2, 1, 0]]  # (rgb->bgr)
        return image, target


def build_transforms():
    PIXEL_MEAN, PIXEL_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # kitti, RGB order
    normalize_transform = Normalize(mean=PIXEL_MEAN, std=PIXEL_STD, to_bgr=False)

    transform = Compose(
        [
            ToTensor(),
            normalize_transform,
        ]
    )

    return transform

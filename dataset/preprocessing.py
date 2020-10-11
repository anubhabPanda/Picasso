import albumentations as A
from albumentations.pytorch import ToTensor
import numpy as np


def transformations(transformations=None, augmentations=None):
    transforms_list = [
    # convert the data to torch.FloatTensor with values within the range [0.0 ,1.0]
    ToTensor()
    ]

    if transformations is not None:
        transforms_list = transformations + transforms_list

    if augmentations is not None:
        transforms_list = augmentations + transforms_list
    
    return  A.Compose(transforms_list)

    
    
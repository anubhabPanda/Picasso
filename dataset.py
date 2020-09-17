import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import config
import torch
import utils


def data_loader(data, batch_size, num_workers, cuda, shuffle):
    """Create data loader
    Args:
        data: Dataset.
        batch_size: int
            Number of images to considered in each batch.
        num_workers: int
            How many subprocesses to use for data loading.
        cuda: bool
            True is GPU is available.
        shuffle: bool
    
    Returns:
        DataLoader instance.
    """

    loader_args = {
        'shuffle': shuffle,
        'batch_size': batch_size
    }

    # If GPU exists
    if cuda:
        loader_args['num_workers'] = num_workers
        loader_args['pin_memory'] = True
    
    return torch.utils.data.DataLoader(data, **loader_args)


def torch_datasets(train, download=True, transforms=None, augmentations=None, dataset_name="data"):
    """
    Returns a dataset from torchvision library. 
    Downloads the dataset in the folder sapecified by dataset_name variable

    Args
    ----
    train:bool
    download:bool
    transforms:list of transforms
        By default ToTensor is applied.
    dataset_name:str

    Returns
    -------
    data: torch Dataset
    """
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), dataset_name)

    
    transforms = utils.transformations(transforms, augmentations)

    if dataset_name.lower() in ['cifar10', 'mnist']:
        if dataset_name.lower() == 'cifar10':
            data = datasets.CIFAR10(
                data_path, train=train, download=download, transform=transforms
            )
        elif dataset_name.lower() == 'mnist':
            data = datasets.MNIST(
                data_path, train=train, download=download, transform=transforms
            )

        return data
    else:
        print("Enter a correct dataset name")




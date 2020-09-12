import torchvision.datasets as datasets

def torch_datasets(path, train, download, transforms=None):
    """
    Returns a dataset from torchvision library
    """
    return  datasets.CIFAR10(path, train, download, transforms)

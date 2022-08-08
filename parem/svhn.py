import torch
import numpy as np
from torch.utils.data import TensorDataset
import PIL
import torchvision.transforms as transforms
from torchvision.datasets import SVHN


def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """

    def __getitem__(self, index):
        data = cls.__getitem__(self, index)
        return data[0], index

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })

TensorDataset = dataset_with_indices(TensorDataset)

def get_svhn(root_path, n_images):
    dataset = SVHN(root_path, split='train', download=True)

    # Transform dataset into torch tensors with values in [-1, 1]:
    transform = transforms.Compose([PIL.Image.fromarray,
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5)),
                                    ])
    tensors = torch.stack([transform(np.moveaxis(dataset.data, 1, -1)[i])
                           for i in range(n_images)], axis=0)

    # Add indexing functionality to dataset class:
    dataset = TensorDataset(tensors)
    return dataset

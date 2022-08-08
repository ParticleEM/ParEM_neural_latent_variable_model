import torch
import numpy as np
from torch.utils.data import TensorDataset
import PIL
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

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

def get_mnist(root_path, n_images):
    dataset = MNIST("/content/", train=True, download=True)

    # Transform dataset into torch tensors with values in [-1, 1]:
    transform = transforms.Compose([PIL.Image.fromarray,
                                    transforms.Resize((32,32)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5),
                                                          (0.5)),
                                    ])
    tensors = torch.stack([transform(dataset.train_data[i].numpy())
                            for i in range(n_images)], axis=0)

    # Add indexing functionality to dataset class:
    dataset = TensorDataset(tensors)
    return dataset

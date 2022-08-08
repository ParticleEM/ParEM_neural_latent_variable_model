import os
import torch
from torch.utils.data import TensorDataset
import PIL
import torchvision.transforms as transforms
from torchvision.datasets import SVHN
import numpy as np


def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """

    def __getitem__(self, index):
        data = cls.__getitem__(self, index)
        return data, index

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })

class SingleImagesFolderMTDataset(torch.utils.data.Dataset):
    def __init__(self, root, cache, transform=None, workers=16, split_size=200, protocol=None, num_images=None):
        if num_images != None:
          split_size = min(split_size, num_images)
        if cache is not None and os.path.exists(cache):
            with open(cache, 'rb') as f:
                self.images = pickle.load(f)
        else:
            self.transform = transform if not transform is None else lambda x: x
            self.images = []

            def split_seq(seq, size):
                newseq = []
                splitsize = 1.0 / size * len(seq)
                for i in range(size):
                    newseq.append(seq[int(round(i * splitsize)):int(round((i + 1) * splitsize))])
                return newseq

            def map(path_imgs):
                imgs_0 = [self.transform(np.array(PIL.Image.open(os.path.join(root, p_i)))) for p_i in path_imgs]
                imgs_1 = [self.compress(img) for img in imgs_0]

                print('.', end='')
                return imgs_1

            path_imgs = os.listdir(root)
            if num_images:
                path_imgs = path_imgs[:num_images]

            n_splits = len(path_imgs) // split_size
            path_imgs_splits = split_seq(path_imgs, n_splits)

            from multiprocessing.dummy import Pool as ThreadPool
            pool = ThreadPool(workers)
            results = pool.map(map, path_imgs_splits)
            pool.close()
            pool.join()

            for r in results:
                self.images.extend(r)

            if cache is not None:
                with open(cache, 'wb') as f:
                    pickle.dump(self.images, f, protocol=protocol)

        print('Total number of images {}'.format(len(self.images)))

    def __getitem__(self, item):
        return self.decompress(self.images[item]), item

    def __len__(self):
        return len(self.images)

    @staticmethod
    def compress(img):
        return img

    @staticmethod
    def decompress(output):
        return output

def get_celeba(data_path, n_images):
    dataset = SingleImagesFolderMTDataset(root=data_path,
                                          cache=None,
                                          num_images=n_images,
                                          transform=transforms.Compose([
                                              PIL.Image.fromarray,
                                              transforms.Resize(32),
                                              transforms.CenterCrop(32),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5),
                                                                   (0.5, 0.5, 0.5)),
                                          ]))
    return dataset

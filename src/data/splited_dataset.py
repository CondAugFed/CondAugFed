import os
from typing import  Optional, Callable
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset
from PIL import Image
from src.data.utils import *

# =======================================
# Base Dataset class.
# =======================================

class BaseDataset(VisionDataset):
    def __init__(self,
                 root,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 **kwargs):

        self.transform = transform
        self.target_transform = target_transform
        self.root = root
        self.files = []

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return None, None

# =======================================
# For non-IID federated learning on MNIST
# =======================================

class BaseSplitDataset(BaseDataset):
    def __init__(self, root, transform):
        super(BaseSplitDataset, self).__init__(root, transform)

        self.images = []
        self.labels = []
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        in_ = self.images[index]
        label_ = self.labels[index]

        # convert to PIL.Image to apply torch.transform
        in_ = Image.fromarray(in_, mode='L')

        if self.transform is not None:
            in_ = self.transform(in_)

        return in_, label_


class SplitMnistDataset(BaseSplitDataset):
    def __init__(self,
                 root,
                 subset,
                 transform,
                 mode,
                 train=False):
        super(SplitMnistDataset, self).__init__(root, transform)

        if train:
            root = os.path.join(root, f'split_dataset/{mode}/MNIST/Train')
            file_path = os.path.join(root, f"subset_{subset}.pt")
            self.images, self.labels = torch.load(file_path)
        else:
            root = os.path.join(root, f'split_dataset/{mode}/MNIST/Test')
            file_path = os.path.join(root, 'subset_0.pt')
            self.images, self.labels = torch.load(file_path)

def getMnistDataLoader(root,
                        subset,
                        batch_size,
                        num_workers,
                        resize=[28,28],
                        mode='dirichlet',
                        train=False,
                        ):
    
    # Following transforms are taken from 
    # https://github.com/med-air/FedBN/blob/master/federated/fed_digits.py
    # to make fair experiments.
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(resize, Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = SplitMnistDataset(root,
                                 subset,
                                 transform,
                                 mode,
                                 train,
                                 )

    dataloader = DataLoader(dataset, 
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=train,
                            drop_last=train)

    return dataloader


# =======================================
# For cifar10 federated laerning.
# =======================================

class SplitCifar10Dataset(BaseSplitDataset):
    def __init__(self,
                 root,
                 subset,
                 transform,
                 mode,
                 train=False):
        super(SplitCifar10Dataset, self).__init__(root, transform)

        if train:
            root = os.path.join(root, f'split_dataset/{mode}/cifar10/Train')
            file_path = os.path.join(root, f"subset_{subset}.pt")
            self.images, self.labels = torch.load(file_path)
        else:
            root = os.path.join(root, f'split_dataset/{mode}/cifar10/Test')
            file_path = os.path.join(root, 'subset_0.pt')
            self.images, self.labels = torch.load(file_path)

    def __getitem__(self, index):
        in_ = self.images[index]
        label_ = self.labels[index]

        # convert to PIL.Image to apply torch.transform
        in_ = Image.fromarray(in_, mode='RGB')

        if self.transform is not None:
            in_ = self.transform(in_)

        return in_, label_

def getCifar10DataLoader(root,
                        subset,
                        batch_size,
                        num_workers,
                        resize=[32,32],
                        mode='dirichlet',
                        train=False,
                        ):
    
    # Following transforms are taken from 
    # https://github.com/med-air/FedBN/blob/master/federated/fed_digits.py
    # to make fair experiments.
    transform = transforms.Compose([
        transforms.Resize(resize, Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


    dataset = SplitCifar10Dataset(root,
                                 subset,
                                 transform,
                                 mode,
                                 train,
                                 )

    dataloader = DataLoader(dataset, 
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=train,
                            drop_last=train)

    return dataloader


# =======================================
# For fMNIST federated laerning.
# =======================================

class SplitfMNISTDataset(BaseSplitDataset):
    def __init__(self,
                 root,
                 subset,
                 transform,
                 mode,
                 train=False):
        super(SplitfMNISTDataset, self).__init__(root, transform)

        if train:
            root = os.path.join(root, f'split_dataset/{mode}/fMNIST/Train')
            file_path = os.path.join(root, f"subset_{subset}.pt")
            self.images, self.labels = torch.load(file_path)
        else:
            root = os.path.join(root, f'split_dataset/{mode}/fMNIST/Test')
            file_path = os.path.join(root, 'subset_0.pt')
            self.images, self.labels = torch.load(file_path)


def getfMNISTDataLoader(root,
                        subset,
                        batch_size,
                        num_workers,
                        resize=[32,32],
                        mode='dirichlet',
                        train=False,
                        ):
    
    # Following transforms are taken from 
    # https://github.com/med-air/FedBN/blob/master/federated/fed_digits.py
    # to make fair experiments.
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(resize, Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


    dataset = SplitfMNISTDataset(root,
                                 subset,
                                 transform,
                                 mode,
                                 train,
                                 )

    dataloader = DataLoader(dataset, 
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=train,
                            drop_last=train)

    return dataloader


# =======================================
# For SVHN federated laerning.
# =======================================

class SplitSVHNDataset(BaseSplitDataset):
    def __init__(self,
                 root,
                 subset,
                 transform,
                 mode,
                 train=False):
        super(SplitSVHNDataset, self).__init__(root, transform)

        if train:
            root = os.path.join(root, f'split_dataset/{mode}/SVHN/Train')
            file_path = os.path.join(root, f"subset_{subset}.pt")
            self.images, self.labels = torch.load(file_path)
        else:
            root = os.path.join(root, f'split_dataset/{mode}/SVHN/Test')
            file_path = os.path.join(root, 'subset_0.pt')
            self.images, self.labels = torch.load(file_path)

    def __getitem__(self, index):
        in_ = self.images[index]
        label_ = self.labels[index]

        # convert to PIL.Image to apply torch.transform
        in_ = Image.fromarray(in_, mode='RGB')

        if self.transform is not None:
            in_ = self.transform(in_)

        return in_, label_

def getSVHNDataLoader(root,
                    subset,
                    batch_size,
                    num_workers,
                    resize=[32,32],
                    mode='dirichlet',
                    train=False,
                    ):
    
    # Following transforms are taken from 
    # https://github.com/med-air/FedBN/blob/master/federated/fed_digits.py
    # to make fair experiments.
    transform = transforms.Compose([
        transforms.Resize(resize, Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = SplitSVHNDataset(root,
                                subset,
                                transform,
                                mode,
                                train,
                                )

    dataloader = DataLoader(dataset, 
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=train,
                            drop_last=train)

    return dataloader

def main():
    l = getSVHNDataLoader(root='/jeongsol/Jeongsol/data/',
                             subset=0,
                            batch_size=2,
                            num_workers=2,
                            mode='dirichlet',
                            train=True,
                            )
    it_loader = iter(l)
    for x,y in it_loader:
        break

    print('>> [Train] Training Dataloader is created successfully.')

    l = getSVHNDataLoader(root='/jeongsol/Jeongsol/data/',
                          subset=0,
                        batch_size=2,
                        num_workers=2,
                        mode='dirichlet',
                        train=False,
                        )
    
    it_loader = iter(l)
    for x,y in it_loader:
        break

    print('>> [Test] Test Dataloader is created successfully.')

    
if __name__=='__main__':
    main()

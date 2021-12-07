import torch
from torchvision import datasets, transforms
from torch.utils.data.dataset import random_split
from torch.utils import data
import params
import torch.utils.data as data_utils
from custom_aug import Synthetic, Synthetic1
from glob import glob
import os
from PIL import Image


class GetLoader(data.Dataset):
    def __init__(self, data_root, data_list, transform=None):
        self.root = data_root
        self.transform = transform

        with open(data_list, 'r') as f:
            data_list = f.read()


        self.img_paths = glob(os.path.join(data_root,'*'))
        self.img_labels = data_list.split('\n')
        self.n_data = len(self.img_labels)

    def __getitem__(self, item):
        imgs = Image.open(self.img_paths[item]).convert('RGB')

        if self.transform:
            imgs = self.transform(imgs)
            labels = int(self.img_labels[item])

        return imgs, labels

    def __len__(self):
        return self.n_data

def get_mnist(train,adp=False,size=0,mode=None):
    """Get usps dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([
        transforms.Resize(params.image_size),
        # Synthetic1(tgt),
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: x.repeat(3, 1, 1))        
        ])

    transform_list = [
        transforms.Resize(params.image_size),
        transforms.ToTensor(),
    ]

    # dataset and data loader
    # print(mode)
    if mode == 'mnist->usps' and train:
        pre_process = transforms.Compose(transform_list)
        mnist_dataset = GetLoader(params.mnist_dataset_root, '/root/jchlwogur/pytorch_adda_mixup/mnist_train.txt',pre_process)
        # mnist_dataset = GetLoader(params.mnist_dataset_root, '/root/jchlwogur/pytorch_adda_mixup/mnist_train.txt',pre_process)
    else:
        if mode == 'mnist->mnist_m' and train:
            transform_list.insert(1,Synthetic())
        else:
            transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
        pre_process = transforms.Compose(transform_list)
        mnist_dataset = datasets.MNIST(root=params.mnist_dataset_root,
                                    train=train,
                                    transform=pre_process,
                                    download=True)

    # label 저장하기 위한 코드
    # print(len(mnist_dataset))
    # if train:
    #     f = open('mnist_train.txt','w')
    #     for _,label in mnist_dataset:
    #         f.write(str(label)+'\n')
    #     f.close()
    if train:
        mnist_dataset,  _   = data_utils.random_split(mnist_dataset, [size,len(mnist_dataset)-size])

    mnist_data_loader = torch.utils.data.DataLoader(
        dataset=mnist_dataset,
        batch_size= params.adp_batch_size if adp else params.batch_size,
        num_workers = 4,
        shuffle=True,
        drop_last=True)
    return mnist_data_loader
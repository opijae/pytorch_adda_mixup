import torch.utils.data as data
from PIL import Image
import os
import params
from torchvision import transforms
import torch 

import torch.utils.data as data_utils


class GetLoader(data.Dataset):
    def __init__(self, data_root, data_list, transform=None):
        self.root = data_root
        self.transform = transform

        f = open(data_list, 'r')
        data_list = f.readlines()
        f.close()

        self.n_data = len(data_list)

        self.img_paths = []
        self.img_labels = []

        for data_ in data_list:
            self.img_paths.append(data_[:-3])
            self.img_labels.append(data_[-2])

    def __getitem__(self, item):
        img_paths, labels = self.img_paths[item], self.img_labels[item]
        imgs = Image.open(os.path.join(self.root, img_paths)).convert('RGB')

        if self.transform:
            imgs = self.transform(imgs)
            labels = int(labels)

        return imgs, labels

    def __len__(self):
        return self.n_data


def get_mnist_m(train,adp=False,size= 0,tgt =None):

    if train == True:
        mode = 'train'
    else:
        mode = 'test'

    train_list = os.path.join(params.mnist_m_dataset_root, 'mnist_m_{}_labels.txt'.format(mode))
        # image pre-processing
    pre_process = transforms.Compose([
        transforms.Resize(params.image_size),
        transforms.ToTensor(),
    ]
   )

    dataset_target = GetLoader(
        data_root=os.path.join(params.mnist_m_dataset_root, 'mnist_m_{}'.format(mode)),
        data_list=train_list,
        transform=pre_process)
        
    if train:
        dataset_target,_ = data_utils.random_split(dataset_target, [size,len(dataset_target)-size])


    dataloader  = torch.utils.data.DataLoader(
        dataset=dataset_target,
        batch_size= params.adp_batch_size if adp else params.batch_size,
        num_workers = 4,
        shuffle=True,
        drop_last=True)
    
    return dataloader
import torch
import torchvision.datasets as datasets

import os
from tqdm import tqdm
train = True

svhn_dataset = datasets.SVHN(root='./svhn',
                                   split='train' if train else 'test',
                                   download=True)

mnist_train = datasets.MNIST(root='MNIST_data/', # 다운로드 경로 지정
                          train=train, # True를 지정하면 훈련 데이터로 다운로드
                          # transform=transforms.ToTensor(), # 텐서로 변환
                          download=True)

usps_dataset = datasets.USPS(root='./USPS',
                                   train=train,
                                   download=True)

folder_path = 'train' if train else 'test'
folder_path = os.path.join('usps',folder_path)

for i,img in tqdm(enumerate(mnist_train)):
    file_name = str(i).zfill(8)
    img[0].save(f'{folder_path}/{file_name}.jpg')
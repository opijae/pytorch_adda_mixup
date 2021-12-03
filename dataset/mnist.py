import torch
from torchvision import datasets, transforms
import torch.utils.data as data_utils


from custom_aug import Synthetic, Synthetic1
import params


def get_mnist(train,adp = False,size = 0,tgt=None):
    """Get MNIST dataset loader."""
    # image pre-processing

    transform_list = [
        transforms.Resize(params.image_size),
        transforms.ToTensor(),
    ]

    if params.domain_shuffle:
        transform_list.insert(1,Synthetic())
    else:
        transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))


    pre_process = transforms.Compose(transform_list)                                 



    # dataset and data loader
    mnist_dataset = datasets.MNIST(root=params.mnist_dataset_root,
                                   train=train,
                                   transform=pre_process,                                   
                                   download=True)
    if train:
        mnist_dataset,_ = data_utils.random_split(mnist_dataset, [size,len(mnist_dataset)-size])

    mnist_data_loader = torch.utils.data.DataLoader(
        dataset=mnist_dataset,
        batch_size= params.adp_batch_size if adp else params.batch_size,
        num_workers = 4,
        shuffle=True,
        drop_last=True)
    return mnist_data_loader
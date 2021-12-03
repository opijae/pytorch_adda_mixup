import numpy as np
import cv2
import random
from glob import glob
from .get_roi import synthesis

mnist_m_bg = glob('/root/jchlwogur/pytorch_adda_mixup/cluster_bg/*')[:1000]
# mnist_train = glob('/root/jchlwogur/pytorch_adda_mixup/mnist/train/*')[:1000]
class Synthetic(object):
    """
        Mnist + mnist_m(bg)
        mnist를 mnist_m 처럼 만들기
    """
    def __call__(self, image):
        image = image.convert('RGB')
        np_image = np.array(image)
        h,w = np_image.shape[:2]
        random.shuffle(mnist_m_bg)
        bg = cv2.imread(mnist_m_bg[0])
        bg = cv2.resize(bg,(h,w))
        masked = np.where(np_image ==0 , bg, np_image)
        return masked

class Synthetic1(object):
    """
        usps를 mnist 처럼 만들기
    """
    def __init__(self, tgt):
        if tgt == 'mnist':
            self.tgt_dataset = mnist_train


    def __call__(self, image):
        # print(image)
        image = image.convert('RGB')
        np_image = np.array(image)
        h,w = np_image.shape[:2]
        random.shuffle(self.tgt_dataset)
        tgt_img = cv2.imread(self.tgt_dataset[0])
        tgt_img = cv2.resize(tgt_img,(h,w))
        # cv2.imwrite('/root/jchlwogur/pytorch_adda_mixup/aaa.jpg',synthesis(np_image, tgt_img))
        # print(synthesis(np_image, tgt_img).shape)
        return synthesis(np_image, tgt_img)
import numpy as np
import cv2
import random
from glob import glob


mnist_m_bg = glob('/root/jchlwogur/pytorch_adda_mixup/cluster_bg/*')[:1000]

class Synthetic(object):
    """
        Mnist + mnist_m(bg)
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
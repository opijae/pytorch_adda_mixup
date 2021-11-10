import numpy as np
from sklearn.cluster import KMeans
import cv2
import datetime
import os
from glob import glob
from tqdm import tqdm
import random


directions = [[0,1],[1,0],[0,-1],[-1,0]]
def fill_black(image):
  h,w = image.shape[:2]
  for i in range(h):
    for j in range(w):
      if image[i][j].sum() == 255*3:
        cnt = 0
        temp = np.zeros_like(image[i][j])
        random.shuffle(directions)
        for di,dj in directions:
          ni,nj = di +i , dj + j
          if 0<= ni < h and 0<= nj < w:
            if image[ni][nj].sum() < 255*3:
              image[i][j] = image[ni][nj]
              break
  return image

image_folder = glob('/root/jchlwogur/pytorch_adda_mixup/mnist_m/mnist_m_test/*.png')
for image in tqdm(sorted(image_folder)):
    new_filename = image.split('/')[-1]
    image = cv2.imread(image)
    orig_image = image.copy()
    orig = image.copy()

    # HSV에서 H,V만 사용
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = image[:,:,[0,2]]

    # flatten
    reshaped = image.reshape(image.shape[0] * image.shape[1], image.shape[2])

    # KMeans , 글자와 배경을 분리하기 위해 cluster는 2개
    num_cluster = 2
    kmeans = KMeans(n_clusters = num_cluster, n_init=40, max_iter=500).fit(reshaped)
    # 1D -> 2D
    clustering = np.reshape(np.array(kmeans.labels_, dtype=np.uint8),(image.shape[0], image.shape[1]))

    # 배경의 픽셀 수가 제일 많다고 가정하고 배경 찾기
    # 0인 부분이 배경 1은 숫자
    sortedLabels = sorted([n for n in range(num_cluster)],key=lambda x: -np.sum(clustering == x))
    
    kmeansImage = np.zeros(image.shape[:2], dtype=np.uint8)
    for i, label in enumerate(sortedLabels):
        kmeansImage[clustering == label] = int(255 / num_cluster - 1) * i
    kmeansImage = cv2.cvtColor(kmeansImage, cv2.COLOR_GRAY2BGR)

    bg = np.where(kmeansImage==0, orig_image, 255)
    cv2.imwrite(os.path.join('/root/jchlwogur/pytorch_adda_mixup/cluster_bg',new_filename), fill_black(bg))

import numpy as np
from sklearn.cluster import KMeans
import cv2
import datetime
import os
from glob import glob
from tqdm import tqdm
import random


directions = [[0,1],[1,0],[0,-1],[-1,0]]

tgt_info = []
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


def get_gray_kmeansImage(image):
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
      kmeansImage[clustering == label] = int(255 / (num_cluster - 1)) * i
  return kmeansImage


def get_num_area(image):
  h,w = image.shape[:2]
  start_x,start_y = w, h
  end_x,end_y = 0,0
  for i in range(h):
    for j in range(w):
      if image[i][j] > 0:
        start_x = min(j, start_x)
        start_y = min(i, start_y)
        end_x = max(j, end_x)
        end_y = max(i, end_y)
  return start_x,start_y,end_x,end_y


def synthesis(src_img, tgt_img):
  """
  src_img = label 있음
  tgt_img = label 없음
  domain adaptation의 src와 타켓
  """

  global tgt_info
  random.shuffle(tgt_info)
  if len(tgt_info) < 100:
    tgt_img = cv2.resize(tgt_img,(28,28))
    tgt_img_kmeans = get_gray_kmeansImage(tgt_img)
    tgt_roi = get_num_area(tgt_img_kmeans)
    tgt_info.append((tgt_img,tgt_img_kmeans, tgt_roi))
  else:
    tgt_img,tgt_img_kmeans, tgt_roi = tgt_info[0]
  src_img = cv2.resize(src_img,(28,28))

  result = np.zeros(src_img.shape, dtype=np.uint8)

  src_img_kmeans = get_gray_kmeansImage(src_img)

  src_roi = get_num_area(src_img_kmeans)

  src_roi_image = src_img[src_roi[1]:src_roi[3],src_roi[0]:src_roi[2]]

  result[tgt_roi[1]:tgt_roi[3],tgt_roi[0]:tgt_roi[2]] = cv2.resize(src_roi_image,(tgt_roi[2]- tgt_roi[0],tgt_roi[3]- tgt_roi[1]))

  return result


def save_sythesis(src_folder, tgt_folder):
  directory = '../' + src_folder.split('/')[4] + '_' + tgt_folder.split('/')[4]
  if not os.path.exists(directory):
    os.makedirs(directory)
  src_image_folder = glob(src_folder)
  tgt_image_folder = glob(tgt_folder)

  print(len(tgt_image_folder))
  for image_path in tqdm(sorted(src_image_folder)):
    random.shuffle(tgt_image_folder)
    src_image = cv2.imread(image_path)
    tgt_image = cv2.imread(tgt_image_folder[0])
    new_filename = image_path.split('/')[-1]

    cv2.imwrite(os.path.join(directory,new_filename), synthesis(src_image, tgt_image))

# image_folder = glob('/root/jchlwogur/pytorch_adda_mixup/usps/train/*')
# for image in tqdm(sorted(image_folder)):
#     new_filename = image.split('/')[-1]
#     image = cv2.imread(image)
#     orig_image = image.copy()
#     orig = image.copy()
    
#     kmeansImage = get_gray_kmeansImage(image)
#     kmeansImage = cv2.cvtColor(kmeansImage, cv2.COLOR_GRAY2BGR)

#     bg = np.where(kmeansImage==0, orig_image, 255)
#     # cv2.imwrite(os.path.join('/root/jchlwogur/pytorch_adda_mixup/cluster_bg_usps','orig' +new_filename), orig)
#     # cv2.imwrite(os.path.join('/root/jchlwogur/pytorch_adda_mixup/cluster_bg_usps','bg' +new_filename), bg)
#     cv2.imwrite(os.path.join('/root/jchlwogur/pytorch_adda_mixup/cluster_bg_usps',new_filename), fill_black(bg))
if __name__=="__main__":
  src_folder = '/root/jchlwogur/pytorch_adda_mixup/mnist/train/*'
  tgt_folder = '/root/jchlwogur/pytorch_adda_mixup/usps/train/*'

  save_sythesis(src_folder, tgt_folder)
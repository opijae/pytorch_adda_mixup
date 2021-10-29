import random
import numpy as np
import os
from glob import glob
import cv2


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

if __name__ == '__main__':
    file_list = glob('mnist_m/bg/*')
    for image_path in file_list:
        image = cv2.imread(image_path)
        cv2.imwrite(image_path.replace('bg','filled_bg'),fill_black(image))

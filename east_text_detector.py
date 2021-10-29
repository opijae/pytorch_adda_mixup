import cv2
from tqdm import tqdm
import numpy as np
def predictions(prob_score, geo):
	(numR, numC) = prob_score.shape[2:4]
	boxes = []
	confidence_val = []

	for y in range(0, numR):
		scoresData = prob_score[0, 0, y]
		x0 = geo[0, 0, y]
		x1 = geo[0, 1, y]
		x2 = geo[0, 2, y]
		x3 = geo[0, 3, y]
		anglesData = geo[0, 4, y]

		for i in range(0, numC):
			if scoresData[i] < 0.5:
				continue

			(offX, offY) = (i * 4.0, y * 4.0)
			angle = anglesData[i]
			cos = np.cos(angle)
			sin = np.sin(angle)

			h = x0[i] + x2[i]
			w = x1[i] + x3[i]

			endX = int(offX + (cos * x1[i]) + (sin * x2[i]))
			endY = int(offY - (sin * x1[i]) + (cos * x2[i]))
			startX = int(endX - w)
			startY = int(endY - h)

			boxes.append((startX, startY, endX, endY))
			confidence_val.append(scoresData[i])

	return (boxes, confidence_val)


cnt = 0
net = cv2.dnn.readNet('frozen_east_text_detection.pb')
layerNames = [
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"]
for i in tqdm(range(1000)):
    # set image path and export folder directory
    file_name = str(i).zfill(8)
    # image = f'mnist_m/mnist_m_test/{file_name}.png' # can be filepath, PIL image or numpy array
    image = f'usps/train/{file_name}.jpg' 
    # output_dir = 'outputs/'
    # image = np.array(svhn_dataset[27000][0])
    image = cv2.imread(image)
    image = cv2.resize(image, (320, 320))

    blob = cv2.dnn.blobFromImage(image, 1.0, (320, 320),(123.68, 116.78, 103.94), swapRB=True, crop=False)


    
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    (boxes1, confidence_val) = predictions(scores, geometry)
    # boxes1 = non_max_suppression(np.array(boxes1), probs=confidence_val)
    boxes1=np.array(boxes1,int)
    # print(boxes1)
    if len(boxes1) > 0:
        cnt += 1
print(cnt)
from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
    export_detected_regions,
    export_extra_results,
    empty_cuda_cache
)
from tqdm import tqdm
import os
import cv2
import numpy as np
# def 

# load models
refine_net = load_refinenet_model(cuda=True)
craft_net = load_craftnet_model(cuda=True)
cnt = 0

# main_folder = 'underline'
# folder_length = len(os.listdir())
for i in tqdm(range(len(os.listdir('mnist_m/mnist_m_test')))):
    # set image path and export folder directory
    file_name = str(i).zfill(8)
    # /root/jchlwogur/pytorch_adda_mixup/mnist_m/mnist_m_test
    image = f'mnist_m/mnist_m_test/{file_name}.png' # can be filepath, PIL image or numpy array
    output_dir = 'outputs/'

    # read image
    image = read_image(image)
    # perform prediction
    prediction_result = get_prediction(
        image=image,
        craft_net=craft_net,
        refine_net=refine_net,
        text_threshold=0.5,
        link_threshold=0.4,
        low_text=0.4,
        cuda=True,
        long_size=1280
    )
    if len(prediction_result["boxes"])>0:
        pts = np.array(prediction_result["boxes"][0],np.int32)
        # print(pts)
        height, width = image.shape[:2]
        # print(height,width)
        area = (pts[:,0].max() - pts[:,0].min()) * (pts[:,1].max() - pts[:,1].min())
        if area < height * width /4:
            continue
        cnt += 1
        img = cv2.fillConvexPoly(image, pts, (255,255,255))
        # img = cv2.polylines(image, [pts], True, (0,0,0),-1)
        cv2.imwrite(f'mnist_m/bg/{file_name}.png',img)
        # print(prediction_result["boxes"])
# # export heatmap, detection points, box visualization
# export_extra_results(
#     image=image,
#     regions=prediction_result["boxes"],
#     heatmaps=prediction_result["heatmaps"],
#     output_dir=output_dir
# )

# # unload models from gpu
empty_cuda_cache()

print(cnt)
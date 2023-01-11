#import inference_scripts.inference_RAFT as inference_RAFT
#import generate_matrix_and_stats.matrix_from_file_path_and_model_instance as matrix_from_file_path_and_model_instance
import argparse
from imageio import imread,imwrite
import utils.flow
# import cv2
# import numpy as np
import os
# import matplotlib.pyplot as plt
# from PIL import Image
# frL_pth='/home/ssavian/training/KITTI2015/training/image_2/000193_10.png'
# frR_pth='/home/ssavian/training/KITTI2015/training/image_2/000193_11.png'
# flo_pth='/home/ssavian/training/KITTI2015/training/flow_occ/000193_10.png'
#
# dest_path='/home/ssavian/training/plots_ironspeed/visualization'
# if not os.path.exists(dest_path):
#     os.makedirs(dest_path)
# train_chairs_pth='/home/ssavian/training/RAFT_trained_models/first_run/raft-chairs.pth'
# train_things_pth='/home/ssavian/optical_flow_networks/RAFT/models/raft-things.pth'
# train_kitti_pth='/home/ssavian/optical_flow_networks/RAFT/models/raft-kitti.pth'
#
# train_pth= train_kitti_pth
# type='kitti_'
#

##save everything
#destination
def save_image(image,name,OF=False,dest_path='/home/ssavian/training/plots_ironspeed/temp'):
    path=os.path.join(dest_path,name)
    if OF:
        imwrite(path, utils.flow.flow_to_png_middlebury(image))
        return path
    imwrite(path, image)
    return path

# def save_matrix(image,name,OF=False,dest_path='/home/ssavian/training/plots_ironspeed/temp'):
#     path=os.path.join(dest_path,name)
#     if OF:
#         plt.imsave(path,np.multiply((invalid_areas[...,0].astype(np.float32)),255).astype(np.uint8),cmap='bwr')
#         return path
#     imwrite(path, image)
#     return path
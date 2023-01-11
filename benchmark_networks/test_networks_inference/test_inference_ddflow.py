import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import inference_scripts.inference_DDFlow_v2 as inference_DDflow
from imageio import imread,imwrite
import cv2
from utils.flow import flow_to_png_middlebury
#set PYTHONPATH='pwd'


rotate_sintel = False
note = 'DDflow_flying_distillation'
mode = 'clean'
results_pth = os.path.join('/home/ssavian/training/plots/a_framework_results_90/',note)
dataset_pth = '/home/ssavian/training/sintel/'
if rotate_sintel:
    fake_input_1 = np.zeros([1024, 436, 3])
else:
    fake_input_1 = np.zeros([436,1024,3])
model_DDflow = inference_DDflow.inference([fake_input_1,fake_input_1],model_checkpoint_path = '/home/ssavian/optical_flow_networks/DDFlow/models/FlyingChairs/data_distillation',rotate_sintel = True)


sintel_pth = '/home/ssavian/training/sintel/training/clean/alley_1'
im1_pth = os.path.join(sintel_pth,"frame_0001.png")
im2_pth = os.path.join(sintel_pth,"frame_0002.png")
im3_pth = os.path.join(sintel_pth,"frame_0003.png")
gnd_pth = '/home/ssavian/training/sintel/training/flow/alley_1/frame_0002.flo'
gnd_vis = cv2.readOpticalFlow(gnd_pth)
output_png_pth = '/home/ssavian/training/plots_ironspeed/test_gnd_ddflow.png'
imwrite(output_png_pth, flow_to_png_middlebury(gnd_vis))

fr1 = np.asarray(imread(im1_pth))
fr2 = np.asarray(imread(im2_pth))
inputs = [fr1.copy(), fr2.copy()]
output = model_DDflow(inputs)
output_png_pth = '/home/ssavian/training/plots_ironspeed/test_ddflow.png'
imwrite(output_png_pth, flow_to_png_middlebury(output))

matlab_pth = '/home/ssavian/training/img_formation_mat/TCSVT_repository/balanced_dataset_textured_std_60_radius_50_limit_dis1000_samples/data'
im1_pth = os.path.join(matlab_pth,"pair486_L.png")
im2_pth = os.path.join(matlab_pth,"pair486_R.png")
fr1 = np.asarray(imread(im1_pth))[0:222,0:221,:]
fr2 = np.asarray(imread(im2_pth))[0:222,0:221,:]
inputs = [fr1.copy(), fr2.copy()]

tf.reset_default_graph()
model_DDflow_mat = inference_DDflow.inference([fr1,fr2],model_checkpoint_path = '/home/ssavian/optical_flow_networks/DDFlow/models/FlyingChairs/data_distillation',rotate_sintel = True)

output = model_DDflow_mat(inputs)
output_png_pth = '/home/ssavian/training/plots_ironspeed/test_ddflow_mat.png'
imwrite(output_png_pth, flow_to_png_middlebury(output))

# class InputPadder:
# def __init__(self, dims, mode='sintel'):
#     self.ht, self.wd = dims[-2:]
#     pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
#     pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
#     if mode == 'sintel':
#         self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
#     else:
#         self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]
#
#
# def pad(self, *inputs):
#     return [F.pad(x, self._pad, mode='replicate') for x in inputs]
#
#
# def unpad(self, x):
#     ht, wd = x.shape[-2:]
#     c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
#     return x[..., c[0]:c[1], c[2]:c[3]]
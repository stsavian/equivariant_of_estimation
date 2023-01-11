# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np
#import os
import logging
import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
# logging.getLogger('tensorflow').setLevel(logging.FATAL)
import sys
import time
import cv2

from six.moves import xrange
from scipy import misc, io
from imageio import imwrite, imread
import tensorflow as tf
from tensorflow.contrib import slim

import matplotlib.pyplot as plt
import networks.DDFlow.ddflow_model as ddflow_model
from networks.DDFlow.network import pyramid_processing
from networks.SelFlow.datasets import BasicDataset
from networks.DDFlow.utils import average_gradients, lrelu, occlusion, rgb_bgr#,mvn
from networks.DDFlow.data_augmentation import flow_resize
from networks.DDFlow.flowlib import flow_to_color, write_flo
from networks.DDFlow.warp import tf_warp
from utils.flow import flow_to_png_middlebury
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#tf.compat.v1.logging.set_verbosity(tf.logging.ERROR)

tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
## Input
sintel_pth = '/home/ssavian/training/sintel/training/clean/alley_1'
im1_pth = os.path.join(sintel_pth,"frame_0001.png")
im2_pth = os.path.join(sintel_pth,"frame_0002.png")
im3_pth = os.path.join(sintel_pth,"frame_0003.png")
gnd_pth = '/home/ssavian/training/sintel/training/flow/alley_1/frame_0002.flo'
gnd_vis = cv2.readOpticalFlow(gnd_pth)
output_png_pth = '/home/ssavian/training/plots/test_gnd_ddflow.png'
imwrite(output_png_pth, flow_to_png_middlebury(np.transpose(gnd_vis, (2, 0, 1))))

img0 = tf.image.decode_png(tf.read_file(im1_pth), channels=3)
img0 = tf.cast(img0, tf.float32)
img1 = tf.image.decode_png(tf.read_file(im2_pth), channels=3)
img1 = tf.cast(img1, tf.float32)
img2 = tf.image.decode_png(tf.read_file(im3_pth), channels=3)
img2 = tf.cast(img2, tf.float32)


img0 = img0 / 255.
img1 = img1 / 255.
img2 = img2 / 255.

img_shape = tf.shape(img1)
h = img_shape[0] #is not h, it's probably row number
w = img_shape[1]

img0 = tf.expand_dims(img0, 0)
img1 = tf.expand_dims(img1, 0)
img2 = tf.expand_dims(img2, 0)

new_h = tf.where(tf.equal(tf.mod(h, 64), 0), h, (tf.to_int32(tf.floor(h / 64) + 1)) * 64)
new_w = tf.where(tf.equal(tf.mod(w, 64), 0), w, (tf.to_int32(tf.floor(w / 64) + 1)) * 64)
#tf.reset_default_graph()

batch_img0 = tf.image.resize_images(img0, [new_h, new_w], method=1, align_corners=True)
batch_img1 = tf.image.resize_images(img1, [new_h, new_w], method=1, align_corners=True)

flow_est = pyramid_processing(batch_img0, batch_img1, train=False, trainable=False, regularizer=None, is_scale=True)



restore_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
saver = tf.train.Saver(var_list=restore_vars)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

model_checkpoint_path = '/home/ssavian/optical_flow_networks/DDFlow/models/FlyingChairs/data_distillation'
saver.restore(sess, model_checkpoint_path)
output = flow_est['full_res'].eval(session=sess)
#output_color = flow_fw_color.eval(session=sess)

temp = output[0,:,:,:].transpose([2, 0, 1])
tf.get_variable_scope().reuse_variables()

#tf.reset_default_graph()
imwrite('/home/ssavian/training/plots_iron/test_ddflow_ouput.png', flow_to_png_middlebury(temp))
#imwrite('/home/ssavian/training/plots/test_selflow_ouput.png', (output_color[0,:,:,:]*255).astype(np.uint8))


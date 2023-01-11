# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
import os
import networks.DDFlow.ddflow_model as ddflowmodel
from networks.DDFlow.network import pyramid_processing
os.environ["CUDA_VISIBLE_DEVICES"]="0"
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
#https://stackoverflow.com/questions/50942550/initializing-tensorflow-session-in-a-class-constructor
#https://stackoverflow.com/questions/51175837/tensorflow-runs-out-of-memory-while-computing-how-to-find-memory-leaks
import time
class inference(object):
    def __init__(self,inputs,model_checkpoint_path = '/home/ssavian/optical_flow_networks/DDFlow/models/FlyingChairs/data_distillation',rotate_sintel = False):
        self.model_checkpoint_path = model_checkpoint_path
        self.rotate_sintel = rotate_sintel
        self.img1_p = tf.placeholder(tf.float32, inputs[0].shape)
        self.img2_p = tf.placeholder(tf.float32, inputs[0].shape)

        img1 = self.img1_p#.astype(np.float32)
        img2 = self.img2_p#.astype(np.float32)
        img1 = img1 / 255.
        img2 = img2 / 255.

        img_shape = tf.shape(img1)
        h = img_shape[0]  # is not h, it's probably row number
        w = img_shape[1]


        #img0 = tf.expand_dims(img0, 0)
        img1 = tf.expand_dims(img1, 0)
        img2 = tf.expand_dims(img2, 0)

        # img1_p = tf.placeholder(tf.float32, img1.shape)
        # img2_p = tf.placeholder(tf.float32, img1.shape)

        new_h = tf.where(tf.equal(tf.mod(h, 64), 0), h, (tf.to_int32(tf.floor(h / 64) + 1)) * 64)
        new_w = tf.where(tf.equal(tf.mod(w, 64), 0), w, (tf.to_int32(tf.floor(w / 64) + 1)) * 64)

        batch_img0 = tf.image.resize_images(img1, [new_h, new_w], method=1, align_corners=True)
        batch_img1 = tf.image.resize_images(img2, [new_h, new_w], method=1, align_corners=True)

        # self.img1_p = tf.placeholder(tf.float32, batch_img0.shape)
        # self.img2_p = tf.placeholder(tf.float32, batch_img1.shape)

        # flow_est = pyramid_processing(self.img1_p, self.img2_p, train=False, trainable=False, regularizer=None,
        #                               is_scale=True)
        self.flow_est = pyramid_processing(batch_img0, batch_img1, train=False, trainable=False, regularizer=None,
                                      is_scale=True)

        self.restore_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(var_list=self.restore_vars)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.saver.restore(self.sess, self.model_checkpoint_path)
        tf.get_variable_scope().reuse_variables()
        tf.get_default_graph().finalize()
    def __call__(self, inputs):
        start_time = time.time()

        output =  self.sess.run(self.flow_est['full_res'],feed_dict = {self.img1_p:inputs[0],self.img2_p:inputs[1]})

        temp = output[0, :, :, :].transpose([2, 0, 1])
        tf.get_variable_scope().reuse_variables()
        #output = flow_est['full_res']
        #temp = tf.make_ndarray(output[0, :, :, :]).transpose([2, 0, 1])
        print("--- %s seconds ---" % (time.time() - start_time))
        # if self.rotate_sintel:
        #     out = np.transpose(temp, (1, 2, 0))[:,6:-6,:]
        # else:
        #     out = np.transpose(temp, (1, 2, 0))[6:-6,:,:]
        out = np.transpose(temp, (1, 2, 0))

        rows_delta = out.shape[0] - inputs[0].shape[0]
        cols_delta = out.shape[1] - inputs[0].shape[1]
        if rows_delta ==0 and cols_delta ==0:
             out = out
        elif cols_delta == 0:
           out = out[int(rows_delta / 2):int(-rows_delta / 2), :, :]
        elif rows_delta == 0:
            out= out[:, int(cols_delta / 2):int(-cols_delta / 2), :]
        else:
            out = out[int(rows_delta/2):int(-rows_delta/2),int(cols_delta/2):int(-cols_delta/2),:]

        return out




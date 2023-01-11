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
class inference(object):
    def __init__(self,model_checkpoint_path = '/home/ssavian/optical_flow_networks/DDFlow/models/FlyingChairs/data_distillation'):
        self.model_checkpoint_path = model_checkpoint_path


    def __call__(self, inputs):
        img1 = inputs[0].astype(np.float32)
        img2 = inputs[1].astype(np.float32)
        img1 = img1 / 255.
        img2 = img2 / 255.

        img_shape = tf.shape(img1)
        h = img_shape[0]  # is not h, it's probably row number
        w = img_shape[1]

        #img0 = tf.expand_dims(img0, 0)
        img1 = tf.expand_dims(img1, 0)
        img2 = tf.expand_dims(img2, 0)

        new_h = tf.where(tf.equal(tf.mod(h, 64), 0), h, (tf.to_int32(tf.floor(h / 64) + 1)) * 64)
        new_w = tf.where(tf.equal(tf.mod(w, 64), 0), w, (tf.to_int32(tf.floor(w / 64) + 1)) * 64)

        batch_img0 = tf.image.resize_images(img1, [new_h, new_w], method=1, align_corners=True)
        batch_img1 = tf.image.resize_images(img2, [new_h, new_w], method=1, align_corners=True)

        flow_est = pyramid_processing(batch_img0, batch_img1, train=False, trainable=False, regularizer=None,
                                      is_scale=True)

        restore_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        saver = tf.train.Saver(var_list=restore_vars)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())


        saver.restore(sess, self.model_checkpoint_path)
        output = flow_est['full_res'].eval(session=sess)
        # output_color = flow_fw_color.eval(session=sess)

        temp = output[0, :, :, :].transpose([2, 0, 1])
        tf.get_variable_scope().reuse_variables()
        #output = flow_est['full_res']
        #temp = tf.make_ndarray(output[0, :, :, :]).transpose([2, 0, 1])

        return np.transpose(temp, (1, 2, 0))[6:-6,:,:]




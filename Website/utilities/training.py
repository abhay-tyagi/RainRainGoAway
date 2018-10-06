#!/usr/bin/env python2
# -*- coding: utf-8 -*-


# This is a re-implementation of training code of our paper:
# X. Fu, J. Huang, D. Zeng, Y. Huang, X. Ding and J. Paisley. “Removing Rain from Single Images via a Deep Detail Network”, CVPR, 2017.
# author: Xueyang Fu (fxy@stu.xmu.edu.cn)

import os
import re
import random
import numpy as np
import tensorflow as tf
import matplotlib.image as img
import matplotlib.pyplot as plt
from Website.utilities.GuidedFilter import guided_filter



##################### Select GPU device ####################################
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
############################################################################

tf.reset_default_graph()

##################### Network parameters ###################################
num_feature = 16             # number of feature maps
num_channels = 3             # number of input's channels 
patch_size = 64              # patch size 
KernelSize = 3               # kernel size 
learning_rate = 0.1          # learning rate
iterations = int(2.1*1e5)        # iterations
batch_size = 20              # batch size
save_model_path = "./model/" # saved model's path
model_name = 'model-epoch'   # saved model's name
############################################################################

# network structure
def inference(images, is_training):
    regularizer = tf.contrib.layers.l2_regularizer(scale = 1e-10)
    initializer = tf.contrib.layers.xavier_initializer()

    base = guided_filter(images, images, 15, 1, nhwc=True) # using guided filter for obtaining base layer
    detail = images - base   # detail layer

   #  layer 1
    with tf.variable_scope('layer_1'):
         output = tf.layers.conv2d(detail, num_feature, KernelSize, padding = 'same', kernel_initializer = initializer, 
                                   kernel_regularizer = regularizer, name='conv_1')
         output = tf.layers.batch_normalization(output, training=is_training, name='bn_1')
         output_shortcut = tf.nn.relu(output, name='relu_1')
  
   #  layers 2 to 25
    for i in range(12):
        with tf.variable_scope('layer_%d'%(i*2+2)):	
             output = tf.layers.conv2d(output_shortcut, num_feature, KernelSize, padding='same', kernel_initializer = initializer, 
                                       kernel_regularizer = regularizer, name=('conv_%d'%(i*2+2)))
             output = tf.layers.batch_normalization(output, training=is_training, name=('bn_%d'%(i*2+2)))	
             output = tf.nn.relu(output, name=('relu_%d'%(i*2+2)))


        with tf.variable_scope('layer_%d'%(i*2+3)): 
             output = tf.layers.conv2d(output, num_feature, KernelSize, padding='same', kernel_initializer = initializer,
                                       kernel_regularizer = regularizer, name=('conv_%d'%(i*2+3)))
             output = tf.layers.batch_normalization(output, training=is_training, name=('bn_%d'%(i*2+3)))
             output = tf.nn.relu(output, name=('relu_%d'%(i*2+3)))

        output_shortcut = tf.add(output_shortcut, output)   # shortcut

   # layer 26
    with tf.variable_scope('layer_26'):
         output = tf.layers.conv2d(output_shortcut, num_channels, KernelSize, padding='same',   kernel_initializer = initializer, 
                                   kernel_regularizer = regularizer, name='conv_26')
         neg_residual = tf.layers.batch_normalization(output, training=is_training, name='bn_26')

    final_out = tf.add(images, neg_residual)

    return final_out
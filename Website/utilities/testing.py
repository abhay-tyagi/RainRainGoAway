#!/usr/bin/env python2
# -*- coding: utf-8 -*-


# This is a re-implementation of testing code of this paper:
# X. Fu, J. Huang, D. Zeng, Y. Huang, X. Ding and J. Paisley. “Removing Rain from Single Images via a Deep Detail Network”, CVPR, 2017.
# author: Xueyang Fu (fxy@stu.xmu.edu.cn)



import os
import numpy as np
import tensorflow as tf
import Website.utilities.training as Network
import matplotlib.image as img
import matplotlib.pyplot as plt
from os import listdir
import cv2


#os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # select GPU device

def improveVideo(file):
	video_name = file
	vidcap = cv2.VideoCapture('./video/raingoaway.mp4')
	success, frame = vidcap.read()
	count = 0
	success = True

	height, width, layers = frame.shape
	fourcc = cv2.VideoWriter_fourcc(*'MJPG')
	video = cv2.VideoWriter(video_name, fourcc, 10.0, (width,height))

	saver = None
	config = None
	sess = None
	output = None

	while success:
		print(i, " Frame")
		ori = frame
		if np.max(ori) > 1:
		   ori = ori/255.0

		input_tensor = np.expand_dims(ori[:,:,:], axis = 0)


		image = tf.placeholder(tf.float32, shape=(1, input_tensor.shape[1], input_tensor.shape[2], 3))
		output = Network.inference(image, is_training = False)	
		saver = tf.train.Saver()
		config = tf.ConfigProto()
		config.gpu_options.per_process_gpu_memory_fraction = 0.8
		config.gpu_options.allow_growth = True

		sess = tf.Session(config=config)
		if tf.train.get_checkpoint_state('./model/'):  
			ckpt = tf.train.latest_checkpoint('./model/')
			saver.restore(sess, ckpt)
			print ("Loading model")
		else:
			saver.restore(sess, "./model/test-model/model") # try a pre-trained model 
			print ("Loading pre-trained model")

		final_output = sess.run(output, feed_dict={image: input_tensor})
		final_output[np.where(final_output < 0. )] = 0.
		final_output[np.where(final_output > 1. )] = 1.
		derained = final_output[0,:,:,:]
		video.write(np.uint8(derained))
		img.imsave('./outputframes/' +  file, derained)
		success, frame = vidcap.read()

	cv2.destroyAllWindows()
	video.release()
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 22:52:51 2017

@author: Tu Bui tb00083@surrey.ac.uk
"""

import sys,os
from PIL import Image
import StringIO
import math
import subprocess
import caffe
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.io import savemat
import time
from class_utils import sketch_process

MODEL_WEIGHTS_PATH = 'triplet1_InceptionV1_InceptionV1_halfshare_inception4e_ld256_triplet_sketchy_iter_31200.caffemodel'
MODEL_SPEC_PATH = 'model/deploy_sketch_net1_InceptionV1_InceptionV1_halfshare_inception4e_ld256_triplet_sketchy.prototxt'



GPU_DEV = 0
LAYER_DIMS=256
mean_pixel = np.array([104, 117, 123],dtype=np.float32)[:,None,None]
skt_preprocess = sketch_process(shape = (224,224))

def get_net(caffemodel, deploy_file, use_gpu=True):
    """
    Returns an instance of caffe.Net

    Arguments:
    caffemodel -- path to a .caffemodel file
    deploy_file -- path to a .prototxt file

    Keyword arguments:
    use_gpu -- if True, use the GPU for inference
    """
    if use_gpu:
        caffe.set_mode_gpu()
        caffe.set_device(GPU_DEV)

    # load a new model
    return caffe.Net(deploy_file, caffe.TEST, weights = caffemodel)


def extractitem(net, mean_pixel, fname):
  
    DATA_LAYER = net.inputs[0]
    net.blobs[DATA_LAYER].reshape(1,3,224,224) 
    try:
       skt_preprocess.read_query(fname)
       input_image = skt_preprocess.process()
       sys.stdout.flush()
       net.blobs[DATA_LAYER].data[...] = input_image
       sys.stdout.flush()
       _ = net.forward()
       sys.stdout.flush()
       blobname=net.blobs.keys()[-1] #should be feat_p for image and feat_a for sketch
       prediction = net.blobs[blobname].data.squeeze()
    
    
    except Exception as e:
       s=str(e)
       print('WARNING: Image was unusable %s' % fname)
       print(s)
       prediction = np.zeros(LAYER_DIMS).astype(np.float32)
    
    return prediction


if __name__ == "__main__":
    net = get_net(MODEL_WEIGHTS_PATH, MODEL_SPEC_PATH)
    sample_img = 'samples/horse.png'
    feat = extractitem(net, mean_pixel, sample_img)
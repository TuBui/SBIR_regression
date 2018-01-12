# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 15:37:50 2016
all class utilities
@author: tb00083
"""

from bwmorph import bwmorph_thin
from scipy import misc
import numpy as np
import StringIO
from PIL import Image
import tempfile
import os

class sketch_process(object):
  """Pre-process the query before putting it to caffe"""
  def __init__(self, shape = (224,224), scale = 1.0, rot = 0,togray = True,max_dim=200):
    self.mean = np.array([104, 117, 123],dtype=np.float32)[:,None,None]   #mean to substract
    self.shape = np.array(shape)        #output shape
    self.scale = scale                  #scale apply to input image
    self.rot  = rot                     #rotation
    self.gray = togray                  #convert to gray
    self.max_dim = max_dim              #see method pre_process()
    
    self.mean_crop = self.mean
    
  def read_query(self,query_file):
    """read image and convert to numpy"""
    self.im = misc.imread(query_file, flatten = self.gray)

  def read_query_var(self,query_var):
    """read image from variable and convert to numpy"""
    with tempfile.NamedTemporaryFile() as temp:
       temp.write(query_var)
       temp.flush()
       self.im = misc.imread(temp.name, flatten = self.gray)
       temp.close()
    
  def process(self):
    """skeletonise, mean subtract, scale, crop"""
    """Note: we don't perform rotation here"""
    #crop to edges
    img = self.im < 50
    nz = np.nonzero(img)
    if nz[0].size:
      ymin = max(0,nz[0].min() - 1)
      ymax = min(img.shape[0],nz[0].max()+1)
      xmin = max(0,nz[1].min()-1)
      xmax = min(img.shape[1],nz[1].max()+1)
      img = img[ymin:ymax,xmin:xmax].astype(np.float32)
    else:
      print('Opps! Blank query after pre-process. Make sure u use black colour to draw.')
    #resize to max_dim
    zf = float(self.max_dim)/max(img.shape)
    img = misc.imresize(img,zf)  #this automatically convert to [0,255] range
    #misc.imsave('skt1.png',img)
    
    #skeletonise sketch
    img = img > 50
    img = bwmorph_thin(img)
    img = np.float32(255*(1-img))
    
    #padding
    p = (self.shape - np.array(img.shape))/2
    img = np.pad(img,((p[0],self.shape[0]-p[0]-img.shape[0]),(p[1],self.shape[1]-p[1]-img.shape[1])),
                 'constant',constant_values = 255)
    img = img[None,...]
    img = np.repeat(img,3, axis=0)
    #mean substraction & scale
    img -= self.mean_crop
    img *= self.scale
    
    return img[None,...]   #output shape 1x1xHxW

class image_process(object):
  """Pre-process the query before putting it to caffe"""
  def __init__(self, shape = (224,224), scale = 1.0, rot = 0):
    self.mean = np.array([104, 117, 123],dtype=np.float32)[:,None,None]   #mean to substract
    self.shape = np.array(shape)        #output shape
    self.scale = scale                  #scale apply to input image
    self.rot  = rot                     #rotation
  
  def read_query(self,query_file):
    """read image and convert to numpy"""
    self.im = None
    self.im = misc.imread(query_file, mode = 'RGB')

  def read_query_var(self,query_var):
    """read image from variable and convert to numpy"""
    with tempfile.NamedTemporaryFile() as temp:
       temp.write(query_var)
       temp.flush()
       self.im = misc.imread(temp.name, mode = 'RGB')
       temp.close()
    
  def process(self):
    """mean subtract, scale, crop"""
    """Note: we don't perform rotation here"""
    #resize to max_dim
    zf = float(256.)/max(self.im.shape[:2])
    img = misc.imresize(self.im,zf)  #this automatically convert to [0,255] range
    
    #padding to 256x256
    p = (256 - np.array(img.shape[:2]))/2
    img = np.pad(img,((p[0],256-p[0]-img.shape[0]),(p[1],256-p[1]-img.shape[1]),(0,0)),
                 'edge')
    
    #BGR swap and transpose
    img = img[:,:,::-1].transpose(2,0,1).astype(np.float32)
    
    #mean substraction & scale
    img -= self.mean
    img *= self.scale
    
    #crop
    img = img[:,16:240,16:240]
    return img[None,...]  #output shape 1x3xHxW
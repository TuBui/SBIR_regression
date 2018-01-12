# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 12:51:10 2016

@author: tb00083
"""

import numpy as np
import caffe
import h5py as h5
import os
import scipy.io as sio
from datetime import timedelta

def mat2py_imdb(mat_file):
  """
  Convert matlab .mat file version 7.3 to numpy array
  You must know the structure of the mat file before hand.
  Here is for imdb mat file only
  """
  assert os.path.isfile(mat_file), 'Mat file {} not exist.'.format(mat_file)
  f = h5.File(mat_file)
  data = np.array(f['images']['data'])
  labels = np.array(f['images']['labels'])
  #img_mean = np.array(f['images']['data_mean'])
  #matlab store data column wise so we need to transpose it
  return data.transpose().astype(np.float32), labels.astype(np.float32)

def mat2py_mean(mat_file):
  """
  Convert matlab .mat file version 7.3 to numpy array
  You must know the structure of the mat file before hand.
  Here is for mat file containing matrix data_mean only
  """
  assert os.path.isfile(mat_file), 'Mat file {} not exist.'.format(mat_file)
  f = h5.File(mat_file)
  data_mean = np.array(f['data_mean'])
  return data_mean.transpose()

def biproto2py(binary_file):
  """
  read binaryproto (usually mean image) to array
  """
  blob = caffe.proto.caffe_pb2.BlobProto()
  data = open( binary_file , 'rb' ).read()
  blob.ParseFromString(data)
  arr = np.array( caffe.io.blobproto_to_array(blob) )
  #out = np.ascontiguousarray(out.transpose(1,2,0))
  out = np.ascontiguousarray(arr)
  out = out.astype(np.float32)
  return out
  
def py2mat(pydict,out_mat):
  """
  save python object (must be a dictionary) to .mat file
  """
  sio.savemat(out_mat,pydict)

def read_mean(mean_file):
  """
  return mean value whether it is pixel mean, scalar or image mean
  """
  if mean_file==0:
      img_mean = 0
  elif type(mean_file) is list:
    img_mean = np.array(mean_file, dtype = np.float32)
  elif mean_file[-4:] == '.mat':
    img_mean = mat2py_mean(mean_file)
  elif mean_file[-12:] == '.binaryproto':
    img_mean = biproto2py(mean_file).squeeze()
  else:
    assert 0, 'Invalid format for mean_file {}'.format(mean_file)
    
  return img_mean
  
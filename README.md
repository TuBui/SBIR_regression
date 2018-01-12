# SBIR regression
This repo contains code for the C&G paper "[Sketching out the details: Sketch-based image retrieval using convolutional neural networks with multi-stage regression](https://doi.org/10.1016/j.cag.2017.12.006)" 

# Dependencies
You will need to compile [Caffe](https://github.com/BVLC/caffe) with customized L2 normalize layer. Check [caffe_utils/README.md](caffe_utils/README.md) for instructions.

Alternatively, you can use standard Caffe, just remove the normalize layer in model/*.prototxt, then normalise the output manually using e.g. numpy.

# Pretrained model
Pretrained model (and dataset) can be downloaded [here (to be updated soon)](http://www.cvssp.org).

# Feature extraction

Check [getfeat_img.py](getfeat_img.py) and [getfeat_skt.py](getfeat_skt.py) for examples of extracting features from a raw image/sketch.

# Reference
```
@article{bui2018sketching,
  title={Sketching out the Details: Sketch-based Image Retrieval using Convolutional Neural Networks with Multi-stage Regression},
  author={Bui, Tu and Ribeiro, Leonardo and Ponti, Moacir and Collomosse, John},
  journal={Computers \& Graphics},
  year={2018},
  publisher={Elsevier}
}
```
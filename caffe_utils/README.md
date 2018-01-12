# Caffe installation with L2 normalize layer
## Step 1
Download caffe
```
git clone https://github.com/BVLC/caffe.git
```

## Step 2
Copy the L2 normalize source files (credit: [happynear](https://github.com/happynear/caffe-windows.git)) into caffe source code. Specifically, put the header file ([normalize_layer.hpp](normalize_layer.hpp)) to `include/caffe/layers` and the main files ([normalize_layer.cpp](normalize_layer.cpp) and [normalize_layer.cu](normalize_layer.cu)) into `src/caffe/layers`

## Step 3
Install caffe as normal.
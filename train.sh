#!/bin/bash

export PYCAFFE_ROOT=/usr/local/caffe/python
export PYTHONPATH=$PYCAFFE_ROOT:$PYTHONPATH
export PATH=/usr/local/caffe/build/tools:$PYCAFFE_ROOT:$PATH

python train.py $@

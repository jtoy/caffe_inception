import os
import glob
import random
import numpy as np
import cv2
import caffe
from caffe.proto import caffe_pb2
import lmdb
import subprocess
import argparse

#Size of images
IMAGE_WIDTH = 226
IMAGE_HEIGHT = 226

def create_solver(train_path,iteration):
    filename = '/data/model_cache/solver.prototxt'
    with open('solver_template', 'r') as f:
      text = f.read()
    text = text.replace('$train_path', train_path)
    text = text.replace('$iter', iteration)
    with open(filename, 'w') as f:
      f.write(text)
    print "Succesfully created solver.proto file"

def create_train(train_path, eval_path, classes, mean_train,mean_val):
    filename = '/data/model_cache/train.prototxt'
    with open('train_proto_template', 'r') as f:
      text = f.read()
    text = text.replace('$train_lmdb', train_path)
    text = text.replace('$eval_lmdb', eval_path)
    text = text.replace('$mean_train', mean_train)
    text = text.replace('$mean_val', mean_val)
    text = text.replace('$class_count', classes)
    with open(filename, 'w') as f:
      f.write(text)
    print "Succesfully created train_val.proto file"
    return filename

def create_deploy(train_path, classes):
    filename = '/data/model_cache/deploy.prototxt'
    with open('deploy_proto_template', 'r') as f:
      text = f.read()
    text = text.replace('$class_count', classes)
    with open(filename, 'w') as f:
      f.write(text)
    print "Succesfully created deploy.proto file"

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)
    return img


def make_datum(img, label):
    #image is numpy.ndarray format. BGR instead of RGB
    return caffe_pb2.Datum(
        channels=3,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        label=label,
        data=np.rollaxis(img, 2).tostring())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Generate Caffe models from a given image directory")
    parser.add_argument('--image_path', default='/data/input/', help='input image dir e.g:/path/images/classes')
    #parser.add_argument('--snap', default='2500',type = str, help='When to trigger capture snapshot')
    parser.add_argument('--iter', default='5000', type = str, help='Number of iterations')
    args = parser.parse_args()
    train_lmdb = '/tmp/caffe/images/train_lmdb'
    validation_lmdb = '/tmp/caffe/images/validation_lmdb'

    #make train.txt and val.txt
    imagedir = "/data/input"
    file = open('/tmp/train.txt','w')
    x = -2
    for parent,dirnames,filenames in os.walk(imagedir):
        for dirname in filenames:
            if dirname != "input.txt":
                file.write(parent.replace(imagedir+"/","") + "/" + dirname)
                file.write(' '+"%d"%x)
                file.write('\n')
        x=x+1
    file.close()

    data_path = filter(os.path.isdir, [os.path.join(imagedir,f) for f in os.listdir(imagedir)])[0]

    print "Finished creating train.txt"
    print "Creating val.txt"
    code = subprocess.call('cp /tmp/train.txt /tmp/val.txt', shell=True)
    if code != 0:
        raise Exception("Failed to create val.txt")
    print "Creating train_lmdb"
    os.system('rm -rf  ' + train_lmdb)
    os.system('rm -rf  ' + validation_lmdb)
    code = subprocess.call('/usr/local/caffe/build/tools/convert_imageset --resize_height=256 --resize_width=256 --shuffle /data/input/ /tmp/train.txt /tmp/caffe/images/train_lmdb', shell=True)
    if code != 0:
        raise Exception("Failed to create train_lmdb")
    code = subprocess.call('/usr/local/caffe/build/tools/convert_imageset --resize_height=256 --resize_width=256 --shuffle /data/input/ /tmp/val.txt /tmp/caffe/images/validation_lmdb', shell=True)
    if code != 0:
        raise Exception("Failed to create validation_lmdb")
    print '\nComputing mean training images'
    code = subprocess.call('/usr/local/caffe/build/tools/compute_image_mean /tmp/caffe/images/train_lmdb /tmp/caffe/images/train_lmdb/train.binaryproto', shell=True)
    if code != 0:
        raise Exception("Failed to compute training mean")
    print '\nComputing mean validation images'
    code = subprocess.call('/usr/local/caffe/build/tools/compute_image_mean /tmp/caffe/images/validation_lmdb /tmp/caffe/images/validation_lmdb/validate.binaryproto', shell=True)
    if code != 0:
        raise Exception("Failed to compute validatin mean")
    print '\nFinished processing all images'
    path = create_train(train_lmdb, validation_lmdb, str(len(os.walk(data_path).next()[1])), '/tmp/caffe/images/train_lmdb/train.binaryproto','/tmp/caffe/images/validation_lmdb/validate.binaryproto')
    create_solver('/data/model_cache/train.prototxt', args.iter)
    create_deploy('/data/model_cache/deploy.prototxt', str(len(os.walk(data_path).next()[1])))
    os.chdir("/data/model_cache/")
    print ("starting training now")
    code = subprocess.call('/usr/local/caffe/distribute/bin/caffe.bin train -gpu 0 -solver solver.prototxt -weights /tmp/google/bvlc_googlenet.caffemodel', shell=True)
    if code != 0:
        raise Exception("Creating TensorFiles Failed")
    print ("Models saved at:")
    print (os.getcwd)

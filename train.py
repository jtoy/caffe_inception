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

def create_solver(train_path,iteration,snap='5000'):
    filename = '/data/model_cache/solver.prototxt'
    with open('solver_template', 'r') as f:
      text = f.read()
    text = text.replace('$train_path', train_path)
    text = text.replace('$iter', iteration)
    text = text.replace('$snap', snap)
    with open(filename, 'w') as f:
      f.write(text)
    print "Succesfully created solver.proto file"

def create_train(train_path, eval_path, classes,mean_train,mean_val):
    filename = '/data/model_cache/train.prototxt'
    with open('train_proto_template', 'r') as f:
      text = f.read()
    text = text.replace('$train_lmdb', train_path)
    text = text.replace('$eval_lmdb', eval_path)
    text = text.replace('$mean_train', mean_train)
    text = text.replace('$mean_val', mean_val)
    text = text.replace('$class_count', '1000')
    with open(filename, 'w') as f:
      f.write(text)
    print "Succesfully created train_val.proto file"
    return filename

def create_deploy(train_path):
    filename = '/data/model_cache/deploy.prototxt'
    with open('deploy_proto_template', 'r') as f:
      text = f.read()
    text = text.replace('$class_count', '1000')
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
	parser = argparse.ArgumentParser(
        description = "Generate Caffe models from a given image directory")
        parser.add_argument('--image_path', default='/data/input/', help='input image dir e.g:/path/images/classes')
	parser.add_argument('--snap', default='2500',type = str, help='When to trigger capture snapshot')
	parser.add_argument('--iter', default='5000', type = str, help='Number of iterations')
	train_lmdb = '/tmp/caffe/images/train_lmdb'
	validation_lmdb = '/tmp/caffe/images/validation_lmdb'

	train_d = []
	args = parser.parse_args()
	temp_dir = '/tmp/input/'
	if os.path.exists(temp_dir):
			print 'Directory Exist'
        else:
			os.makedirs(temp_dir)
	train_dir = temp_dir
	if os.path.exists('/tmp/caffe/images/train_lmdb'):
            print 'Directory Exist'
        else:
            os.makedirs(train_lmdb)
            os.makedirs(validation_lmdb)
	os.system('cp -r /data/input/* ' + temp_dir)
	trainlist = os.listdir(temp_dir)
	os.chdir(temp_dir)
	train_d_1 = []
	rem_dir = []
	for a in trainlist:
		if os.path.isdir(a):
			rem_dir.append(a)
			train_d_1.append([img for img in glob.glob(temp_dir+a+'/*')])
	train_data_1 = [val for sublist in train_d_1 for val in sublist]
	count = 0
	for a in train_data_1:
		os.system('mkdir -p %s;mv %s %s;'%(count,a,count))
		count +=1
	for a in rem_dir:
		os.system('rm -rf '+a)
		print ("removing %s" % a)
	os.system('rm -rf  ' + train_lmdb)
	os.system('rm -rf  ' + validation_lmdb)
	t_dir = os.listdir(train_dir)
	for a in t_dir:
	   train_d.append([img for img in glob.glob(train_dir+a+'/*')])
	#for b in v_dir:
	   #test_d.append([img for img in glob.glob(val_dir+b+'/*')])
	train_data = [val for sublist in train_d for val in sublist]

	#Shuffle train_data
	random.shuffle(train_data)
	os.chdir('/home/ubuntu/experiment/')
	print 'Creating train_lmdb'
	in_db = lmdb.open(train_lmdb, map_size=int(1e12))
	with in_db.begin(write=True) as in_txn:
	    for in_idx, img_path in enumerate(train_data):
	       try:
		  img = cv2.imread(img_path, cv2.IMREAD_COLOR)
		  img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
		  for a,b in enumerate(t_dir):
			if b in img_path:
				label = a
		  datum = make_datum(img, label)
		  in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
		  print '{:0>5d}'.format(in_idx) + ':' + img_path
	       except:
		  print ("failed")
	in_db.close()

	print '\nCreating validation_lmdb'
	in_db = lmdb.open(validation_lmdb, map_size=int(1e12))
	with in_db.begin(write=True) as in_txn:
	    for in_idx, img_path in enumerate(train_data):
	      try:
		img = cv2.imread(img_path, cv2.IMREAD_COLOR)
		img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
		for a,b in enumerate(t_dir):
			if b in img_path:
				label = a
		datum = make_datum(img, label)
		in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
		print '{:0>5d}'.format(in_idx) + ':' + img_path
	      except:
		print ('valiadtion failed')
	in_db.close()
        print '\nComputing mean training images'
        code = subprocess.call('/usr/local/caffe/build/tools/compute_image_mean /tmp/caffe/images/train_lmdb /tmp/caffe/images/train_lmdb/train.binaryproto', shell=True)
	if code != 0:
           raise Exception("Failed to compute training mean")
        print '\nComputing mean validation images'
        code = subprocess.call('/usr/local/caffe/build/tools/compute_image_mean /tmp/caffe/images/train_lmdb /tmp/caffe/images/validation_lmdb/validate.binaryproto', shell=True)
	if code != 0:
           raise Exception("Failed to compute validatin mean")
	print '\nFinished processing all images'
	path = create_train(train_lmdb, validation_lmdb,str(len(t_dir)),'/tmp/caffe/images/train_lmdb/train.binaryproto',
			'/tmp/caffe/images/validation_lmdb/validate.binaryproto')
	create_solver('/data/model_cache/train.prototxt', args.iter, snap=args.snap)
	create_deploy('/data/model_cache/deploy.prototxt')
	os.chdir("/data/model_cache/")
	print ("starting training now")
	code = subprocess.call('/usr/local/caffe/distribute/bin/caffe.bin train -gpu 0 -solver solver.prototxt -weights /tmp/google/bvlc_googlenet.caffemodel', shell=True)
	if code != 0:
           raise Exception("Creating TensorFiles Failed")
	print ("Models saved at:")
	print (os.getcwd)

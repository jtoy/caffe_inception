# import the necessary packages
from __future__ import print_function
from google.protobuf import text_format
from cStringIO import StringIO
from PIL import Image
import scipy.ndimage as nd
import numpy as np
import caffe
import os

class BatCountry:
	def __init__(self, base_path, deploy_path=None, model_path=None,
		patch_model="/data/input/tmp.prototxt", mean=(104.0, 116.0, 122.0),
		channels=(2, 1, 0)):
		# if the deploy path is None, set the default
		if deploy_path is None:
			deploy_path = base_path + "/deploy.prototxt"

		# if the model path is None, set it to the default GoogleLeNet model
		if model_path is None:
			model_path = base_path + "/bvlc_googlenet.caffemodel"

		# check to see if the model should be patched to compute gradients
		if patch_model:
			model = caffe.io.caffe_pb2.NetParameter()
			text_format.Merge(open(deploy_path).read(), model)
			model.force_backward = True
			f = open(patch_model, "w")
			f.write(str(model))
			f.close()

		# load the network and store the patched model path
		self.net = caffe.Classifier(patch_model, model_path, mean=np.float32(mean),
			channel_swap=channels)
		self.patch_model = patch_model

	def dream(self, image, iter_n=10, octave_n=4, octave_scale=1.4,
		end="inception_4c/output", clip=True, step_fn=None, objective_fn=None,
		preprocess_fn=None, deprocess_fn=None, verbose=True, visualize=False,
		**step_params):
		# if a step function has not been supplied, initialize it as the
		# standard gradient ascent step
		if step_fn is None:
			step_fn = BatCountry.gradient_ascent_step

		# if the objective function has not been supplied, initialize it
		# as the L2 objective
		if objective_fn is None:
			objective_fn = BatCountry.L2_objective

		# if the preprocess function has not been supplied, initialize it
		if preprocess_fn is None:
			preprocess_fn = BatCountry.preprocess

		# if the deprocess function has not been supplied, initialize it
		if deprocess_fn is None:
			deprocess_fn = BatCountry.deprocess

		# initialize the visualization list
		visualizations = []

		# prepare base images for all octaves
		octaves = [preprocess_fn(self.net, image)]

		for i in xrange(octave_n - 1):
			octaves.append(nd.zoom(octaves[-1], (1, 1.0 / octave_scale,
				1.0 / octave_scale), order=1))

		# allocate image for network-produced details
		detail = np.zeros_like(octaves[-1])
		src = self.net.blobs["data"]

		for octave, octave_base in enumerate(octaves[::-1]):
			h, w = octave_base.shape[-2:]

			if octave > 0:
				# upscale details from the previous octave
				h1, w1 = detail.shape[-2:]
				detail = nd.zoom(detail, (1, 1.0 * h/ h1, 1.0 * w / w1), order=1)

 			# resize the network's input image size
			src.reshape(1, 3, h, w)
			src.data[0] = octave_base + detail

			for i in xrange(iter_n):
				step_fn(self.net, end=end, clip=clip, objective_fn=objective_fn,
					**step_params)

				# visualization
				vis = deprocess_fn(self.net, src.data[0])

				# adjust image contrast if clipping is disabled
				if not clip:
					vis = vis * (255.0 / np.percentile(vis, 99.98))

				if verbose:
					print("octave={}, iter={}, layer={}, image_dim={}".format(octave,
						i, end, vis.shape))

				# check to see if the visualization list should be
				# updated
				if visualize:
					k = "octave_{}-iter_{}-layer_{}".format(octave, i,
						end.replace("/", "_"))
					visualizations.append((k, vis))

			# extract details produced on the current octave
			detail = src.data[0] - octave_base

		# grab the resulting image
		r = deprocess_fn(self.net, src.data[0])

		# check to see if the visualizations should be included
		if visualize:
			r = (r, visualizations)

		return r

	def classdream(self, base_img, octaves, random_crop=False, visualize=False, focus=None, clip=True, **step_params):
		image = BatCountry.preprocess(self.net, base_img)

		w = self.net.blobs['data'].width
		h = self.net.blobs['data'].height

		src = self.net.blobs['data']
		src.reshape(1,3,h,w)

		for e,o in enumerate(octaves):
			if 'scale' in o:
				image = nd.zoom(image, (1,o['scale'],o['scale']))
			_,imw,imh = image.shape

			layer = o['layer']

			for i in xrange(o['iter_n']):
				if imw > w:
					if random_crop:
						mid_x = (imw-w)/2.
						width_x = imw-w
						ox = np.random.normal(mid_x, width_x*0.3, 1)
						ox = int(np.clip(ox,0,imw-w))
						mid_y = (imh-h)/2.
						width_y = imh - h
						oy = np.random.normal(mid_y, width_y*0.3, 1)
						oy = int(np.clip(oy,0,imh-h))

						src.data[0] = image[:,ox:ox+w,oy:oy+h]
					else:
						ox = (imw-w)/2.
						oy = (imh-h)/2.
						src.data[0] = image[:,ox:ox+w,oy:oy+h]
				else:
					ox = 0
					oy = 0
					src.data[0] = image.copy()

				sigma = o['start_sigma'] + ((o['end_sigma'] - o['start_sigma']) * i) / o['iter_n']
				step_size = o['start_step_size'] + ((o['end_step_size'] - o['start_step_size']) * i) / o['iter_n']

				BatCountry.class_step(self.net, end=layer, clip=clip, focus=focus, sigma=sigma, step_size=step_size)

				if visualize:
					print("Doing nothing...")

				if i % 10 == 0:
					print("finished step {} in octave {}".format(i, e))

				image[:,ox:ox+w,oy:oy+h] = src.data[0]
		return BatCountry.deprocess(self.net, image)




	@staticmethod
	def gradient_ascent_step(net, step_size=1.5, end="inception_4c/output",
		jitter=32, clip=True, objective_fn=None, **objective_params):
		# if the objective function is None, initialize it as
		# the standard L2 objective
		if objective_fn is None:
			objective_fn = BatCountry.L2_objective

		# input image is stored in Net's 'data' blob
		src = net.blobs["data"]
		dst = net.blobs[end]

		# apply jitter shift
		ox, oy = np.random.randint(-jitter, jitter + 1, 2)
		src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2)

		net.forward(end=end)
		objective_fn(dst, **objective_params)
		net.backward(start=end)
		g = src.diff[0]

		# apply normalized ascent step to the input image
		src.data[:] += step_size / np.abs(g).mean() * g

 		# unshift image
		src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2)

		# unshift image
		if clip:
			bias = net.transformer.mean["data"]
			src.data[:] = np.clip(src.data, -bias, 255 - bias)

	def layers(self):
		# return the layers of the network
		return self.net._layer_names

	def cleanup(self):
		# remove the patched model from disk
		os.remove(self.patch_model)

	def prepare_guide(self, image, end="inception_4c/output", maxW=224, maxH=224,
		preprocess_fn=None):
		# if the preprocess function has not been supplied, initialize it
		if preprocess_fn is None:
			preprocess_fn = BatCountry.preprocess

		# grab dimensions of input image
		(w, h) = image.size

		# GoogLeNet was trained on images with maximum width and heights
		# of 224 pixels -- if either dimension is larger than 224 pixels,
		# then we'll need to do some resizing
		nW, nH = 244, 244
		if w != 244 or h != 244:
			image = np.float32(image.resize((nW, nH), Image.BILINEAR))

		(src, dst) = (self.net.blobs["data"], self.net.blobs[end])
		src.reshape(1, 3, nH, nW)
		src.data[0] = preprocess_fn(self.net, image)
		self.net.forward(end=end)
		guide_features = dst.data[0].copy()

		return guide_features

	@staticmethod
	def L2_objective(dst):
		dst.diff[:] = dst.data

	@staticmethod
	def guided_objective(dst, objective_features):
		x = dst.data[0].copy()
		y = objective_features
		ch = x.shape[0]
		x = x.reshape(ch,-1)
		y = y.reshape(ch,-1)

		# compute the matrix of dot-products with guide features
		A = x.T.dot(y)

		# select ones that match best
		dst.diff[0].reshape(ch, -1)[:] = y[:,A.argmax(1)]

	@staticmethod
	def preprocess(net, img):
		return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean["data"]

	@staticmethod
	def deprocess(net, img):
		return np.dstack((img + net.transformer.mean["data"])[::-1])


	# Added for class visualization
	# PLEASE MODIFY ME

	@staticmethod
	def blur(img, sigma):
		if sigma > 0:
			img[0] = nd.filters.gaussian_filter(img[0], sigma, order=0)
			img[1] = nd.filters.gaussian_filter(img[1], sigma, order=0)
			img[2] = nd.filters.gaussian_filter(img[2], sigma, order=0)
		return img

	@staticmethod
	def class_step(net, step_size=1, end='inception_4c/output', clip=True, focus=None, sigma=None):
		src = net.blobs['data']
		dst = net.blobs[end]

		net.forward(end=end)

		one_hot = np.zeros_like(dst.data)
		one_hot.flat[focus] = 1.
		dst.diff[:] = one_hot

		net.backward(start=end)
		g = src.diff[0]

		src.data[:] += step_size/np.abs(g).mean() * g

		if clip:
			bias = net.transformer.mean['data']
			src.data[:] = np.clip(src.data, -bias, 255-bias)

		#src.data[0] = BatCountry.blur(src.data[0], sigma)

		dst.diff.fill(0.)

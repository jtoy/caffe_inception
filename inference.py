# python inference.py --base-model $CAFFE_ROOT/models/bvlc_googlenet \
#	--image initial_images/clouds.jpg \
#	--layer 'inception_4c/output'
#	--output examples/output/seeded/clouds_and_starry_night.jpg

# import the necessary packages
from batcountry import BatCountry
from PIL import Image
import numpy as np
import argparse
import os
import glob
import time
import shutil
# Set GPU mode
import caffe
caffe.set_mode_gpu()
caffe.set_device(0)
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--base-model", default='/data/model_cache/', help="base model path")
ap.add_argument("--layer", nargs='+', default="['inception_4c/output','inception_3a/pool_proj','inceptoin_3a/output','inception_4b/3x3_reduce']",
	help="layer of CNN to use")
ap.add_argument("--image", required=True, help="path to base image")
ap.add_argument("--output",  help="path to output image")
ap.add_argument("--iteration_count",default=20, type=int, help="iterations for dreaming")
ap.add_argument("--guide", help="Image to guide deep dream")
ap.add_argument("--mixlayer", help="Layer to mix")
ap.add_argument("--classtoshow", help="Specific image to show")
args = ap.parse_args()
if args.output == None:
  args.output = "/data/output/"+ str(int(time.time())) + ".jpg"

# Rename model file so inference script can pick it up
search_dir = args.base_model
files = filter(os.path.isfile, glob.glob(search_dir + "*.caffemodel"))
files.sort(key=lambda x: os.path.getmtime(x))
model_file = files[-1]
shutil.move(model_file, args.base_model+'/bvlc_googlenet.caffemodel')
# we can't stop here...
if args.classtoshow:
	bc = BatCountry(args.base_model, deploy_path='/data/model_cache/deploy_class.prototxt')
else:
	bc = BatCountry(args.base_model)


for layer in args.layer:
	if args.guide:
		features = bc.prepare_guide(Image.open(args.guide), end=layer)
		image = bc.dream(np.float32(Image.open(args.image)), end=layer,
    iter_n=args.iteration_count, objective_fn=BatCountry.guided_objective,
    objective_features=features,)

	elif args.mixlayer:
		mixed_features = bc.prepare_guide(Image.open(args.image), end=args.mixlayer)
		image = bc.dream(np.float32(Image.open(args.image)), end=layer, iter_n=args.iteration_count, objective_fn=BatCountry.guided_objective, objective_features=mixed_features, )

	elif args.classtoshow:
		octaves = [
			{
				'layer':'loss3/classifier_zzzz',
				'iter_n':190,
				'start_sigma':2.5,
				'end_sigma':0.78,
				'start_step_size':11.,
				'end_step_size':11.
			},
			{
				'layer':'loss3/classifier_zzzz',
				'iter_n':150,
				'start_sigma':0.78*1.2,
				'end_sigma':0.78,
				'start_step_size':6.,
				'end_step_size':6.
			},
			{
				'layer':'loss2/classifier_zzzz',
				'iter_n':150,
				'start_sigma':0.78*1.2,
				'end_sigma':0.44,
				'start_step_size':6.,
				'end_step_size':3.
			},
			{
				'layer':'loss1/classifier_zzzz',
				'iter_n':150,
				'start_sigma':0.44,
				'end_sigma':0.304,
				'start_step_size':3.,
				'end_step_size':3.
			}
		]

		image = bc.classdream(np.float32(Image.open(args.image)), octaves, focus=int(args.classtoshow), random_crop=True, visualize=False)

	else:
		image = bc.dream(np.float32(Image.open(args.image)), end=layer,
		iter_n=args.iteration_count)
bc.cleanup()

# write the output image to file
result = Image.fromarray(np.uint8(image))
result.save(args.output)
print ("result saved at:")
print (args.output)

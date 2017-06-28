#Inception and Deep-dream with Caffe

My experiments with trying to get deep dream to hallucinate and merge different classes and layers together.
This repo can be used to train cutom models just by providing a set of images.

# training 

python train.py /ROOT/dataset/classes/


The script assumes that images are broken down into sperate categories and placed in their respective folder. We use
bvlc_googlenet network architecture for training.

Once you the trained models, you can use Inference.py to generate deep-dream like images.

python inference.py --base-model $CAFFE_ROOT/models/bvlc_googlenet \
	--image initial_images/clouds.jpg \
	--layer 'inception_4c/output'
	--output examples/output/seeded/clouds_and_starry_night.jpg

If you want to dig deep and are intrested in how each layer in the network perform, you can use:

python inference_debug.py --base-model $CAFFE_ROOT/models/bvlc_googlenet \
	--image initial_images/clouds.jpg \
	--output examples/output/
	
#Input data format

Train.py script assumes that your input data(images) is alway inside a directory and image formate is JPEG/JPG. (Due to a somatic feature which strips the image directory we have to add the images as per /tmp/(class-dir)/images )

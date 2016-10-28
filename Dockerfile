FROM ubuntu:14.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python-dev \
        python-numpy \
        python-pip \
	curl \
	python-opencv \
        python-scipy

ENV CUDA_RUN https://developer.nvidia.com/compute/cuda/8.0/prod/local_installers/cuda_8.0.44_linux-run

RUN cd /opt && \
  wget $CUDA_RUN && \
  chmod +x cuda_8.0.44_linux-run && \
  mkdir nvidia_installers && \
  ./cuda_8.0.44_linux-run -extract=`pwd`/nvidia_installers && \
  cd nvidia_installers && \
  ./cuda-linux64-rel-8.0.44-21122537.run -noprompt && \
   ./NVIDIA-Linux-x86_64-367.48.run -s -N --no-kernel-module

ENV CAFFE_ROOT=/usr/local/caffe
WORKDIR $CAFFE_ROOT

# FIXME: clone a specific git tag and use ARG instead of ENV once DockerHub supports this.
ENV CLONE_TAG=master

RUN git clone -b ${CLONE_TAG} --depth 1 https://github.com/umerebryx/caffe.git . && \
    cp Makefile.config.example Makefile.config && \	
    for req in $(cat python/requirements.txt) pydot; do pip install $req; done && \
    mkdir build && cd build && \
    cmake .. && \
    cd .. && \	
    make -j"$(nproc)" && \
    make test && \
    make runtest && \
    make distribute && \
    make pycaffe


ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

ENV CUDNN_VERSION 5
LABEL com.nvidia.cudnn.version="5"

RUN CUDNN_DOWNLOAD_SUM=a87cb2df2e5e7cc0a05e266734e679ee1a2fadad6f06af82a76ed81a23b102c8 && \
    curl -fsSL http://developer.download.nvidia.com/compute/redist/cudnn/v5.1/cudnn-8.0-linux-x64-v5.1.tgz -O && \
    echo "$CUDNN_DOWNLOAD_SUM  cudnn-8.0-linux-x64-v5.1.tgz" | sha256sum -c --strict - && \
    tar -xzf cudnn-8.0-linux-x64-v5.1.tgz -C /usr/local --wildcards 'cuda/lib64/libcudnn.so.*' && \
    rm cudnn-8.0-linux-x64-v5.1.tgz && \
    ldconfig


RUN export LIBRARY_PATH=/usr/local/cuda-8.0/lib64
RUN export PATH=$PATH:/usr/local/cuda-8.0/bin

RUN pip install lmdb

RUN mkdir /tmp/google/ &&  \
    cd /tmp/google/ && wget http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel

WORKDIR /home/ubuntu/experiment/


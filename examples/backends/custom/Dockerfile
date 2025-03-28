FROM nvcr.io/nvidia/pytorch:23.12-py3 AS pytorch-base

FROM radiusaiinc/trtserver:23.12-sa-sep9-12.01-false

RUN apt-get update && apt-get install -y \
	sudo \
	rapidjson-dev \ 
	libgeos-dev \
	libeigen3-dev \
	libopenblas-dev \
	liblapack-dev \
	libarmadillo-dev \
	nlohmann-json3-dev \
	&& apt-get clean

#install xtensor
RUN sudo apt-get install libxtensor-dev -y


#Install libtorch
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/* 

# RUN wget -O /tmp/libtorch.zip "https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.2.0%2Bcu121.zip" \
# 	&& unzip /tmp/libtorch.zip -d /opt \
# 	&& rm /tmp/libtorch.zip 


RUN mkdir /tmp/cuda-download
RUN wget -O /tmp/cuda-download/cuda-keyring_1.1-1_all.deb "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
RUN sudo dpkg -i /tmp/cuda-download/cuda-keyring_1.1-1_all.deb \
    && sudo apt-get update \
    && sudo apt-get -y install cuda-toolkit-12-2

# extra libraries for reading conf yaml file
RUN sudo apt install libboost-all-dev libyaml-cpp-dev -y

# remove torch
RUN pip uninstall torch torchvision torchaudio -y

RUN mkdir -p /usr/local/lib/python3.10/dist-packages && \
    mkdir -p /usr/local/lib/python3.10/dist-packages/torch/lib

# get torch from nvidia container
COPY --from=pytorch-base /usr/local/lib/python3.10/dist-packages/torch /usr/local/lib/python3.10/dist-packages/torch
COPY --from=pytorch-base /usr/local/lib/python3.10/dist-packages/torch_tensorrt /usr/local/lib/python3.10/dist-packages/torch_tensorrt
COPY --from=pytorch-base /usr/local/lib/python3.10/dist-packages/torchdata /usr/local/lib/python3.10/dist-packages/torchdata
COPY --from=pytorch-base /usr/local/lib/python3.10/dist-packages/torchgen /usr/local/lib/python3.10/dist-packages/torchgen
COPY --from=pytorch-base /usr/local/lib/python3.10/dist-packages/torchtext /usr/local/lib/python3.10/dist-packages/torchtext
COPY --from=pytorch-base /usr/local/lib/python3.10/dist-packages/torchvision /usr/local/lib/python3.10/dist-packages/torchvision

ENV LD_LIBRARY_PATH /usr/local/lib/python3.10/dist-packages/torch/lib:/opt/tritonserver/lib:$LD_LIBRARY_PATH
# build torch from source with cxx11 abi enabled
# Install mkl-static and mkl-include
RUN pip install mkl-static mkl-include

# fix libmkl errors by using the three shared libraries from pytorch backend in tritonserver
# RUN mkdir -p /usr/local/lib/python3.10/dist-packages/torch/lib
RUN cp --preserve=links /opt/tritonserver/backends/pytorch/libmkl_intel_lp64.so.1 /usr/local/lib/python3.10/dist-packages/torch/lib
RUN cp --preserve=links /opt/tritonserver/backends/pytorch/libmkl_gnu_thread.so.1 /usr/local/lib/python3.10/dist-packages/torch/lib
RUN cp --preserve=links /opt/tritonserver/backends/pytorch/libmkl_core.so.1 /usr/local/lib/python3.10/dist-packages/torch/lib

# Clone the PyTorch repository
# RUN git clone --depth 1 --branch v2.1.2 https://github.com/pytorch/pytorch.git /pytorch

# Set working directory to PyTorch source
# WORKDIR /pytorch

# Configure git
# RUN git config --global --add safe.directory "*"

# Update submodules
# RUN git submodule sync && \
#     git submodule update --init --recursive

# Build and install PyTorch (separate layer)
# RUN export _GLIBCXX_USE_CXX11_ABI=1 && python3 /pytorch/setup.py develop

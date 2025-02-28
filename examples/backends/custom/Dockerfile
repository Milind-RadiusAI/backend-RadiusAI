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

RUN wget -O /tmp/libtorch.zip "https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.2.0%2Bcu121.zip" \
	&& unzip /tmp/libtorch.zip -d /opt \
	&& rm /tmp/libtorch.zip 


RUN mkdir /tmp/cuda-download
RUN wget -O /tmp/cuda-download/cuda-keyring_1.1-1_all.deb "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
RUN sudo dpkg -i /tmp/cuda-download/cuda-keyring_1.1-1_all.deb \
    && sudo apt-get update \
    && sudo apt-get -y install cuda-toolkit-12-4

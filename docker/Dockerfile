# ==================================================================
# module list
# ------------------------------------------------------------------
# python        3.7    (apt)
# pytorch       1.4    (pip)
# ==================================================================
FROM nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04

RUN rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update

# ==================================================================
# tools
# ------------------------------------------------------------------
RUN apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        wget \
        git \
        vim \
	    nano \
        libx11-dev \
        fish \
        libsparsehash-dev \
        software-properties-common \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgl1-mesa-glx

# ==================================================================
# python
# ------------------------------------------------------------------
WORKDIR /M3D-RPN
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
        python3.7 \
        python3.7-dev \
        python3-pip \
        python3-setuptools
RUN ln -s /usr/bin/python3.7 /usr/local/bin/python3
RUN ln -s /usr/bin/python3.7 /usr/local/bin/python
COPY requirements.txt requirements.txt
RUN python -m pip --no-cache-dir install --upgrade -r requirements.txt

# ==================================================================
# config & cleanup
# ------------------------------------------------------------------
WORKDIR /root
RUN ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

# Install cmake v3.13.2
RUN apt-get purge -y cmake && \
    mkdir /root/temp && \
    cd /root/temp && \
    wget https://cmake.org/files/v3.13/cmake-3.13.2.tar.gz && \
    tar -xzvf cmake-3.13.2.tar.gz && \
    cd cmake-3.13.2 && \
    bash ./bootstrap && \
    make && \
    make install && \
    cmake --version && \
    rm -rf /root/temp


# Install Boost geometry
RUN wget https://dl.bintray.com/boostorg/release/1.68.0/source/boost_1_68_0.tar.gz
RUN tar xzvf boost_1_68_0.tar.gz
RUN cp -r ./boost_1_68_0/boost /usr/include
RUN rm -rf ./boost_1_68_0
RUN rm -rf ./boost_1_68_0.tar.gz

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# nvidia runtime
COPY --from=nvidia/opengl:1.0-glvnd-runtime-ubuntu16.04 \
 /usr/lib/x86_64-linux-gnu \
 /usr/lib/x86_64-linux-gnu

COPY --from=nvidia/opengl:1.1-glvnd-runtime-ubuntu16.04 \
 /usr/local/share/glvnd/egl_vendor.d/10_nvidia.json \
 /usr/local/share/glvnd/egl_vendor.d/10_nvidia.json

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# ==================================================================
# M3D-RPN Framework
# ------------------------------------------------------------------
WORKDIR /M3D-RPN
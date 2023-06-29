FROM ubuntu:20.04
LABEL maintainer="Ignacio Vizzo <ivizzo@uni-bonn.de>"

# setup environment
ENV TERM xterm
ENV DEBIAN_FRONTEND=noninteractive

# install essentials
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-numpy \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install OpenVDB Dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    libblosc-dev \
    libboost-all-dev \
    libilmbase-dev \
    libsnappy1v5 \
    libtbb-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install OpenVDB from source, with Python bindings
RUN git clone https://github.com/nachovizzo/openvdb.git -b nacho/fix_background_inactive \
  && cd openvdb \
  && mkdir build && cd build \
  && cmake -DOPENVDB_BUILD_PYTHON_MODULE=ON -DUSE_NUMPY=ON .. \
  && make -j$(nproc) all install \
  && rm -rf /openvdb

# Install Open3D dependencies: https://github.com/intel-isl/Open3D/issues/3388
RUN apt-get update && apt-get install --no-install-recommends -y \
    libgl1 \
    libgomp1 \
    libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade \
    black \
    np \
    open3d


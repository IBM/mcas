#!/bin/bash
#
# Packages for Ubuntu 18.04 LTS
apt-get update
apt-get install -y --no-install-recommends build-essential cmake

./install-tzdata-noninteractive.sh

# install Rust compiler and runtime
./install-rust.sh

apt-get install -y --no-install-recommends \
        autoconf automake ca-certificates cmake gcc g++ git make python3 python3-numpy libtool-bin pkg-config \
        libnuma-dev \
        libboost-system-dev libboost-iostreams-dev libboost-program-options-dev \
        libboost-filesystem-dev libboost-date-time-dev \
        libaio-dev libssl-dev libibverbs-dev librdmacm-dev \
        libudev-dev \
        libboost-python-dev libkmod-dev libjson-c-dev libbz2-dev \
        libelf-dev \
        libgoogle-perftools-dev libcurl4-openssl-dev \
        linux-headers-generic \
        uuid-dev golang gnutls-dev libgnutls30 \
        lcov libzmq5-dev libczmq-dev \
        python3-setuptools python3-pip \
        libc6-dev libstdc++-6-dev

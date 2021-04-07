#!/bin/bash

# required subscriptions
#
# subscription-manager repos --enable codeready-builder-for-rhel-8-x86_64-rpms

# enable epel repo
#
dnf -y install dnf-plugins-core
dnf -y install https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm

dnf -y update

# libuuid which gmp-devel mpfr-devel CUnit CUnit-devel
dnf -y --nodocs --setopt=install_weak_deps=False install automake \
    cmake git make gcc-c++ make libtool \
    openssl-devel python3 python3-devel kmod-libs pkg-config bash-completion \
    kmod-devel libudev-devel json-c-devel uuid-devel \
    boost boost-devel boost-python3 boost-python3-devel \
    elfutils-libelf-devel \
    gperftools-devel \
    libaio-devel \
    libcurl-devel \
    librdmacm-devel librdmacm \
    libuuid-devel \
    numactl-devel \
    python3-devel python3-pip \
    rapidjson-devel \
    openssl-devel golang gnutls gnutls-devel \
    zeromq-devel czmq-devel czmq 

dnf clean packages
df -h /
dnf -y --nodocs --setopt=install_weak_deps=False install kernel-devel

# install Rust compiler and runtime
./install-rust.sh


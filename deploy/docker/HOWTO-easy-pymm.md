## Setup Docker Environment

- Follow [Ubuntu install instructions](https://docs.docker.com/install/linux/docker-ce/ubuntu/) 
or [Fedora install instructions](https://docs.docker.com/engine/install/fedora/) to install docker.

- Add your user id into the docker group for non-root execution.

# Getting started with PyMM

This document explains how to get started with PyMM using docker containers.  PyMM is a local-only
distribution of MCAS memory management that provides Python3 language integration. The advantage
of using the container version is that you don't need to build the MCAS distribution.  You
can download the image and simply run!  Furthermore, the container is not bloated with the
complete MCAS distribution, which is not needed if you are only using PyMM.

First, you need to ensure you have a pmem mounted partition (e.g., /mnt/pmem0).  Make
sure it has suitable permissions (see https://www.redhat.com/sysadmin/supplemental-groups-podman-containers).

```bash
sudo chcon -t container_file_t /mnt/pmem0
```

## Starting container from existing Dockerhub image

There are Ubuntu 18 and Fedora Core 32 based images available on docker hub.  The '-v' option
is needed to pass through the persistent memory mount to the container.

You may need the option --annotation run.oci.keep_original_groups=1

```bash
$ docker run -it -v /mnt/pmem0:/mnt/pmem0 dwaddington/pymm:latest

mcasuser@ce09291b91a0:~$ python3 -i sample.py 
Created shelf OK.
Created array 'x' on shelf OK. Data at ('0x10000300018', 800000000)
[-0.89107643 -0.96512335  0.59666344 ...  0.63990706 -0.21330142
 -0.67030775]
Sorted array 'x' on shelf OK. Data at ('0x10000300018', 800000000)
[-0.99999999 -0.99999998 -0.99999998 ...  0.99999997  0.99999998
  0.99999999]
Use s and s.x to access shelf...
>>> 
>>> s.used
74
```

To force a "re-pull" from docker io, you can delete your images:

```bash
docker rmi $(docker images -a -q) -f
```

Or selectively delete as follows:
```bash
docker images -a | grep "pattern" | awk '{print $3}' | xargs docker rmi
```

## Building your own container (you should not need to do this)

Use the provided Dockerfile (e.g. Dockerfile.pymm-ubuntu-18) to build your own container.

On Docker Hub (https://hub.docker.com/) create an account and a registry.

- Build image (e.g.):
```bash
docker build -t <your-docker-username>/pymm:ubuntu18 -f Dockerfile.pymm-ubuntu-18 .
```

- (Optional) Push image to Docker Hub, e.g.:

```bash
docker login docker.io
docker push <your-docker-username>/pymm:ubuntu18
```
282101ba-daca-4def-9080-2bf6b4c83644

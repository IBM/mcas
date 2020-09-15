# MCAS

Memory Centric Active Storage (MCAS) is a high-performance key-value store
explicitly designed for persistent memory.  Beyond a conventional
key-value store, MCAS provides the ability to provide custom in-store
compute, ultimately reducing data movement across the network and
improving performance.  Beyond existing volatile memory stores, data
is durable across node failures or reset events.

The key attributes of the solution are:

1. High-performance key-value store based on hash-table primary index and optional secondary indices.
2. Support for Intel Optane DC Persistent Memory or conventional DRAM (without persistence).
3. Support for both RDMA and traditional TCP/IP network transports.
4. Zero-copy transfer capable with RDMA and GPU-Direct capable.
5. Support for C++ based in-store compute plugins.
6. Support for C++ and Python clients. 

## Documentation

* [Web Pages](https://ibm.github.io/mcas/)
* [Quick Start](./info/quick_start.md)
* [Overview](./info/MCAS_overview.md)
* [More documentation](./info/index.md)
* [Trello - IBM only](https://trello.com/b/vXomq2Xg/mcas-core)


## Run dependencies for your OS 

``` bash
cd deps
./install-<Your-OS-Version>.sh
cd ../
``` 

## How to Build

Check out source (for example public version):

``` bash
git clone https://github.com/IBM/mcas.git
```

### Update submodules
```bash
cd mcas
git submodule update --init --recursive
```

### Configure

Create build directory at root level.  We normally use `mcas/build` (The deadult build is in debug mode)

```bash
mkdir build
cd build
cmake -DBUILD_KERNEL_SUPPORT=ON -DFLATBUFFERS_BUILD_TESTS=0 -DTBB_BUILD_TESTS=0 -DBUILD_PYTHON_SUPPORT=1 -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/dist ..
```

Or perform a Release build (which will be much faster):

```bash
mkdir build
cd build
cmake -DBUILD_KERNEL_SUPPORT=ON -DFLATBUFFERS_BUILD_TESTS=0 -DTBB_BUILD_TESTS=0 -DBUILD_PYTHON_SUPPORT=1 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/dist ..
```

Sometimes we build with an alternate compiler:

```bash
mkdir clang
cd clang
cmake -DBUILD_KERNEL_SUPPORT=ON -DFLATBUFFERS_BUILD_TESTS=0 -DTBB_BUILD_TESTS=0 -DBUILD_PYTHON_SUPPORT=1 -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/dist ..
```

Or with code coverage:

```bash
mkdir coverage
cd coverage
cmake -DBUILD_KERNEL_SUPPORT=ON -DFLATBUFFERS_BUILD_TESTS=0 -DTBB_BUILD_TESTS=0 -DBUILD_PYTHON_SUPPORT=1 -DCMAKE_BUILD_TYPE=Debug -DCODE_COVERAGE=1 -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/dist ..
```

### One-time build
```bash
make bootstrap
```

### Normal build
```bash
make -j
make install 
```



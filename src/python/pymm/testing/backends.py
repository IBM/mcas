import pymm
import numpy as np
import math
import torch

from inspect import currentframe, getframeinfo
line = lambda : currentframe().f_back.f_lineno

def fail(msg):
    print(colored(255,0,0,msg))
    raise RuntimeError(msg)

def colored(r, g, b, text):
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)

def print_error(*args):
    print(colored(255,0,0,*args))

def log(*args):
    print(colored(0,255,255,*args))
    


# based on https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

def test_default():
    log("Running shelf with default backend ...")
    s = pymm.shelf('myShelf',size_mb=8,pmem_path='/mnt/pmem0',force_new=True)
    s.items
    log("OK!")

def test_dram_mapstore():
    log("Running shelf with mapstore and default MM plugin ...")
    s = pymm.shelf('myShelf2',size_mb=8,backend='mapstore')
    s.x = pymm.ndarray((10,10,))
    print(s.items)
    log("OK!")

def test_dram_mapstore_jemalloc():
    log("Running shelf with mapstore and jemalloc MM plugin ...")
    s = pymm.shelf('myShelf3',size_mb=8,backend='mapstore',mm_plugin='libmm-plugin-jemalloc.so')
    s.x = pymm.ndarray((10,10,))
    s.y = pymm.ndarray((100,100,))
    print(s.items)
    log("OK!")

def test_dram_mapstore_rcalb():
    log("Running shelf with mapstore and rcalb MM plugin ...")
    s = pymm.shelf('myShelf4',size_mb=8,backend='mapstore',mm_plugin='libmm-plugin-rcalb.so')
    s.x = pymm.ndarray((10,10,))
    s.y = pymm.ndarray((100,100,))
    print(s.items)
    log("OK!")
    

def test_hstore_mr_rcalb():
    log("Running shelf with hstore-mr and rcalb MM plugin ...")
    s = pymm.shelf('myShelf5',size_mb=8,backend='hstore-mr',pmem_path="/mnt/pmem0",mm_plugin='libmm-plugin-rcalb.so')
    s.x = pymm.ndarray((10,10,))
    s.y = pymm.ndarray((100,100,))
    print(s.items)
    log("OK!")

def test_hstore_mc_ccpm():
    log("Running shelf with hstore-cc and ccpm MM plugin ...")
    s = pymm.shelf('myShelf6',size_mb=8,backend='hstore-mc',pmem_path="/mnt/pmem0",mm_plugin='libmm-plugin-ccpm.so')
    s.x = pymm.ndarray((10,10,))
    s.y = pymm.ndarray((100,100,))
    print(s.items)
    log("OK!")
    
# run with DAX_RESET=1
#
test_default()
test_dram_mapstore()
test_dram_mapstore_jemalloc()
test_dram_mapstore_rcalb()
test_hstore_mr_rcalb()
test_hstore_mc_ccpm()


log("All tests OK!")

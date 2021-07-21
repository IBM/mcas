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


s = pymm.shelf('testShelf', pmem_path="/mnt/pmem0", size_mb=1024*32, force_new=True)

def test_list_construction(shelf):

    log("Testing: pymm.linked_list construction")
    shelf.x = pymm.linked_list()
    print(shelf.x)

    log("Testing: pymm.linked_list append method")
    shelf.x.append(123)
    shelf.x.append(1.321)
    shelf.x.append(np.ndarray((3,3,),dtype=np.uint8))
    
    print(shelf.x[0])
    print(shelf.x[1])
    print(shelf.x[2])
    shelf.x.append("Hello list!")

    if (shelf.x[0] != 123 or
        shelf.x[1] != 1.321):
        raise RuntimeError("data integrity check failed")

    print(shelf.x[3])
    if (shelf.x[3] != "Hello list!"):
        raise RuntimeError("string recall check failed")

    print(shelf.items)
    
    shelf.x[3] = "Goodbye";
    if (shelf.x[3] != "Goodbye"):
        raise RuntimeError("string change check failed")

    print("Length of list:{}".format(len(shelf.x)))
    

test_list_construction(s)

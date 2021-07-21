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
    

def test_ndarray(s):

    log("Testing: np.ndarray RHS ...")
    s.w = np.ndarray((100,100),dtype=np.uint8)
    s.w = np.ndarray([2,2,2],dtype=np.uint8)
    w = np.ndarray([2,2,2],dtype=np.uint8)
    s.w.fill(9)
    w.fill(9)    
    if np.array_equal(s.w, w) != True:
        raise RuntimeError('ndarray equality failed')

    log("Testing: pymm.ndarray RHS ...")
    s.x = pymm.ndarray((100,100),dtype=np.uint8)
    s.x = pymm.ndarray([2,2,2],dtype=np.uint8)
    x = np.ndarray([2,2,2],dtype=np.uint8)
    s.x.fill(8)
    x.fill(8)    
    if np.array_equal(s.x, x) != True:
        raise RuntimeError('ndarray equality failed')

    s.erase('x')
    s.erase('w')
    log("Testing: ndarray OK!")


def test_torch_tensor(s):

    log("Testing: torch.tensor RHS ...")
    s.w = torch.tensor((100,100,100),dtype=torch.uint8)
    s.w = torch.tensor([2,2,2],dtype=torch.uint8)
    w = torch.tensor([2,2,2],dtype=torch.uint8)
    s.w.fill_(9)
    w.fill_(9)
    print(s.w)
    print(w)

    if torch.equal(s.w, w) != True:
        raise RuntimeError('ndarray equality failed')

    log("Testing: pymm.ndarray RHS ...")
    s.x = pymm.torch_tensor((100,100,100),dtype=torch.uint8)
    s.x = pymm.torch_tensor([[1,1,1],[2,2,2]],dtype=torch.uint8)
    x = torch.tensor([[1,1,1],[2,2,2]],dtype=torch.uint8)
    s.x.fill_(8)
    s.x[1] = 9
    x.fill_(8)
    x[1] = 9
    print(s.x)
    print(x)
    if torch.equal(s.x, x) != True:
        raise RuntimeError('ndarray equality failed')

    s.erase('x')
    s.erase('w')
    log("Testing: ndarray OK!")


# based on https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

s = pymm.shelf('myShelf',size_mb=1024,pmem_path='/mnt/pmem0',force_new=True)
#test_ndarray(s)
#test_torch_tensor(s)

print(colored(255,255,255,"OK!"))

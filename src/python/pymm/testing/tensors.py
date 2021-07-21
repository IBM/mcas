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
    

def test_torch_tensor(s):
    
    log("Testing: torch_tensor")
    n = torch.Tensor(np.arange(0,1000))
    s.x = torch.Tensor(np.arange(0,1000)) #pymm.torch_tensor(n)

    # shelf type S
    if str(type(s.x)) != "<class 'pymm.torch_tensor.shelved_torch_tensor'>":
        fail('type check failed')

    log("Testing: torch_tensor sum={}".format(sum(s.x)))        
    if s.x.sum() != 499500:
        fail('summation check failed')

    slice_sum = sum(s.x[10:20])
    log("Testing: torch_tensor slice sum={}".format(slice_sum))
    if slice_sum != 145:
        fail('slice summation check failed')

    # shelf type S after in-place operation
    if str(type(s.x)) != "<class 'pymm.torch_tensor.shelved_torch_tensor'>":
        fail('type check failed')

    # shelf type S * NS (non-shelf type)
    if str(type(s.x * n)) != "<class 'torch.Tensor'>":
        fail('type check failed')

    # shelf type NS * S
    if str(type(n * s.x)) != "<class 'torch.Tensor'>":
        fail('type check failed')

    # shelf type S * shelf type S
    if str(type(s.x * s.x)) != "<class 'torch.Tensor'>":
        fail('type check failed')
    
    s.x += 1
    s.x *= 2
    s.x -= 0.4
    s.x /= 2

    s.erase('x')
    log("Testing: torch_tensor OK!")


# based on https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

s = pymm.shelf('myShelf',pmem_path='/mnt/pmem0',size_mb=1024,force_new=True)

test_torch_tensor(s)

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

    s.r = np.ndarray((3,4,))
    s.r.fill(1)
    print(id(s.r))
    
    if (s.r.shape[0] != 3 or s.r.shape[1] != 4):
        raise RuntimeError('shape check failed')

    s.r2 = s.r.reshape((2,6,))
    print(id(s.r2))

    if (s.r2.shape[0] != 2 or s.r2.shape[1] != 6):
        raise RuntimeError('shape check failed')

    s.r3 = s.r.reshape((1,-1,))
    print(id(s.r3))

    
    s.r2.fill(2)
    s.r3.fill(3)
    print(s.r)
    print(s.r2)
    print(s.r3)
    
    s.erase('r')
    s.erase('r2')
    s.erase('r3')
            
    log("Testing: ndarray OK!")

    
def test_tf_example(shelf):
    import tensorflow as tf
    (X_train, _), (X_test, _) = tf.keras.datasets.mnist.load_data()

    shelf.X = np.vstack([X_train, X_test])
    shelf.X = shelf.X.reshape(shelf.X.shape[0], -1).astype(np.float32)
    log("Testing: tf_example OK!")

# based on https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

s = pymm.shelf('myShelf',size_mb=1024,pmem_path='/mnt/pmem0',force_new=True)

test_ndarray(s)
test_tf_example(s)

print(colored(255,255,255,"OK!"))

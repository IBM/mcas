import pymm
import numpy as np
import math

def fail(msg):
    print(colored(255,0,0,msg))
    raise RuntimeError(msg)

def colored(r, g, b, text):
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)

def print_error(*args):
    print(colored(255,0,0,*args))

def log(*args):
    print(colored(0,255,255,*args))
    

def test_string(s):

    log("Testing: string")
    s.x = pymm.string("Hello world!")
    if str(s.x) != "Hello world!":
        print_error("FAIL: string value check")

    s.y = "Good Day sir!"
    if s.y != "Good Day sir!":
        print_error("FAIL: string value check 2")
        
    print(s)

    if "Day" not in s.y:
        print_error("FAIL: __contains__ result failed")

    if "foobar" in s.y:
        print_error("FAIL: __contains__ result failed")

    if s.y.capitalize() != "Good day sir!":
        print(">", s.y.capitalize())
        print_error("FAIL: string capitalize failed")

    if s.x.count("world") != 1:
        print_error("FAIL: count failed")

    s.x += " Brilliant!"
    print("Modified string >{}<".format(s.x))
        
    log("Testing: string OK!")


def test_float_number(s):    
    log("Testing: float number")

    s.n = pymm.float_number(700.001)

    # in-place ops
    s.n *= 2
    s.n /= 2

    print(s.n)
    print(s.n * 2)
    print(s.n * 1.1)
    
    print(2 * s.n)
    print(1.1 * s.n)

    print(s.n / 2.11)
    print(2.11 / s.n)

    print(s.n // 2.11)
    print(2.11 // s.n)

    print(s.n - 2.11)
    print(2.11 - s.n)
    
    if s.n * 2.0 != 1400.002:
        raise RuntimeError('arithmetic error')

    # from implicit cast
    s.m = 700.001
    if s.m != s.n:
        raise RuntimeError('equality error')
    
    log("Testing: number OK!")


def test_integer_number(s):    
    log("Testing: integer number")

    s.n = pymm.integer_number(700)
    print(s.n)
    
    # in-place ops
    s.n *= 2
    print(s.n)

    x = s.n * 2.1111
    print(x)
    if x != 2955.54:
        raise RuntimeError("integer arithmetic value check result")

    s.n += 2
    print(s.n)
    s.n /= 2
    print(s.n)
    
    log("Testing: integer number OK!")

#--- main ---
    
s = pymm.shelf('myShelf',pmem_path='/mnt/pmem0',size_mb=1024,force_new=True)

test_string(s)
test_float_number(s)
test_integer_number(s)



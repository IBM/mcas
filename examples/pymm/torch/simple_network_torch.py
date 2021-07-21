import pymm
import numpy as np
import math
import torch

# based on https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

s = pymm.shelf('myShelf',size_mb=1024,pmem_path='/mnt/pmem0',force_new=True)

dtype = torch.float

# device = torch.device("cuda:0") # Uncomment this to run on GPU

# Create random input and output data
s.x = torch.linspace(-math.pi, math.pi, 2000, dtype=dtype)

# create reference to shelf object; use as_tensor to get the subclass representation
# so that it looks like a Tensor type. this will prevent transactions happening where
# they are not needed
#x = s.x.as_tensor() 
x = s.x
print (type(s.x))

s.y = torch.sin(s.x)

# Randomly initialize weights
a = torch.randn((), dtype=dtype)
b = torch.randn((), dtype=dtype)
c = torch.randn((), dtype=dtype)
d = torch.randn((), dtype=dtype)

# sanity checks
if type(s.x * a) != torch.Tensor:
    raise RuntimeError('unexpected type result: {}'.format(type(s.x * a)))
if type(a * s.x) != torch.Tensor:
    raise RuntimeError('unexpected type result: {}'.format(type(a * s.x)))


learning_rate = 1e-6
for t in range(2000):
    # Forward pass: compute predicted y
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # Compute and print loss
    loss = (y_pred - s.y).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)

    # Backprop to compute gradients of a, b, c, d with respect to loss
    grad_y_pred = 2.0 * (y_pred - s.y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

     # Uwdate weights using gradient descent
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d


print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')




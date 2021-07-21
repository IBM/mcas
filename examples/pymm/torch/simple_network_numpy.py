import pymm
import numpy as np
import math

# based on https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

s = pymm.shelf('myShelf',size_mb=1024,pmem_path='/mnt/pmem0',force_new=True)


# Create random input and output data
s.x = np.linspace(-math.pi, math.pi, 2000)
print (type(s.x))
s.y = np.sin(s.x)


# Randomly initialize weights
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

learning_rate = 1e-6
for t in range(2000):
    # Forward pass: compute predicted y
    # y = a + b x + c x^2 + d x^3
    y_pred = a + b * s.x + c * s.x ** 2 + d * s.x ** 3

    # Compute and print loss
    loss = np.square(y_pred - s.y).sum()
    if t % 100 == 99:
        print(t, loss)

    # Backprop to compute gradients of a, b, c, d with respect to loss
    grad_y_pred = 2.0 * (y_pred - s.y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * s.x).sum()
    grad_c = (grad_y_pred * s.x ** 2).sum()
    grad_d = (grad_y_pred * s.x ** 3).sum()

    # Update weights
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')

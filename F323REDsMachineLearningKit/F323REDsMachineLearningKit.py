'''
Author : F323RED
Date : 2020/10/17
Version : 1.0.0
Describes : A set of functions for machine learning projects.
            Hopefully everything should be work.
'''

import numpy as np

# Some common activation functions
def Sigmoid(x) :
    return (1 / (1 + np.exp(-x)))   # From 0 to 1, smoothly.

def Stair(x) :
    return x > 0                    # Either 1 or 0. Like a stair.

def ReLU(x) :
    return np.maximum(x, 0)         # Any postive real number.

# Output process
def Softmax(x) :
    c = np.max(x)               # To prevent e^x overflow
    exp_a = np.exp(x - c)
    sum_exp = np.sum(exp_a)
                                # The sum of every elements is 1.
    return exp_a / sum_exp      # Represent chance of this answer.

# Common loss functions
# y : predict, t : correct answer
def MeanSquareError(y, t) :
    # Compatiable with batch process
    if y.ndim == 1 :
        t = t.reshape(1, t.size)        
        y = y.reshape(1, y.size)

    # Less is better.
    return 0.5 * np.sum((y - t) ** 2) / y.shape[0]

def CrossEntropyError(y, t) :
    # Compatiable with batch process
    if y.ndim == 1 :
        t = t.reshape(1, t.size)        
        y = y.reshape(1, y.size)

    # Less is better.
    return -np.sum(t * np.log(y + 10e-7))  / y.shape[0]

# Some sort of differentiation stuff
def NumericalDiff(f, x) :
    h = 1e-4

    return (f(x+h) - f(x-h)) / 2*h      # A simple differentiation

def NumeriaclGradient(f, x) :
    h = 1e-4
    grad = np.zeros_like(x)

    for i in range(x.shape[0]) :
        temp = np.copy(x[i])
        
        x[i] = temp + h
        a = f(x[i])

        x[i] = temp - h
        b = f(x[i])

        x[i] = temp
        grad[i] = (a - b) / (2 * h)


    return grad;                        # Get the gradient at x

def GradientDescent(f, init_x, learnRate=0.01, step=100) :
    x = init_x

    for i in range(step) :
        grad = NumeriaclGradient(f, x)
        x -= learnRate * grad

    return x
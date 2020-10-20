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
    if x.ndim == 1:
        x = x.reshape(1, x.size)
    
    result_sum_exp = []
    for i in x:
        c = np.max(i)                           # To prevent e^x overflow
        exp_a = np.exp(i - c)
        sum_exp = np.sum(exp_a)
        result_sum_exp.append(exp_a / sum_exp)  # The sum of every elements is 1.
                                
    return np.array(result_sum_exp)     # Represent chance of this answer.

# Common loss functions
# y : predict, t : correct answer
def MeanSquareError(y, t) :
    # Compatiable with batch process
    if y.ndim == 1 :
        t = t.reshape(1, t.size)        
        y = y.reshape(1, y.size)

    # Less is better.
    return (0.5 * np.sum((y - t) ** 2)) / y.shape[0]

def CrossEntropyError(y, t) :
    # Compatiable with batch process
    if y.ndim == 1 :
        t = t.reshape(1, t.size)        
        y = y.reshape(1, y.size)

    # Less is better.
    return -np.sum(t * np.log(y + 1e-4))  / y.shape[0]

# Some sort of differentiation stuff
def NumericalDiff(f, x) :
    h = 1e-3

    return (f(x+h) - f(x-h)) / 2*h      # A simple differentiation

def NumericalGradient(f, x) :
    h = 1e-3
    grad = np.zeros_like(x)

    for i in range(x.shape[0]) :
        if(x.ndim > 1):
            for j in range(x.shape[1]):
                temp = np.copy(x[i][j])
        
                x[i][j] = temp + h
                a = f(x[i][j])

                x[i][j] = temp - h
                b = f(x[i][j])

                x[i][j] = temp
                grad[i][j] = (a - b) / (2 * h)

        else:
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
        grad = NumericalGradient(f, x)
        x -= learnRate * grad

    return x

# Debug
#lr = 0.1
#x = np.array([[1, 2, 8], [2, 5, 6]], dtype=float)
#W1 = np.array([[1, 2], 
#              [3, 4], 
#              [5, 6]], dtype=float)
#b1 = np.array([6, 9], dtype=float)
#t = np.array([1, 0], dtype=float)

#print("x", x)
#def pre():
#    a1 = np.dot(x, W1) + b1
#    z1 = Softmax(a1)
#    return z1

#def f(W):
#    y = pre()
#    return CrossEntropyError(y, t)

#for i in range(100):
#    dW = NumericalGradient(f, W1)
#    #print("dW", dW)
#    db = NumericalGradient(f, b1)
#    #print("db", db)

#    W1 -= lr * dW
#    b1 -= lr * db
#    print("loop", i+1)
#    print("W1", W1)
#    print("b1", b1)
#    print(CrossEntropyError(pre(), t))
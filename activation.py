# activation.py - Activation Functions (ReLU, Sigmoid, Custom)
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def relu(x):
    return max(0, x)
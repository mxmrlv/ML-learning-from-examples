import math


def unipolar_sigmoid(x):
    return 1 / (1 + math.exp(-x))


def bipolar_sigmoid(x):
    return (2 / (1 + math.exp(-x))) - 1


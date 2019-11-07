import numpy as np


def euclidean_distance(a, b):
    return np.sqrt(np.power(a[0] - b[0], 2) + np.power(a[1] - b[1], 2))


def slope(a, b):
    return (a[1] - b[1]) / (a[0] - b[0])

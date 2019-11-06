import numpy as np


def euclidean_distance(a, b):
    return np.sqrt(np.power(a[0] - b[0], 2) + np.power(a[1] - b[1], 2))


def per_elem_diff(a, b):
    return a[0] - b[0], a[1] - b[1]

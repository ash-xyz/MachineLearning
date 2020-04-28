import numpy as np
import random

class Network(object):
    def __init__(self, sizes):
        self.num_layers = sizes
        self.sizes = sizes
        self.biases = [np.randomn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

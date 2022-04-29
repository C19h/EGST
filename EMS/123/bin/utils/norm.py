import numpy as np
class Norm():
    def __init__(self, x):
        self.max = x.max(axis = 0)
        self.min = x.min(axis = 0)

    def encode(self, x):
        return (x - self.min) / (self.max - self.min)

    def decode(self, x):
        return x * (self.max - self.min) + self.min

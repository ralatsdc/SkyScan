import numpy as np

class Vector:
    def __init__(self, v):
        self.v = v

    def __add__(self, other):
        return Vector(self.v + other.v)

    def __sub__(self, other):
        return Vector(self.v - other.v)

    def smul(self, scalar):
        return Vector(scalar * self.v)

    def dot(self, other):
        return np.dot(self.v, other.v)

    def cross(self, other):
        return Vector(np.cross(self.v, other.v))

    def __eq__(self, other):
        return np.equal(self.v, other.v).all()

import numpy as np

class Vector:
    def __init__(self, b, c, d):
        self.b = b
        self.c = c
        self.d = d

    def __add__(self, other):
        """
        if type(other) is not Vector:
            raise TypeError("Only Vectors can be added to Vectors")
        """
        return Vector(
            self.b + other.b,
            self.c + other.c,
            self.d + other.d,
        )

    def __sub__(self, other):
        """
        if type(other) is not Vector:
            raise TypeError("Only Vectors can be subtracted from Vectors")
        """
        return Vector(
            self.b - other.b,
            self.c - other.c,
            self.d - other.d,
        )

    def smul(self, scalar):
        if not (type(scalar) is int or type(scalar) is float):
            raise TypeError("Vectors can only be multiplied by scalars")
        return Vector(
            self.b * scalar,
            self.c * scalar,
            self.d * scalar,
        )

    def dot(self, other):
        """
        if type(other) is not Vector:
            raise TypeError("Dot product only defined for Vector arguments")
        """
        return self.b * other.b + self.c * other.c + self.d * other.d

    def cross(self, other):
        """
        if type(other) is not Vector:
            raise TypeError("Cross product only defined for Vector arguments")
        """
        return Vector(
            self.c * other.d - self.d * other.c,
            self.d * other.b - self.b * other.d,
            self.b * other.c - self.c * other.b,
        )

    def __eq__(self, other):
        """
        if type(other) is not Vector:
            raise TypeError("Vectors can only be compared to Vectors")
        """
        return self.b == other.b and self.c == other.c and self.d == other.d

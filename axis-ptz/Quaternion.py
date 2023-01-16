import math

import numpy as np

from Vector import Vector


class Quaternion:
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def get_scalar(self):
        return self.a

    def get_vector(self):
        return Vector(self.b, self.c, self.d)

    def conjugate(self):
        return Quaternion(self.a, -self.b, -self.c, -self.d)

    def norm(self):
        return math.sqrt(self.a**2 + self.b**2 + self.c**2 + self.d**2)

    def __sub__(self, other):
        """
        if type(other) is not Quaternion:
            raise TypeError("Only Quaternions can be subtracted from Quaternions")
        """
        return Quaternion(
            self.a - other.a,
            self.b - other.b,
            self.c - other.c,
            self.d - other.d,
        )

    def __mul__(self, other):
        """
        if type(other) is not Quaternion:
            raise TypeError("Quaternions can only be multiplied by Quaternions")
        """
        self_s = float(self.get_scalar())
        other_s = float(other.get_scalar())
        self_v = self.get_vector()
        other_v = other.get_vector()
        s = self_s * other_s - self_v.dot(other_v)
        v = other_v.smul(self_s) + self_v.smul(other_s) + self_v.cross(other_v)
        return Quaternion(s, v.b, v.c, v.d)

    def __eq__(self, other):
        """
        if type(other) is not Quaternion:
            raise TypeError("Quaternions can only be compared to Quaternions")
        """
        return (
            self.a == other.a
            and self.b == other.b
            and self.c == other.c
            and self.d == other.d
        )

    @staticmethod
    def as_quaternion(s, v):
        """Construct a quaternion given a scalar and vector.

        Parameters
        ----------
        s : float
            A scalar value
        v : list or numpy.ndarray
            A vector of floats

        Returns
        -------
        Quaternion.Quaternion
            A quaternion with the specified scalar and vector parts
        """
        """
        if type(s) != float:
            raise Exception("Scalar part is not a float")
        if len(v) != 3 or not all(
            [type(e) == float or type(e) == np.float64 for e in v]
        ):
            raise Exception("Vector part is not an iterable of three floats")
        """
        return Quaternion(s, v[0], v[1], v[2])

    @staticmethod
    def as_rotation_quaternion(d_omega, u):
        """Construct a rotation quaternion given an angle and
        direction of rotation.

        Parameters
        ----------
        d_omega : float
            An angle [deg]
        u : list or numpy.ndarray
            A vector of floats

        Returns
        -------
        Quaternion.Quaternion
            A rotation quaternion with the specified angle and direction
        """
        """
        if type(d_omega) != float:
            raise Exception("Angle is not a float")
        if len(u) != 3 or not all(
            [type(e) == float or type(e) == np.float64 for e in u]
        ):
            raise Exception("Vector part is not an iterable of three floats")
        """
        r_omega = math.radians(d_omega)
        v = [math.sin(r_omega / 2) * e for e in u]
        return Quaternion(math.cos(r_omega / 2), v[0], v[1], v[2])

    @staticmethod
    def as_vector(q):
        """Return the vector part of a quaternion, provided the scalar
        part is nearly zero.

        Parameters
        ----------
        q : Quaternion.Quaternion
            A vector quaternion

        Returns
        -------
        numpy.ndarray
            A vector of floats
        """
        if math.fabs(q.a) > 1e-12:
            raise Exception("Quaternion is not a vector quaternion")
        return np.array([q.b, q.c, q.d])

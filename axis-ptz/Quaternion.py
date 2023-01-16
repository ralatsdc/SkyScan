import math

import numpy as np
import numpy.linalg.linalg

from Vector import Vector


class Quaternion:
    def __init__(self, a):
        self.a = a

    def get_scalar(self):
        return self.a[0]

    def get_vector(self):
        return self.a[1:]

    def conjugate(self):
        return Quaternion(np.append(self.a[0], -self.a[1:]))

    def norm(self):
        return numpy.linalg.norm(self.a)

    def __sub__(self, other):
        return Quaternion(self.a - other.a)

    def __mul__(self, other):
        self_s = self.get_scalar()
        other_s = other.get_scalar()
        self_v = self.get_vector()
        other_v = other.get_vector()
        s = self_s * other_s - np.dot(self_v, other_v)
        v = self_s * other_v + other_s * self_v + np.cross(self_v, other_v)
        return Quaternion(np.append(s, v))

    def __eq__(self, other):
        return np.equal(self.a, other.a).all()

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
        return Quaternion(np.append(s, v))

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
        return Quaternion(np.array([math.cos(r_omega / 2), v[0], v[1], v[2]]))

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
        if math.fabs(q.a[0]) > 1e-12:
            raise Exception("Quaternion is not a vector quaternion")
        return q.get_vector()

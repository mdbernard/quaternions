import numpy as np


class Quaternion:
    """A base quaternion class with universal quaternion properties."""

    def __init__(self, w, x, y, z):
        """
        Instantiate a quaternion from its component parts.

        Args:
            w (float): scalar component
            x (float): first vector component
            y (float): second vector component
            z (float): third vector component
        """
        self.__scalar = w
        self.__vector = np.array([x, y, z])

    @property
    def array(self):
        """Get the quaternion as a scalar-first array."""
        return np.array([self.w, self.x, self.y, self.z])

    @property
    def w(self):
        return self.__scalar

    @property
    def v(self):
        return self.__vector

    @property
    def x(self):
        return self.v[0]

    @property
    def y(self):
        return self.v[1]

    @property
    def z(self):
        return self.v[2]

    def normalize(self):
        """Normalize the quaternion to have unit length."""
        mag = np.linalg.norm(self.array)
        if mag > 0:
            arr_norm = self.array/mag
            self.__scalar = arr_norm[0]
            self.__vector = arr_norm[1:]
        else:
            raise ValueError(('Cannot normalize quaternion with non-positive '
                              'magnitude.'))


class UnitQuaternion(Quaternion):
    """A quaternion class that enforces unit magnitude."""

    def __init__(self, w, x, y, z, tol=1e-5):
        """
        Create the quaternion with the closest floating point values to the
        inputted components such that the magnitude is as close to 1.0 as
        possible.

        Args:
            w (float): scalar component
            x (float): first vector component
            y (float): second vector component
            z (float): third vector component
            tol (float): if the difference between 1.0 and the norm of the
                inputted components is greater than tol, raise an Exception
        """
        norm = np.linalg.norm([w, x, y, z])
        if norm == 0:
            raise ValueError(('Cannot create UnitQuaternion from components '
                              'with zero magnitude.'))
        if np.abs(1.0 - norm) > tol:
            raise ValueError(('The magnitude of the inputted components is '
                              'greater than the tolerance for rounding to 1.'))

        w, x, y, z = np.array([w, x, y, z])/norm
        super().__init__(w, x, y, z)

        self.__tol = tol

    @property
    def is_unit(self):
        return np.abs(1.0 - np.linalg.norm(self.array)) < self.__tol

    @property
    def tol(self):
        return self.__tol

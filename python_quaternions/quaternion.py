import numpy as np


class Quaternion:
    def __init__(self, w: float, x: float, y: float, z: float) -> None:
        """
        Instantiate a quaternion from its component parts.

        Args:
            w (float): scalar component
            x (float): first vector component
            y (float): second vector component
            z (float): third vector component
        """
        self.w = w
        self.v = np.array([x, y, z])

    @property
    def array(self):
        """Get the quaternion as a scalar-first array."""
        return np.array([self.w, self.x, self.y, self.z])

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
            self.w = arr_norm[0]
            self.v = arr_norm[1:]
        else:
            raise ValueError('Cannot normalize quaternion with magnitude less than 0')

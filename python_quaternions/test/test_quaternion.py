import pytest
import numpy as np

from python_quaternions.quaternion import Quaternion, UnitQuaternion


def assert_array_equal(array_0, array_1, tol=0.0):
    """
    Utility function asserting whether two numpy arrays are componentwise equal,
    within a tolerance.
    """
    assert np.array_equal(array_0.shape, array_1.shape)
    for arr_0_elem, arr_1_elem in zip(array_0, array_1):
        assert np.abs(arr_0_elem - arr_1_elem) <= tol



def test_quaternion_init():
    """Test that all properties of a quaternion are set correctly."""
    # arrange
    w = 0.5
    x = 0.5
    y = 0.5
    z = 0.5

    # act
    q = Quaternion(w, x, y, z)

    # assert
    assert q.w == 0.5
    assert_array_equal(q.v, np.array([x, y, z]))
    assert_array_equal(q.array, np.array([w, x, y, z]))
    assert q.x == 0.5
    assert q.y == 0.5
    assert q.z == 0.5


def test_quaternion_normalize():
    """Test that normalizing a quaternion with non-zero magnitude works."""
    # arrange
    w = 1.0
    x = -1.0
    y = 1.0
    z = -1.0

    q = Quaternion(w, x, y, z)

    # act
    q.normalize()

    # assert
    assert q.w == 0.5
    assert q.x == -0.5
    assert q.y == 0.5
    assert q.z == -0.5


def test_quaternion_normalize_zero_magnitude():
    """
    Test that normalizing a quaternion with zero magnitude raises the
    correct error.
    """
    # arrange
    w = 0.0
    x = 0.0
    y = 0.0
    z = 0.0

    q = Quaternion(w, x, y, z)

    err_msg = 'Cannot normalize quaternion with non-positive magnitude.'

    # act/assert
    with pytest.raises(ValueError, match=err_msg):
        q.normalize()


def test_unit_quaternion_zero_error_input():
    # arrange
    w = 1.0
    x = 0.0
    y = 0.0
    z = 0.0

    # act
    q = UnitQuaternion(w, x, y, z)

    # assert
    assert q.w == 1.0
    assert q.x == 0.0
    assert q.y == 0.0
    assert q.z == 0.0
    assert q.is_unit
    assert q.tol == 1e-5


def test_unit_quaternion_zero_magnitude():
    # arrange
    w = 0.0
    x = 0.0
    y = 0.0
    z = 0.0

    err_msg = ('Cannot create UnitQuaternion from components with zero '
               'magnitude.')

    # act/assert
    with pytest.raises(ValueError, match=err_msg):
        q = UnitQuaternion(w, x, y, z)


def test_unit_quaternion_within_tolerance_input():
    # arrange
    w = 1 - 1e-5  # within default tolerance of UnitQuaternion
    x = 0.0
    y = 0.0
    z = 0.0

    # act
    q = UnitQuaternion(w, x, y, z)

    # assert
    assert q.w == 1.0
    assert q.x == 0.0
    assert q.y == 0.0
    assert q.z == 0.0


def test_unit_quaternion_out_of_tolerance_input():
    # arrange
    w = 1.1
    x = 1.1
    y = 1.1
    z = 1.1

    err_msg = ('The magnitude of the inputted components is greater than the '
               'tolerance for rounding to 1.')

    # act/assert
    with pytest.raises(ValueError, match=err_msg):
        q = UnitQuaternion(w, x, y, z)

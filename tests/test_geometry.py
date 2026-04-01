"""Tests for core/geometry.py — pure math, no file I/O."""

import numpy as np
import pytest

from molconvert.core.geometry import (
    unit,
    bond_length,
    bond_angle_deg,
    dihedral_deg,
    place_atom,
)


# ------------------------------------------------------------------ #
#  unit                                                                #
# ------------------------------------------------------------------ #

def test_unit_basic():
    v = np.array([3.0, 0.0, 0.0])
    u = unit(v)
    np.testing.assert_allclose(u, [1.0, 0.0, 0.0])


def test_unit_normalises_to_length_one():
    v = np.array([1.0, 2.0, 3.0])
    assert abs(np.linalg.norm(unit(v)) - 1.0) < 1e-12


def test_unit_zero_raises():
    with pytest.raises(ValueError):
        unit(np.array([0.0, 0.0, 0.0]))


# ------------------------------------------------------------------ #
#  bond_length                                                         #
# ------------------------------------------------------------------ #


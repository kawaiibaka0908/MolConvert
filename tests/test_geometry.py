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

def test_bond_length_along_x():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.5, 0.0, 0.0])
    assert abs(bond_length(a, b) - 1.5) < 1e-10


def test_bond_length_3d():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 6.0, 3.0])
    # distance = sqrt(9 + 16 + 0) = 5
    assert abs(bond_length(a, b) - 5.0) < 1e-10


def test_bond_length_symmetric():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert abs(bond_length(a, b) - bond_length(b, a)) < 1e-12


# ------------------------------------------------------------------ #
#  bond_angle_deg                                                      #
# ------------------------------------------------------------------ #

def test_bond_angle_90():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0])
    c = np.array([0.0, 1.0, 0.0])
    assert abs(bond_angle_deg(a, b, c) - 90.0) < 1e-10


def test_bond_angle_180():
    a = np.array([-1.0, 0.0, 0.0])
    b = np.array([ 0.0, 0.0, 0.0])
    c = np.array([ 1.0, 0.0, 0.0])
    assert abs(bond_angle_deg(a, b, c) - 180.0) < 1e-10


def test_bond_angle_60():
    # Equilateral triangle — angle at origin = 60°
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0])
    c = np.array([0.5, np.sqrt(3) / 2, 0.0])
    assert abs(bond_angle_deg(a, b, c) - 60.0) < 1e-8


def test_bond_angle_in_range():
    rng = np.random.default_rng(42)
    for _ in range(50):
        a, b, c = rng.standard_normal((3, 3))
        angle = bond_angle_deg(a, b, c)
        assert 0.0 <= angle <= 180.0


# ------------------------------------------------------------------ #
#  dihedral_deg                                                        #
# ------------------------------------------------------------------ #

def test_dihedral_zero():
    # All atoms in the x-y plane, cis configuration → 0°
    a = np.array([0.0, 1.0, 0.0])
    b = np.array([0.0, 0.0, 0.0])
    c = np.array([1.0, 0.0, 0.0])
    d = np.array([1.0, 1.0, 0.0])
    assert abs(dihedral_deg(a, b, c, d)) < 1e-8



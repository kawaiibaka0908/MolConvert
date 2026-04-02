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


def test_dihedral_180():
    # trans configuration
    a = np.array([0.0,  1.0, 0.0])
    b = np.array([0.0,  0.0, 0.0])
    c = np.array([1.0,  0.0, 0.0])
    d = np.array([1.0, -1.0, 0.0])
    assert abs(abs(dihedral_deg(a, b, c, d)) - 180.0) < 1e-8


def test_dihedral_90():
    a = np.array([0.0, 1.0, 0.0])
    b = np.array([0.0, 0.0, 0.0])
    c = np.array([1.0, 0.0, 0.0])
    d = np.array([1.0, 0.0, 1.0])   # rotated 90° out of plane
    assert abs(abs(dihedral_deg(a, b, c, d)) - 90.0) < 1e-8


def test_dihedral_range():
    rng = np.random.default_rng(7)
    for _ in range(50):
        a, b, c, d = rng.standard_normal((4, 3))
        di = dihedral_deg(a, b, c, d)
        assert -180.0 <= di <= 180.0


def test_dihedral_collinear_returns_zero():
    # b and c collinear with a — undefined dihedral, should not raise
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    c = np.array([2.0, 0.0, 0.0])
    d = np.array([3.0, 0.0, 0.0])
    result = dihedral_deg(a, b, c, d)
    assert result == 0.0


# ------------------------------------------------------------------ #
#  place_atom (NeRF round-trip)                                        #
# ------------------------------------------------------------------ #

def test_place_atom_recovers_position():
    """
    Given three anchor atoms and a known fourth, compute its IC from the
    geometry functions, then reconstruct with place_atom — must match.
    """
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.5, 0.0, 0.0])
    c = np.array([2.0, 1.2, 0.0])
    d_true = np.array([3.0, 1.5, 0.8])

    bl = bond_length(c, d_true)
    ba = bond_angle_deg(b, c, d_true)
    di = dihedral_deg(a, b, c, d_true)

    d_recon = place_atom(a, b, c, bl, ba, di)
    np.testing.assert_allclose(d_recon, d_true, atol=1e-8)


def test_place_atom_correct_bond_length():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.5, 0.0, 0.0])
    c = np.array([2.0, 1.2, 0.0])
    d = place_atom(a, b, c, bond_length=1.4, bond_angle_deg=110.0, dihedral_deg=45.0)
    assert abs(bond_length(c, d) - 1.4) < 1e-8


def test_place_atom_correct_bond_angle():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.5, 0.0, 0.0])
    c = np.array([2.0, 1.2, 0.0])
    d = place_atom(a, b, c, bond_length=1.4, bond_angle_deg=109.5, dihedral_deg=60.0)
    assert abs(bond_angle_deg(b, c, d) - 109.5) < 1e-6


def test_place_atom_correct_dihedral():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.5, 0.0, 0.0])
    c = np.array([2.0, 1.2, 0.0])
    for target_di in [-120.0, -60.0, 0.0, 60.0, 120.0, 180.0]:
        d = place_atom(a, b, c, bond_length=1.4, bond_angle_deg=109.5,
                       dihedral_deg=target_di)
        recovered = dihedral_deg(a, b, c, d)
        assert abs(recovered - target_di) < 1e-6, (
            f"dihedral {target_di}° → recovered {recovered}°"
        )

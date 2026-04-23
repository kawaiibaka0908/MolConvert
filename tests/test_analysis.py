"""Tests for analysis/rmsd.py — kabsch_align, kabsch_rmsd, rmsd, rmsd_molecules."""

import copy
from pathlib import Path

import numpy as np
import pytest

from molconvert.parsers.pdb_parser import parse_pdb
from molconvert.builders.reconstruct import reconstruct
from molconvert.analysis import (
    rmsd,
    rmsd_molecules,
    per_atom_deviation,
    ic_summary,
    ICSummary,
)
from molconvert.analysis.rmsd import kabsch_align, kabsch_superpose, kabsch_rmsd

MINI_PDB = str(Path(__file__).parent / "data" / "mini.pdb")


@pytest.fixture
def mol_orig():
    return parse_pdb(MINI_PDB)[0]


@pytest.fixture
def mol_recon(mol_orig):
    return reconstruct(mol_orig)


# ------------------------------------------------------------------ #
#  kabsch_align — input validation                                     #
# ------------------------------------------------------------------ #

def test_kabsch_align_raises_on_non_3d():
    bad = np.ones((5, 2))
    good = np.ones((5, 3))
    with pytest.raises(ValueError, match="N, 3"):
        kabsch_align(good, bad)
    with pytest.raises(ValueError, match="N, 3"):
        kabsch_align(bad, good)


def test_kabsch_align_raises_on_shape_mismatch():
    c1 = np.ones((5, 3))
    c2 = np.ones((6, 3))
    with pytest.raises(ValueError, match="Shape mismatch"):
        kabsch_align(c1, c2)


def test_kabsch_align_raises_when_fewer_than_3_points():
    c1 = np.ones((2, 3))
    c2 = np.ones((2, 3))
    with pytest.raises(ValueError, match="3"):
        kabsch_align(c1, c2)


def test_kabsch_align_accepts_list_input():
    c1 = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    c2 = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    R, t, r = kabsch_align(c1, c2)
    assert r == pytest.approx(0.0, abs=1e-10)


# ------------------------------------------------------------------ #
#  kabsch_align — return types and shapes                              #
# ------------------------------------------------------------------ #

def test_kabsch_align_returns_tuple_of_three():
    c = np.random.default_rng(0).standard_normal((10, 3))
    result = kabsch_align(c, c)
    assert len(result) == 3


def test_kabsch_align_rotation_is_3x3():
    c = np.random.default_rng(1).standard_normal((10, 3))
    R, t, _ = kabsch_align(c, c)
    assert R.shape == (3, 3)


def test_kabsch_align_translation_is_3():
    c = np.random.default_rng(2).standard_normal((10, 3))
    R, t, _ = kabsch_align(c, c)
    assert t.shape == (3,)


def test_kabsch_align_rmsd_is_scalar():
    c = np.random.default_rng(3).standard_normal((10, 3))
    _, _, r = kabsch_align(c, c)
    assert isinstance(r, float)


# ------------------------------------------------------------------ #
#  kabsch_align — Test 1: identity (coords1 == coords2 → RMSD ≈ 0)   #
# ------------------------------------------------------------------ #

def test_kabsch_identity_rmsd_zero():
    rng = np.random.default_rng(42)
    coords = rng.standard_normal((20, 3))
    R, t, r = kabsch_align(coords, coords)
    assert r == pytest.approx(0.0, abs=1e-10)


def test_kabsch_identity_rotation_is_identity_matrix():
    rng = np.random.default_rng(42)
    coords = rng.standard_normal((20, 3))
    R, t, _ = kabsch_align(coords, coords)
    assert np.allclose(R, np.eye(3), atol=1e-10)


# ------------------------------------------------------------------ #
#  kabsch_align — Test 2: pure translation (RMSD ≈ 0 after alignment) #
# ------------------------------------------------------------------ #

def test_kabsch_translated_structure_rmsd_zero():
    rng = np.random.default_rng(7)
    coords1 = rng.standard_normal((15, 3))
    shift = np.array([3.5, -2.1, 7.8])
    coords2 = coords1 + shift
    R, t, r = kabsch_align(coords1, coords2)
    assert r == pytest.approx(0.0, abs=1e-9)


def test_kabsch_translated_structure_rotation_is_identity():
    rng = np.random.default_rng(8)
    coords1 = rng.standard_normal((15, 3))
    coords2 = coords1 + np.array([1.0, 2.0, 3.0])
    R, t, _ = kabsch_align(coords1, coords2)
    assert np.allclose(R, np.eye(3), atol=1e-9)


def test_kabsch_translation_vector_recovers_shift():
    rng = np.random.default_rng(9)
    coords1 = rng.standard_normal((12, 3))
    shift = np.array([5.0, -3.0, 2.0])
    coords2 = coords1 + shift
    R, t, _ = kabsch_align(coords1, coords2)
    # Applying R and t to coords2 should reproduce coords1
    aligned = coords2 @ R + t
    assert np.allclose(aligned, coords1, atol=1e-9)


# ------------------------------------------------------------------ #
#  kabsch_align — Test 3: pure rotation (RMSD ≈ 0 after alignment)    #
# ------------------------------------------------------------------ #

def _rotation_matrix_x(angle_deg: float) -> np.ndarray:
    a = np.radians(angle_deg)
    return np.array([
        [1, 0,          0         ],
        [0, np.cos(a), -np.sin(a) ],
        [0, np.sin(a),  np.cos(a) ],
    ])


def _rotation_matrix_z(angle_deg: float) -> np.ndarray:
    a = np.radians(angle_deg)
    return np.array([
        [ np.cos(a), -np.sin(a), 0],
        [ np.sin(a),  np.cos(a), 0],
        [ 0,          0,         1],
    ])


def test_kabsch_rotated_structure_rmsd_zero():
    rng = np.random.default_rng(13)
    coords1 = rng.standard_normal((20, 3))
    Rtrue = _rotation_matrix_x(37.0) @ _rotation_matrix_z(55.0)
    coords2 = coords1 @ Rtrue.T      # rotate coords2 away from coords1
    R, t, r = kabsch_align(coords1, coords2)
    assert r == pytest.approx(0.0, abs=1e-9)


def test_kabsch_recovered_rotation_matches_true():
    rng = np.random.default_rng(14)
    coords1 = rng.standard_normal((20, 3))
    Rtrue = _rotation_matrix_z(90.0)
    coords2 = coords1 @ Rtrue.T
    R, t, _ = kabsch_align(coords1, coords2)
    # R should equal Rtrue (the rotation that maps coords2 back to coords1)
    assert np.allclose(R, Rtrue, atol=1e-9)


def test_kabsch_aligned_coords_match_reference():
    rng = np.random.default_rng(15)
    coords1 = rng.standard_normal((25, 3))
    Rtrue = _rotation_matrix_x(120.0)
    coords2 = coords1 @ Rtrue.T
    R, t, _ = kabsch_align(coords1, coords2)
    aligned = coords2 @ R + t
    assert np.allclose(aligned, coords1, atol=1e-9)


# ------------------------------------------------------------------ #
#  kabsch_align — Test 4: noise (RMSD small but non-zero)             #
# ------------------------------------------------------------------ #


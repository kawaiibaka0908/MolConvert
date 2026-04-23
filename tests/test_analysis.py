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



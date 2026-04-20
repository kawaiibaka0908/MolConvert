"""
Tests for SDF ↔ PDB / SDF ↔ JSON / SDF ↔ ZMAT conversions.

Covers:
  - to_sdf builder  : MoleculeIC → SDF string / file
  - bond inference  : correct atom/bond counts, no phantom bonds
  - SDF round-trip  : parse → to_sdf atom/bond counts preserved
  - CLI paths       : .sdf --to pdb/json/sdf/zmat
                      .pdb --to sdf
                      .zmat --to sdf
  - --remove-hydrogens flag with SDF input
  - Error paths     : missing file, unsupported extension
"""

import json
from pathlib import Path

import pytest

from molconvert.parsers.sdf_parser import parse_sdf
from molconvert.parsers.pdb_parser import parse_pdb
from molconvert.builders.reconstruct import reconstruct
from molconvert.builders.to_sdf import (
    molecule_to_sdf,
    save_sdf,
    molecules_to_sdf,
    save_sdf_multi,
)
from molconvert.analysis import rmsd_molecules
from molconvert.cli.main import run_convert

MINI_SDF = str(Path(__file__).parent / "data" / "mini.sdf")
MINI_PDB = str(Path(__file__).parent / "data" / "mini.pdb")


# ------------------------------------------------------------------ #
#  Fixtures                                                            #
# ------------------------------------------------------------------ #

@pytest.fixture
def ethanol():
    return parse_sdf(MINI_SDF)[0]   # 9 atoms (C, C, O, 6H)


@pytest.fixture
def acetone():
    return parse_sdf(MINI_SDF)[1]   # 4 atoms (C, C, C, O)


@pytest.fixture
def ethanol_sdf(ethanol):
    return molecule_to_sdf(ethanol)


@pytest.fixture
def acetone_sdf(acetone):
    return molecule_to_sdf(acetone)


# ------------------------------------------------------------------ #
#  sdf_parser — new reference fields (bond_to / angle_to / dihedral_to)
# ------------------------------------------------------------------ #

def test_sdf_parser_bond_to_set(ethanol):
    assert ethanol.atoms[1].bond_to == 1
    assert ethanol.atoms[2].bond_to == 2
    assert ethanol.atoms[3].bond_to == 3



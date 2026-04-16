"""Tests for builders/reconstruct.py."""

import copy
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from molconvert.parsers.pdb_parser import parse_pdb
from molconvert.builders.reconstruct import reconstruct, to_pdb, save_pdb
from molconvert.core.internal_coords import MoleculeIC, AtomIC

MINI_PDB = str(Path(__file__).parent / "data" / "mini.pdb")


# ------------------------------------------------------------------ #
#  Fixtures                                                            #
# ------------------------------------------------------------------ #

@pytest.fixture
def mol_orig():
    return parse_pdb(MINI_PDB)[0]


@pytest.fixture

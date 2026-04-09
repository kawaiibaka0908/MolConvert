"""Tests for parsers/sdf_parser.py."""

from pathlib import Path

import numpy as np
import pytest

from molconvert.parsers.sdf_parser import parse_sdf
from molconvert.builders.reconstruct import reconstruct
from molconvert.analysis import rmsd_molecules

MINI_SDF = str(Path(__file__).parent / "data" / "mini.sdf")


# ------------------------------------------------------------------ #
#  Basic parsing                                                       #
# ------------------------------------------------------------------ #

def test_parse_sdf_returns_list():
    result = parse_sdf(MINI_SDF)
    assert isinstance(result, list)


def test_parse_sdf_molecule_count():
    # mini.sdf contains 2 records: ethanol and acetone
    result = parse_sdf(MINI_SDF)
    assert len(result) == 2


def test_parse_sdf_source_fmt():
    for mol in parse_sdf(MINI_SDF):
        assert mol.source_fmt == "sdf"


def test_parse_sdf_molecule_names_unique():
    mols = parse_sdf(MINI_SDF)
    names = [m.name for m in mols]
    assert len(names) == len(set(names))



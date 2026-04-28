"""Tests for cli/main.py — run_convert and run_rmsd."""

import json
import sys
import tempfile
from pathlib import Path

import pytest

from molconvert.cli.main import run_convert, run_rmsd

MINI_PDB = str(Path(__file__).parent / "data" / "mini.pdb")


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def capture_stdout(func, args, capsys):
    """Run func(args), return (stdout, stderr) strings."""
    func(args)
    captured = capsys.readouterr()
    return captured.out, captured.err


# ------------------------------------------------------------------ #
#  convert — json output (default)                                     #
# ------------------------------------------------------------------ #

def test_convert_json_stdout(capsys):
    run_convert([MINI_PDB])
    out, _ = capsys.readouterr()
    data = json.loads(out)
    assert "atoms" in data
    assert len(data["atoms"]) == 5


def test_convert_json_has_ic_fields(capsys):
    run_convert([MINI_PDB])
    out, _ = capsys.readouterr()
    data = json.loads(out)
    atom3 = data["atoms"][3]
    assert atom3["bond_length"] is not None
    assert atom3["bond_angle"]  is not None
    assert atom3["dihedral"]    is not None



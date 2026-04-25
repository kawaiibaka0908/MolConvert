"""
CLI entry points for molconvert.

Commands
--------
convert  : Convert molecular structure files between formats.
           Supported input  : .pdb, .sdf, .zmat
           Supported output : json (internal IR), pdb, sdf, zmat, internal (alias for json)
rmsd     : RMSD between two PDB files, or a round-trip self-test on one file.

Entry points (setup.py)
-----------------------
    convert  →  molconvert.cli.main:run_convert
    rmsd     →  molconvert.cli.main:run_rmsd
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..parsers.pdb_parser import parse_pdb
from ..parsers.sdf_parser import parse_sdf
from ..builders.reconstruct import reconstruct, to_pdb, save_pdb
from ..builders.to_sdf import molecule_to_sdf, molecules_to_sdf
from ..converters.json_to_zmat import molecule_to_zmat
from ..converters.zmat_to_json import load_zmat
from ..analysis import rmsd_molecules, per_atom_deviation, ic_summary


# ------------------------------------------------------------------ #
#  convert                                                             #
# ------------------------------------------------------------------ #


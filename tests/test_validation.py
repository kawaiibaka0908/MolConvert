"""Tests for analysis/validation.py -- chemical validation module."""

from pathlib import Path

import pytest
import numpy as np

from molconvert.parsers.sdf_parser import parse_sdf
from molconvert.parsers.pdb_parser import parse_pdb
from molconvert.builders.reconstruct import reconstruct
from molconvert.analysis.validation import (
    validate_molecule,
    validate_molecules,
    format_report,
    ValidationReport,
    ValidationIssue,
)
from molconvert.core.internal_coords import MoleculeIC, AtomIC

MINI_SDF = str(Path(__file__).parent / "data" / "mini.sdf")
MINI_PDB = str(Path(__file__).parent / "data" / "mini.pdb")


# ------------------------------------------------------------------ #
#  Fixtures                                                            #
# ------------------------------------------------------------------ #

@pytest.fixture
def ethanol_mol():
    """Parse ethanol from the mini SDF, reconstruct, and return."""
    mols = parse_sdf(MINI_SDF)
    assert len(mols) > 0, "mini.sdf should contain at least one molecule"
    return reconstruct(mols[0])


@pytest.fixture
def collision_mol():
    """A molecule with two carbon atoms at the exact same position."""
    mol = MoleculeIC(name="collision", source_fmt="test")
    mol.atoms.append(AtomIC(
        atom_serial=1, atom_name="C1", residue_name="MOL",
        chain_id="A", residue_seq=1, element="C",
        cart_x=0.0, cart_y=0.0, cart_z=0.0,
    ))
    mol.atoms.append(AtomIC(
        atom_serial=2, atom_name="C2", residue_name="MOL",
        chain_id="A", residue_seq=1, element="C",
        cart_x=0.0, cart_y=0.0, cart_z=0.0,
    ))
    return mol


@pytest.fixture
def pdb_mol():
    """Parse the first chain from mini.pdb."""
    mols = parse_pdb(MINI_PDB)
    assert len(mols) > 0, "mini.pdb should contain at least one molecule"
    return reconstruct(mols[0])


# ------------------------------------------------------------------ #
#  Valid molecule tests                                                 #
# ------------------------------------------------------------------ #

def test_valid_molecule_passes(ethanol_mol):
    report = validate_molecule(ethanol_mol)
    assert report.is_valid is True
    assert len(report.issues) == 0


def test_valid_molecule_has_bonds(ethanol_mol):
    report = validate_molecule(ethanol_mol)
    assert report.bond_count > 0


def test_valid_molecule_bond_orders(ethanol_mol):
    report = validate_molecule(ethanol_mol)
    assert "SINGLE" in report.perceived_bond_orders
    assert report.perceived_bond_orders["SINGLE"] > 0


# ------------------------------------------------------------------ #
#  Atom collision tests                                                #
# ------------------------------------------------------------------ #

def test_atom_collision_detected(collision_mol):
    report = validate_molecule(collision_mol)
    collision_issues = [i for i in report.issues if i.issue_type == "atom_collision"]
    assert len(collision_issues) >= 1


def test_collision_has_error_severity(collision_mol):
    report = validate_molecule(collision_mol)
    collision_issues = [i for i in report.issues if i.issue_type == "atom_collision"]
    assert any(i.severity == "error" for i in collision_issues)


# ------------------------------------------------------------------ #
#  validate_molecules (batch)                                          #
# ------------------------------------------------------------------ #

def test_validate_molecules_list(ethanol_mol, collision_mol):
    reports = validate_molecules([ethanol_mol, collision_mol])
    assert isinstance(reports, list)
    assert len(reports) == 2
    assert all(isinstance(r, ValidationReport) for r in reports)


# ------------------------------------------------------------------ #
#  format_report                                                       #
# ------------------------------------------------------------------ #

def test_format_report_pass(ethanol_mol):
    report = validate_molecule(ethanol_mol)
    text = format_report([report])
    assert "[PASS]" in text


def test_format_report_fail(collision_mol):
    report = validate_molecule(collision_mol)
    text = format_report([report])
    assert "[FAIL]" in text


def test_format_report_verbose(collision_mol):
    report = validate_molecule(collision_mol)
    text = format_report([report], verbose=True)
    assert "Suggestion:" in text


# ------------------------------------------------------------------ #
#  Data-class structure                                                #
# ------------------------------------------------------------------ #

def test_validation_report_structure(ethanol_mol):
    report = validate_molecule(ethanol_mol)
    assert hasattr(report, "molecule_name")
    assert hasattr(report, "is_valid")
    assert hasattr(report, "issues")
    assert hasattr(report, "bond_count")
    assert hasattr(report, "perceived_bond_orders")
    assert isinstance(report.issues, list)
    assert isinstance(report.perceived_bond_orders, dict)


# ------------------------------------------------------------------ #
#  Valence violation detection                                         #
# ------------------------------------------------------------------ #

def test_valence_violation_detected():
    """Create a molecule where a carbon has 5 close neighbours, forcing
    RDKit to perceive 5 bonds and flag a valence violation."""
    mol = MoleculeIC(name="bad_valence", source_fmt="test")
    # Central carbon at origin
    mol.atoms.append(AtomIC(
        atom_serial=1, atom_name="C1", residue_name="MOL",
        chain_id="A", residue_seq=1, element="C",
        cart_x=0.0, cart_y=0.0, cart_z=0.0,
    ))
    # Surround with 5 hydrogens at bonding distance (~1.09 A)
    positions = [
        (1.09, 0.0, 0.0),
        (-1.09, 0.0, 0.0),
        (0.0, 1.09, 0.0),
        (0.0, -1.09, 0.0),
        (0.0, 0.0, 1.09),
    ]
    for i, (x, y, z) in enumerate(positions, start=2):
        mol.atoms.append(AtomIC(
            atom_serial=i, atom_name=f"H{i}",
            residue_name="MOL", chain_id="A", residue_seq=1,
            element="H", cart_x=x, cart_y=y, cart_z=z,
        ))

    report = validate_molecule(mol)
    # Should have at least one valence violation or sanitization failure
    issue_types = [iss.issue_type for iss in report.issues]
    has_valence_issue = (
        "valence_violation" in issue_types
        or "sanitization_failure" in issue_types
    )
    assert has_valence_issue, (
        f"Expected a valence violation, got issues: {report.issues}"
    )


def test_valence_violation_has_suggestion():
    """Valence violation issues should include a non-empty suggestion."""
    mol = MoleculeIC(name="bad_valence2", source_fmt="test")
    mol.atoms.append(AtomIC(
        atom_serial=1, atom_name="C1", residue_name="MOL",
        chain_id="A", residue_seq=1, element="C",
        cart_x=0.0, cart_y=0.0, cart_z=0.0,
    ))
    for i, (x, y, z) in enumerate([
        (1.09, 0.0, 0.0), (-1.09, 0.0, 0.0),
        (0.0, 1.09, 0.0), (0.0, -1.09, 0.0),
        (0.0, 0.0, 1.09),
    ], start=2):
        mol.atoms.append(AtomIC(
            atom_serial=i, atom_name=f"H{i}",
            residue_name="MOL", chain_id="A", residue_seq=1,
            element="H", cart_x=x, cart_y=y, cart_z=z,
        ))

    report = validate_molecule(mol)
    # Every issue must have a non-empty suggestion
    for issue in report.issues:
        assert issue.suggestion, f"Issue missing suggestion: {issue}"


# ------------------------------------------------------------------ #
#  format_report readability                                           #
# ------------------------------------------------------------------ #

def test_format_report_produces_readable_output(ethanol_mol):
    report = validate_molecule(ethanol_mol)
    text = format_report([report])
    # Should be a non-empty string
    assert len(text) > 0
    # Should mention the molecule name
    assert ethanol_mol.name in text


# ------------------------------------------------------------------ #
#  PDB molecule validates                                              #
# ------------------------------------------------------------------ #

def test_pdb_molecule_validates(pdb_mol):
    """Small PDB fragments may not pass RDKit validation (dangling bonds),
    but validate_molecule must return a properly structured report."""
    report = validate_molecule(pdb_mol)
    assert report.molecule_name == pdb_mol.name
    assert isinstance(report.is_valid, bool)
    assert isinstance(report.issues, list)



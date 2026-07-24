"""
Chemical Validation module.

Validates MoleculeIC structures for common chemical problems:
  - Atom collisions (overlapping / unusually close atoms)
  - Valence violations (incorrect bond perception from bad coordinates)
  - RDKit sanitization failures

Each check produces a structured ValidationReport with human-readable
suggestions for corrective action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from rdkit import Chem

from ..core.internal_coords import MoleculeIC
from ..core.rdkit_bridge import molecule_to_rdmol


# ------------------------------------------------------------------ #
#  Data classes                                                        #
# ------------------------------------------------------------------ #

@dataclass
class ValidationIssue:
    """A single problem detected during validation."""

    severity: str       # "error" or "warning"
    atom_index: int     # 0-based index in mol.atoms
    atom_name: str
    element: str
    issue_type: str     # "valence_violation" | "atom_collision" | "sanitization_failure"
    description: str    # Human-readable description
    suggestion: str     # Correction suggestion


@dataclass
class ValidationReport:
    """Aggregated result of all validation checks for one molecule."""

    molecule_name: str
    is_valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    bond_count: int = 0
    perceived_bond_orders: dict[str, int] = field(default_factory=dict)


# ------------------------------------------------------------------ #
#  Public API                                                          #
# ------------------------------------------------------------------ #

def validate_molecule(mol: MoleculeIC) -> ValidationReport:
    """
    Run all validation checks on *mol* and return a ValidationReport.

    Checks performed (in order):
      1. Atom collision detection
      2. Bond perception + valence violation check
      3. Full RDKit sanitization
    """
    report = ValidationReport(molecule_name=mol.name, is_valid=True)

    # -- 1. Atom collision check -----------------------------------------
    _check_collisions(mol, report)

    # -- 2. Bond perception + valence check ------------------------------
    rdmol = _perceive_bonds(mol, report)
    if rdmol is None:
        # molecule_to_rdmol failed; issue already recorded
        report.is_valid = False
        return report

    _check_valence(rdmol, mol, report)

    # -- 3. Sanitization check -------------------------------------------
    _check_sanitization(rdmol, mol, report)

    # Final verdict
    if any(issue.severity == "error" for issue in report.issues):
        report.is_valid = False

    return report


def validate_molecules(mols: list[MoleculeIC]) -> list[ValidationReport]:
    """Validate every molecule in *mols* and return a list of reports."""
    return [validate_molecule(m) for m in mols]


def format_report(reports: list[ValidationReport], verbose: bool = False) -> str:
    """
    Format validation reports as a human-readable string.

    Parameters
    ----------
    reports : list[ValidationReport]
        Reports to format.
    verbose : bool
        If True, list every issue with its suggestion.
    """
    lines: list[str] = []
    for rpt in reports:
        if rpt.is_valid:
            order_parts = [
                f"{count} {kind.lower()}" for kind, count in rpt.perceived_bond_orders.items()
            ]
            order_str = ", ".join(order_parts) if order_parts else "none"
            lines.append(
                f"[PASS] {rpt.molecule_name} "
                f"({rpt.bond_count} bonds: {order_str})"
            )
        else:
            lines.append(
                f"[FAIL] {rpt.molecule_name} - "
                f"{len(rpt.issues)} issue(s) found"
            )

        if verbose:
            for issue in rpt.issues:
                lines.append(
                    f"  [{issue.severity}] {issue.atom_name} ({issue.element}) "
                    f"at index {issue.atom_index}: {issue.description}"
                )
                lines.append(f"    Suggestion: {issue.suggestion}")

    return "\n".join(lines)


# ------------------------------------------------------------------ #
#  Internal helpers                                                    #
# ------------------------------------------------------------------ #

def _check_collisions(mol: MoleculeIC, report: ValidationReport) -> None:
    """Detect atom pairs that are too close together."""
    n = len(mol.atoms)
    for i in range(n):
        ai = mol.atoms[i]
        pos_i = ai.position
        if pos_i is None:
            continue
        for j in range(i + 1, n):
            aj = mol.atoms[j]
            pos_j = aj.position
            if pos_j is None:
                continue

            dist = float(np.linalg.norm(pos_i - pos_j))

            if dist < 0.4:
                report.issues.append(ValidationIssue(
                    severity="error",
                    atom_index=i,
                    atom_name=ai.atom_name,
                    element=ai.element,
                    issue_type="atom_collision",
                    description=(
                        f"Atom collision: {ai.atom_name} and {aj.atom_name} "
                        f"are {dist:.3f} A apart (overlapping)"
                    ),
                    suggestion=(
                        "Remove duplicate atom or fix coordinates"
                    ),
                ))
            elif dist < 0.8 and ai.element != "H" and aj.element != "H":
                report.issues.append(ValidationIssue(
                    severity="warning",
                    atom_index=i,
                    atom_name=ai.atom_name,
                    element=ai.element,
                    issue_type="atom_collision",
                    description=(
                        f"Atoms unusually close: {ai.atom_name} and "
                        f"{aj.atom_name} are {dist:.3f} A apart"
                    ),
                    suggestion=(
                        "Check coordinates; atoms may need to be separated"
                    ),
                ))


def _perceive_bonds(
    mol: MoleculeIC, report: ValidationReport
) -> Optional[Chem.RWMol]:
    """Build an RDKit molecule and populate bond statistics in *report*."""
    try:
        rdmol = molecule_to_rdmol(mol)
    except Exception as exc:
        report.issues.append(ValidationIssue(
            severity="error",
            atom_index=0,
            atom_name=mol.atoms[0].atom_name if mol.atoms else "?",
            element=mol.atoms[0].element if mol.atoms else "?",
            issue_type="sanitization_failure",
            description=f"Failed to build RDKit molecule: {exc}",
            suggestion="Check that all atoms have valid coordinates and elements",
        ))
        return None

    # Bond statistics
    bond_order_counts: dict[str, int] = {}
    for bond in rdmol.GetBonds():
        bt = bond.GetBondType()
        name = str(bt).rsplit(".", maxsplit=1)[-1]  # e.g. "SINGLE"
        bond_order_counts[name] = bond_order_counts.get(name, 0) + 1

    report.bond_count = rdmol.GetNumBonds()
    report.perceived_bond_orders = bond_order_counts

    return rdmol


def _valence_suggestion(element: str, degree: int) -> str:
    """Return a context-specific suggestion for a valence violation."""
    if element == "N" and degree == 4:
        return (
            "Consider assigning +1 formal charge "
            "(quaternary ammonium)"
        )
    if element == "O" and degree == 3:
        return (
            "Consider assigning +1 formal charge (oxonium) "
            "or check for a coordinate error"
        )
    if element == "C" and degree >= 5:
        return (
            "Check for atom collision or incorrect coordinates "
            "causing false bond perception"
        )
    if element == "S" and degree == 3:
        return (
            "May be a sulfonium ion (+1 charge) or need a "
            "double bond (e.g. sulfoxide)"
        )
    return "Atom exceeds maximum allowed valence; check coordinates"


def _check_valence(
    rdmol: Chem.RWMol, mol: MoleculeIC, report: ValidationReport
) -> None:
    """Flag atoms whose perceived valence exceeds chemical expectations."""
    for idx, rdatom in enumerate(rdmol.GetAtoms()):
        try:
            has_violation = rdatom.HasValenceViolation()
        except Exception:
            continue
        if has_violation:
            atom_ic = mol.atoms[idx]
            degree = rdatom.GetTotalDegree()
            element = rdatom.GetSymbol()
            report.issues.append(ValidationIssue(
                severity="error",
                atom_index=idx,
                atom_name=atom_ic.atom_name,
                element=atom_ic.element,
                issue_type="valence_violation",
                description=(
                    f"Valence violation on {atom_ic.atom_name} "
                    f"({element}): total degree = {degree}"
                ),
                suggestion=_valence_suggestion(element, degree),
            ))


def _check_sanitization(
    rdmol: Chem.RWMol, mol: MoleculeIC, report: ValidationReport
) -> None:
    """Run full RDKit sanitization and capture any errors."""
    try:
        Chem.SanitizeMol(rdmol)
    except Exception as exc:
        report.issues.append(ValidationIssue(
            severity="error",
            atom_index=0,
            atom_name=mol.atoms[0].atom_name if mol.atoms else "?",
            element=mol.atoms[0].element if mol.atoms else "?",
            issue_type="sanitization_failure",
            description=f"RDKit sanitization failed: {exc}",
            suggestion="Review molecule coordinates and bond perception",
        ))

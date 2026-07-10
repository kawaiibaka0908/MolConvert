"""
RDKit Bridge — shared utility for MoleculeIC ↔ RDKit conversions.

This module provides two core functions used by multiple downstream components:

1. ``molecule_to_rdmol`` — converts a MoleculeIC (with Cartesian coords) into a
   fully-bonded RDKit ``RWMol`` using ``rdDetermineBonds.DetermineBonds``.

2. ``tripos_atom_type`` — maps an RDKit atom to its Tripos SYBYL atom type
   string (e.g. ``C.3``, ``N.ar``, ``O.co2``).

The IR (MoleculeIC) stores no bond information by design. These functions bridge
that gap at the output / validation stage without modifying the IR itself.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

from .internal_coords import MoleculeIC


# ------------------------------------------------------------------ #
#  MoleculeIC  →  RDKit RWMol                                         #
# ------------------------------------------------------------------ #

def molecule_to_rdmol(mol: MoleculeIC) -> Chem.RWMol:
    """
    Build an RDKit ``RWMol`` from a MoleculeIC's Cartesian coordinates.

    The returned molecule has full bond connectivity and bond orders
    perceived by ``rdDetermineBonds.DetermineBonds``.

    Parameters
    ----------
    mol : MoleculeIC
        Must have ``cart_x``, ``cart_y``, ``cart_z`` set on every atom
        (i.e. after reconstruction or from a parser that stores positions).

    Returns
    -------
    Chem.RWMol
        An editable RDKit molecule with a single 3-D conformer and
        perceived bonds / bond orders.

    Raises
    ------
    ValueError
        If any atom in *mol* lacks Cartesian coordinates.
    """
    rwmol = Chem.RWMol()

    # --- 1. Add atoms ---------------------------------------------------
    for atom in mol.atoms:
        if atom.cart_x is None or atom.cart_y is None or atom.cart_z is None:
            raise ValueError(
                f"Atom {atom.atom_serial} ({atom.atom_name}) has no Cartesian "
                "coordinates. Run reconstruction before calling molecule_to_rdmol."
            )
        rd_atom = Chem.Atom(atom.element)
        rwmol.AddAtom(rd_atom)

    # --- 2. Set 3-D conformer -------------------------------------------
    conf = Chem.Conformer(len(mol.atoms))
    for i, atom in enumerate(mol.atoms):
        conf.SetAtomPosition(i, (atom.cart_x, atom.cart_y, atom.cart_z))
    conf.Set3D(True)
    rwmol.AddConformer(conf, assignId=True)

    # --- 3. Perceive bonds + bond orders --------------------------------
    rdDetermineBonds.DetermineBonds(rwmol)

    return rwmol


# ------------------------------------------------------------------ #
#  Tripos SYBYL atom type mapping                                      #
# ------------------------------------------------------------------ #

def tripos_atom_type(rdatom: Chem.Atom) -> str:
    """
    Map an RDKit atom to its Tripos SYBYL atom type string.

    The mapping uses hybridisation and aromaticity from RDKit, plus a few
    special-case checks for common functional groups (amide nitrogen,
    carboxylate oxygen, etc.).

    Parameters
    ----------
    rdatom : Chem.Atom
        An atom that belongs to a sanitised RDKit molecule.

    Returns
    -------
    str
        Tripos atom type, e.g. ``"C.3"``, ``"N.ar"``, ``"O.co2"``,
        ``"H"``, ``"Fe"``.
    """
    elem = rdatom.GetSymbol()

    # Hydrogen — always just "H" in Tripos
    if elem == "H":
        return "H"

    # Aromatic — only C.ar and N.ar are standard Tripos SYBYL types
    if rdatom.GetIsAromatic() and elem in ("C", "N"):
        return f"{elem}.ar"

    # Special-case: amide nitrogen  (N bonded to a C=O carbon)
    if elem == "N":
        if _is_amide_nitrogen(rdatom):
            return "N.am"

    # Special-case: carboxylate / carboxyl oxygen  (O bonded to C that
    # itself is bonded to another O)
    if elem == "O":
        if _is_carboxylate_oxygen(rdatom):
            return "O.co2"

    # General hybridisation mapping
    hyb = rdatom.GetHybridization()
    hyb_map = {
        Chem.HybridizationType.SP3: f"{elem}.3",
        Chem.HybridizationType.SP2: f"{elem}.2",
        Chem.HybridizationType.SP:  f"{elem}.1",
    }
    if hyb in hyb_map:
        return hyb_map[hyb]

    # Fallback — bare element symbol (metals, noble gases, etc.)
    return elem


# ------------------------------------------------------------------ #
#  Private helpers for special-case detection                          #
# ------------------------------------------------------------------ #

def _is_amide_nitrogen(rdatom: Chem.Atom) -> bool:
    """
    Return True if *rdatom* is an amide nitrogen — i.e. bonded to a carbon
    that in turn has a double bond to an oxygen.
    """
    for bond in rdatom.GetBonds():
        nbr = bond.GetOtherAtom(rdatom)
        if nbr.GetSymbol() != "C":
            continue
        # Check if this carbon has a C=O
        for cbond in nbr.GetBonds():
            if cbond.GetBondType() == Chem.BondType.DOUBLE:
                other = cbond.GetOtherAtom(nbr)
                if other.GetSymbol() == "O":
                    return True
    return False


def _is_carboxylate_oxygen(rdatom: Chem.Atom) -> bool:
    """
    Return True if *rdatom* is a carboxylate oxygen — bonded to a carbon
    that has at least one other oxygen neighbour connected via a double bond.
    """
    for bond in rdatom.GetBonds():
        nbr = bond.GetOtherAtom(rdatom)
        if nbr.GetSymbol() != "C":
            continue
        # Check the carbon's other bonds for a C=O double bond to oxygen
        has_double_o = False
        o_count = 0
        for nbr_bond in nbr.GetBonds():
            other = nbr_bond.GetOtherAtom(nbr)
            if other.GetSymbol() == "O":
                o_count += 1
                if nbr_bond.GetBondType() == Chem.BondType.DOUBLE:
                    has_double_o = True
        if o_count >= 2 and has_double_o:
            return True
    return False

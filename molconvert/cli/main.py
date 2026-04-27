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

def run_convert(argv: list[str] | None = None) -> None:
    """Entry point: convert <INPUT> --to <FORMAT> [options]"""
    parser = argparse.ArgumentParser(
        prog="convert",
        description=(
            "Convert molecular structure files between formats.\n\n"
            "  convert protein.pdb --to zmat\n"
            "  convert molecule.zmat --to pdb\n"
            "  convert molecule.zmat --to internal\n"
            "  convert input.pdb                   (default: json)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", metavar="INPUT",
                        help="Input file (.pdb, .sdf, or .zmat).")
    parser.add_argument("-o", "--output", metavar="PATH", default=None,
                        help="Output file path. Defaults to stdout.")
    parser.add_argument("--to", dest="to_fmt",
                        choices=["json", "pdb", "sdf", "zmat", "internal"],
                        default=None,
                        help="Output format: json, pdb, sdf, zmat, or internal "
                             "(internal is an alias for json). Default: json.")
    # Keep -f/--format as a backward-compatible alias.
    parser.add_argument("-f", "--format", dest="fmt_legacy",
                        choices=["json", "pdb", "sdf", "zmat", "internal"],
                        default=None,
                        help=argparse.SUPPRESS)
    # PDB-specific options (ignored for .sdf / .zmat input)
    parser.add_argument("--model", type=int, default=0, metavar="N",
                        help="MODEL record index to read (default: 0, PDB only).")
    parser.add_argument("--include-hetatm", action="store_true",
                        help="Also process HETATM ligand records (PDB only).")
    parser.add_argument("--chain", metavar="ID", default=None,
                        help="Only output the specified chain, e.g. A (PDB only).")
    # SDF-specific options (ignored for .pdb / .zmat input)
    parser.add_argument("--remove-hydrogens", action="store_true",
                        help="Strip hydrogen atoms before conversion (SDF input only).")
    parser.add_argument("--summary", action="store_true",
                        help="Print IC statistics to stderr after conversion.")

    args = parser.parse_args(argv)

    # Resolve output format: --to takes priority over legacy -f/--format
    out_fmt = args.to_fmt or args.fmt_legacy or "json"
    if out_fmt == "internal":
        out_fmt = "json"

    # Auto-detect input format from file extension
    in_ext = Path(args.input).suffix.lower()
    if in_ext == ".pdb":
        in_fmt = "pdb"
    elif in_ext == ".sdf":
        in_fmt = "sdf"
    elif in_ext == ".zmat":
        in_fmt = "zmat"
    else:
        _die(
            f"Cannot detect input format for '{args.input}'. "
            "Supported extensions: .pdb, .sdf, .zmat"
        )

    # ------------------------------------------------------------------ #
    #  Load molecules                                                      #
    # ------------------------------------------------------------------ #

    if in_fmt == "pdb":
        try:
            molecules = parse_pdb(
                args.input,
                model_id=args.model,
                include_hetatm=args.include_hetatm,
            )
        except Exception as exc:
            _die(f"Failed to parse '{args.input}': {exc}")

        if not molecules:
            _die("No ATOM records found in the file.")

        if args.chain:
            molecules = [m for m in molecules if m.name.endswith(f"_{args.chain}")]
            if not molecules:
                _die(f"Chain '{args.chain}' not found in '{args.input}'.")

    elif in_fmt == "sdf":
        try:
            molecules = parse_sdf(
                args.input,
                remove_hydrogens=args.remove_hydrogens,
            )
        except Exception as exc:
            _die(f"Failed to parse '{args.input}': {exc}")

        if not molecules:
            _die("No valid molecule records found in the file.")

    else:  # zmat
        try:
            mol = load_zmat(args.input)
        except Exception as exc:
            _die(f"Failed to parse '{args.input}': {exc}")
        molecules = [mol]

    # ------------------------------------------------------------------ #
    #  Build output                                                        #
    # ------------------------------------------------------------------ #

    chunks: list[str] = []

    for mol in molecules:
        if out_fmt == "json":
            chunks.append(mol.to_json())

        elif out_fmt in ("pdb", "sdf"):
            # Both PDB and SDF output need full Cartesian positions.
            # reconstruct() is a no-op when positions are already set (PDB/SDF
            # input); it computes positions from IC when they are not (ZMAT input).
            try:
                mol_r = reconstruct(mol)
            except Exception as exc:
                _die(f"Reconstruction failed for '{mol.name}': {exc}")

            if out_fmt == "pdb":
                model_id = None if len(molecules) == 1 else molecules.index(mol) + 1
                chunks.append(to_pdb(mol_r, model_id=model_id))
            else:
                chunks.append(molecule_to_sdf(mol_r))

        elif out_fmt == "zmat":
            chunks.append(molecule_to_zmat(mol))

        if args.summary:
            summary = ic_summary(mol)
            print(f"\n[{mol.name}]", file=sys.stderr)
            print(summary, file=sys.stderr)

    output_text = "\n".join(chunks)

    # ------------------------------------------------------------------ #
    #  Write output                                                        #
    # ------------------------------------------------------------------ #

    if args.output:
        try:
            with open(args.output, "w") as fh:
                fh.write(output_text)
                fh.write("\n")
            print(f"Wrote {args.output}", file=sys.stderr)
        except OSError as exc:
            _die(f"Cannot write to '{args.output}': {exc}")
    else:
        print(output_text)


# ------------------------------------------------------------------ #
#  rmsd                                                                #
# ------------------------------------------------------------------ #


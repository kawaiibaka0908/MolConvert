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

def run_rmsd(argv: list[str] | None = None) -> None:
    """Entry point: rmsd <file1.pdb> <file2.pdb | --self> [options]"""
    parser = argparse.ArgumentParser(
        prog="rmsd",
        description="Compute RMSD between two PDB files.\n"
                    "Use --self for a round-trip reconstruction test on a "
                    "single file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("file1", metavar="FILE1.pdb",
                        help="First PDB file (or the only file with --self).")
    parser.add_argument("file2", metavar="FILE2.pdb", nargs="?", default=None,
                        help="Second PDB file. Omit when using --self.")
    parser.add_argument("--self", dest="self_test", action="store_true",
                        help="Round-trip test: parse FILE1, reconstruct from IC, "
                             "compare reconstructed vs original.")
    parser.add_argument("--filter", metavar="NAMES", default=None,
                        help="Comma-separated atom names to include "
                             "(e.g. 'CA' or 'N,CA,C'). Default: all atoms.")
    parser.add_argument("--per-atom", action="store_true",
                        help="Print per-atom deviation table.")
    parser.add_argument("--no-superpose", action="store_true",
                        help="Skip Kabsch superposition; compare coordinates as-is. "
                             "Useful when structures share the same reference frame "
                             "(e.g. round-trip --self tests).")
    parser.add_argument("--model", type=int, default=0, metavar="N",
                        help="MODEL record index (default: 0).")
    parser.add_argument("--chain", metavar="ID", default=None,
                        help="Restrict to a single chain (e.g. A).")

    args = parser.parse_args(argv)

    # --- Validate argument combinations ---
    if args.self_test and args.file2:
        _die("Provide either --self or FILE2, not both.")
    if not args.self_test and args.file2 is None:
        _die("Provide FILE2.pdb, or use --self for a round-trip test.")

    atom_filter = (
        [n.strip() for n in args.filter.split(",")]
        if args.filter else None
    )

    # --- Load molecule(s) ---
    mol1 = _load_single(args.file1, args.model, args.chain)

    if args.self_test:
        # Round-trip: reconstruct from IC, compare to original
        try:
            mol2 = reconstruct(mol1)
        except Exception as exc:
            _die(f"Reconstruction failed: {exc}")
        label1 = f"{args.file1} (original)"
        label2 = f"{args.file1} (reconstructed)"
    else:
        mol2 = _load_single(args.file2, args.model, args.chain)
        label1 = args.file1
        label2 = args.file2

    superpose = not args.no_superpose

    # --- Compute RMSD ---
    try:
        r = rmsd_molecules(mol1, mol2, atom_filter=atom_filter, superpose=superpose)
    except ValueError as exc:
        _die(str(exc))

    filter_label = f" [{args.filter}]" if args.filter else ""
    method_label = "" if superpose else " (no superposition)"
    print(f"RMSD{filter_label}{method_label}: {r:.4f} Å")
    print(f"  {label1}")
    print(f"  {label2}")

    # --- Per-atom table ---
    if args.per_atom:
        try:
            deviations = per_atom_deviation(mol1, mol2, atom_filter=atom_filter, superpose=superpose)
        except ValueError as exc:
            _die(str(exc))

        print()
        print(f"{'Chain':>5}  {'Res':>4}  {'ResSeq':>6}  {'Atom':>4}  {'Dev (Å)':>8}")
        print("-" * 36)
        for rec in deviations:
            print(
                f"{rec['chain_id']:>5}  "
                f"{rec['residue_name']:>4}  "
                f"{rec['residue_seq']:>6}  "
                f"{rec['atom_name']:>4}  "
                f"{rec['deviation']:>8.4f}"
            )
        print("-" * 36)
        max_dev = max(r["deviation"] for r in deviations)
        print(f"Max deviation: {max_dev:.4f} Å")


# ------------------------------------------------------------------ #
#  Shared helpers                                                      #
# ------------------------------------------------------------------ #

def _load_single(path: str, model_id: int, chain: str | None):
    """Parse a PDB and return the first (or chain-filtered) MoleculeIC."""
    try:
        molecules = parse_pdb(path, model_id=model_id)
    except Exception as exc:
        _die(f"Failed to parse '{path}': {exc}")

    if not molecules:
        _die(f"No ATOM records found in '{path}'.")

    if chain:
        molecules = [m for m in molecules if m.name.endswith(f"_{chain}")]
        if not molecules:
            _die(f"Chain '{chain}' not found in '{path}'.")

    if len(molecules) > 1 and not chain:
        chains = ", ".join(m.name.split("_")[-1] for m in molecules)
        print(
            f"[info] '{path}' has multiple chains ({chains}); "
            "using the first. Use --chain to select one.",
            file=sys.stderr,
        )

    return molecules[0]



"""
CLI entry points for molconvert.

Commands
--------
convert  : Convert molecular structure files between formats.
           Supported input  : .pdb, .sdf, .zmat, .log, .out, .mol2
           Supported output : json, pdb, sdf, zmat, internal, mol2, xyz, gro, gjf, inp
rmsd     : RMSD between two PDB files, or a round-trip self-test on one file.

Entry points (setup.py)
-----------------------
    convert  ->  molconvert.cli.main:run_convert
    rmsd     ->  molconvert.cli.main:run_rmsd
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..parsers.pdb_parser import parse_pdb
from ..parsers.sdf_parser import parse_sdf
from ..parsers.gaussian_parser import parse_gaussian
from ..parsers.gamess_parser import parse_gamess
from ..parsers.mol2_parser import parse_mol2
from ..builders.reconstruct import reconstruct, to_pdb, save_pdb
from ..builders.to_sdf import molecule_to_sdf, molecules_to_sdf
from ..builders.to_mol2 import molecule_to_mol2
from ..builders.to_xyz import molecule_to_xyz
from ..builders.to_gro import molecule_to_gro
from ..builders.to_gaussian import molecule_to_gaussian
from ..builders.to_gamess import molecule_to_gamess
from ..converters.json_to_zmat import molecule_to_zmat
from ..converters.zmat_to_json import load_zmat
from ..analysis import rmsd_molecules, per_atom_deviation, ic_summary
from ..analysis.validation import validate_molecules, format_report


# Supported input extensions and the formats they map to.
_INPUT_EXT_MAP = {
    ".pdb": "pdb",
    ".sdf": "sdf",
    ".zmat": "zmat",
    ".log": "gaussian",
    ".mol2": "mol2",
    # .out is ambiguous (Gaussian or GAMESS) — resolved via --input-format
}

# Supported output format tokens.
_OUTPUT_FORMATS = [
    "json", "pdb", "sdf", "zmat", "internal",
    "mol2", "xyz", "gro", "gjf", "inp",
]

# Map output-format token to file extension for batch mode.
_FMT_TO_EXT = {
    "json": ".json",
    "pdb": ".pdb",
    "sdf": ".sdf",
    "zmat": ".zmat",
    "mol2": ".mol2",
    "xyz": ".xyz",
    "gro": ".gro",
    "gjf": ".gjf",
    "inp": ".inp",
}

# Extensions we can auto-detect when globbing a directory for batch input.
_BATCH_EXTENSIONS = {".pdb", ".sdf", ".zmat", ".log", ".out", ".mol2"}


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
            "  convert molecule.mol2 --to sdf\n"
            "  convert calc.log --to pdb\n"
            "  convert calc.out --input-format gamess --to xyz\n"
            "  convert data_dir/ --to mol2 -o output_dir/\n"
            "  convert molecule.pdb --to mol2 --validate\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", metavar="INPUT",
                        help="Input file or directory (batch mode).")
    parser.add_argument("-o", "--output", metavar="PATH", default=None,
                        help="Output file path (single file) or directory (batch).")
    parser.add_argument("--to", dest="to_fmt",
                        choices=_OUTPUT_FORMATS,
                        default=None,
                        help="Output format. Default: json.")
    # Keep -f/--format as a backward-compatible alias.
    parser.add_argument("-f", "--format", dest="fmt_legacy",
                        choices=["json", "pdb", "sdf", "zmat", "internal"],
                        default=None,
                        help=argparse.SUPPRESS)

    # --- Input format disambiguation ---
    parser.add_argument("--input-format", dest="input_fmt",
                        choices=["gaussian", "gamess"],
                        default=None,
                        help="Explicitly specify input format for .out files.")
    parser.add_argument("--step", default="last",
                        help="Geometry step to extract (gaussian/gamess): "
                             "'last', 'all', or integer N (0-indexed).")

    # --- PDB-specific options ---
    parser.add_argument("--model", type=int, default=0, metavar="N",
                        help="MODEL record index to read (default: 0, PDB only).")
    parser.add_argument("--include-hetatm", action="store_true",
                        help="Also process HETATM ligand records (PDB only).")
    parser.add_argument("--chain", metavar="ID", default=None,
                        help="Only output the specified chain, e.g. A (PDB only).")

    # --- SDF-specific options ---
    parser.add_argument("--remove-hydrogens", action="store_true",
                        help="Strip hydrogen atoms before conversion (SDF input only).")

    # --- Gaussian / GAMESS input builder options ---
    parser.add_argument("--charge", type=int, default=0,
                        help="Molecular charge (gjf/inp output only). Default: 0.")
    parser.add_argument("--multiplicity", type=int, default=1,
                        help="Spin multiplicity (gjf/inp output only). Default: 1.")
    parser.add_argument("--route", default="# HF/6-31G(d) opt",
                        help="Gaussian route section (gjf output only).")

    # --- Batch mode ---
    parser.add_argument("--recursive", action="store_true",
                        help="Recurse into subdirectories in batch mode.")

    # --- Validation ---
    parser.add_argument("--validate", action="store_true",
                        help="Run chemical validation after conversion.")
    parser.add_argument("--report", metavar="PATH", default=None,
                        help="Write detailed validation report to file.")

    # --- Summary ---
    parser.add_argument("--summary", action="store_true",
                        help="Print IC statistics to stderr after conversion.")

    args = parser.parse_args(argv)

    # Resolve output format: --to takes priority over legacy -f/--format
    out_fmt = args.to_fmt or args.fmt_legacy or "json"
    if out_fmt == "internal":
        out_fmt = "json"

    # ------------------------------------------------------------------ #
    #  Batch mode: input is a directory                                    #
    # ------------------------------------------------------------------ #
    input_path = Path(args.input)
    if input_path.is_dir():
        _run_batch(args, input_path, out_fmt)
        return

    # ------------------------------------------------------------------ #
    #  Single file mode                                                    #
    # ------------------------------------------------------------------ #

    # Detect input format
    in_fmt = _detect_input_format(args.input, args.input_fmt)

    # Load molecules
    molecules = _load_molecules(args, in_fmt)

    # Build output and optionally validate
    _convert_and_write(args, molecules, out_fmt)


# ------------------------------------------------------------------ #
#  Batch mode                                                          #
# ------------------------------------------------------------------ #

def _run_batch(args, input_dir: Path, out_fmt: str) -> None:
    """Process all supported files in *input_dir*."""
    # Determine output directory
    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = input_dir

    # Collect input files
    glob_fn = input_dir.rglob if args.recursive else input_dir.glob
    input_files = sorted(
        f for f in glob_fn("*")
        if f.is_file() and f.suffix.lower() in _BATCH_EXTENSIONS
    )

    if not input_files:
        _die(f"No supported files found in '{input_dir}'.")

    all_reports = []
    converted = 0
    failed = 0

    for fpath in input_files:
        try:
            in_fmt = _detect_input_format(str(fpath), args.input_fmt)
            molecules = _load_molecules_from_path(fpath, in_fmt, args)

            # Build output text
            out_text = _build_output(molecules, out_fmt, args)

            # Determine output file path
            out_ext = _FMT_TO_EXT.get(out_fmt, f".{out_fmt}")
            out_file = out_dir / (fpath.stem + out_ext)
            out_file.write_text(out_text + "\n", encoding="utf-8")

            converted += 1
            print(f"  [OK] {fpath.name} -> {out_file.name}", file=sys.stderr)

            # Validation
            if args.validate:
                from ..analysis.validation import validate_molecules, format_report
                reports = validate_molecules(
                    [reconstruct(m) for m in molecules]
                )
                all_reports.extend(reports)

        except SystemExit as exc:
            # _die() raises SystemExit(1) — in batch mode, skip the file.
            # Only count non-zero exits as failures.
            if exc.code != 0:
                failed += 1
        except Exception as exc:
            failed += 1
            print(f"  [FAIL] {fpath.name}: {exc}", file=sys.stderr)

    # Summary
    print(
        f"\nBatch complete: {converted} converted, {failed} failed "
        f"(out of {len(input_files)} files).",
        file=sys.stderr,
    )

    # Validation report
    if args.validate and all_reports:
        summary = format_report(all_reports, verbose=True)
        print("\n--- Validation Report ---", file=sys.stderr)
        print(summary, file=sys.stderr)
        if args.report:
            Path(args.report).write_text(summary + "\n", encoding="utf-8")
            print(f"Report written to {args.report}", file=sys.stderr)


# ------------------------------------------------------------------ #
#  Single-file conversion core                                         #
# ------------------------------------------------------------------ #

def _convert_and_write(args, molecules, out_fmt: str) -> None:
    """Build output string from molecules, write to file/stdout, validate."""
    output_text = _build_output(molecules, out_fmt, args)

    # Write output
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

    # IC summary
    if args.summary:
        for mol in molecules:
            summary = ic_summary(mol)
            print(f"\n[{mol.name}]", file=sys.stderr)
            print(summary, file=sys.stderr)

    # Validation
    if args.validate:
        reconstructed = []
        for mol in molecules:
            try:
                reconstructed.append(reconstruct(mol))
            except Exception:
                reconstructed.append(mol)  # use as-is if reconstruct fails

        reports = validate_molecules(reconstructed)
        summary_text = format_report(reports, verbose=True)
        print("\n--- Validation Report ---", file=sys.stderr)
        print(summary_text, file=sys.stderr)

        if args.report:
            Path(args.report).write_text(summary_text + "\n", encoding="utf-8")
            print(f"Report written to {args.report}", file=sys.stderr)


def _build_output(molecules, out_fmt: str, args) -> str:
    """Build the output text for a list of molecules in the given format."""
    chunks: list[str] = []

    for mol in molecules:
        if out_fmt == "json":
            chunks.append(mol.to_json())

        elif out_fmt == "zmat":
            chunks.append(molecule_to_zmat(mol))

        else:
            # All other formats need Cartesian positions.
            try:
                mol_r = reconstruct(mol)
            except Exception as exc:
                _die(f"Reconstruction failed for '{mol.name}': {exc}")

            if out_fmt == "pdb":
                model_id = None if len(molecules) == 1 else molecules.index(mol) + 1
                chunks.append(to_pdb(mol_r, model_id=model_id))
            elif out_fmt == "sdf":
                chunks.append(molecule_to_sdf(mol_r))
            elif out_fmt == "mol2":
                chunks.append(molecule_to_mol2(mol_r))
            elif out_fmt == "xyz":
                chunks.append(molecule_to_xyz(mol_r))
            elif out_fmt == "gro":
                chunks.append(molecule_to_gro(mol_r))
            elif out_fmt == "gjf":
                chunks.append(molecule_to_gaussian(
                    mol_r,
                    route=args.route,
                    charge=args.charge,
                    multiplicity=args.multiplicity,
                ))
            elif out_fmt == "inp":
                chunks.append(molecule_to_gamess(
                    mol_r,
                    charge=args.charge,
                    multiplicity=args.multiplicity,
                ))

    return "\n".join(chunks)


# ------------------------------------------------------------------ #
#  Input format detection                                              #
# ------------------------------------------------------------------ #

def _detect_input_format(filepath: str, explicit_fmt: str | None) -> str:
    """Resolve the input format from the file extension and --input-format."""
    ext = Path(filepath).suffix.lower()

    if ext == ".out":
        if explicit_fmt:
            return explicit_fmt
        _die(
            f"Ambiguous '.out' extension for '{filepath}'. "
            "Use --input-format gaussian or --input-format gamess."
        )

    if ext in _INPUT_EXT_MAP:
        return _INPUT_EXT_MAP[ext]

    if explicit_fmt:
        return explicit_fmt

    _die(
        f"Cannot detect input format for '{filepath}'. "
        "Supported extensions: .pdb, .sdf, .zmat, .log, .out, .mol2. "
        "Use --input-format for ambiguous extensions."
    )


# ------------------------------------------------------------------ #
#  Molecule loading                                                    #
# ------------------------------------------------------------------ #

def _load_molecules(args, in_fmt: str) -> list:
    """Load molecules from a single file based on parsed CLI args."""
    return _load_molecules_from_path(Path(args.input), in_fmt, args)


def _load_molecules_from_path(fpath: Path, in_fmt: str, args) -> list:
    """Load molecules from *fpath* using format *in_fmt*."""
    filepath = str(fpath)

    if in_fmt == "pdb":
        try:
            molecules = parse_pdb(
                filepath,
                model_id=getattr(args, "model", 0),
                include_hetatm=getattr(args, "include_hetatm", False),
            )
        except Exception as exc:
            _die(f"Failed to parse '{filepath}': {exc}")

        if not molecules:
            _die(f"No ATOM records found in '{filepath}'.")

        chain = getattr(args, "chain", None)
        if chain:
            molecules = [m for m in molecules if m.name.endswith(f"_{chain}")]
            if not molecules:
                _die(f"Chain '{chain}' not found in '{filepath}'.")

        return molecules

    elif in_fmt == "sdf":
        try:
            molecules = parse_sdf(
                filepath,
                remove_hydrogens=getattr(args, "remove_hydrogens", False),
            )
        except Exception as exc:
            _die(f"Failed to parse '{filepath}': {exc}")

        if not molecules:
            _die(f"No valid molecule records found in '{filepath}'.")
        return molecules

    elif in_fmt == "zmat":
        try:
            mol = load_zmat(filepath)
        except Exception as exc:
            _die(f"Failed to parse '{filepath}': {exc}")
        return [mol]

    elif in_fmt == "gaussian":
        step = getattr(args, "step", "last")
        try:
            molecules = parse_gaussian(filepath, step=step)
        except Exception as exc:
            _die(f"Failed to parse '{filepath}': {exc}")

        if not molecules:
            _die(f"No geometry blocks found in '{filepath}'.")
        return molecules

    elif in_fmt == "gamess":
        step = getattr(args, "step", "last")
        try:
            molecules = parse_gamess(filepath, step=step)
        except Exception as exc:
            _die(f"Failed to parse '{filepath}': {exc}")

        if not molecules:
            _die(f"No geometry blocks found in '{filepath}'.")
        return molecules

    elif in_fmt == "mol2":
        try:
            molecules = parse_mol2(filepath)
        except Exception as exc:
            _die(f"Failed to parse '{filepath}': {exc}")

        if not molecules:
            _die(f"No molecules found in '{filepath}'.")
        return molecules

    else:
        _die(f"Unsupported input format: '{in_fmt}'.")


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
    print(f"RMSD{filter_label}{method_label}: {r:.4f} \u00c5")
    print(f"  {label1}")
    print(f"  {label2}")

    # --- Per-atom table ---
    if args.per_atom:
        try:
            deviations = per_atom_deviation(mol1, mol2, atom_filter=atom_filter, superpose=superpose)
        except ValueError as exc:
            _die(str(exc))

        print()
        print(f"{'Chain':>5}  {'Res':>4}  {'ResSeq':>6}  {'Atom':>4}  {'Dev (\u00c5)':>8}")
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
        print(f"Max deviation: {max_dev:.4f} \u00c5")


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


def _die(message: str) -> None:
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(1)

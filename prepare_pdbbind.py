"""Create a PertaBind manifest from a licensed PDBbind installation.

This script does not download or redistribute PDBbind. It parses Kd/Ki/IC50 values from
an INDEX file and converts molar measurements to pK = -log10(K[M]), as described in the
manuscript. Inequality-qualified measurements are skipped by default.
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd


AFFINITY = re.compile(
    r"(?P<kind>Kd|Ki|IC50)\s*=\s*(?P<relation>[<>~]?)\s*(?P<value>[0-9.eE+-]+)\s*"
    r"(?P<unit>fM|pM|nM|uM|µM|mM|M)\b"
)
UNIT_TO_MOLAR = {"fM": 1e-15, "pM": 1e-12, "nM": 1e-9,
                 "uM": 1e-6, "µM": 1e-6, "mM": 1e-3, "M": 1.0}


def parse_index(path: Path, allow_inequalities: bool = False) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            pdb_id = line.split()[0].lower()
            match = AFFINITY.search(line)
            if not match:
                continue
            relation = match.group("relation")
            if relation in ("<", ">") and not allow_inequalities:
                continue
            molar = float(match.group("value")) * UNIT_TO_MOLAR[match.group("unit")]
            if not math.isfinite(molar) or molar <= 0:
                continue
            rows.append({
                "sample_id": pdb_id,
                "pdb_id": pdb_id,
                "label": -math.log10(molar),
                "measurement_type": match.group("kind"),
                "measurement_relation": relation,
                "measurement_molar": molar,
                "source_line": line_number,
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-file", required=True)
    parser.add_argument("--pdbbind-root", required=True)
    parser.add_argument("--apo-dir", required=True,
                        help="Directory containing ESMFold outputs named <PDB_ID>.pdb")
    parser.add_argument("--smiles-csv", required=True,
                        help="CSV with pdb_id,smiles; do not infer chemistry from coordinates")
    parser.add_argument("--membership",
                        help="Optional published membership CSV with pdb_id and split/fold")
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-inequalities", action="store_true")
    parser.add_argument("--keep-missing", action="store_true")
    args = parser.parse_args()

    frame = parse_index(Path(args.index_file), args.allow_inequalities)
    smiles = pd.read_csv(args.smiles_csv)
    if not {"pdb_id", "smiles"}.issubset(smiles):
        raise ValueError("--smiles-csv must contain pdb_id,smiles")
    smiles["pdb_id"] = smiles["pdb_id"].astype(str).str.lower()
    frame = frame.merge(smiles[["pdb_id", "smiles"]], on="pdb_id", how="left")
    if args.membership:
        membership = pd.read_csv(args.membership)
        membership["pdb_id"] = membership["pdb_id"].astype(str).str.lower()
        frame = frame.merge(membership, on="pdb_id", how="left")
    root, apo_dir = Path(args.pdbbind_root).resolve(), Path(args.apo_dir).resolve()
    frame["apo_pdb"] = frame["pdb_id"].map(lambda value: str(apo_dir / f"{value}.pdb"))
    frame["holo_pdb"] = frame["pdb_id"].map(
        lambda value: str(root / value / f"{value}_protein.pdb")
    )
    frame["bound_ligand"] = frame["pdb_id"].map(
        lambda value: str(root / value / f"{value}_ligand.sdf")
    )
    frame["sequence"] = ""
    frame["task"] = "affinity"
    frame["source"] = "PDBbind"
    required_paths = ["apo_pdb", "holo_pdb", "bound_ligand"]
    frame["files_present"] = frame[required_paths].applymap(lambda value: Path(value).exists()).all(axis=1)
    valid = frame["smiles"].notna() & frame["files_present"]
    if not args.keep_missing:
        frame = frame[valid].copy()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    print(f"Wrote {len(frame)} rows ({int(valid.sum())} complete) to {destination}")


if __name__ == "__main__":
    main()


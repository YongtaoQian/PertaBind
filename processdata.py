"""Build PertaBind tensor bundles from structures and a CSV manifest.

The preprocessor deliberately stores simple tensors and provenance rather than pickled
RDKit objects, keeping caches stable across RDKit releases.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import AllChem, ChemicalFeatures
from rdkit.Chem.rdchem import HybridizationType
from rdkit import RDConfig
from tqdm import tqdm

from utils import package_versions, save_json, sha256_file

RDLogger.DisableLog("rdApp.warning")

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "C", "PYL": "K",
}
AA_ORDER = "ACDEFGHIKLMNPQRSTVWYX"
ELEMENTS = ("C", "N", "O", "S", "P", "F", "CL", "BR", "I", "B", "SE")
HYBRIDIZATIONS = (
    HybridizationType.SP, HybridizationType.SP2, HybridizationType.SP3,
    HybridizationType.SP3D, HybridizationType.SP3D2,
)
BACKBONE = {"N", "CA", "C", "O", "OXT"}
HYDROPHOBIC_AA = set("AVILMFWYPC")


def one_hot(value, vocabulary: Sequence, unknown: bool = True) -> List[float]:
    size = len(vocabulary) + int(unknown)
    result = [0.0] * size
    try:
        result[vocabulary.index(value)] = 1.0
    except ValueError:
        if unknown:
            result[-1] = 1.0
    return result


def tokenize_sequence(sequence: str) -> torch.Tensor:
    mapping = {aa: i + 1 for i, aa in enumerate(AA_ORDER)}
    sequence = re.sub(r"[^A-Za-z]", "", sequence).upper()
    return torch.tensor([mapping.get(aa, mapping["X"]) for aa in sequence], dtype=torch.long)


def parse_pdb(path: str | Path) -> Tuple[List[Dict], List[Dict]]:
    """Parse the first PDB model without a BioPython dependency."""
    atoms: List[Dict] = []
    seen = set()
    in_first_model = True
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            record = line[:6].strip()
            if record == "MODEL":
                model = int(line[10:14].strip() or "1")
                in_first_model = model == 1
                continue
            if record == "ENDMDL" and in_first_model:
                break
            if not in_first_model or record != "ATOM":
                continue
            altloc = line[16:17]
            if altloc not in (" ", "A", "1"):
                continue
            atom_name = line[12:16].strip()
            resname = line[17:20].strip().upper()
            chain = line[21:22].strip() or "_"
            resseq = line[22:26].strip()
            insertion = line[26:27].strip()
            key = (chain, resseq, insertion, atom_name)
            if key in seen:
                continue
            seen.add(key)
            try:
                coord = np.array([float(line[30:38]), float(line[38:46]),
                                  float(line[46:54])], dtype=np.float32)
                occupancy = float(line[54:60].strip() or 1.0)
                bfactor = float(line[60:66].strip() or 0.0)
            except ValueError:
                continue
            element = line[76:78].strip().upper()
            if not element:
                element = re.sub(r"[^A-Za-z]", "", atom_name)[:1].upper()
            atoms.append({
                "name": atom_name, "resname": resname, "aa": AA3_TO_1.get(resname, "X"),
                "chain": chain, "resseq": resseq, "insertion": insertion,
                "coord": coord, "occupancy": occupancy, "bfactor": bfactor,
                "element": element,
            })
    if not atoms:
        raise ValueError(f"No protein ATOM records found in {path}")
    residues: List[Dict] = []
    residue_map: Dict[Tuple[str, str, str], int] = {}
    for atom in atoms:
        key = (atom["chain"], atom["resseq"], atom["insertion"])
        if key not in residue_map:
            residue_map[key] = len(residues)
            residues.append({"key": key, "aa": atom["aa"], "atoms": []})
        residue_index = residue_map[key]
        atom["residue_index"] = residue_index
        residues[residue_index]["atoms"].append(atom)
    return atoms, residues


def residue_graph(path: str | Path) -> Tuple[Dict, str, List[Dict]]:
    atoms, residues = parse_pdb(path)
    features, positions = [], []
    for residue in residues:
        named = {a["name"]: a for a in residue["atoms"]}
        heavy = [a for a in residue["atoms"] if a["element"] != "H"]
        if not heavy:
            continue
        center = named.get("CA", heavy[0])["coord"]
        aa = residue["aa"]
        bfactor = float(np.mean([a["bfactor"] for a in heavy])) / 100.0
        occupancy = float(np.mean([a["occupancy"] for a in heavy]))
        backbone_complete = float(all(name in named for name in ("N", "CA", "C", "O")))
        # Secondary structure, SASA and depth were named in the paper but their exact
        # calculation was not. Slots are retained as zero-valued, replaceable descriptors.
        feature = one_hot(aa, list(AA_ORDER), unknown=False)
        feature += [backbone_complete, bfactor, occupancy, 0.0, 0.0, 0.0, 1.0, bfactor]
        features.append(feature)
        positions.append(center)
    sequence = "".join(residue["aa"] for residue in residues)
    graph = {
        "x": torch.tensor(np.asarray(features), dtype=torch.float32),
        "pos": torch.tensor(np.asarray(positions), dtype=torch.float32),
        "edge_index": torch.empty((2, 0), dtype=torch.long),
    }
    return graph, sequence, residues


def normalize_smiles(smiles: str) -> Tuple[str, Chem.Mol]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    Chem.SanitizeMol(mol)
    normalized = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    return normalized, mol


def generate_conformer(mol: Chem.Mol, seed: int) -> Chem.Mol:
    mol = Chem.AddHs(Chem.Mol(mol))
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = int(seed) & 0x7FFFFFFF
    parameters.useRandomCoords = True
    status = AllChem.EmbedMolecule(mol, parameters)
    if status != 0:
        raise ValueError("RDKit ETKDG failed to generate a conformer")
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        else:
            AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass
    return Chem.RemoveHs(mol)


def load_bound_molecule(path: str | Path) -> Chem.Mol:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".sdf", ".sd"):
        supplier = Chem.SDMolSupplier(str(path), removeHs=True, sanitize=True)
        mol = next((m for m in supplier if m is not None), None)
    elif suffix == ".mol2":
        mol = Chem.MolFromMol2File(str(path), removeHs=True, sanitize=True)
    elif suffix in (".mol", ".mdl"):
        mol = Chem.MolFromMolFile(str(path), removeHs=True, sanitize=True)
    elif suffix in (".pdb", ".ent"):
        mol = Chem.MolFromPDBFile(str(path), removeHs=True, sanitize=True)
    else:
        raise ValueError(f"Unsupported ligand format: {path}")
    if mol is None or not mol.GetNumConformers():
        raise ValueError(f"Could not read 3D ligand: {path}")
    return mol


def donor_acceptor_sets(mol: Chem.Mol) -> Tuple[set, set]:
    factory = ChemicalFeatures.BuildFeatureFactory(str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"))
    donors, acceptors = set(), set()
    for feature in factory.GetFeaturesForMol(mol):
        if feature.GetFamily() == "Donor":
            donors.update(feature.GetAtomIds())
        elif feature.GetFamily() == "Acceptor":
            acceptors.update(feature.GetAtomIds())
    return donors, acceptors


def chemical_atom_feature(atom: Chem.Atom, donors: set, acceptors: set) -> List[float]:
    symbol = atom.GetSymbol().upper()
    feature = one_hot(symbol, list(ELEMENTS), unknown=True)
    feature += one_hot(min(atom.GetTotalDegree(), 5), list(range(6)), unknown=False)
    feature += [
        float(atom.GetFormalCharge()) / 3.0,
        float(atom.GetIsAromatic()),
        float(atom.GetIdx() in donors),
        float(atom.GetIdx() in acceptors),
        float(atom.IsInRing()),
    ]
    feature += one_hot(atom.GetHybridization(), list(HYBRIDIZATIONS), unknown=True)
    return feature  # 29 dimensions


def ligand_graph(mol: Chem.Mol) -> Dict:
    donors, acceptors = donor_acceptor_sets(mol)
    conf = mol.GetConformer()
    x, pos = [], []
    for atom in mol.GetAtoms():
        feature = chemical_atom_feature(atom, donors, acceptors) + [0.0] * 6
        point = conf.GetAtomPosition(atom.GetIdx())
        x.append(feature)
        pos.append([point.x, point.y, point.z])
    edges, edge_features = [], []
    bond_types = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE,
                  Chem.BondType.AROMATIC]
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = one_hot(bond.GetBondType(), bond_types, unknown=True)
        bf += [float(bond.GetIsConjugated()), float(bond.IsInRing())]
        edges.extend([(i, j), (j, i)])
        edge_features.extend([bf, bf])
    edge_index = (torch.tensor(edges, dtype=torch.long).t().contiguous()
                  if edges else torch.empty((2, 0), dtype=torch.long))
    return {
        "x": torch.tensor(x, dtype=torch.float32),
        "pos": torch.tensor(pos, dtype=torch.float32),
        "edge_index": edge_index,
        "edge_attr": torch.tensor(edge_features, dtype=torch.float32) if edge_features
                     else torch.empty((0, 7), dtype=torch.float32),
    }


def protein_atom_feature(atom: Dict, shell_index: int) -> List[float]:
    feature = one_hot(atom["element"], list(ELEMENTS), unknown=True)
    # Pad to the same 29-dimensional chemical block used for ligand atoms.
    feature += [0.0] * (29 - len(feature))
    feature += [
        1.0,
        float(atom["name"] in BACKBONE),
        float(atom["aa"] in HYDROPHOBIC_AA),
    ]
    feature += one_hot(shell_index, [0, 1, 2], unknown=False)
    return feature


def pocket_graph(holo_pdb: str | Path, bound_mol: Chem.Mol,
                 shells: Sequence[float]) -> Tuple[Dict, Dict]:
    protein_atoms, residues = parse_pdb(holo_pdb)
    conf = bound_mol.GetConformer()
    ligand_positions = np.asarray([
        list(conf.GetAtomPosition(i)) for i in range(bound_mol.GetNumAtoms())
    ], dtype=np.float32)
    residue_shell = {}
    retained = []
    for old_index, residue in enumerate(residues):
        coords = np.asarray([a["coord"] for a in residue["atoms"] if a["element"] != "H"])
        if not len(coords):
            continue
        distance = float(np.linalg.norm(coords[:, None, :] - ligand_positions[None, :, :], axis=-1).min())
        if distance <= shells[-1]:
            shell = 0 if distance <= shells[0] else (1 if distance <= shells[1] else 2)
            residue_shell[old_index] = (len(retained), shell, distance)
            retained.append(residue)
    if not retained:
        raise ValueError("No holo residues found within the 8-Angstrom pocket")

    x, pos, node_type, residue_index, shell_values = [], [], [], [], []
    for atom in protein_atoms:
        if atom["element"] == "H" or atom["residue_index"] not in residue_shell:
            continue
        new_residue, shell, _ = residue_shell[atom["residue_index"]]
        x.append(protein_atom_feature(atom, shell))
        pos.append(atom["coord"])
        node_type.append(0)
        residue_index.append(new_residue)
        shell_values.append(shell)
    donors, acceptors = donor_acceptor_sets(bound_mol)
    for atom in bound_mol.GetAtoms():
        x.append(chemical_atom_feature(atom, donors, acceptors) + [0.0] * 6)
        point = conf.GetAtomPosition(atom.GetIdx())
        pos.append([point.x, point.y, point.z])
        node_type.append(1)
        residue_index.append(-1)
        shell_values.append(-1)
    graph = {
        "x": torch.tensor(np.asarray(x), dtype=torch.float32),
        "pos": torch.tensor(np.asarray(pos), dtype=torch.float32),
        "edge_index": torch.empty((2, 0), dtype=torch.long),
        "node_type": torch.tensor(node_type, dtype=torch.long),
        "residue_index": torch.tensor(residue_index, dtype=torch.long),
        "shell": torch.tensor(shell_values, dtype=torch.long),
    }
    residue_records = [None] * len(retained)
    for old_index, (new_index, shell, distance) in residue_shell.items():
        residue = residues[old_index]
        chain, resseq, insertion = residue["key"]
        residue_records[new_index] = {
            "pocket_residue_index": new_index,
            "chain": chain,
            "resseq": resseq,
            "insertion": insertion,
            "amino_acid": residue["aa"],
            "shell": shell,
            "minimum_ligand_distance": distance,
        }
    meta = {
        "num_pocket_residues": len(retained),
        "shell_counts": {str(k): sum(v[1] == k for v in residue_shell.values()) for k in range(3)},
        "residues": residue_records,
    }
    return graph, meta


def resolve_input(value, manifest_dir: Path) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)) or not str(value).strip():
        return None
    path = Path(str(value))
    return str(path if path.is_absolute() else (manifest_dir / path).resolve())


def process_row(record: Dict, manifest_dir: str, output_dir: str, seed: int,
                shells: Sequence[float], overwrite: bool = False) -> Dict:
    manifest_dir = Path(manifest_dir)
    output_dir = Path(output_dir)
    sample_id = str(record["sample_id"])
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id)
    destination = output_dir / "samples" / f"{safe_id}.pt"
    if destination.exists() and not overwrite:
        return {**record, "processed_path": str(destination.relative_to(output_dir)), "status": "cached"}

    apo_path = resolve_input(record.get("apo_pdb"), manifest_dir)
    holo_path = resolve_input(record.get("holo_pdb"), manifest_dir)
    ligand_path = resolve_input(record.get("bound_ligand"), manifest_dir)
    if not apo_path:
        raise ValueError(f"{sample_id}: apo_pdb is required")
    normalized_smiles, base_mol = normalize_smiles(record["smiles"])
    sample_seed = seed ^ int.from_bytes(sample_id.encode("utf-8")[:4].ljust(4, b"\0"), "little")
    free_mol = generate_conformer(base_mol, sample_seed)
    apo, pdb_sequence, _ = residue_graph(apo_path)
    sequence = str(record.get("sequence", "") or "").strip().upper() or pdb_sequence
    if not sequence:
        raise ValueError(f"{sample_id}: no sequence found")

    holo = bound_graph = pocket = None
    pocket_meta = {}
    input_hashes = {"apo_pdb": sha256_file(apo_path)}
    if holo_path and ligand_path:
        holo, _, _ = residue_graph(holo_path)
        bound_mol = load_bound_molecule(ligand_path)
        bound_graph = ligand_graph(bound_mol)
        pocket, pocket_meta = pocket_graph(holo_path, bound_mol, shells)
        input_hashes.update({"holo_pdb": sha256_file(holo_path),
                             "bound_ligand": sha256_file(ligand_path)})

    payload = {
        "format_version": 1,
        "sample_id": sample_id,
        "sequence_tokens": tokenize_sequence(sequence),
        "smiles": normalized_smiles,
        "label": float(record["label"]),
        "task": str(record.get("task", "affinity") or "affinity"),
        "apo": apo,
        "holo": holo,
        "free_ligand": ligand_graph(free_mol),
        "bound_ligand": bound_graph,
        "pocket": pocket,
        "metadata": {
            "mutation": record.get("mutation"),
            "source": record.get("source"),
            "input_hashes": input_hashes,
            "pocket": pocket_meta,
            "shells_angstrom": list(map(float, shells)),
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return {**record, "smiles": normalized_smiles,
            "processed_path": str(destination.relative_to(output_dir)), "status": "ok"}


def validate_manifest(frame: pd.DataFrame) -> None:
    required = {"sample_id", "smiles", "label", "apo_pdb"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")
    if frame["sample_id"].astype(str).duplicated().any():
        duplicates = frame.loc[frame["sample_id"].astype(str).duplicated(), "sample_id"].tolist()
        raise ValueError(f"Duplicate sample_id values: {duplicates[:10]}")
    labels = pd.to_numeric(frame["label"], errors="coerce")
    if labels.isna().any():
        raise ValueError("Every row must have a numeric label")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--shells", type=float, nargs=3, default=(4.0, 6.0, 8.0))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = Path(args.manifest).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(manifest)
    validate_manifest(frame)
    records = frame.to_dict("records")
    successes, failures = [], []

    kwargs = dict(manifest_dir=str(manifest.parent), output_dir=str(output), seed=args.seed,
                  shells=tuple(args.shells), overwrite=args.overwrite)
    if args.workers == 1:
        iterator = ((record, None) for record in records)
        for record, _ in tqdm(iterator, total=len(records), desc="preprocess"):
            try:
                successes.append(process_row(record, **kwargs))
            except Exception as exc:
                failures.append({"sample_id": record.get("sample_id"), "error": repr(exc),
                                 "traceback": traceback.format_exc()})
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_record = {executor.submit(process_row, record, **kwargs): record for record in records}
            for future in tqdm(as_completed(future_to_record), total=len(records), desc="preprocess"):
                record = future_to_record[future]
                try:
                    successes.append(future.result())
                except Exception as exc:
                    failures.append({"sample_id": record.get("sample_id"), "error": repr(exc),
                                     "traceback": traceback.format_exc()})

    index = pd.DataFrame(successes)
    if len(index):
        order = {str(v): i for i, v in enumerate(frame["sample_id"])}
        index["_order"] = index["sample_id"].astype(str).map(order)
        index.sort_values("_order").drop(columns="_order").to_csv(output / "index.csv", index=False)
    else:
        pd.DataFrame(columns=list(frame.columns) + ["processed_path", "status"]).to_csv(
            output / "index.csv", index=False)
    pd.DataFrame(failures).to_csv(output / "failures.csv", index=False)
    save_json({
        "manifest": str(manifest), "manifest_sha256": sha256_file(manifest),
        "seed": args.seed, "shells_angstrom": args.shells,
        "num_input": len(records), "num_success": len(successes), "num_failure": len(failures),
        "versions": package_versions(), "rdkit_version": rdBase.rdkitVersion,
    }, output / "preprocess_metadata.json")
    print(json.dumps({"success": len(successes), "failure": len(failures),
                      "index": str(output / "index.csv")}, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

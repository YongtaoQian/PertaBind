"""Virtual screening with the distilled, holo-independent PertaBind student."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import collate_samples
from model import build_model
from processdata import generate_conformer, ligand_graph, normalize_smiles
from utils import load_checkpoint, load_config, recursive_to, resolve_device, set_seed


def build_screen_sample(target: dict, compound_id: str, smiles: str, seed: int) -> dict:
    normalized, molecule = normalize_smiles(smiles)
    conformer = generate_conformer(molecule, seed)
    return {
        "sample_id": str(compound_id),
        "sequence_tokens": target["sequence_tokens"],
        "apo": target["apo"],
        "holo": None,
        "free_ligand": ligand_graph(conformer),
        "bound_ligand": None,
        "pocket": None,
        "label": 0.0,
        "task": target.get("task", "affinity"),
        "smiles": normalized,
        "metadata": {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--target", required=True,
                        help="A processed .pt sample supplying sequence and apo structure")
    parser.add_argument("--library", required=True,
                        help="CSV containing compound_id and smiles")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed = int(cfg["seed"])
    set_seed(seed, bool(cfg.get("deterministic", True)))
    device = resolve_device(cfg.get("device", "auto"))
    target = torch.load(args.target, map_location="cpu", weights_only=False)
    library = pd.read_csv(args.library)
    missing = {"compound_id", "smiles"} - set(library)
    if missing:
        raise ValueError(f"Library is missing columns: {sorted(missing)}")
    samples, failures = [], []
    for row_index, row in tqdm(library.iterrows(), total=len(library), desc="conformers"):
        try:
            samples.append(build_screen_sample(target, row.compound_id, row.smiles, seed + row_index))
        except Exception as exc:
            failures.append({"compound_id": row.compound_id, "smiles": row.smiles,
                             "error": repr(exc)})
    if not samples:
        raise RuntimeError("No compounds could be processed")
    batch_size = args.batch_size or int(cfg["stage2"]["batch_size"])
    loader = DataLoader(samples, batch_size=batch_size, shuffle=False, collate_fn=collate_samples)
    predictions = []
    for checkpoint_index, checkpoint in enumerate(args.checkpoints):
        model = build_model(cfg).to(device)
        load_checkpoint(checkpoint, model, map_location=device)
        model.eval()
        values = []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"model {checkpoint_index + 1}", leave=False):
                batch = recursive_to(batch, device)
                values.extend(model(batch, student_only=True)["student_prediction"].cpu().tolist())
        predictions.append(values)
    matrix = np.asarray(predictions).T
    result = pd.DataFrame({
        "compound_id": [s["sample_id"] for s in samples],
        "smiles": [s["smiles"] for s in samples],
        "score_mean": matrix.mean(1),
        "score_sd": matrix.std(1),
    })
    for i in range(matrix.shape[1]):
        result[f"model_{i}"] = matrix[:, i]
    # Higher pK is better for affinity; for a delta-delta-G model direction is use-dependent.
    ascending = cfg["data"].get("task", "affinity") == "ddg"
    result = result.sort_values("score_mean", ascending=ascending).reset_index(drop=True)
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    if failures:
        pd.DataFrame(failures).to_csv(destination.with_suffix(".failures.csv"), index=False)
    print(f"Ranked {len(result)} compounds; {len(failures)} failures; wrote {destination}")


if __name__ == "__main__":
    main()


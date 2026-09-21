"""Export teacher-side residue perturbation scores and shell summaries."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from dataset import collate_samples
from model import build_model
from utils import load_checkpoint, load_config, recursive_to, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample", required=True, help="Processed .pt with holo inputs")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg.get("device", "auto"))
    sample = torch.load(args.sample, map_location="cpu", weights_only=False)
    if sample.get("pocket") is None:
        raise ValueError("Teacher attribution requires holo protein and bound ligand inputs")
    batch = recursive_to(collate_samples([sample]), device)
    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()
    with torch.no_grad():
        output = model(batch)
    scores = output["residue_scores"].cpu().numpy()
    shells = output["residue_shell"].cpu().numpy()
    frame = pd.DataFrame({
        "pocket_residue_index": range(len(scores)),
        "shell": shells,
        "shell_name": [["direct_0_4A", "near_4_6A", "distal_6_8A"][int(v)] for v in shells],
        "perturbation_score": scores,
    })
    residue_metadata = sample.get("metadata", {}).get("pocket", {}).get("residues", [])
    if len(residue_metadata) == len(frame):
        annotations = pd.DataFrame(residue_metadata).drop(
            columns=["pocket_residue_index", "shell"], errors="ignore"
        )
        frame = pd.concat([frame, annotations], axis=1)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    print(f"Wrote {len(frame)} residue scores to {destination}")


if __name__ == "__main__":
    main()

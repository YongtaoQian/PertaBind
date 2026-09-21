"""Student-only predictions for an existing processed index."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import PertaBindDataset, collate_samples
from model import build_model
from utils import load_checkpoint, load_config, recursive_to, resolve_device, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]), bool(cfg.get("deterministic", True)))
    device = resolve_device(cfg.get("device", "auto"))
    dataset = PertaBindDataset(args.index)
    loader = DataLoader(dataset, batch_size=int(cfg["stage2"]["batch_size"]), shuffle=False,
                        num_workers=int(cfg["data"].get("num_workers", 0)),
                        collate_fn=collate_samples)
    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            batch = recursive_to(batch, device)
            prediction = model(batch, student_only=True)["student_prediction"].cpu().tolist()
            rows.extend({"sample_id": sample_id, "prediction": value}
                        for sample_id, value in zip(batch["sample_id"], prediction))
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination, index=False)
    print(f"Wrote {len(rows)} predictions to {destination}")


if __name__ == "__main__":
    main()


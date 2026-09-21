"""Evaluate one checkpoint or a five-fold model set on a fixed benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PertaBindDataset, collate_samples, subset_frame
from metric import all_metrics
from model import build_model
from utils import load_checkpoint, load_config, recursive_to, resolve_device, set_seed


@torch.no_grad()
def predict_model(model, loader, device):
    model.eval()
    ids, truth, predictions = [], [], []
    for batch in tqdm(loader, desc="evaluate", leave=False):
        batch = recursive_to(batch, device)
        output = model(batch, student_only=True)
        ids.extend(batch["sample_id"])
        truth.extend(batch["label"].cpu().numpy().tolist())
        predictions.extend(output["student_prediction"].cpu().numpy().tolist())
    return ids, np.asarray(truth), np.asarray(predictions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--fold", type=int, default=0,
                        help="Used only to select validation rows when --split=val")
    parser.add_argument("--index", help="Override data.index")
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]), bool(cfg.get("deterministic", True)))
    device = resolve_device(cfg.get("device", "auto"))
    index_path = Path(args.index or cfg["data"]["index"])
    index = pd.read_csv(index_path)
    folds_path = Path(cfg["data"]["folds"])
    folds = pd.read_csv(folds_path) if folds_path.exists() else None
    if args.split in ("train", "val"):
        frame = subset_frame(index, folds, args.fold, args.split)
    elif "split" in index:
        frame = index[index["split"] == args.split].reset_index(drop=True)
    else:
        frame = index
    if not len(frame):
        raise ValueError(f"No rows selected for split {args.split!r}")
    loader = DataLoader(PertaBindDataset(index_path, frame),
                        batch_size=int(cfg["stage2"]["batch_size"]), shuffle=False,
                        num_workers=int(cfg["data"].get("num_workers", 0)),
                        collate_fn=collate_samples)
    result = None
    model_metrics = []
    for model_index, checkpoint in enumerate(args.checkpoints):
        model = build_model(cfg).to(device)
        load_checkpoint(checkpoint, model, map_location=device)
        ids, truth, prediction = predict_model(model, loader, device)
        if result is None:
            result = pd.DataFrame({"sample_id": ids, "label": truth})
        elif ids != result["sample_id"].tolist() or not np.allclose(truth, result["label"]):
            raise RuntimeError("Model predictions are not aligned")
        column = f"model_{model_index}"
        result[column] = prediction
        metrics = all_metrics(truth, prediction, cfg["data"]["task"],
                              float(cfg["data"].get("resistance_threshold", 1.36)))
        model_metrics.append({"checkpoint": checkpoint, **metrics})
    model_columns = [c for c in result if c.startswith("model_")]
    result["prediction"] = result[model_columns].mean(axis=1)
    result["prediction_sd"] = result[model_columns].std(axis=1, ddof=0)
    destination = Path(args.predictions)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    summary = {
        "per_model": model_metrics,
        "paper_style_mean": {
            key: float(np.nanmean([m[key] for m in model_metrics]))
            for key in model_metrics[0] if key != "checkpoint"
        },
        "paper_style_sd": {
            key: float(np.nanstd([m[key] for m in model_metrics], ddof=0))
            for key in model_metrics[0] if key != "checkpoint"
        },
        "ensemble": all_metrics(result["label"], result["prediction"], cfg["data"]["task"],
                                float(cfg["data"].get("resistance_threshold", 1.36))),
    }
    with open(destination.with_suffix(".metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


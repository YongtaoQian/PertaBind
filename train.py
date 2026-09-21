"""Two-stage, fold-specific PertaBind training."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PertaBindDataset, collate_samples, subset_frame
from loss import PertaBindLoss
from metric import all_metrics
from model import build_model
from utils import (EarlyStopping, load_checkpoint, load_config, recursive_to,
                   resolve_device, save_checkpoint, save_json, set_seed)


def build_loader(dataset, batch_size: int, shuffle: bool, workers: int,
                 seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        collate_fn=collate_samples, pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0, generator=generator,
    )


def train_epoch(model, loader, objective, optimizer, device, stage: int,
                grad_clip: float) -> Dict[str, float]:
    model.train()
    totals = defaultdict(float)
    examples = 0
    for batch in tqdm(loader, desc=f"stage {stage} train", leave=False):
        batch = recursive_to(batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        losses = objective(output, batch["label"], batch["has_teacher"], stage=stage)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        count = len(batch["label"])
        examples += count
        for key, value in losses.items():
            totals[key] += float(value) * count
    return {key: value / max(examples, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate_loader(model, loader, device, task: str, threshold: float) -> Tuple[Dict, pd.DataFrame]:
    model.eval()
    ids, labels, predictions = [], [], []
    for batch in tqdm(loader, desc="validate", leave=False):
        batch = recursive_to(batch, device)
        output = model(batch, student_only=True)
        ids.extend(batch["sample_id"])
        labels.extend(batch["label"].cpu().tolist())
        predictions.extend(output["student_prediction"].cpu().tolist())
    metrics = all_metrics(labels, predictions, task, threshold)
    return metrics, pd.DataFrame({"sample_id": ids, "label": labels,
                                  "prediction": predictions})


def run_stage(model, stage: int, cfg: Dict, train_dataset, val_dataset, device,
              fold_dir: Path, start_epoch: int = 0) -> Path:
    stage_cfg = cfg[f"stage{stage}"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(stage_cfg["learning_rate"]),
        betas=tuple(map(float, cfg["optimizer"]["betas"])),
        weight_decay=float(stage_cfg.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=float(cfg["optimizer"].get("scheduler_factor", 0.5)),
        patience=int(cfg["optimizer"].get("scheduler_patience", 5)),
    )
    train_loader = build_loader(train_dataset, int(stage_cfg["batch_size"]), True,
                                int(cfg["data"].get("num_workers", 0)), cfg["seed"] + stage)
    val_loader = build_loader(val_dataset, int(stage_cfg["batch_size"]), False,
                              int(cfg["data"].get("num_workers", 0)), cfg["seed"] + 100 + stage)
    objective = PertaBindLoss(cfg)
    early = EarlyStopping(int(cfg["early_stopping"]["patience"]),
                          float(cfg["early_stopping"].get("min_delta", 0.0)))
    log_path = fold_dir / f"stage{stage}_history.csv"
    checkpoint = fold_dir / f"stage{stage}_best.pt"
    with open(log_path, "w", newline="", encoding="utf-8") as handle:
        writer = None
        for epoch in range(start_epoch, int(stage_cfg["epochs"])):
            train_values = train_epoch(
                model, train_loader, objective, optimizer, device, stage,
                float(cfg["optimizer"].get("grad_clip", 5.0)),
            )
            validation, predictions = evaluate_loader(
                model, val_loader, device, cfg["data"]["task"],
                float(cfg["data"].get("resistance_threshold", 1.36)),
            )
            scheduler.step(validation["rmse"])
            row = {"epoch": epoch, "stage": stage, "lr": optimizer.param_groups[0]["lr"],
                   **{f"train_{k}": v for k, v in train_values.items()},
                   **{f"val_{k}": v for k, v in validation.items()}}
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            improved = early.update(validation["rmse"])
            print(json.dumps(row))
            if improved:
                save_checkpoint(checkpoint, model, optimizer, epoch, cfg, validation, stage)
                predictions.to_csv(fold_dir / f"stage{stage}_validation_predictions.csv", index=False)
            if early.should_stop:
                break
    if not checkpoint.exists():
        raise RuntimeError(f"Training stage {stage} produced no checkpoint")
    load_checkpoint(checkpoint, model, map_location=device)
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--resume", help="Optional checkpoint to load before Stage I")
    parser.add_argument("--skip-stage1", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]), bool(cfg.get("deterministic", True)))
    device = resolve_device(cfg.get("device", "auto"))

    index_path = Path(cfg["data"]["index"])
    index = pd.read_csv(index_path)
    fold_path = Path(cfg["data"]["folds"])
    folds = pd.read_csv(fold_path) if fold_path.exists() else None
    train_frame = subset_frame(index, folds, args.fold, "train")
    val_frame = subset_frame(index, folds, args.fold, "val")
    if not len(train_frame) or not len(val_frame):
        raise ValueError(f"Empty train/validation split for fold {args.fold}")
    # Teacher supervision is mandatory during model development.
    train_dataset = PertaBindDataset(index_path, train_frame)
    val_dataset = PertaBindDataset(index_path, val_frame)
    model = build_model(cfg).to(device)
    if args.resume:
        load_checkpoint(args.resume, model, map_location=device, strict=True)

    fold_dir = Path(cfg["output_dir"]) / f"fold_{args.fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    save_json(cfg, fold_dir / "effective_config.json")
    save_json({"fold": args.fold, "train_samples": train_frame["sample_id"].astype(str).tolist(),
               "validation_samples": val_frame["sample_id"].astype(str).tolist()},
              fold_dir / "split_membership.json")
    if not args.skip_stage1:
        run_stage(model, 1, cfg, train_dataset, val_dataset, device, fold_dir)
    final = run_stage(model, 2, cfg, train_dataset, val_dataset, device, fold_dir)
    # Stable name expected by evaluation scripts.
    payload = torch.load(final, map_location="cpu", weights_only=False)
    torch.save(payload, fold_dir / "best.pt")
    print(f"Best student checkpoint: {fold_dir / 'best.pt'}")


if __name__ == "__main__":
    main()


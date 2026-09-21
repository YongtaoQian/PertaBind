"""Dataset loading and dependency-free batching of variable-sized molecular graphs."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset


class PertaBindDataset(Dataset):
    def __init__(self, index_csv: str | Path, rows: pd.DataFrame | None = None):
        self.index_path = Path(index_csv).resolve()
        frame = pd.read_csv(self.index_path) if rows is None else rows.copy()
        if "processed_path" not in frame:
            raise ValueError("index.csv must contain processed_path")
        self.frame = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> Dict:
        row = self.frame.iloc[index]
        path = Path(str(row.processed_path))
        if not path.is_absolute():
            path = self.index_path.parent / path
        sample = torch.load(path, map_location="cpu", weights_only=False)
        sample["_processed_path"] = str(path)
        return sample


def _batch_graph(graphs: Sequence[Dict | None]) -> Dict | None:
    present = [(i, graph) for i, graph in enumerate(graphs) if graph is not None]
    if not present:
        return None
    x, pos, batch, edge_index, edge_attr = [], [], [], [], []
    offset = 0
    extras: Dict[str, List[torch.Tensor]] = {}
    for sample_index, graph in present:
        n = graph["x"].shape[0]
        x.append(graph["x"].float())
        pos.append(graph["pos"].float())
        batch.append(torch.full((n,), sample_index, dtype=torch.long))
        if graph.get("edge_index") is not None and graph["edge_index"].numel():
            edge_index.append(graph["edge_index"].long() + offset)
            if graph.get("edge_attr") is not None:
                edge_attr.append(graph["edge_attr"].float())
        for key in ("node_type", "residue_index", "shell"):
            if key in graph:
                extras.setdefault(key, []).append(graph[key].long())
        offset += n
    result = {
        "x": torch.cat(x),
        "pos": torch.cat(pos),
        "batch": torch.cat(batch),
        "num_graphs": len(graphs),
        "edge_index": torch.cat(edge_index, dim=1) if edge_index else torch.empty((2, 0), dtype=torch.long),
    }
    if edge_attr:
        result["edge_attr"] = torch.cat(edge_attr)
    for key, values in extras.items():
        result[key] = torch.cat(values)
    return result


def collate_samples(samples: Sequence[Dict]) -> Dict:
    batch_size = len(samples)
    lengths = [min(int(s["sequence_tokens"].numel()), 100_000) for s in samples]
    max_length = max(lengths)
    sequence = torch.zeros((batch_size, max_length), dtype=torch.long)
    sequence_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    for i, sample in enumerate(samples):
        tokens = sample["sequence_tokens"][:max_length].long()
        sequence[i, : len(tokens)] = tokens
        sequence_mask[i, : len(tokens)] = True
    return {
        "sample_id": [str(s["sample_id"]) for s in samples],
        "sequence_tokens": sequence,
        "sequence_mask": sequence_mask,
        "apo": _batch_graph([s.get("apo") for s in samples]),
        "holo": _batch_graph([s.get("holo") for s in samples]),
        "free_ligand": _batch_graph([s.get("free_ligand") for s in samples]),
        "bound_ligand": _batch_graph([s.get("bound_ligand") for s in samples]),
        "pocket": _batch_graph([s.get("pocket") for s in samples]),
        "label": torch.tensor([float(s["label"]) for s in samples], dtype=torch.float32),
        "has_teacher": torch.tensor(
            [s.get("holo") is not None and s.get("pocket") is not None for s in samples],
            dtype=torch.bool,
        ),
        "task": [s.get("task", "affinity") for s in samples],
        "metadata": [s.get("metadata", {}) for s in samples],
    }


def subset_frame(index: pd.DataFrame, folds: pd.DataFrame | None, fold: int, split: str) -> pd.DataFrame:
    frame = index.copy()
    if folds is not None:
        keys = [c for c in ("sample_id", "fold", "split") if c in folds]
        if "sample_id" not in keys:
            raise ValueError("Fold table must include sample_id")
        frame = frame.merge(folds[keys], on="sample_id", how="left", suffixes=("", "_fold"))
    split_col = "split_fold" if "split_fold" in frame else "split"
    fold_col = "fold_fold" if "fold_fold" in frame else "fold"
    if split_col in frame and frame[split_col].notna().any():
        if split == "train" and fold_col in frame:
            # Rows assigned to this fold are validation; other development rows train.
            external = frame[split_col].isin(["test", "external_test"])
            return frame[(frame[fold_col] != fold) & ~external].reset_index(drop=True)
        if split == "val" and fold_col in frame:
            return frame[frame[fold_col] == fold].reset_index(drop=True)
        return frame[frame[split_col] == split].reset_index(drop=True)
    if fold_col not in frame:
        raise ValueError("Need either split or fold assignment")
    if split == "train":
        return frame[frame[fold_col] != fold].reset_index(drop=True)
    if split == "val":
        return frame[frame[fold_col] == fold].reset_index(drop=True)
    raise ValueError(f"No explicit {split!r} rows")


"""Synthetic end-to-end test; no structures, datasets or downloads are required."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset import collate_samples  # noqa: E402
from loss import PertaBindLoss  # noqa: E402
from model import build_model  # noqa: E402


def graph(nodes: int, feature_dim: int = 35):
    return {
        "x": torch.randn(nodes, feature_dim),
        "pos": torch.randn(nodes, 3),
        "edge_index": torch.empty((2, 0), dtype=torch.long),
    }


def pocket():
    result = graph(8)
    result.update({
        "node_type": torch.tensor([0, 0, 0, 0, 0, 0, 1, 1]),
        "residue_index": torch.tensor([0, 0, 1, 1, 2, 2, -1, -1]),
        "shell": torch.tensor([0, 0, 1, 1, 2, 2, -1, -1]),
    })
    return result


def sample(index: int):
    return {
        "sample_id": f"sample_{index}",
        "sequence_tokens": torch.randint(1, 22, (12 + index,)),
        "apo": graph(12 + index, 29),
        "holo": graph(12 + index, 29),
        "free_ligand": graph(5 + index),
        "bound_ligand": graph(5 + index),
        "pocket": pocket(),
        "label": 6.5 + index,
        "task": "affinity",
    }


def configuration():
    return {
        "model": {
            "hidden_dim": 32, "num_rbf": 8, "protein_layers": 1,
            "ligand_layers": 1, "atom_layers": 1, "sequence_layers": 1,
            "sequence_heads": 4, "dropout": 0.0, "protein_cutoff": 12.0,
            "atom_cutoff": 5.0, "shells": [4.0, 6.0, 8.0],
        },
        "loss": {
            "lambda_rank": 0.2, "lambda_align": 0.1, "lambda_distill": 0.5,
            "lambda_teacher": 1.0, "distill_prediction_weight": 1.0,
            "distill_representation_weight": 1.0, "ranking_margin": 0.0,
        },
    }


def test_teacher_student_forward_backward():
    batch = collate_samples([sample(0), sample(1)])
    model = build_model(configuration())
    output = model(batch)
    assert output["student_prediction"].shape == (2,)
    assert output["teacher_prediction"].shape == (2,)
    assert output["residue_scores"].shape == (6,)
    losses = PertaBindLoss(configuration())(output, batch["label"], batch["has_teacher"], 2)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_student_only_without_holo():
    item = sample(0)
    item.update({"holo": None, "bound_ligand": None, "pocket": None})
    batch = collate_samples([item])
    output = build_model(configuration())(batch, student_only=True)
    assert output["student_prediction"].shape == (1,)
    assert "teacher_prediction" not in output


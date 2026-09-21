"""Independent reconstruction of the PertaBind teacher-student architecture.

The module maps manuscript equations 1-18 to explicit PyTorch components. It uses only
core PyTorch and supports variable-size protein, ligand and pocket graphs.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = src.new_zeros((dim_size,) + src.shape[1:])
    if src.numel():
        out.index_add_(0, index, src)
    return out


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    total = scatter_sum(src, index, dim_size)
    count = scatter_sum(src.new_ones((src.shape[0], 1)), index, dim_size).clamp_min(1.0)
    return total / count


def gated_pool(x: torch.Tensor, batch: torch.Tensor, num_graphs: int,
               gate: nn.Module) -> torch.Tensor:
    weights = torch.sigmoid(gate(x))
    return scatter_sum(weights * x, batch, num_graphs) / scatter_sum(
        weights, batch, num_graphs
    ).clamp_min(1e-6)


def radius_edges(pos: torch.Tensor, batch: torch.Tensor, cutoff: float,
                 max_neighbors: int = 64) -> torch.Tensor:
    """Deterministic radius graph; intended for pocket-sized graphs."""
    sources, targets = [], []
    for graph_index in torch.unique(batch, sorted=True).tolist():
        nodes = torch.nonzero(batch == graph_index, as_tuple=False).flatten()
        if len(nodes) < 2:
            continue
        distances = torch.cdist(pos[nodes], pos[nodes])
        distances.fill_diagonal_(float("inf"))
        k = min(max_neighbors, max(1, len(nodes) - 1))
        values, neighbor = torch.topk(distances, k=k, dim=1, largest=False)
        keep = values <= cutoff
        row = torch.arange(len(nodes), device=pos.device)[:, None].expand_as(neighbor)
        targets.append(nodes[row[keep]])
        sources.append(nodes[neighbor[keep]])
    if not sources:
        return torch.empty((2, 0), dtype=torch.long, device=pos.device)
    return torch.stack((torch.cat(sources), torch.cat(targets)), dim=0)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 dropout: float = 0.0, layers: int = 2):
        super().__init__()
        modules: List[nn.Module] = []
        dim = input_dim
        for _ in range(layers - 1):
            modules.extend([nn.Linear(dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout)])
            dim = hidden_dim
        modules.append(nn.Linear(dim, output_dim))
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


class GaussianRBF(nn.Module):
    def __init__(self, num_rbf: int, cutoff: float):
        super().__init__()
        centers = torch.linspace(0.0, cutoff, num_rbf)
        self.register_buffer("centers", centers)
        self.gamma = float(num_rbf) / max(cutoff, 1e-6)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.gamma * (distance[:, None] - self.centers[None, :]).square())


class GeometricMessageLayer(nn.Module):
    """Distance-message layer with an optional E(n)-equivariant coordinate update."""
    def __init__(self, hidden_dim: int, num_rbf: int, cutoff: float, dropout: float,
                 update_coordinates: bool = False):
        super().__init__()
        self.cutoff = cutoff
        self.rbf = GaussianRBF(num_rbf, cutoff)
        self.message = MLP(2 * hidden_dim + num_rbf, hidden_dim, hidden_dim, dropout)
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.update_coordinates = update_coordinates
        self.coordinate_weight = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                                               nn.Linear(hidden_dim, 1), nn.Tanh())

    def forward(self, h: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor,
                edge_index: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if edge_index is None or not edge_index.numel():
            edge_index = radius_edges(pos, batch, self.cutoff)
        if not edge_index.numel():
            return h, pos
        source, target = edge_index
        difference = pos[target] - pos[source]
        distance = difference.norm(dim=-1).clamp_min(1e-8)
        message = self.message(torch.cat([h[source], h[target], self.rbf(distance)], dim=-1))
        aggregate = scatter_mean(message, target, h.shape[0])
        h_new = self.norm(h + self.update(torch.cat([h, aggregate], dim=-1)))
        if self.update_coordinates:
            scalar = self.coordinate_weight(message) / (1.0 + distance[:, None])
            displacement = scatter_mean(scalar * difference, target, h.shape[0])
            pos = pos + 0.1 * displacement
        return h_new, pos


class GeometricGraphEncoder(nn.Module):
    def __init__(self, hidden_dim: int, layers: int, num_rbf: int, cutoff: float,
                 dropout: float, update_coordinates: bool = False):
        super().__init__()
        self.input = nn.Sequential(nn.LazyLinear(hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.layers = nn.ModuleList([
            GeometricMessageLayer(hidden_dim, num_rbf, cutoff, dropout, update_coordinates)
            for _ in range(layers)
        ])
        self.gate = nn.Linear(hidden_dim, 1)

    def forward(self, graph: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.input(graph["x"])
        pos = graph["pos"]
        edges = graph.get("edge_index")
        # Explicit covalent edges are used on the first pass; radius edges add geometry later.
        for index, layer in enumerate(self.layers):
            h, pos = layer(h, pos, graph["batch"], edges if index == 0 else None)
        pooled = gated_pool(h, graph["batch"], int(graph["num_graphs"]), self.gate)
        return pooled, h, pos


class BondMessageLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.message = MLP(2 * hidden_dim, hidden_dim, hidden_dim, dropout)
        self.update = MLP(2 * hidden_dim, hidden_dim, hidden_dim, dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, edge_index):
        if edge_index is None or not edge_index.numel():
            return h
        source, target = edge_index
        messages = self.message(torch.cat([h[source], h[target]], dim=-1))
        aggregate = scatter_mean(messages, target, h.shape[0])
        return self.norm(h + self.update(torch.cat([h, aggregate], dim=-1)))


class BondGraphEncoder(nn.Module):
    def __init__(self, hidden_dim: int, layers: int, dropout: float):
        super().__init__()
        self.input = nn.Sequential(nn.LazyLinear(hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.layers = nn.ModuleList([BondMessageLayer(hidden_dim, dropout) for _ in range(layers)])
        self.gate = nn.Linear(hidden_dim, 1)

    def forward(self, graph: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.input(graph["x"])
        for layer in self.layers:
            h = layer(h, graph.get("edge_index"))
        pooled = gated_pool(h, graph["batch"], int(graph["num_graphs"]), self.gate)
        return pooled, h


class SequenceEncoder(nn.Module):
    def __init__(self, hidden_dim: int, layers: int, heads: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(23, hidden_dim, padding_idx=0)
        block = nn.TransformerEncoderLayer(
            hidden_dim, heads, 4 * hidden_dim, dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(self.embedding(tokens), src_key_padding_mask=~mask)
        h = self.norm(h)
        pooled = (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return pooled, h


class Fusion(nn.Module):
    def __init__(self, hidden_dim: int, inputs: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inputs * hidden_dim, 2 * hidden_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
        )

    def forward(self, *values):
        return self.net(torch.cat(values, dim=-1))


class MultiShellPerturbation(nn.Module):
    """Equations 9-12: shell adaptation, propagation, scores and weighted fusion."""
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.shell_embedding = nn.Embedding(3, hidden_dim)
        self.shared = MLP(2 * hidden_dim, hidden_dim, hidden_dim, dropout)
        self.adapters = nn.ModuleList([MLP(hidden_dim, hidden_dim, hidden_dim, dropout)
                                       for _ in range(3)])
        self.propagate_2_to_1 = MLP(2 * hidden_dim, hidden_dim, hidden_dim, dropout)
        self.propagate_3_to_2 = MLP(2 * hidden_dim, hidden_dim, hidden_dim, dropout)
        self.score = nn.ModuleList([MLP(3 * hidden_dim, hidden_dim, 1, dropout)
                                    for _ in range(3)])
        self.fuse = Fusion(hidden_dim, 3, dropout)

    def forward(self, residue_h: torch.Tensor, residue_batch: torch.Tensor,
                residue_shell: torch.Tensor, protein_shift: torch.Tensor,
                num_graphs: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shell_context = self.shell_embedding(residue_shell.clamp(0, 2))
        base = self.shared(torch.cat([residue_h, shell_context], dim=-1))
        adapted = base.new_zeros(base.shape)
        for shell in range(3):
            mask = residue_shell == shell
            adapted[mask] = self.adapters[shell](base[mask])

        summary = base.new_zeros((num_graphs, 3, base.shape[-1]))
        for shell in range(3):
            mask = residue_shell == shell
            if mask.any():
                summary[:, shell] = scatter_mean(adapted[mask], residue_batch[mask], num_graphs)
        shell_3 = summary[:, 2]
        shell_2 = summary[:, 1] + self.propagate_3_to_2(
            torch.cat([summary[:, 1], shell_3], dim=-1))
        shell_1 = summary[:, 0] + self.propagate_2_to_1(
            torch.cat([summary[:, 0], shell_2], dim=-1))
        summary = torch.stack([shell_1, shell_2, shell_3], dim=1)

        scores = base.new_zeros((len(base),))
        weights = base.new_zeros((len(base),))
        for shell in range(3):
            mask = residue_shell == shell
            if not mask.any():
                continue
            context = summary[residue_batch[mask], shell]
            u = torch.cat([adapted[mask], context, protein_shift[residue_batch[mask]]], dim=-1)
            shell_scores = torch.sigmoid(self.score[shell](u)).squeeze(-1)
            scores[mask] = shell_scores
            # Normalize within each sample and shell, as equation 12 requires.
            for graph_index in torch.unique(residue_batch[mask], sorted=True).tolist():
                group_mask = mask & (residue_batch == graph_index)
                weights[group_mask] = torch.softmax(scores[group_mask], dim=0)

        weighted_shells = []
        for shell in range(3):
            mask = residue_shell == shell
            if mask.any():
                weighted_shells.append(scatter_sum(
                    weights[mask, None] * adapted[mask], residue_batch[mask], num_graphs
                ))
            else:
                weighted_shells.append(base.new_zeros((num_graphs, base.shape[-1])))
        perturbation = self.fuse(*weighted_shells)
        return perturbation, summary.reshape(num_graphs, -1), scores


def pool_pocket_atoms_to_residues(node_h: torch.Tensor, pocket: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    residue_h, residue_batch, residue_shell = [], [], []
    for graph_index in range(int(pocket["num_graphs"])):
        mask = (pocket["batch"] == graph_index) & (pocket["node_type"] == 0)
        if not mask.any():
            continue
        local_index = pocket["residue_index"][mask]
        valid = local_index >= 0
        local_index = local_index[valid]
        local_h = node_h[mask][valid]
        local_shell = pocket["shell"][mask][valid]
        unique, inverse = torch.unique(local_index, sorted=True, return_inverse=True)
        pooled = scatter_mean(local_h, inverse, len(unique))
        shell = torch.empty(len(unique), dtype=torch.long, device=node_h.device)
        for i in range(len(unique)):
            shell[i] = local_shell[inverse == i][0]
        residue_h.append(pooled)
        residue_batch.append(torch.full((len(unique),), graph_index, device=node_h.device,
                                        dtype=torch.long))
        residue_shell.append(shell)
    if not residue_h:
        return (node_h.new_zeros((0, node_h.shape[-1])),
                torch.empty(0, dtype=torch.long, device=node_h.device),
                torch.empty(0, dtype=torch.long, device=node_h.device))
    return torch.cat(residue_h), torch.cat(residue_batch), torch.cat(residue_shell)


class PertaBind(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        cfg = config["model"] if "model" in config else config
        h = int(cfg["hidden_dim"])
        dropout = float(cfg.get("dropout", 0.1))
        num_rbf = int(cfg.get("num_rbf", 24))
        protein_layers = int(cfg.get("protein_layers", 4))
        ligand_layers = int(cfg.get("ligand_layers", 4))
        atom_layers = int(cfg.get("atom_layers", 4))
        self.hidden_dim = h

        self.sequence_encoder = SequenceEncoder(h, int(cfg.get("sequence_layers", 2)),
                                                int(cfg.get("sequence_heads", 8)), dropout)
        self.apo_encoder = GeometricGraphEncoder(h, protein_layers, num_rbf,
                                                 float(cfg.get("protein_cutoff", 12.0)), dropout)
        self.holo_encoder = GeometricGraphEncoder(h, protein_layers, num_rbf,
                                                  float(cfg.get("protein_cutoff", 12.0)), dropout)
        self.ligand_2d_encoder = BondGraphEncoder(h, ligand_layers, dropout)
        self.free_ligand_encoder = GeometricGraphEncoder(h, ligand_layers, num_rbf, 6.0, dropout)
        self.bound_ligand_encoder = GeometricGraphEncoder(h, ligand_layers, num_rbf, 6.0, dropout)
        self.pocket_encoder = GeometricGraphEncoder(
            h, atom_layers, num_rbf, float(cfg.get("atom_cutoff", 5.0)), dropout,
            update_coordinates=True,
        )
        self.protein_fusion = Fusion(h, 2, dropout)  # equation 1
        self.ligand_fusion = Fusion(h, 2, dropout)   # equation 2
        self.apo_alignment = nn.Linear(h, h)
        self.holo_alignment = nn.Linear(h, h)
        self.perturbation = MultiShellPerturbation(h, dropout)

        self.student_encoder = Fusion(h, 2, dropout)  # equation 14
        self.teacher_encoder = Fusion(h, 7, dropout)  # equation 13: shell contributes 3h
        self.student_head = MLP(h, h, 1, dropout, layers=3)
        self.teacher_global_head = MLP(2 * h, h, 1, dropout)
        self.teacher_local_head = MLP(2 * h, h, 1, dropout)
        self.teacher_pert_head = MLP(2 * h, h, 1, dropout)
        self.student_distill_projection = nn.Linear(h, h)
        self.teacher_distill_projection = nn.Linear(h, h)

    def student_forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        sequence, residue_sequence = self.sequence_encoder(batch["sequence_tokens"],
                                                           batch["sequence_mask"])
        apo, apo_nodes, _ = self.apo_encoder(batch["apo"])
        ligand_2d, ligand_2d_nodes = self.ligand_2d_encoder(batch["free_ligand"])
        free_3d, free_nodes, _ = self.free_ligand_encoder(batch["free_ligand"])
        protein_global = self.protein_fusion(sequence, apo)
        ligand_prior = self.ligand_fusion(ligand_2d, free_3d)
        student = self.student_encoder(protein_global, ligand_prior)
        prediction = self.student_head(student).squeeze(-1)
        return {
            "prediction": prediction,
            "student": student,
            "protein_global": protein_global,
            "ligand_prior": ligand_prior,
            "apo": apo,
            "free_3d": free_3d,
            "sequence_nodes": residue_sequence,
            "apo_nodes": apo_nodes,
            "ligand_nodes": free_nodes,
        }

    def forward(self, batch: Dict, student_only: bool = False) -> Dict[str, torch.Tensor]:
        result = self.student_forward(batch)
        result["student_prediction"] = result.pop("prediction")
        if student_only or batch.get("holo") is None or batch.get("pocket") is None:
            return result

        holo, holo_nodes, _ = self.holo_encoder(batch["holo"])
        bound, bound_nodes, _ = self.bound_ligand_encoder(batch["bound_ligand"])
        pocket_global, pocket_nodes, pocket_positions = self.pocket_encoder(batch["pocket"])
        protein_shift = holo - result["apo"]                         # equation 7
        ligand_shift = bound - result["free_3d"]                     # equation 8
        residue_h, residue_batch, residue_shell = pool_pocket_atoms_to_residues(
            pocket_nodes, batch["pocket"]
        )
        perturbation, shell_summary, residue_scores = self.perturbation(
            residue_h, residue_batch, residue_shell, protein_shift,
            int(batch["pocket"]["num_graphs"]),
        )
        teacher = self.teacher_encoder(
            holo, pocket_global, perturbation, bound,
            shell_summary[:, : self.hidden_dim],
            shell_summary[:, self.hidden_dim: 2 * self.hidden_dim],
            shell_summary[:, 2 * self.hidden_dim:],
        )
        # Equation 17 decomposition. Its sum is the teacher prediction.
        global_readout = self.teacher_global_head(torch.cat([holo, bound], -1)).squeeze(-1)
        local_readout = self.teacher_local_head(torch.cat([pocket_global, bound], -1)).squeeze(-1)
        perturb_readout = self.teacher_pert_head(
            torch.cat([perturbation, protein_shift + ligand_shift], -1)
        ).squeeze(-1)
        teacher_prediction = global_readout + local_readout + perturb_readout
        result.update({
            "teacher_prediction": teacher_prediction,
            "teacher": teacher,
            "teacher_components": torch.stack(
                [global_readout, local_readout, perturb_readout], dim=-1
            ),
            "apo_aligned": self.apo_alignment(result["apo"]),
            "holo_aligned": self.holo_alignment(holo),
            "student_distill": self.student_distill_projection(result["student"]),
            "teacher_distill": self.teacher_distill_projection(teacher),
            "protein_shift": protein_shift,
            "ligand_shift": ligand_shift,
            "perturbation": perturbation,
            "shell_summary": shell_summary,
            "residue_scores": residue_scores,
            "residue_batch": residue_batch,
            "residue_shell": residue_shell,
            "holo_nodes": holo_nodes,
            "bound_nodes": bound_nodes,
            "pocket_positions": pocket_positions,
        })
        return result


def build_model(config: Dict) -> PertaBind:
    return PertaBind(config)

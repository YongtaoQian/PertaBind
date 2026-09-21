"""PertaBind objective corresponding to manuscript equations 18 and 19."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def pairwise_ranking_loss(prediction: torch.Tensor, target: torch.Tensor,
                          margin: float = 0.0) -> torch.Tensor:
    if len(target) < 2:
        return prediction.sum() * 0.0
    delta_target = target[:, None] - target[None, :]
    delta_prediction = prediction[:, None] - prediction[None, :]
    upper = torch.triu(torch.ones_like(delta_target, dtype=torch.bool), diagonal=1)
    comparable = upper & (delta_target.abs() > 1e-8)
    if not comparable.any():
        return prediction.sum() * 0.0
    direction = delta_target[comparable].sign()
    return F.relu(margin - direction * delta_prediction[comparable]).mean()


class PertaBindLoss:
    def __init__(self, config: Dict):
        cfg = config["loss"] if "loss" in config else config
        self.lambda_rank = float(cfg.get("lambda_rank", 0.2))
        self.lambda_align = float(cfg.get("lambda_align", 0.1))
        self.lambda_distill = float(cfg.get("lambda_distill", 0.5))
        self.lambda_teacher = float(cfg.get("lambda_teacher", 1.0))
        self.prediction_weight = float(cfg.get("distill_prediction_weight", 1.0))
        self.representation_weight = float(cfg.get("distill_representation_weight", 1.0))
        self.ranking_margin = float(cfg.get("ranking_margin", 0.0))

    def __call__(self, output: Dict[str, torch.Tensor], target: torch.Tensor,
                 has_teacher: torch.Tensor, stage: int = 2) -> Dict[str, torch.Tensor]:
        student = output["student_prediction"]
        affinity_student = F.mse_loss(student, target)
        ranking = pairwise_ranking_loss(student, target, self.ranking_margin)
        zero = affinity_student.detach() * 0.0
        teacher_affinity = zero
        alignment = zero
        distillation_prediction = zero
        distillation_representation = zero

        teacher_mask = has_teacher & torch.isfinite(target)
        if "teacher_prediction" in output and teacher_mask.any():
            teacher_prediction = output["teacher_prediction"][teacher_mask]
            teacher_target = target[teacher_mask]
            teacher_affinity = F.mse_loss(teacher_prediction, teacher_target)
            alignment = F.mse_loss(
                output["apo_aligned"][teacher_mask],
                output["holo_aligned"][teacher_mask],
            )
            distillation_prediction = F.l1_loss(
                student[teacher_mask], output["teacher_prediction"][teacher_mask].detach()
            )
            distillation_representation = F.mse_loss(
                output["student_distill"][teacher_mask],
                output["teacher_distill"][teacher_mask].detach(),
            )

        if stage == 1:
            total = affinity_student + self.lambda_teacher * teacher_affinity + self.lambda_align * alignment
            distillation = zero
        elif stage == 2:
            distillation = (self.prediction_weight * distillation_prediction +
                            self.representation_weight * distillation_representation)
            total = (affinity_student + self.lambda_teacher * teacher_affinity +
                     self.lambda_rank * ranking + self.lambda_align * alignment +
                     self.lambda_distill * distillation)
        else:
            raise ValueError(f"stage must be 1 or 2, got {stage}")
        return {
            "loss": total,
            "affinity_student": affinity_student.detach(),
            "affinity_teacher": teacher_affinity.detach(),
            "ranking": ranking.detach(),
            "alignment": alignment.detach(),
            "distillation": distillation.detach(),
            "distillation_prediction": distillation_prediction.detach(),
            "distillation_representation": distillation_representation.detach(),
        }

